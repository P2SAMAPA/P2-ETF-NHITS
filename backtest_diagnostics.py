"""
backtest_diagnostics.py — Diagnostic Validity Study for the N-HiTS Engine
============================================================================

QUESTION THIS ANSWERS
----------------------
Of the two confidence diagnostics N-HiTS reports (fit_quality, trend_consistency),
which one better identifies moments where path_signal is more aligned with what
actually happened?

METHODOLOGY — READ BEFORE INTERPRETING RESULTS
------------------------------------------------
For each (ticker, window), a model is trained EXACTLY as trainer.py does today
— same data prep, same architecture, same training loop, reusing
nhits_engine.prepare_ticker_training_data() and _train_nhits() directly so
there is no risk of this script's logic drifting from production.

The trained model is then evaluated, IN A SINGLE VECTORIZED FORWARD PASS,
across every timestep in its own training window (not retrained at each
historical date). This is an IN-SAMPLE diagnostic-validity check: it asks
"conditional on how fit_quality/trend_consistency looked at time t, was
path_signal(t) more or less aligned with the realized return at t+H?" —
that question does not require walk-forward retraining to answer honestly.

This is NOT a claim about the engine's out-of-sample predictive skill.
A true walk-forward test (retrain using only data available as of each
historical date) would answer a different, considerably more expensive
question and is not what this script does.

OVERLAP CAVEAT
--------------
Consecutive training rows share most of their target window (each Y_t
overlaps Y_{t+1} in H-1 of its H days), so raw row-by-row correlations are
autocorrelated and will look more significant than they are. This script
reports both:
  - STRIDE-H (non-overlapping target windows): the more defensible number
  - ALL-ROWS (full overlap): larger sample, upward-biased — for reference only
Treat STRIDE-H as the primary result.

OUTPUT
------
- results/diagnostic_backtest_raw.csv     — every (ticker, window, t) row
- results/diagnostic_backtest_report.md   — tercile bucket tables + verdict
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config
import data_manager
from nhits_engine import prepare_ticker_training_data, _train_nhits, NHiTS


def evaluate_model_insample(model: NHiTS, X_norm: np.ndarray, Y_norm: np.ndarray,
                              mu: float, sd: float) -> pd.DataFrame:
    """Vectorized evaluation of an already-trained model across its own
    training set. Returns one row per timestep with signal + diagnostics."""
    pred_norm = model.forward(X_norm)                # (N, H)
    final_residual = model._final_residual            # (N, L)

    forecast_path = pred_norm * sd + mu                # (N, H)
    path_signal = forecast_path.mean(axis=1)           # (N,)
    sign = np.where(path_signal == 0, 1.0, np.sign(path_signal))
    trend_consistency = (np.sign(forecast_path) == sign[:, None]).mean(axis=1)

    input_energy = np.mean(X_norm ** 2, axis=1) + 1e-8
    resid_energy = np.mean(final_residual ** 2, axis=1)
    fit_quality = np.clip(1.0 - resid_energy / input_energy, 0.0, 1.0)

    realized_return = Y_norm.mean(axis=1) * sd + mu    # (N,) raw-return units, same scale as path_signal

    return pd.DataFrame({
        "path_signal": path_signal,
        "trend_consistency": trend_consistency,
        "fit_quality": fit_quality,
        "realized_return": realized_return,
    })


def run_backtest(prices: pd.DataFrame, tickers: list, windows: list,
                  universe_label: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    all_rows = []

    for ticker in tickers:
        for window in windows:
            prepped = prepare_ticker_training_data(prices, ticker, window)
            if prepped is None:
                continue
            X_norm, Y_norm, mu, sd = prepped

            print(f"  Training {ticker} @ {window}d (N={len(X_norm)}) ...")
            try:
                model = _train_nhits(X_norm, Y_norm, rng)
            except Exception as e:
                print(f"    failed: {e}")
                continue

            df = evaluate_model_insample(model, X_norm, Y_norm, mu, sd)
            df["ticker"] = ticker
            df["window"] = window
            df["universe"] = universe_label
            df["t_idx"] = np.arange(len(df))
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table formatter — avoids adding a tabulate dependency."""
    if df.empty:
        return "_(no data)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def tercile_ic_table(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """IC (Pearson corr of path_signal vs realized_return) and hit-rate
    within each tercile of bucket_col."""
    d = df.copy()
    try:
        d["tercile"] = pd.qcut(d[bucket_col], 3, labels=["Low", "Mid", "High"], duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows = []
    for label, grp in d.groupby("tercile", observed=True):
        if len(grp) < 5:
            continue
        ic = grp["path_signal"].corr(grp["realized_return"])
        hit_rate = (np.sign(grp["path_signal"]) == np.sign(grp["realized_return"])).mean()
        rows.append({
            "Tercile": label, "N": len(grp),
            f"Mean {bucket_col}": grp[bucket_col].mean(),
            "IC (corr)": ic, "Hit Rate": hit_rate,
        })
    return pd.DataFrame(rows)


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set — cannot load master data"); return

    df_master = data_manager.load_master_data()

    universes_to_run = ["FI_COMMODITIES", "EQUITY_SECTORS"]
    windows = config.WINDOWS

    combined = []
    for uni in universes_to_run:
        tickers = config.UNIVERSES[uni]
        prices = data_manager.prepare_prices(df_master, tickers)
        available = [t for t in tickers if t in prices.columns]
        print(f"\n=== {uni}: {len(available)} tickers ===")
        res = run_backtest(prices, available, windows, uni)
        if not res.empty:
            combined.append(res)

    if not combined:
        print("No results produced — insufficient data.")
        return

    raw = pd.concat(combined, ignore_index=True)

    Path("results").mkdir(exist_ok=True)
    raw.to_csv("results/diagnostic_backtest_raw.csv", index=False)
    print(f"\nSaved {len(raw)} rows to results/diagnostic_backtest_raw.csv")

    # ── Primary: non-overlapping rows (stride = H) ─────────────────────────────
    H = config.PRED_HORIZON
    stride_rows = []
    for (ticker, window), grp in raw.groupby(["ticker", "window"]):
        stride_rows.append(grp.iloc[::H])
    stride_df = pd.concat(stride_rows, ignore_index=True)

    fit_table_stride    = tercile_ic_table(stride_df, "fit_quality")
    trend_table_stride  = tercile_ic_table(stride_df, "trend_consistency")

    # ── Reference: all overlapping rows ────────────────────────────────────────
    fit_table_all   = tercile_ic_table(raw, "fit_quality")
    trend_table_all = tercile_ic_table(raw, "trend_consistency")

    report_lines = []
    report_lines.append("# N-HiTS Diagnostic Validity Backtest\n")
    report_lines.append(f"Total rows collected: {len(raw)} (all overlapping) / "
                         f"{len(stride_df)} (stride-{H}, non-overlapping)\n")

    report_lines.append("\n## PRIMARY RESULT — stride-H (non-overlapping), fit_quality terciles\n")
    report_lines.append(fit_table_stride.pipe(df_to_markdown))
    report_lines.append("\n\n## PRIMARY RESULT — stride-H (non-overlapping), trend_consistency terciles\n")
    report_lines.append(trend_table_stride.pipe(df_to_markdown))

    report_lines.append("\n\n## Reference only — all overlapping rows, fit_quality terciles\n")
    report_lines.append(fit_table_all.pipe(df_to_markdown))
    report_lines.append("\n\n## Reference only — all overlapping rows, trend_consistency terciles\n")
    report_lines.append(trend_table_all.pipe(df_to_markdown))

    # ── Verdict: compare High-tercile IC spread (High IC - Low IC) ─────────────
    def ic_spread(table):
        if table.empty or "High" not in table["Tercile"].values or "Low" not in table["Tercile"].values:
            return None
        hi = table.loc[table["Tercile"] == "High", "IC (corr)"].values[0]
        lo = table.loc[table["Tercile"] == "Low", "IC (corr)"].values[0]
        return hi - lo

    fit_spread   = ic_spread(fit_table_stride)
    trend_spread = ic_spread(trend_table_stride)

    report_lines.append("\n\n## Verdict\n")
    report_lines.append(f"- fit_quality High-vs-Low IC spread: {fit_spread}")
    report_lines.append(f"- trend_consistency High-vs-Low IC spread: {trend_spread}")
    if fit_spread is not None and trend_spread is not None:
        winner = "fit_quality" if abs(fit_spread) > abs(trend_spread) else "trend_consistency"
        report_lines.append(f"\n**Larger spread (more discriminating diagnostic): {winner}**")
    report_lines.append(
        "\n\nA large POSITIVE spread means the High tercile of that diagnostic has "
        "meaningfully higher IC than the Low tercile — i.e. that diagnostic is doing "
        "real work identifying when path_signal should be trusted. A spread near zero "
        "means that diagnostic doesn't discriminate skill, regardless of how the raw "
        "IC values look in isolation."
    )

    report_text = "\n".join(str(l) for l in report_lines)
    with open("results/diagnostic_backtest_report.md", "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print("\nSaved report to results/diagnostic_backtest_report.md")


if __name__ == "__main__":
    main()
