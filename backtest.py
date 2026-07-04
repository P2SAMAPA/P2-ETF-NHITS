"""
backtest.py — Which diagnostic actually predicts returns: fit_quality or
trend_consistency?

Walks forward through history for each ticker, retraining N-HiTS at each
step using ONLY data available up to that point (prices are truncated —
no lookahead into the future), records the forecast diagnostics, then
checks them against the REALIZED forward return over the next
PRED_HORIZON trading days once it's actually known.

Buckets results into terciles of fit_quality and, separately, terciles of
trend_consistency, and reports within each bucket:
  - hit rate         : fraction of times sign(path_signal) == sign(realized return)
  - correlation      : corr(path_signal, realized_forward_return)
  - long-short spread: mean realized return when path_signal>0 minus when path_signal<0

This answers the question empirically instead of relying on architectural
reasoning about what each diagnostic *should* mean.

Usage:
    python backtest.py --universe EQUITY_SECTORS --window 252 --step 15
    python backtest.py --universe FI_COMMODITIES --window 252 --step 10 --tickers TLT GLD

Note: this retrains a full N-HiTS model at every walk-forward point, so
runtime scales with (number of tickers) x (history length / step). Use
--tickers to test a subset first, and a larger --step for a full-universe
run. Training hyperparameters are the same as production (config.py) so
the result reflects what the live engine actually does.

Results are saved locally to backtest_results.csv (full row-level detail)
and a summary is pushed to HF as walkforward_backtest_{date}.json for the
"🎯 Walk-Forward Validation" dashboard tab — the true out-of-sample
counterpart to the in-sample "📊 Diagnostic Validity" tab.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data_manager
import push_results
from nhits_engine import forecast_and_diagnose


def walk_forward_ticker(prices_full: pd.DataFrame, ticker: str, window: int,
                         step: int, rng: np.random.Generator) -> list:
    """
    Walk forward through history for one ticker, collecting diagnostics +
    realized forward return at each as-of point spaced `step` trading days
    apart. At each point, N-HiTS is trained using only prices truncated up
    to that date (no lookahead) — the identical code path production uses.
    """
    L, H = config.NHITS_LOOKBACK, config.PRED_HORIZON
    ps_full = prices_full[ticker].dropna()
    n_total = len(ps_full)

    min_needed = window + H + L + config.NHITS_BATCH_SIZE * 2 + 5
    if n_total < min_needed + H + 1:
        return []

    records = []
    start_idx = min_needed
    end_idx = n_total - H - 1   # need H future days available to score the forecast

    for as_of_idx in range(start_idx, end_idx, step):
        as_of_date = ps_full.index[as_of_idx]
        prices_upto = prices_full.loc[:as_of_date]

        diag = forecast_and_diagnose(prices_upto, ticker, window, rng)
        if diag is None:
            continue

        # Realized outcome: only knowable now because this is history, not live.
        future_prices = ps_full.iloc[as_of_idx:as_of_idx + H + 1].values
        realized_path = np.diff(np.log(future_prices))
        realized_forward_return = float(np.mean(realized_path))

        path_signal = diag["path_signal"]
        hit = np.nan if path_signal == 0 else int(
            np.sign(path_signal) == np.sign(realized_forward_return)
        )

        records.append({
            "ticker": ticker,
            "as_of_date": str(as_of_date.date()),
            "path_signal": path_signal,
            "trend_consistency": diag["trend_consistency"],
            "fit_quality": diag["fit_quality"],
            "realized_forward_return": realized_forward_return,
            "hit": hit,
        })

    return records


def run_backtest(universe: str, window: int, step: int, tickers_override=None) -> pd.DataFrame:
    df = data_manager.load_master_data()
    tickers = tickers_override or config.UNIVERSES[universe]
    prices = data_manager.prepare_prices(df, tickers)
    available = [t for t in tickers if t in prices.columns]

    rng = np.random.default_rng(123)
    all_records = []
    for ticker in available:
        print(f"Backtesting {ticker}...")
        recs = walk_forward_ticker(prices, ticker, window, step, rng)
        print(f"  {len(recs)} as-of points")
        all_records += recs

    return pd.DataFrame(all_records)


def tercile_analysis(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Bucket by `by` into terciles; report hit-rate / correlation / spread per bucket."""
    d = df.dropna(subset=[by, "path_signal", "realized_forward_return"]).copy()
    if len(d) < 9:
        print(f"Not enough data to tercile by {by} (N={len(d)})")
        return None

    d["tercile"] = pd.qcut(d[by], 3, labels=["Low", "Mid", "High"], duplicates="drop")

    rows = []
    for t in d["tercile"].cat.categories:
        sub = d[d["tercile"] == t]
        if len(sub) == 0:
            continue
        hit_rate = sub["hit"].mean()
        corr = sub["path_signal"].corr(sub["realized_forward_return"])
        long_mean  = sub.loc[sub["path_signal"] > 0, "realized_forward_return"].mean()
        short_mean = sub.loc[sub["path_signal"] < 0, "realized_forward_return"].mean()
        spread = (long_mean - short_mean) if pd.notna(long_mean) and pd.notna(short_mean) else np.nan
        rows.append({
            "Tercile": t, "N": len(sub), "Hit Rate": hit_rate,
            "Corr(signal, realized)": corr, "Long-Short Spread": spread,
        })
    return pd.DataFrame(rows)


def table_to_records(df: pd.DataFrame) -> list:
    """Convert a tercile table to JSON-serializable records (handles Categorical dtype)."""
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, float)):
                rec[k] = None if pd.isna(v) else float(v)
            elif isinstance(v, (np.integer, int)):
                rec[k] = int(v)
            else:
                rec[k] = str(v)
        out.append(rec)
    return out


def ic_spread(table: pd.DataFrame):
    """High-tercile corr minus Low-tercile corr — same comparability metric
    used in backtest_diagnostics.py, so the two tabs can be read side by side."""
    if table is None or table.empty:
        return None
    vals = table.set_index("Tercile")["Corr(signal, realized)"]
    if "High" not in vals.index or "Low" not in vals.index:
        return None
    return float(vals["High"] - vals["Low"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="EQUITY_SECTORS", choices=list(config.UNIVERSES.keys()))
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=15,
                         help="trading days between walk-forward retraining points")
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="optional subset of tickers instead of the full universe")
    args = parser.parse_args()

    print(f"Running N-HiTS backtest: universe={args.universe} window={args.window} step={args.step}")
    results = run_backtest(args.universe, args.window, args.step, args.tickers)

    if results.empty:
        print("No backtest results — insufficient history for the chosen window/step.")
        return

    results.to_csv("backtest_results.csv", index=False)
    print(f"\nSaved {len(results)} rows to backtest_results.csv")

    print("\n=== By fit_quality tercile ===")
    fq = tercile_analysis(results, "fit_quality")
    if fq is not None:
        print(fq.to_string(index=False))

    print("\n=== By trend_consistency tercile ===")
    tc = tercile_analysis(results, "trend_consistency")
    if tc is not None:
        print(tc.to_string(index=False))

    print("\n=== Overall (no bucketing) ===")
    overall_hit  = results["hit"].mean()
    overall_corr = results["path_signal"].corr(results["realized_forward_return"])
    print(f"Hit rate: {overall_hit:.3f}   Corr(signal, realized): {overall_corr:.3f}   N={len(results)}")

    # ── JSON summary, pushed to HF so the dashboard can display it ────────────
    today = datetime.now().strftime("%Y-%m-%d")
    fit_spread   = ic_spread(fq)
    trend_spread = ic_spread(tc)
    winner = None
    if fit_spread is not None and trend_spread is not None:
        winner = "fit_quality" if abs(fit_spread) > abs(trend_spread) else "trend_consistency"

    summary = {
        "run_date": today,
        "universe": args.universe,
        "window": int(args.window),
        "step": int(args.step),
        "tickers_used": sorted(results["ticker"].unique().tolist()),
        "as_of_date_range": [
            str(results["as_of_date"].min()), str(results["as_of_date"].max()),
        ],
        "total_rows": int(len(results)),
        "overall_hit_rate": None if pd.isna(overall_hit) else float(overall_hit),
        "overall_corr": None if pd.isna(overall_corr) else float(overall_corr),
        "fit_quality_terciles":       table_to_records(fq),
        "trend_consistency_terciles": table_to_records(tc),
        "fit_quality_spread":       fit_spread,
        "trend_consistency_spread": trend_spread,
        "winner": winner,
    }

    json_path = Path(f"results/walkforward_backtest_{today}.json")
    json_path.parent.mkdir(exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    push_results.push_daily_result(json_path)


if __name__ == "__main__":
    main()
