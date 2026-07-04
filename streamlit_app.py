import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="N-HiTS Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1c1c3c; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #3a5a78; }
.etf-card    { background:linear-gradient(135deg,#1c1c3c 0%,#3a5a78 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1c1c3c 0%,#264653 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏔️ N-HiTS Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Challu et al. (2022) Neural Hierarchical Interpolation · '
    'Multi-rate pooling + hierarchical interpolation + doubly-residual stacking · '
    'Pure univariate forecaster, analytical backprop · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## N-HiTS Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Architecture:** lookback={config.NHITS_LOOKBACK} | horizon={config.PRED_HORIZON} | "
    f"pools={config.POOL_SIZES}")
st.sidebar.markdown(
    f"**Training:** epochs={config.NHITS_EPOCHS} | lr={config.NHITS_LR} | "
    f"batch={config.NHITS_BATCH_SIZE} | backcast_weight={config.BACKCAST_WEIGHT}")
st.sidebar.markdown(
    f"**Weights:** Forecast {config.WEIGHT_FORECAST:.0%} | "
    f"Consistency {config.WEIGHT_CONSISTENCY:.0%} | "
    f"Fit {config.WEIGHT_FIT:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "nhits_engine_2")
tab2_path = find_latest(files, "nhits_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Best Window per ETF", "🔍 Explore by Window",
    "📊 Diagnostic Validity", "🎯 Walk-Forward Validation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Hierarchical Interpolation Forecast Signal")

    with st.expander("N-HiTS Methodology", expanded=True):
        st.markdown("""
N-HiTS extends N-BEATS with two ideas aimed at efficient long-horizon
forecasting, stacked with doubly-residual connections:

**1. Multi-rate signal sampling** — each stack max-pools the lookback
window at a different rate before its MLP sees it:

```
pool sizes (coarse → fine): [8, 4, 1]
```

A large pool smooths the input down to its trend; pool=1 leaves it at full
resolution. Each stack specializes in a different frequency band purely
through what resolution it's shown.

**2. Hierarchical interpolation** — each stack outputs a small number of
basis coefficients (fewer for coarse stacks, more for fine stacks), which
are linearly interpolated up to the full backcast/forecast length. A
coarse stack literally cannot represent high-frequency detail — it has
too few coefficients — which keeps its contribution smooth by construction.

**3. Doubly residual stacking** (from N-BEATS): each stack predicts a
backcast that is SUBTRACTED from the residual before the next stack sees
it, and a forecast that is SUMMED across all stacks:

```
r_0 = x
for i in 1..S:
    pooled_i         = MaxPool(r_{i-1}, pool_i)
    theta_b, theta_f = MLP_i(pooled_i)
    backcast_i       = Interpolate(theta_b, target_len=L)
    forecast_i       = Interpolate(theta_f, target_len=H)
    r_i              = r_{i-1} - backcast_i
y_hat = sum_i forecast_i
```

**Pure univariate — no macro conditioning.** Unlike DDB, the Decision
Transformer, or OT-FM elsewhere in this suite, N-HiTS as specified in the
original paper takes no exogenous inputs. Its entire contribution is
architectural: how the series is decomposed and reconstructed across
temporal resolutions, not what extra information it's fed.

**Signal:**

```
score = 0.50*path_signal + 0.30*trend_consistency*sign(path_signal) + 0.20*fit_quality
```

- `path_signal` — mean of the forecasted H-step return path
- `trend_consistency` — fraction of forecast steps agreeing in sign: do the
  coarse (trend) and fine (detail) stacks agree on direction?
- `fit_quality` — 1 - (residual energy / input energy), explicitly
  regularized via an auxiliary backcast loss during training (vanilla
  N-BEATS/N-HiTS leaves this purely incidental — this engine trains it
  directly so it's a real, informative diagnostic)
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">N-HiTS score = {etf['nhits_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
  <div class="etf-score">consistency = {etf.get('trend_consistency', float('nan')):.2f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    cons  = info.get("trend_consistency", None) if isinstance(info, dict) else None
                    fit   = info.get("fit_quality", None) if isinstance(info, dict) else None
                    rows.append({
                        "ETF": t, "N-HiTS Score": score, "Best Window (d)": win,
                        "Trend Consistency": cons, "Fit Quality": fit,
                    })
                df = pd.DataFrame(rows).sort_values("N-HiTS Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Challu et al. (2022) N-HiTS · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore N-HiTS Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — short training set; few samples for the hierarchy to specialize on; reactive, noisier
- **126d** — 6-month window; recommended minimum for a stable decomposition
- **252d** — 1-year window; most stable multi-rate decomposition; recommended primary signal
- **504d** — 2-year window; structural regime decomposition; slow-moving signal
        """)

    st.markdown(f"### N-HiTS Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">N-HiTS score = {etf['nhits_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
  <div class="etf-score">consistency = {etf.get('trend_consistency', float('nan')):.2f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["ETF", "N-HiTS Score", "Path Signal", "Trend Consistency", "Fit Quality"],
                )
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Diagnostic Validity
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Diagnostic Validity — Does fit_quality or trend_consistency actually predict returns?")

    diag_path = find_latest(files, "diagnostic_backtest_")
    if not diag_path:
        st.warning(
            "No diagnostic backtest found yet. This is a separate, on-demand "
            "analysis — it does not run on the daily schedule. Trigger the "
            "**\"N-HiTS Diagnostic Backtest\"** workflow manually from the "
            "GitHub Actions tab (`python backtest_diagnostics.py`) to generate it."
        )
        st.stop()

    diag = load_json(diag_path)
    if "error" in diag:
        st.error(f"Error loading diagnostic backtest: {diag['error']}")
        st.stop()

    with st.expander("Methodology", expanded=True):
        st.markdown("""
For each (ticker, window), a model is trained exactly as `trainer.py` does
in production, then evaluated **in-sample, across every timestep in its own
training window** — this asks *"conditional on how a diagnostic looked at
time t, was `path_signal(t)` more aligned with what happened at t+H?"*

This is an in-sample diagnostic-validity check, not a claim about
out-of-sample trading skill — see `backtest.py` in this repo for the more
expensive true walk-forward version (retrains at every historical date
using only data available as of that date).

Consecutive rows share most of their target window, so results are shown
both **stride-H** (non-overlapping — the defensible number) and
**all-rows** (overlapping — larger sample, inflated, reference only).
        """)

    st.markdown(f"**Run date:** `{diag.get('run_date','?')}`  ·  "
                f"**Rows:** {diag.get('total_rows_stride_h','?')} stride-H / "
                f"{diag.get('total_rows_all_overlapping','?')} all-overlapping")

    winner = diag.get("winner")
    fit_spread = diag.get("fit_quality_spread")
    trend_spread = diag.get("trend_consistency_spread")
    if winner:
        st.success(
            f"**More discriminating diagnostic: `{winner}`**  "
            f"(fit_quality High-vs-Low spread = {fit_spread:.4f}, "
            f"trend_consistency High-vs-Low spread = {trend_spread:.4f})"
        )

    st.markdown("### Primary result — stride-H (non-overlapping)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By fit_quality tercile**")
        rows = diag.get("fit_quality_terciles_primary", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No data")
    with col2:
        st.markdown("**By trend_consistency tercile**")
        rows = diag.get("trend_consistency_terciles_primary", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No data")

    with st.expander("Reference only — all overlapping rows (inflated, upward-biased)"):
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**By fit_quality tercile**")
            rows = diag.get("fit_quality_terciles_allrows_ref", [])
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with col4:
            st.markdown("**By trend_consistency tercile**")
            rows = diag.get("trend_consistency_terciles_allrows_ref", [])
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "A large positive High-vs-Low IC spread means that diagnostic's High "
        "tercile has meaningfully higher correlation with realized returns "
        "than its Low tercile — i.e. it's doing real work identifying when "
        "path_signal should be trusted. A spread near zero means it doesn't "
        "discriminate skill, regardless of the raw IC values in isolation."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Walk-Forward Validation (true out-of-sample)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🎯 Walk-Forward Validation — True Out-of-Sample Test")

    wf_path = find_latest(files, "walkforward_backtest_")
    if not wf_path:
        st.warning(
            "No walk-forward backtest found yet. This is a separate, on-demand "
            "analysis that retrains N-HiTS at every historical point using only "
            "data available as of that date — genuinely out-of-sample, unlike "
            "the in-sample check in the previous tab. Trigger the "
            "**\"N-HiTS Walk-Forward Backtest\"** workflow manually from the "
            "GitHub Actions tab to generate it."
        )
        st.stop()

    wf = load_json(wf_path)
    if "error" in wf:
        st.error(f"Error loading walk-forward backtest: {wf['error']}")
        st.stop()

    with st.expander("Methodology", expanded=True):
        st.markdown("""
Unlike the **📊 Diagnostic Validity** tab (which evaluates an already-trained
model in-sample, on the same data it was trained on), this tab retrains
N-HiTS **at every historical walk-forward point using only data available
as of that date** — prices are truncated, so there is no lookahead — then
checks the forecast against the return that actually, subsequently
happened.

This is the more expensive but more trustworthy test. If the numbers here
are noticeably weaker than the in-sample tab, that's expected: it means
some of the in-sample correlation was the model fitting its own training
set rather than genuine predictive skill.
        """)

    st.markdown(
        f"**Run date:** `{wf.get('run_date','?')}`  ·  "
        f"**Universe:** `{wf.get('universe','?')}`  ·  "
        f"**Window:** {wf.get('window','?')}d  ·  "
        f"**Step:** {wf.get('step','?')}d  ·  "
        f"**Tickers:** {len(wf.get('tickers_used', []))}"
    )
    date_range = wf.get("as_of_date_range", ["?", "?"])
    st.caption(f"As-of dates spanning {date_range[0]} → {date_range[1]}  ·  "
               f"{wf.get('total_rows','?')} walk-forward points")

    overall_hit  = wf.get("overall_hit_rate")
    overall_corr = wf.get("overall_corr")
    m1, m2 = st.columns(2)
    m1.metric("Overall hit rate", f"{overall_hit:.3f}" if overall_hit is not None else "N/A")
    m2.metric("Overall corr(signal, realized)", f"{overall_corr:.3f}" if overall_corr is not None else "N/A")

    winner = wf.get("winner")
    fit_spread = wf.get("fit_quality_spread")
    trend_spread = wf.get("trend_consistency_spread")
    if winner:
        st.success(
            f"**More discriminating diagnostic (out-of-sample): `{winner}`**  "
            f"(fit_quality High-vs-Low spread = {fit_spread:.4f}, "
            f"trend_consistency High-vs-Low spread = {trend_spread:.4f})"
        )

    st.markdown("### By diagnostic tercile")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By fit_quality tercile**")
        rows = wf.get("fit_quality_terciles", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No data")
    with col2:
        st.markdown("**By trend_consistency tercile**")
        rows = wf.get("trend_consistency_terciles", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No data")

    st.caption(
        "Compare these numbers against the 📊 Diagnostic Validity tab. A large "
        "drop from in-sample to walk-forward is normal and expected — it's the "
        "gap between how well the model fits its own training data and how "
        "well it actually predicts what it hasn't seen."
    )
