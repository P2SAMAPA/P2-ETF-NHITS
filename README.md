# 🏔️ P2-ETF-NHITS

**N-HiTS Engine (Neural Hierarchical Interpolation) — Challu et al. (2022)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine trains an **N-HiTS** model per ETF — a pure univariate time
series forecaster that decomposes the lookback window across multiple
temporal resolutions (via multi-rate pooling), reconstructs it hierarchically
(via interpolated basis coefficients), and stacks the decomposition with
doubly-residual connections inherited from N-BEATS. Unlike every other
generative/RL engine in this suite, it takes **no macro conditioning** —
its entire contribution is architectural, not exogenous.

---

## Theory

### Multi-Rate Signal Sampling

Each stack max-pools the lookback window at a different rate before its MLP
sees it:

```
pool sizes (coarse → fine): [8, 4, 1]
```

A large pool (8) smooths the input down to its trend; pool=1 leaves it at
full resolution. Each stack specializes in a frequency band purely through
what resolution it's shown — no explicit frequency-domain transform needed.

### Hierarchical Interpolation

Each stack outputs a **small** number of basis coefficients — fewer for
coarse stacks, more for fine stacks — which are linearly interpolated up to
the full backcast/forecast length. A coarse stack literally cannot
represent high-frequency detail; it doesn't have the coefficients for it.
This is what keeps its contribution smooth by construction, rather than by
regularization.

### Doubly Residual Stacking

Inherited from N-BEATS: each stack predicts a backcast that is subtracted
from the residual before the next stack sees it, and a forecast that is
summed across all stacks:

```
r_0 = x                              (the lookback window)
for i in 1..S:
    pooled_i          = MaxPool(r_{i-1}, pool_i)
    theta_b, theta_f  = MLP_i(pooled_i)
    backcast_i        = Interpolate(theta_b, target_len=L)
    forecast_i        = Interpolate(theta_f, target_len=H)
    r_i               = r_{i-1} - backcast_i
y_hat = sum_i forecast_i
```

Built from scratch with manual forward/backward through pooling
(gradient routed to the max-argmax position only), the linear interpolation
matrices, and the doubly-residual chain — no autograd framework.

### Backcast Reconstruction Is Trained Explicitly

In vanilla N-BEATS/N-HiTS, the final residual shrinking is only ever an
**incidental** byproduct of forecast training — nothing directly pressures
it to get small. This engine adds a small auxiliary loss on the final
residual (`BACKCAST_WEIGHT`) so that reconstruction quality is a real,
optimized signal rather than an accident of whatever helped the forecast.

### Score Construction

```
score = 0.50*path_signal + 0.30*trend_consistency*sign(path_signal) + 0.20*fit_quality
```

| Component | Meaning |
|-----------|---------|
| path_signal | Mean of the forecasted H-step return path — the direct multi-horizon forecast |
| trend_consistency | Fraction of forecast steps agreeing in sign — do the coarse (trend) and fine (detail) stacks agree on direction? |
| fit_quality | 1 - (residual energy / input energy) — explicitly regularized reconstruction quality |

---

## Distinction from Other Engines in the Suite

| Engine | Conditioning | Core mechanism |
|--------|--------------|-----------------|
| DDB | Macro-implied target | Diffusion bridge SDE |
| Decision Transformer | Return-to-go | Causal sequence modelling (offline RL) |
| OT-FM | Lagged returns + macro | Flow matching with optimal coupling |
| **N-HiTS (this engine)** | **None — pure univariate** | **Multi-rate hierarchical decomposition** |

N-HiTS is the only engine in this group that forecasts a full multi-step
path rather than a single scalar signal, and the only one whose
distinguishing idea is entirely architectural rather than about what
external information it conditions on.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-NHITS/
├── config.py          # Universes, N-HiTS hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── nhits_engine.py     # Core: multi-rate pooling, hierarchical interpolation, doubly-residual stacking
├── trainer.py          # Orchestrator
├── push_results.py     # HfApi.upload_file wrapper
├── streamlit_app.py     # Two-tab Streamlit dashboard
├── us_calendar.py      # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml   # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-NHITS
cd P2-ETF-NHITS
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-nhits-results`

---

## References

- Challu, C. et al. (2022). N-HiTS: Neural Hierarchical Interpolation for
  Time Series Forecasting. AAAI 2023.
- Oreshkin, B. et al. (2019). N-BEATS: Neural Basis Expansion Analysis for
  Interpretable Time Series Forecasting. ICLR 2020.
