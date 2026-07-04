import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-nhits-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── N-HiTS hyperparameters ────────────────────────────────────────────────────
# Challu et al. (2022) "N-HiTS: Neural Hierarchical Interpolation for Time
# Series Forecasting". Extends N-BEATS with two ideas aimed specifically at
# long-horizon forecasting efficiency:
#
#   1. MULTI-RATE SIGNAL SAMPLING — each stack max-pools the lookback window
#      at a different rate before its MLP sees it. Coarse stacks (large pool)
#      see a smoothed, downsampled view and specialize in long-term trend;
#      fine stacks (pool=1) see full resolution and specialize in short-term
#      detail. This is a genuine frequency decomposition, not just repeated
#      copies of the same block at the same resolution (as in N-BEATS).
#
#   2. HIERARCHICAL INTERPOLATION — each stack predicts a SMALL number of
#      basis coefficients (fewer for coarse stacks, more for fine stacks)
#      which are then linearly interpolated up to the full backcast/forecast
#      length. This keeps coarse stacks smooth by construction and lets fine
#      stacks add detail only where the data supports it.
#
# Both ideas are combined with N-BEATS's DOUBLY RESIDUAL STACKING: each
# stack predicts a backcast (reconstruction of its input) that is subtracted
# from the residual before the next stack sees it, and a forecast that is
# summed across all stacks to form the final multi-step prediction.
#
# Unlike every other generative/RL engine in this suite (DDB, DT, OT-FM),
# N-HiTS is a PURE UNIVARIATE FORECASTER as specified in the original paper
# — no macro conditioning, no noise, no return-to-go. Its entire
# contribution is architectural: how it decomposes and reconstructs the
# time series itself across multiple temporal resolutions.

NHITS_LOOKBACK = 16     # L: input window length (must divide evenly by POOL_SIZES)
PRED_HORIZON   = 21     # H: forecast horizon — N-HiTS predicts the FULL H-step
                        # path at once, not just its mean (unlike other engines)

# 3 stacks, coarse -> fine. Pool size 8 sees the trend at 1/8 resolution;
# pool size 1 sees full-resolution short-term detail.
POOL_SIZES = [8, 4, 1]

HIDDEN_DIM = 32
N_HIDDEN   = 2       # hidden layers per stack's MLP

NHITS_EPOCHS     = 60
NHITS_LR         = 3e-3
NHITS_BATCH_SIZE = 32

# Backcast reconstruction is only ever an INCIDENTAL byproduct of forecast
# training in vanilla N-BEATS/N-HiTS — nothing directly pressures the final
# residual to shrink. That makes it a poor diagnostic on its own. A small
# explicit auxiliary loss on the final residual is added during training so
# that fit_quality (below) reflects something the model actually optimizes,
# not an accident of whatever helped the forecast.
BACKCAST_WEIGHT = 0.15

# ── Score construction ────────────────────────────────────────────────────────
# path_signal      : mean of the forecasted H-step return path — the direct
#                    multi-horizon forecast itself
# trend_consistency: fraction of steps in the forecast path that share the
#                    same sign as path_signal — measures whether the coarse
#                    (trend) and fine (detail) stacks agree on direction,
#                    which is specifically what the hierarchical
#                    decomposition is supposed to produce when it works
# fit_quality      : 1 - (residual energy after all stacks / input energy) —
#                    how well the doubly-residual decomposition explains the
#                    lookback window itself; a poor reconstruction means the
#                    extrapolation shouldn't be trusted

WEIGHT_FORECAST     = 0.50
WEIGHT_CONSISTENCY  = 0.30
WEIGHT_FIT           = 0.20

TOP_N = 3
