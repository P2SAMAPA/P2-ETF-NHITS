"""
nhits_engine.py — N-HiTS Engine (Neural Hierarchical Interpolation)
========================================================================

Theory
------
**N-HiTS (Challu et al. 2022)** extends N-BEATS with two ideas aimed at
efficient long-horizon forecasting, stacked S deep with doubly residual
connections:

1. **Multi-rate signal sampling** — before stack i's MLP sees the current
   residual, it is max-pooled with a stack-specific kernel size (pool_i).
   A large pool (e.g. 8) smooths the input down to its trend; a pool of 1
   leaves it at full resolution. Each stack therefore specializes in a
   different frequency band, purely through what resolution it's shown —
   not through any explicit frequency-domain transform.

2. **Hierarchical interpolation** — each stack's MLP outputs a SMALL number
   of basis coefficients (fewer for coarse/high-pool stacks, more for
   fine/low-pool stacks), which are linearly interpolated up to the full
   backcast length (L) and forecast horizon (H). A coarse stack literally
   cannot represent high-frequency detail — it has too few coefficients —
   which is what keeps its contribution smooth by construction.

3. **Doubly residual stacking** (inherited from N-BEATS): each stack
   predicts a backcast (reconstruction of its input) that is SUBTRACTED
   from the residual before the next stack sees it, so later stacks only
   have to explain what earlier stacks missed. Forecasts from all stacks
   are SUMMED to form the final H-step prediction.

    r_0 = x                              (the lookback window)
    for i in 1..S:
        pooled_i          = MaxPool(r_{i-1}, pool_i)
        theta_b, theta_f  = MLP_i(pooled_i)
        backcast_i        = Interpolate(theta_b, target_len=L)
        forecast_i        = Interpolate(theta_f, target_len=H)
        r_i               = r_{i-1} - backcast_i
    y_hat = sum_i forecast_i

Unlike DDB, the Decision Transformer, or OT-FM elsewhere in this suite,
N-HiTS as specified is a PURE UNIVARIATE FORECASTER — no macro
conditioning, no noise, no return-to-go. Its entire contribution is
architectural: how the time series is decomposed and reconstructed across
multiple temporal resolutions, not what exogenous information it's fed.

**Score construction**

    score = 0.50*path_signal + 0.30*trend_consistency*sign(path_signal) + 0.20*fit_quality

| Component          | Meaning                                                             |
|----------------------|------------------------------------------------------------------------|
| path_signal          | Mean of the forecasted H-step return path                            |
| trend_consistency    | Fraction of forecast steps agreeing in sign — do the coarse/fine stacks agree? |
| fit_quality          | 1 - (residual energy / input energy) — reconstruction quality, explicitly regularized via an auxiliary backcast loss (BACKCAST_WEIGHT) rather than left as an incidental byproduct |

References
----------
- Challu, C. et al. (2022). N-HiTS: Neural Hierarchical Interpolation for
  Time Series Forecasting. AAAI 2023.
- Oreshkin, B. et al. (2019). N-BEATS: Neural Basis Expansion Analysis for
  Interpretable Time Series Forecasting. ICLR 2020.
"""

import numpy as np
import pandas as pd
from typing import List

import config


# ── Basic differentiable layers (manual forward/backward) ─────────────────────

class Linear:
    def __init__(self, in_d: int, out_d: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_d)
        self.W = rng.normal(0, scale, (in_d, out_d))
        self.b = np.zeros(out_d)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY: np.ndarray):
        X = self.X
        X2  = X.reshape(-1, X.shape[-1])
        dY2 = dY.reshape(-1, dY.shape[-1])
        dW  = X2.T @ dY2
        db  = dY2.sum(axis=0)
        dX  = dY @ self.W.T
        return dX, dW, db


def maxpool1d_forward(x: np.ndarray, k: int):
    """x: (B,L), non-overlapping windows of size k. Returns pooled (B,L//k) + cache."""
    B, L = x.shape
    Lp = L // k
    xr = x[:, :Lp * k].reshape(B, Lp, k)
    idx = np.argmax(xr, axis=2)
    pooled = np.max(xr, axis=2)
    return pooled, (B, L, k, Lp, idx)


def maxpool1d_backward(dpooled: np.ndarray, cache):
    B, L, k, Lp, idx = cache
    dx = np.zeros((B, Lp, k))
    b_idx  = np.arange(B)[:, None]
    lp_idx = np.arange(Lp)[None, :]
    dx[b_idx, lp_idx, idx] = dpooled
    return dx.reshape(B, Lp * k)


def build_interp_matrix(r: int, T: int) -> np.ndarray:
    """Linear interpolation matrix M (T,r): out = theta @ M.T upsamples r->T points."""
    if r == 1:
        return np.ones((T, 1))
    src = np.linspace(0, r - 1, r)
    tgt = np.linspace(0, r - 1, T)
    M = np.zeros((T, r))
    for t, pos in enumerate(tgt):
        j = int(np.floor(pos))
        j = min(j, r - 2)
        frac = pos - j
        M[t, j]     = 1 - frac
        M[t, j + 1] = frac
    return M


# ── N-HiTS stack ────────────────────────────────────────────────────────────────

class NHiTSStack:
    def __init__(self, L: int, H: int, pool: int, rng: np.random.Generator):
        self.pool = pool
        self.Lp = L // pool
        theta_b_size = max(1, L // pool)
        theta_f_size = max(1, H // pool)

        Hd = config.HIDDEN_DIM
        self.L1 = Linear(self.Lp, Hd, rng)
        self.L2 = Linear(Hd, Hd, rng)
        self.Wb = Linear(Hd, theta_b_size, rng)
        self.Wf = Linear(Hd, theta_f_size, rng)

        self.M_back = build_interp_matrix(theta_b_size, L)
        self.M_fore = build_interp_matrix(theta_f_size, H)

    def forward(self, residual: np.ndarray):
        pooled, pool_cache = maxpool1d_forward(residual, self.pool)
        h1 = np.tanh(self.L1.forward(pooled))
        h2 = np.tanh(self.L2.forward(h1))
        theta_b = self.Wb.forward(h2)
        theta_f = self.Wf.forward(h2)
        backcast = theta_b @ self.M_back.T
        forecast = theta_f @ self.M_fore.T
        self.cache = (pool_cache, h1, h2)
        return backcast, forecast

    def backward(self, dbackcast: np.ndarray, dforecast: np.ndarray):
        pool_cache, h1, h2 = self.cache
        dtheta_b = dbackcast @ self.M_back
        dtheta_f = dforecast @ self.M_fore

        dh2_b, dWb_W, dWb_b = self.Wb.backward(dtheta_b)
        dh2_f, dWf_W, dWf_b = self.Wf.backward(dtheta_f)
        dh2 = dh2_b + dh2_f

        dz2 = dh2 * (1 - h2 ** 2)
        dh1, dW2, db2 = self.L2.backward(dz2)
        dz1 = dh1 * (1 - h1 ** 2)
        dpooled, dW1, db1 = self.L1.backward(dz1)

        dresidual = maxpool1d_backward(dpooled, pool_cache)
        grads = {
            "L1": (dW1, db1), "L2": (dW2, db2),
            "Wb": (dWb_W, dWb_b), "Wf": (dWf_W, dWf_b),
        }
        return dresidual, grads


# ── Full N-HiTS model ───────────────────────────────────────────────────────────

class NHiTS:
    def __init__(self, L: int, H: int, pools: List[int], rng: np.random.Generator):
        self.L, self.H = L, H
        self.stacks = [NHiTSStack(L, H, p, rng) for p in pools]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, L). Returns total_forecast: (B, H)."""
        residual = x.copy()
        total_forecast = np.zeros((x.shape[0], self.H))
        self._last_shape = x.shape
        for stack in self.stacks:
            backcast, forecast = stack.forward(residual)
            residual = residual - backcast
            total_forecast = total_forecast + forecast
        self._final_residual = residual
        return total_forecast

    def backward(self, dtotal_forecast: np.ndarray, dfinal_residual: np.ndarray = None):
        """Doubly-residual backprop: dr_{i-1} = dr_i (passthrough) + stack_i's own gradient.
        dfinal_residual carries gradient from the auxiliary backcast reconstruction loss."""
        dr = np.zeros(self._last_shape) if dfinal_residual is None else dfinal_residual
        all_grads = [None] * len(self.stacks)
        for i in reversed(range(len(self.stacks))):
            dbackcast = -dr
            dforecast = dtotal_forecast   # broadcast: total_forecast = sum of all stacks' forecasts
            dr_from_stack, grads = self.stacks[i].backward(dbackcast, dforecast)
            dr = dr + dr_from_stack
            all_grads[i] = grads
        return all_grads

    # ── Adam over all stacks' params ───────────────────────────────────────────

    def _param_list(self):
        params = []
        for stack in self.stacks:
            params += [
                (stack.L1, "W"), (stack.L1, "b"),
                (stack.L2, "W"), (stack.L2, "b"),
                (stack.Wb, "W"), (stack.Wb, "b"),
                (stack.Wf, "W"), (stack.Wf, "b"),
            ]
        return params

    def init_adam(self):
        return [(np.zeros_like(getattr(o, a)), np.zeros_like(getattr(o, a)))
                for o, a in self._param_list()]

    def apply_adam(self, all_grads, state, step, lr,
                    b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        flat = []
        for grads in all_grads:
            flat += [
                grads["L1"][0], grads["L1"][1],
                grads["L2"][0], grads["L2"][1],
                grads["Wb"][0], grads["Wb"][1],
                grads["Wf"][0], grads["Wf"][1],
            ]

        params = self._param_list()
        for i, ((obj, attr), grad) in enumerate(zip(params, flat)):
            m, v = state[i]
            m[:] = b1 * m + (1 - b1) * grad
            v[:] = b2 * v + (1 - b2) * grad ** 2
            mh = m / (1 - b1 ** step)
            vh = v / (1 - b2 ** step)
            update = lr * mh / (np.sqrt(vh) + eps)
            setattr(obj, attr, getattr(obj, attr) - update)


# ── Training ───────────────────────────────────────────────────────────────────

def _train_nhits(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> NHiTS:
    """X: (N,L) lookback windows. Y: (N,H) future return paths."""
    N = len(X)
    B = config.NHITS_BATCH_SIZE
    if N < B:
        raise ValueError("insufficient samples for N-HiTS training")

    model = NHiTS(config.NHITS_LOOKBACK, config.PRED_HORIZON, config.POOL_SIZES, rng)
    state = model.init_adam()
    step = 0

    for epoch in range(config.NHITS_EPOCHS):
        idx = rng.permutation(N)
        epoch_loss, n_b = 0.0, 0

        for i in range(0, N, B):
            bi = idx[i:i + B]
            if len(bi) < 4:
                continue

            X_b, Y_b = X[bi], Y[bi]
            pred = model.forward(X_b)
            resid = pred - Y_b
            forecast_loss = float(np.mean(resid ** 2))

            final_res = model._final_residual
            recon_loss = float(np.mean(final_res ** 2))
            loss = forecast_loss + config.BACKCAST_WEIGHT * recon_loss

            dtotal_forecast = 2.0 * resid / resid.size
            dfinal_residual = 2.0 * config.BACKCAST_WEIGHT * final_res / final_res.size

            grads = model.backward(dtotal_forecast, dfinal_residual)
            step += 1
            model.apply_adam(grads, state, step, lr=config.NHITS_LR)

            epoch_loss += loss
            n_b += 1

        if (epoch + 1) % 15 == 0:
            print(f"    epoch {epoch+1}/{config.NHITS_EPOCHS}  loss={epoch_loss/max(n_b,1):.6f}")

    return model


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_nhits_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Train an N-HiTS model per ETF (pure univariate — no macro conditioning,
    faithful to the original architecture) and extract a multi-horizon
    forecast signal. Returns cross-sectional z-scores.
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    L, H = config.NHITS_LOOKBACK, config.PRED_HORIZON
    min_rows = window + H + L + config.NHITS_BATCH_SIZE * 2 + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    rng = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        ps = prices[ticker].dropna()
        if len(ps) < min_rows:
            continue

        log_ret = np.log(ps / ps.shift(1)).dropna().values
        T = len(log_ret)
        start = max(L, T - window - H)
        end = T - H
        n = end - start
        if n < config.NHITS_BATCH_SIZE * 2:
            continue

        X = np.stack([log_ret[t - L:t] for t in range(start, end)])
        Y = np.stack([log_ret[t:t + H] for t in range(start, end)])

        seg = log_ret[start - L:end + H]
        mu, sd = seg.mean(), seg.std() + 1e-8
        X_norm = (X - mu) / sd
        Y_norm = (Y - mu) / sd

        print(f"    Training N-HiTS for {ticker} (N={n})")
        try:
            model = _train_nhits(X_norm, Y_norm, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            continue

        # ── Inference: forecast the next H steps from today's lookback ────────
        x_today = ((log_ret[-L:] - mu) / sd)[None, :]
        forecast_norm = model.forward(x_today)[0]
        forecast_path = forecast_norm * sd + mu

        path_signal = float(np.mean(forecast_path))
        sign = np.sign(path_signal) if path_signal != 0 else 1.0
        trend_consistency = float(np.mean(np.sign(forecast_path) == sign))

        final_residual = model._final_residual[0]
        fit_quality = float(1.0 - np.clip(
            np.mean(final_residual ** 2) / (np.mean(x_today ** 2) + 1e-8), 0.0, 1.0
        ))

        print(f"    {ticker}: path_signal={path_signal:.5f}  "
              f"consistency={trend_consistency:.3f}  fit={fit_quality:.3f}")

        composite = (
            config.WEIGHT_FORECAST     * path_signal
            + config.WEIGHT_CONSISTENCY * trend_consistency * sign
            + config.WEIGHT_FIT          * fit_quality
        )
        raw_scores[ticker] = {
            "composite": composite,
            "path_signal": path_signal,
            "trend_consistency": trend_consistency,
            "fit_quality": fit_quality,
        }

    cols = ["score", "path_signal", "trend_consistency", "fit_quality"]
    if not raw_scores:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(raw_scores).T
    mu_s, std_s = df["composite"].mean(), df["composite"].std()
    if std_s < 1e-10:
        df["score"] = 0.0
    else:
        df["score"] = (df["composite"] - mu_s) / std_s
    return df[cols]
    return (scores - mu_s) / std_s
