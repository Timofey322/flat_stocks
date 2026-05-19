from __future__ import annotations

import numpy as np
import pandas as pd


def nadaraya_watson_envelope(
    close: pd.Series,
    bandwidth: float = 8.0,
    multiplier: float = 2.5,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Causal Nadaraya–Watson envelope (no future bars).

    At bar *t* only ``close[0:t+1]`` enters the Gaussian-kernel regression.
    Upper/lower bands are ``estimate ± multiplier * std(residuals)`` in the
    causal window. ``nw_position`` is 0 at the lower band and 1 at the upper.
    """

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    prices = close.astype(float).to_numpy()
    n = len(prices)
    window = lookback or max(int(bandwidth * 4), 20)

    nw_mid = np.full(n, np.nan)
    nw_upper = np.full(n, np.nan)
    nw_lower = np.full(n, np.nan)

    for t in range(n):
        start = max(0, t - window + 1)
        segment = prices[start : t + 1]
        m = len(segment)
        if m < 3:
            continue

        ages = np.arange(m - 1, -1, -1, dtype=float)
        weights = np.exp(-(ages**2) / (2 * bandwidth**2))
        estimate = float(np.dot(weights, segment) / weights.sum())
        residuals = segment - estimate
        band = multiplier * float(np.std(residuals, ddof=0))
        if band <= 0:
            band = multiplier * float(np.mean(np.abs(residuals)))

        nw_mid[t] = estimate
        nw_upper[t] = estimate + band
        nw_lower[t] = estimate - band

    width = nw_upper - nw_lower
    position = (prices - nw_lower) / np.where(width > 0, width, np.nan)

    return pd.DataFrame(
        {
            "nw_mid": nw_mid,
            "nw_upper": nw_upper,
            "nw_lower": nw_lower,
            "nw_position": position,
        },
        index=close.index,
    )
