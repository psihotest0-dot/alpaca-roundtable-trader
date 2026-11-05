"""The one statistic the measurements stand on: the mean of a daily series whose observations
overlap, with a standard error that knows it.

A signal measured over 20 days is checked every day, so neighbouring observations share 19 of
their 20 days. Counting them as independent makes the t-value about sqrt(20) too large. The
Newey-West estimate (Bartlett weights, lag = horizon - 1) accounts for that.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HacMean:
    mean: float
    se: float
    t: float
    n: int  # observations that count


def hac_mean(values: np.ndarray, mask: np.ndarray, lag: int) -> HacMean:
    """The mean of `values` where `mask` is true (and the value exists), with a Newey-West standard
    error. A day left out stays in the time axis, so the lags still mean "days apart"."""
    values = np.asarray(values, dtype=float)
    keep = np.asarray(mask, dtype=bool) & np.isfinite(values)
    n = int(keep.sum())
    if n < 2:
        return HacMean(float("nan"), float("nan"), float("nan"), n)
    mean = float(values[keep].mean())
    deviation = np.where(keep, values - mean, 0.0)
    total = float(deviation @ deviation)
    for step in range(1, min(lag, len(deviation) - 1) + 1):
        weight = 1.0 - step / (lag + 1)
        total += 2.0 * weight * float(deviation[step:] @ deviation[:-step])
    se = float(np.sqrt(max(total, 0.0))) / n
    t = mean / se if se > 0 else float("nan")
    return HacMean(mean, se, t, n)
