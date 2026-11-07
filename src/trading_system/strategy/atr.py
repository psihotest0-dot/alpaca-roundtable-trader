"""The ATR of a window of bars, as the live code computes it (for the ATR exit candidate,
docs/strategy-research-plan.md, section 13.1). Nothing in the live path reads this."""

import math
from typing import Optional

import pandas as pd

from trading_system.strategy.config import RUNTIME_STRATEGY, StrategyConfig
from trading_system.strategy.trend import calculate_indicators


def atr_and_close(
    window: pd.DataFrame, config: StrategyConfig = RUNTIME_STRATEGY
) -> Optional[tuple[float, float]]:
    """(ATR, close) at the LAST bar of `window`: the 15-minute bars up to and including the signal
    bar, which is what the live loop's indicators see (`calculate_indicators`, Wilder, `atr_period`).
    None when it is not defined (too few bars, or no usable numbers)."""
    if window is None or len(window) <= config.atr_period:
        return None
    indicators = calculate_indicators(window, config)
    atr = float(indicators["atr"].iloc[-1])
    close = float(indicators["close"].iloc[-1])
    if not (math.isfinite(atr) and math.isfinite(close)) or atr <= 0 or close <= 0:
        return None
    return atr, close
