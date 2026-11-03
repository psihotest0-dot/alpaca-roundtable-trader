"""The strategy the live loop runs: a 15-minute EMA cross-up with the RSI in a band, confirmed by
the 1-hour trend (`strategy.trend.generate_signals`), as a plugin."""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.domain.models import StrategySignal
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import RUNTIME_STRATEGY, StrategyConfig
from trading_system.strategy.trend import calculate_indicators, generate_signals

# A bar whose indicators are this close (relative) to a line is passed on to the strategy, which
# decides; so the hint below never drops a signal over a rounding difference.
_MARGIN = 1e-6


class TrendCrossStrategy(BaseStrategy):
    name = "trend_cross"

    def __init__(self, config: StrategyConfig = RUNTIME_STRATEGY):
        self.config = config

    @property
    def version(self) -> str:
        return self.config.strategy_version

    def generate_signals(
        self,
        data_15m: Dict[str, pd.DataFrame],
        data_1h: Dict[str, pd.DataFrame],
        eval_time: datetime,
    ) -> List[StrategySignal]:
        return generate_signals(data_15m, data_1h, self.config, eval_time)

    def candidate_bars(
        self, bars_15m: pd.DataFrame, bars_1h: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """Possible where the fast EMA crosses above the slow one and the RSI is in its band
        (both read off indicators computed once for the whole history)."""
        config = self.config
        ind = calculate_indicators(bars_15m, config, is_htf=False)
        fast = ind["ema_fast"].to_numpy(dtype=float)
        slow = ind["ema_slow"].to_numpy(dtype=float)
        rsi = ind["rsi"].to_numpy(dtype=float)
        margin = _MARGIN * np.abs(slow)
        crossed = np.zeros(len(bars_15m), dtype=bool)
        crossed[1:] = (fast[:-1] <= slow[:-1] + margin[:-1]) & (fast[1:] > slow[1:] - margin[1:])
        in_band = (rsi >= config.rsi_lower_bound - _MARGIN) & (
            rsi <= config.rsi_upper_bound + _MARGIN
        )
        flags: np.ndarray = crossed & in_band
        return flags
