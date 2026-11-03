"""What a strategy is, so that several can be run and compared on the same data.

A strategy looks at closed bars and says "long this symbol now". It does not size, price or
place anything: that is the risk engine's and the execution engine's job, the same for every
strategy. Two things follow from that and are part of the contract:

- It judges each symbol on its own, from that symbol's bars only. The backtest relies on it to
  ask about a few symbols instead of all of them (see `candidate_bars`).
- It uses closed bars only and sees nothing after `eval_time`.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.domain.models import StrategySignal


class BaseStrategy(ABC):
    #: The name it is listed and chosen by (`backtest --strategy NAME`).
    name: str = ""

    @property
    @abstractmethod
    def version(self) -> str:
        """Identifies the strategy and its parameters. It goes into every signal's id, so a
        changed parameter is a different strategy in the records."""

    @abstractmethod
    def generate_signals(
        self,
        data_15m: Dict[str, pd.DataFrame],
        data_1h: Dict[str, pd.DataFrame],
        eval_time: datetime,
    ) -> List[StrategySignal]:
        """LONG signals for the bars that had closed by `eval_time`. `data_15m` and `data_1h`
        hold, per symbol, the bars the strategy is shown (the live loop's window)."""

    def candidate_bars(
        self, bars_15m: pd.DataFrame, bars_1h: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """Optional speed-up for the backtest. For one symbol's 15-minute bars (sorted, one row
        per instant; `bars_1h` are its hourly bars, same shape), a boolean array: True where a
        signal is POSSIBLE on that bar. It may say
        True for a bar that gives no signal, but must never say False for one that does: the
        strategy still decides. None (the default) means no hint, ask about every bar."""
        return None
