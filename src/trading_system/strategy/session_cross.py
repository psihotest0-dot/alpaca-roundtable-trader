"""Two candidates to repair the live strategy (docs/strategy-research-plan.md, section 12).

`trend_cross` as the live loop runs it, with the bars cleaned first:

* **Only bars of the regular session reach the indicators.** The stored 15-minute bars include
  08:00 to 09:15 and 16:00 to 16:45 New York time, about 150 to 190 per slot against 384 in the
  session, and they carry 0.0 % of the volume: a few prints at stray prices, which then sit in the
  EMAs and the RSI. The 15-minute bars kept are those starting 09:30 to 15:45, the hourly bars those
  starting 09:00 to 15:00 (the 09:00 hour holds the trades of 09:30 to 10:00 and the trend reads
  only its closing price). The 08:00 and 16:00 hourly bars are dropped.
* **`skip_open_minutes`:** no entry on a signal bar that starts within that many minutes of the
  open (30 in `trend_cross_session_late`). The first two bars carry 9.2 % and 6.2 % of the day's
  volume and the literature puts the widest spreads and the wildest prices there.

The rule itself, its parameters and its stop and target are untouched: this is the same strategy
shown cleaner data and a later start, nothing tuned. The live loop does not use this class; see
the registry.
"""

from datetime import datetime
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_system.data.bars import normalize_bars
from trading_system.domain.models import StrategySignal
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import RUNTIME_STRATEGY, StrategyConfig
from trading_system.strategy.trend import generate_signals
from trading_system.strategy.trend_cross import TrendCrossStrategy

MARKET_TZ = ZoneInfo("America/New_York")

# Minutes after New York midnight.
SESSION_OPEN = 9 * 60 + 30
SESSION_CLOSE = 16 * 60
FIRST_HOURLY_START = 9 * 60  # the hour that holds 09:30 to 10:00


def _minutes(instants: pd.DatetimeIndex) -> np.ndarray:
    """The New York minute of the day of each instant (the clock change is handled by the zone)."""
    local = instants.tz_convert(MARKET_TZ)
    return np.asarray(local.hour * 60 + local.minute)


def _instants(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """The start of each row as a UTC time, from the `timestamp` column or the index."""
    if "timestamp" in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index
    raise ValueError("bars need a `timestamp` column or a DatetimeIndex")


def session_mask_15m(instants: pd.DatetimeIndex) -> np.ndarray:
    minutes = _minutes(instants)
    return (minutes >= SESSION_OPEN) & (minutes < SESSION_CLOSE)


def session_mask_1h(instants: pd.DatetimeIndex) -> np.ndarray:
    minutes = _minutes(instants)
    return (minutes >= FIRST_HOURLY_START) & (minutes < SESSION_CLOSE)


def _in_session(
    frame: pd.DataFrame, mask_of: Callable[[pd.DatetimeIndex], np.ndarray]
) -> pd.DataFrame:
    if frame.empty:
        return frame
    canonical = normalize_bars(frame)
    return canonical[mask_of(canonical.index)]


class SessionTrendCrossStrategy(BaseStrategy):
    def __init__(
        self,
        name: str,
        skip_open_minutes: int = 0,
        config: StrategyConfig = RUNTIME_STRATEGY,
    ):
        if skip_open_minutes < 0:
            raise ValueError("skip_open_minutes cannot be negative")
        self.name = name
        self.skip_open_minutes = skip_open_minutes
        suffix = "-session" + (f"-after{skip_open_minutes}" if skip_open_minutes else "")
        # A different version is a different strategy in the records (it goes into the signal id).
        self.config = config.model_copy(
            update={"strategy_version": config.strategy_version + suffix}
        )

    @property
    def version(self) -> str:
        return self.config.strategy_version

    def generate_signals(
        self,
        data_15m: Dict[str, pd.DataFrame],
        data_1h: Dict[str, pd.DataFrame],
        eval_time: datetime,
    ) -> List[StrategySignal]:
        clean_15m = {s: _in_session(df, session_mask_15m) for s, df in data_15m.items()}
        clean_1h = {s: _in_session(df, session_mask_1h) for s, df in data_1h.items()}
        signals = generate_signals(clean_15m, clean_1h, self.config, eval_time)
        if not self.skip_open_minutes:
            return signals
        earliest = SESSION_OPEN + self.skip_open_minutes
        return [s for s in signals if self._starts_late_enough(s, earliest)]

    @staticmethod
    def _starts_late_enough(signal: StrategySignal, earliest: int) -> bool:
        start = pd.DatetimeIndex([pd.Timestamp(signal.closed_bar_timestamp)])
        start = start.tz_localize("UTC") if start.tz is None else start
        return bool(_minutes(start)[0] >= earliest)

    def candidate_bars(
        self, bars_15m: pd.DataFrame, bars_1h: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """`trend_cross`'s hint computed on the session bars only and put back on the rows of the
        full frame (False for a bar outside the session), minus the bars before the late start."""
        instants = _instants(bars_15m)
        keep = session_mask_15m(instants)
        flags = np.zeros(len(bars_15m), dtype=bool)
        if keep.sum() < 2:
            return flags
        inner = TrendCrossStrategy(self.config).candidate_bars(bars_15m[keep])
        if inner is None:
            return None
        flags[np.flatnonzero(keep)] = inner
        if self.skip_open_minutes:
            flags &= _minutes(instants) >= SESSION_OPEN + self.skip_open_minutes
        return flags
