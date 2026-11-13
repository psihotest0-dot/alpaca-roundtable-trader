"""The two strategies of the original Money_bot, ported (Phase 3.3a).

Source: `tools/indicators.py` of https://github.com/psihotest0-dot/Money-bot (commit c7560c6),
classes `EMACrossoverStrategy` (the "stable" bot) and `StrictEMACrossoverStrategy` (the
"risky" bot), with the indicators as `bot_engine.py` computes them. Despite the names they
are not crossovers but a trend with a pullback to the fast EMA:

    LONG when  fast EMA > slow EMA            (uptrend)
          and  |close - fast EMA| / fast EMA <= pullback_pct   (the price is back at the EMA)
          and  close > the hourly EMA 200
          [strict only] and  volume >= 0.8 x its 20-bar average

Stable: EMA 9/21, pullback 0.5 %. Strict: EMA 20/50, pullback 1 %, with the volume filter.

What was changed, on purpose (the plan's rules for the port):
- LONG only: the original also sold short.
- Closed bars only: the original judged the last row, which could be a bar still forming.
- A signal is valid for one bar (like the other strategy), and the risk engine, not the
  strategy, sizes the position and places the stop and target.

What was kept exactly: the EMAs and the volume average use `min_periods` equal to their
length (so they are empty until that many bars exist), the hourly EMA 200 uses
`min_periods=200`, the hourly bars used are those that started before the hour of the 15-minute
bar (`index < floor('h')`), and a signal is given on every bar where the conditions hold, not
only on the first (the original acted on every evaluation while it held no position).

One difference in a corner: with fewer than 200 hourly bars the original's EMA 200 is NaN and
the stable bot's filter `close <= NaN` is False, so it let the signal through; here a missing
EMA 200 means no signal, as it does for the strict bot.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from trading_system.data.bars import normalize_bars
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.runtime.schedule import BAR
from trading_system.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

# The hint for the backtest accepts a bar whose numbers are this close (relative) to a line.
# The hourly EMA 200 decays slowly, so the same average computed over the whole history and
# over the window it is shown (`htf_history_days`, 90 days) can differ in the fifth decimal: its margin is wider.
_MARGIN = 1e-6
_HTF_MARGIN = 1e-3


class EmaCrossoverConfig(BaseModel):
    strategy_version: str
    short_window: int = Field(gt=0)
    long_window: int = Field(gt=0)
    # How far the close may be from the fast EMA, as a fraction of it (0.005 = 0.5 %).
    pullback_pct: float = Field(gt=0, le=0.5)
    # The strict bot only: the bar's volume must be at least `min_relative_volume` times
    # its average over `volume_window` bars.
    volume_window: Optional[int] = Field(default=None, gt=0)
    min_relative_volume: float = 0.8
    htf_ema_period: int = Field(default=200, gt=0)


EMA_CROSSOVER = EmaCrossoverConfig(
    strategy_version="ema-crossover-9-21-0.5pct-v1",
    short_window=9,
    long_window=21,
    pullback_pct=0.005,
)
STRICT_EMA_CROSSOVER = EmaCrossoverConfig(
    strategy_version="strict-ema-crossover-20-50-1pct-v1",
    short_window=20,
    long_window=50,
    pullback_pct=0.01,
    volume_window=20,
)
# The same two with the distance from the fast EMA widened to 5 % (the owner's wish to look at
# a looser entry): they are not the originals, they say what the looser entry does.
EMA_CROSSOVER_5PCT = EMA_CROSSOVER.model_copy(
    update={"strategy_version": "ema-crossover-9-21-5pct-v1", "pullback_pct": 0.05}
)
STRICT_EMA_CROSSOVER_5PCT = STRICT_EMA_CROSSOVER.model_copy(
    update={"strategy_version": "strict-ema-crossover-20-50-5pct-v1", "pullback_pct": 0.05}
)


class EmaCrossoverStrategy(BaseStrategy):
    name = "ema_crossover"

    # The hourly EMA 200 needs 200 hourly bars (`min_periods`); the stored bars hold about 5 per
    # calendar day, so the live loop's 30 days (about 150) are not enough and the strategy would
    # give no signal at all, without an error (found 2026-11-13, after HTF_LOOKBACK_DAYS went from
    # 90 to 30). 90 days is the window both strategies were measured with (-1.35 % and -0.81 %
    # over 180 days), so those baselines can still be reproduced.
    htf_history_days = 90

    def __init__(self, config: EmaCrossoverConfig = EMA_CROSSOVER, name: Optional[str] = None):
        self.config = config
        self._warned_short_history = False
        if name:
            self.name = name

    @property
    def version(self) -> str:
        return self.config.strategy_version

    def _indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """The columns the original adds before it judges: short_ema, long_ema (and
        volume_sma for the strict bot), each empty until its window is full."""
        config = self.config
        out = pd.DataFrame(index=bars.index)
        out["short_ema"] = (
            bars["close"]
            .ewm(span=config.short_window, min_periods=config.short_window, adjust=False)
            .mean()
        )
        out["long_ema"] = (
            bars["close"]
            .ewm(span=config.long_window, min_periods=config.long_window, adjust=False)
            .mean()
        )
        if config.volume_window:
            out["volume_sma"] = (
                bars["volume"]
                .rolling(window=config.volume_window, min_periods=config.volume_window)
                .mean()
            )
        return out

    def _htf_ema(self, hourly: pd.DataFrame) -> pd.Series:
        period = self.config.htf_ema_period
        return hourly["close"].ewm(span=period, min_periods=period, adjust=False).mean()

    def generate_signals(
        self,
        data_15m: Dict[str, pd.DataFrame],
        data_1h: Dict[str, pd.DataFrame],
        eval_time: datetime,
    ) -> List[StrategySignal]:
        config = self.config
        signals: List[StrategySignal] = []
        for symbol, bars in data_15m.items():
            hourly = data_1h.get(symbol)
            if bars.empty or hourly is None or hourly.empty:
                continue
            bars = normalize_bars(bars)
            hourly = normalize_bars(hourly)
            # The original's guards (it asks the hourly frame for as many rows as the slow EMA).
            if len(bars) < config.long_window + 1 or len(hourly) < config.long_window + 1:
                continue
            if len(hourly) < config.htf_ema_period and not self._warned_short_history:
                # "No signal because the hourly EMA is undefined" is not "no signal found".
                self._warned_short_history = True
                logger.warning(
                    "%s: %s has %d hourly bars, fewer than the %d the hourly EMA needs; "
                    "there can be no signal until the window is longer (HTF window too short)",
                    self.name,
                    symbol,
                    len(hourly),
                    config.htf_ema_period,
                )

            closed = (bars["timestamp"] + BAR <= eval_time).to_numpy()
            if not closed.any():
                continue
            i = int(np.flatnonzero(closed)[-1])
            ind = self._indicators(bars)
            short, long_, close = (
                float(ind["short_ema"].iloc[i]),
                float(ind["long_ema"].iloc[i]),
                float(bars["close"].iloc[i]),
            )
            if not (short > long_):  # also False while either average is still empty
                continue
            if not abs(close - short) / short <= config.pullback_pct:
                continue
            if config.volume_window:
                average = float(ind["volume_sma"].iloc[i])
                if np.isnan(average) or float(bars["volume"].iloc[i]) < average * (
                    config.min_relative_volume
                ):
                    continue

            bar_start = bars["timestamp"].iloc[i]
            htf_rows = (hourly["timestamp"] < bar_start.floor("h")).to_numpy()
            if not htf_rows.any():
                continue
            htf_ema = float(self._htf_ema(hourly).iloc[int(np.flatnonzero(htf_rows)[-1])])
            if np.isnan(htf_ema) or not close > htf_ema:
                continue

            signals.append(
                StrategySignal(
                    version=config.strategy_version,
                    symbol=symbol,
                    timeframe="15m",
                    closed_bar_timestamp=bar_start,
                    direction=SignalDirection.LONG,
                    # Valid for the bar after the one that produced it, like the other strategy.
                    expires_at=bar_start + BAR + BAR,
                    confidence=1.0,
                )
            )
        return signals

    def candidate_bars(
        self, bars_15m: pd.DataFrame, bars_1h: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """Possible where the uptrend, the pullback, the volume and the hourly EMA 200 all
        (nearly) hold, from indicators computed once for the whole history."""
        if bars_1h is None or bars_1h.empty:
            return None
        config = self.config
        ind = self._indicators(bars_15m)
        short = ind["short_ema"].to_numpy(dtype=float)
        long_ = ind["long_ema"].to_numpy(dtype=float)
        close = bars_15m["close"].to_numpy(dtype=float)

        flags = (short > long_ * (1 - _MARGIN)) & (
            np.abs(close - short) <= config.pullback_pct * short * (1 + _MARGIN)
        )
        if config.volume_window:
            average = ind["volume_sma"].to_numpy(dtype=float)
            volume = bars_15m["volume"].to_numpy(dtype=float)
            flags &= volume >= average * config.min_relative_volume * (1 - _MARGIN)

        # The hourly average at each 15-minute bar: the last hourly bar that started before
        # the hour the bar belongs to.
        htf_ema = self._htf_ema(bars_1h).to_numpy(dtype=float)
        row = bars_1h.index.searchsorted(bars_15m.index.floor("h"), side="left") - 1
        usable = row >= 0
        reference = np.where(usable, htf_ema[np.where(usable, row, 0)], np.nan)
        flags &= close > reference * (1 - _HTF_MARGIN)
        out: np.ndarray = flags
        return out
