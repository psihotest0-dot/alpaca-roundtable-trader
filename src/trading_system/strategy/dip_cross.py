"""`dip_in_uptrend`: buy a sharp dip inside a rising hourly trend (docs/strategy-research-plan.md,
section 14, variant 19, registered on 2026-11-09 before any code and before any signal).

The live strategy waits for a 15-minute EMA cross-up with the RSI in a band of 30 to 70 and so
never enters when the RSI is below 30. This candidate is the opposite reading of the same
indicators:

* the last closed hourly bar has EMA(12) above EMA(26) (the live strategy's own trend test);
* the last closed 15-minute bar has RSI(14) below 30 (Wilder, the live strategy's own function);
* no entry on a signal bar that starts before 10:00 New York time (the first two bars are the
  wildest, as in `trend_cross_session_late`);
* only bars of the regular session reach the indicators (as in `trend_cross_session`).

The numbers are fixed and are not tuned; another number is another variant. There is NO cross-up
requirement, so the same dip can signal on several bars in a row; the judge plays the signals out
with one open trade per symbol and strategy, which is what makes that harmless. The live loop does
not use this class; it only runs forward in the observer, without orders and without a question to
the Oracle (the registry says the same).
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.data.bars import normalize_bars
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import RUNTIME_STRATEGY, StrategyConfig
from trading_system.strategy.session_cross import (
    SESSION_OPEN,
    _in_session,
    _instants,
    _minutes,
    session_mask_1h,
    session_mask_15m,
)
from trading_system.strategy.trend import calculate_indicators

RSI_BELOW = 30.0  # a dip: the live strategy's lower RSI bound, read from the other side
SKIP_OPEN_MINUTES = 30  # no entry on a signal bar that starts before 10:00 New York

# A bar whose RSI is this close to the line is passed on to the strategy, which decides, so the
# backtest's hint never drops a signal over a rounding difference.
_MARGIN = 1e-6


class DipInUptrendStrategy(BaseStrategy):
    name = "dip_in_uptrend"

    def __init__(self, config: StrategyConfig = RUNTIME_STRATEGY):
        # A different version is a different strategy in the records (it goes into the signal id).
        self.config = config.model_copy(
            update={"strategy_version": config.strategy_version + "-dip"}
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
        config = self.config
        signals: List[StrategySignal] = []
        for symbol, raw_15m in data_15m.items():
            raw_1h = data_1h.get(symbol)
            if raw_1h is None or raw_15m.empty or raw_1h.empty:
                continue
            df_15 = _in_session(raw_15m, session_mask_15m)
            df_1h = _in_session(raw_1h, session_mask_1h)
            if len(df_15) < config.min_warmup_bars or len(df_1h) < config.min_warmup_bars:
                continue
            df_15 = calculate_indicators(normalize_bars(df_15), config, is_htf=False)
            df_1h = calculate_indicators(normalize_bars(df_1h), config, is_htf=True)

            # Closed bars only: a bar is closed once its start plus its length has passed.
            closed_15 = df_15[df_15["timestamp"] + pd.Timedelta(minutes=15) <= eval_time]
            closed_1h = df_1h[df_1h["timestamp"] + pd.Timedelta(hours=1) <= eval_time]
            if closed_15.empty or closed_1h.empty:
                continue

            last_bar = closed_15.iloc[-1]
            rsi = last_bar["rsi"]
            if pd.isna(rsi) or not rsi < RSI_BELOW:
                continue

            last_time = last_bar["timestamp"]
            if _minutes(pd.DatetimeIndex([last_time]))[0] < SESSION_OPEN + SKIP_OPEN_MINUTES:
                continue

            # Only hourly bars that had closed by the 15-minute bar's close: nothing from after it.
            last_close = last_time + pd.Timedelta(minutes=15)
            aligned = closed_1h[closed_1h["timestamp"] + pd.Timedelta(hours=1) <= last_close]
            if aligned.empty:
                continue
            last_1h = aligned.iloc[-1]
            htf_fast, htf_slow = last_1h["ema_fast"], last_1h["ema_slow"]
            if pd.isna(htf_fast) or pd.isna(htf_slow) or not htf_fast > htf_slow:
                continue

            signals.append(
                StrategySignal(
                    version=config.strategy_version,
                    symbol=symbol,
                    timeframe="15m",
                    closed_bar_timestamp=last_time,
                    direction=SignalDirection.LONG,
                    # Valid for the bar after the one that produced it (as in the live strategy).
                    expires_at=last_close + pd.Timedelta(minutes=15),
                    confidence=1.0,
                )
            )
        return signals

    def candidate_bars(
        self, bars_15m: pd.DataFrame, bars_1h: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """Possible where the session RSI is below the line and the bar starts late enough, put
        back on the rows of the full frame (False outside the session)."""
        instants = _instants(bars_15m)
        keep = session_mask_15m(instants)
        flags = np.zeros(len(bars_15m), dtype=bool)
        if keep.sum() < 2:
            return flags
        inner = calculate_indicators(normalize_bars(bars_15m[keep]), self.config, is_htf=False)
        rsi = inner["rsi"].to_numpy(dtype=float)
        flags[np.flatnonzero(keep)] = rsi < RSI_BELOW + _MARGIN
        flags &= _minutes(instants) >= SESSION_OPEN + SKIP_OPEN_MINUTES
        return flags
