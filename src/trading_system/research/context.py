"""The state of the world at every signal (docs/strategy-research-plan.md, section 16.1, "A1").

A `signal_context` event is journalled for each signal that a loop handles: the stock's volatility
(ATR in percent), the time of the bar, the market (SPY above its 200-day average, its 20-day volatility
and how high that is in its own history), the stock's group in the paper3 portfolio and whether it
reports earnings within two sessions. Nothing is traded or refused because of it: the later splits
(section 16.3) need no new trial if the fields were recorded first.

Best effort, by design: whatever goes wrong while a context is built is logged and counted, and the
loop goes on. A field that cannot be known is `None`, never a guess, and nothing is filled in later.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_system.data.bars import normalize_bars
from trading_system.data.earnings_calendar import EarningsCalendar
from trading_system.domain.models import StrategySignal
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import calculate_indicators

logger = logging.getLogger(__name__)

SIGNAL_CONTEXT = "signal_context"
MARKET_TZ = ZoneInfo("America/New_York")
MIN_SPY_DAYS = 220  # the 200-day average and some history for the percentile
VOL_WINDOW = 20


@dataclass(frozen=True)
class MarketState:
    spy_above_200d: Optional[bool] = None
    spy_vol20_ann_pct: Optional[float] = None
    spy_vol20_pctile: Optional[float] = None
    spy_asof: Optional[str] = None


def compute_market_state(spy: Optional[pd.DataFrame], today: date) -> MarketState:
    """The market on the last completed day before `today` (New York date) from SPY's daily bars.
    Fewer than MIN_SPY_DAYS bars give all `None`."""
    if spy is None or spy.empty:
        return MarketState()
    local = pd.DatetimeIndex(spy.index).tz_convert(MARKET_TZ)
    keep = np.array([d < today for d in local.date])
    closes = spy.loc[keep, "close"].astype(float)
    if len(closes) < MIN_SPY_DAYS:
        return MarketState()
    last_day = local[keep][-1].date()
    log_returns = np.log(closes).diff().dropna()
    rolling = log_returns.rolling(VOL_WINDOW).std().dropna()
    now_vol = float(rolling.iloc[-1])
    return MarketState(
        spy_above_200d=bool(closes.iloc[-1] > closes.iloc[-200:].mean()),
        spy_vol20_ann_pct=round(now_vol * np.sqrt(252) * 100, 3),
        spy_vol20_pctile=round(float((rolling <= now_vol).mean() * 100), 2),
        spy_asof=last_day.isoformat(),
    )


def bar_facts(
    bars_15m: Optional[pd.DataFrame], closed_bar: pd.Timestamp, config: StrategyConfig
) -> tuple[Optional[float], Optional[float]]:
    """(ATR(14) of the 15-minute bars up to the signal bar as a percent of its close, that close)."""
    if bars_15m is None or bars_15m.empty:
        return None, None
    indicators = calculate_indicators(normalize_bars(bars_15m), config, is_htf=False)
    row = indicators[indicators["timestamp"] == closed_bar]
    if row.empty:
        return None, None
    close = float(row["close"].iloc[-1])
    atr = row["atr"].iloc[-1]
    if close <= 0 or pd.isna(atr):
        return None, close
    return round(float(atr) / close * 100, 4), close


def minutes_since_open(closed_bar: pd.Timestamp) -> int:
    """Minutes from 09:30 New York time to the START of the signal bar (the first bar is 0)."""
    local = closed_bar.tz_convert(MARKET_TZ)
    return int((local.hour * 60 + local.minute) - (9 * 60 + 30))


def _utc(instant: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(instant)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class ContextRecorder:
    """Journals a `signal_context` for each signal it is given. Never raises."""

    def __init__(
        self,
        audit_repo: Any,
        source: str,
        *,
        config: StrategyConfig,
        groups: Mapping[str, str],
        spy_loader: Callable[[], Optional[pd.DataFrame]],
        calendar: Optional[EarningsCalendar] = None,
    ) -> None:
        self.audit_repo = audit_repo
        self.source = source
        self.config = config
        self.groups = dict(groups)
        self._spy_loader = spy_loader
        self.calendar = calendar
        self._state: Optional[MarketState] = None
        self._state_day: Optional[date] = None
        self._seen: set[tuple[str, str, str]] = set()
        self.errors = 0

    def _market(self, today: date) -> MarketState:
        if self._state is not None and self._state_day == today:
            return self._state
        try:
            state = compute_market_state(self._spy_loader(), today)
        except Exception as e:  # e.g. the research file is being written; try again next time
            logger.warning(f"Could not read SPY for the signal context: {e}")
            self.errors += 1
            return MarketState()
        if (
            state.spy_asof is not None
        ):  # not enough history is not cached, so a later read can fill it
            self._state, self._state_day = state, today
        return state

    def record(
        self,
        signals: Iterable[StrategySignal],
        bars_15m: Mapping[str, pd.DataFrame],
        when: datetime,
        source: Optional[str] = None,
    ) -> int:
        """One event per new (symbol, bar, source). Returns how many were written."""
        written = 0
        label = source or self.source
        today = when.astimezone(MARKET_TZ).date()
        for signal in signals:
            try:
                closed_bar = _utc(signal.closed_bar_timestamp)
                key = (signal.symbol, closed_bar.isoformat(), label)
                if key in self._seen:
                    continue
                self._seen.add(key)
                atr_pct, close = bar_facts(bars_15m.get(signal.symbol), closed_bar, self.config)
                state = self._market(today)
                details: dict[str, Any] = {
                    "symbol": signal.symbol,
                    "closed_bar": closed_bar.isoformat(),
                    "source": label,
                    "group": self.groups.get(signal.symbol, "other"),
                    "minutes_since_open": minutes_since_open(closed_bar),
                    "close": close,
                    "atr_pct": atr_pct,
                    "spy_above_200d": state.spy_above_200d,
                    "spy_vol20_ann_pct": state.spy_vol20_ann_pct,
                    "spy_vol20_pctile": state.spy_vol20_pctile,
                    "spy_asof": state.spy_asof,
                    **self._earnings(signal.symbol, closed_bar),
                }
                self.audit_repo.record(SIGNAL_CONTEXT, details, when=when)
                written += 1
            except Exception as e:  # a context must never stop a loop
                logger.warning(f"Could not journal the context of {signal.symbol}: {e}")
                self.errors += 1
        return written

    def _earnings(self, symbol: str, closed_bar: pd.Timestamp) -> dict[str, Any]:
        if self.calendar is None:
            return {"earnings_known": False, "earnings_in_2_sessions": None, "earnings_dates": []}
        try:
            day = closed_bar.tz_convert(MARKET_TZ).date()
            info = self.calendar.lookup(symbol, day)
        except Exception as e:
            logger.warning(f"Could not look up earnings for {symbol}: {e}")
            self.errors += 1
            return {"earnings_known": False, "earnings_in_2_sessions": None, "earnings_dates": []}
        return {
            "earnings_known": info.known,
            "earnings_in_2_sessions": info.in_2_sessions,
            "earnings_dates": info.dates,
        }
