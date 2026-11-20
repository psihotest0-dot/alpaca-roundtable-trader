"""The state of the world at a signal (research/context.py, plan section 16.1)."""

from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_system.data.earnings_calendar import EarningsInfo
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.research.context import (
    SIGNAL_CONTEXT,
    ContextRecorder,
    MarketState,
    bar_facts,
    compute_market_state,
    minutes_since_open,
)
from trading_system.strategy.config import RUNTIME_STRATEGY

TODAY = date(2026, 11, 20)
WHEN = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)  # 12:00 New York


def spy_bars(n: int = 400, drift: float = 0.0004, vol: float = 0.01, seed: int = 3) -> pd.DataFrame:
    """n daily bars ending the day before TODAY, indexed by the UTC time at 05:00 as Alpaca does."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp(TODAY) - pd.Timedelta(days=1), periods=n)
    closes = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    index = pd.DatetimeIndex([d + pd.Timedelta(hours=5) for d in days]).tz_localize("UTC")
    return pd.DataFrame(
        {"close": closes, "open": closes, "high": closes, "low": closes}, index=index
    )


def signal(symbol: str = "AAA", bar: str = "2026-11-20T16:45:00+00:00") -> StrategySignal:
    return StrategySignal(
        version="1.0.0",
        symbol=symbol,
        timeframe="15m",
        closed_bar_timestamp=pd.Timestamp(bar).to_pydatetime(),
        direction=SignalDirection.LONG,
        expires_at=WHEN + timedelta(minutes=30),
    )


def bars_15m(until: str = "2026-11-20T16:45:00+00:00", n: int = 120) -> pd.DataFrame:
    end = pd.Timestamp(until)
    index = pd.date_range(end=end, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(5)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


class FakeAudit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any], datetime]] = []

    def record(self, event_type: str, details: dict[str, Any], *, when: datetime) -> None:
        self.events.append((event_type, details, when))


class FakeCalendar:
    def __init__(self, info: EarningsInfo | Exception) -> None:
        self.info = info
        self.asked: list[tuple[str, date]] = []

    def lookup(self, symbol: str, day: date) -> EarningsInfo:
        self.asked.append((symbol, day))
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


def recorder(
    audit: FakeAudit,
    *,
    spy: Any = None,
    calendar: Any = None,
    groups: dict[str, str] | None = None,
) -> ContextRecorder:
    bars = spy_bars() if spy is None else spy
    return ContextRecorder(
        audit,
        "paper3",
        config=RUNTIME_STRATEGY,
        groups=groups or {"AAA": "risky"},
        spy_loader=(lambda: bars) if not callable(bars) else bars,
        calendar=calendar,
    )


# --------------------------------------------------------------------------- market state


def test_market_state_reads_the_last_completed_day_and_the_trend() -> None:
    up = compute_market_state(spy_bars(drift=0.002), TODAY)
    down = compute_market_state(spy_bars(drift=-0.002), TODAY)

    assert up.spy_asof == "2026-11-19" and up.spy_above_200d is True
    assert down.spy_above_200d is False
    assert up.spy_vol20_ann_pct is not None and 0 < up.spy_vol20_ann_pct < 100


def test_market_state_leaves_out_the_day_in_progress() -> None:
    bars = spy_bars()
    today_bar = bars.iloc[[-1]].copy()
    today_bar.index = today_bar.index + pd.Timedelta(days=1)  # a bar dated TODAY
    today_bar["close"] = 1.0  # would flip everything if it were read
    state = compute_market_state(pd.concat([bars, today_bar]), TODAY)

    assert state == compute_market_state(bars, TODAY)


def test_the_percentile_is_the_share_of_calmer_or_equal_days_in_the_history() -> None:
    state = compute_market_state(spy_bars(), TODAY)
    bars = spy_bars()
    returns = np.log(bars["close"]).diff().dropna()
    rolling = returns.rolling(20).std().dropna()
    expected = float((rolling <= rolling.iloc[-1]).mean() * 100)

    assert state.spy_vol20_pctile == pytest.approx(expected, abs=0.01)


def test_market_state_is_none_with_too_little_history() -> None:
    assert compute_market_state(spy_bars(n=150), TODAY) == MarketState()
    assert compute_market_state(None, TODAY) == MarketState()
    assert compute_market_state(spy_bars().iloc[0:0], TODAY) == MarketState()


# --------------------------------------------------------------------------- the bar


def test_minutes_since_open_counts_to_the_start_of_the_bar() -> None:
    assert minutes_since_open(pd.Timestamp("2026-11-20T14:30:00+00:00")) == 0  # 09:30 New York
    assert minutes_since_open(pd.Timestamp("2026-11-20T16:45:00+00:00")) == 135  # 11:45
    assert minutes_since_open(pd.Timestamp("2026-07-20T13:30:00+00:00")) == 0  # summer time


def test_atr_percent_is_atr_of_the_signal_bar_over_its_close() -> None:
    bars = bars_15m()
    atr_pct, close = bar_facts(bars, pd.Timestamp("2026-11-20T16:45:00+00:00"), RUNTIME_STRATEGY)

    assert close == pytest.approx(float(bars["close"].iloc[-1]))
    assert atr_pct is not None and 0.1 < atr_pct < 1.0  # bars are 0.4 % wide, ATR about that

    earlier = bar_facts(bars, pd.Timestamp("2026-11-20T15:00:00+00:00"), RUNTIME_STRATEGY)
    assert earlier[1] != close  # the row of that bar, not the last one


def test_a_bar_that_is_not_there_gives_nothing() -> None:
    assert bar_facts(bars_15m(), pd.Timestamp("2025-01-01T15:00:00+00:00"), RUNTIME_STRATEGY) == (
        None,
        None,
    )
    assert bar_facts(None, pd.Timestamp("2026-11-20T16:45:00+00:00"), RUNTIME_STRATEGY) == (
        None,
        None,
    )


# --------------------------------------------------------------------------- the recorder


def test_a_signal_gets_every_registered_field() -> None:
    audit = FakeAudit()
    calendar = FakeCalendar(
        EarningsInfo(
            known=True, in_2_sessions=True, dates=[{"date": "2026-11-23", "timing": "pre"}]
        )
    )

    written = recorder(audit, calendar=calendar).record([signal()], {"AAA": bars_15m()}, WHEN)

    assert written == 1
    ((kind, details, when),) = audit.events
    assert kind == SIGNAL_CONTEXT and when == WHEN
    assert set(details) == {
        "symbol",
        "closed_bar",
        "source",
        "group",
        "minutes_since_open",
        "close",
        "atr_pct",
        "spy_above_200d",
        "spy_vol20_ann_pct",
        "spy_vol20_pctile",
        "spy_asof",
        "earnings_known",
        "earnings_in_2_sessions",
        "earnings_dates",
    }
    assert details["symbol"] == "AAA" and details["source"] == "paper3"
    assert details["group"] == "risky" and details["minutes_since_open"] == 135
    assert details["closed_bar"] == "2026-11-20T16:45:00+00:00"
    assert details["spy_asof"] == "2026-11-19" and details["atr_pct"] is not None
    assert details["earnings_known"] is True and details["earnings_in_2_sessions"] is True
    assert calendar.asked == [("AAA", date(2026, 11, 20))]  # the New York date of the bar


def test_a_stock_outside_the_portfolio_is_group_other() -> None:
    audit = FakeAudit()

    recorder(audit).record([signal("ZZZ")], {"ZZZ": bars_15m()}, WHEN)

    assert audit.events[0][1]["group"] == "other"


def test_the_source_can_be_named_per_call_for_the_candidates() -> None:
    audit = FakeAudit()

    recorder(audit).record([signal()], {"AAA": bars_15m()}, WHEN, source="candidate:dip_in_uptrend")

    assert audit.events[0][1]["source"] == "candidate:dip_in_uptrend"


def test_the_same_bar_from_the_same_source_is_written_once() -> None:
    audit = FakeAudit()
    rec = recorder(audit)

    rec.record([signal()], {"AAA": bars_15m()}, WHEN)
    rec.record([signal()], {"AAA": bars_15m()}, WHEN)
    rec.record([signal()], {"AAA": bars_15m()}, WHEN, source="observer")

    assert [e[1]["source"] for e in audit.events] == ["paper3", "observer"]


def test_without_a_calendar_or_with_a_broken_one_the_earnings_are_not_known_not_none() -> None:
    audit = FakeAudit()
    recorder(audit).record([signal()], {"AAA": bars_15m()}, WHEN)
    broken = FakeCalendar(RuntimeError("down"))
    rec = recorder(audit, calendar=broken)
    rec.record([signal("BBB")], {"BBB": bars_15m()}, WHEN)

    for _, details, _ in audit.events:
        assert details["earnings_known"] is False
        assert details["earnings_in_2_sessions"] is None and details["earnings_dates"] == []
    assert rec.errors == 1


def test_spy_that_cannot_be_read_leaves_the_fields_empty_and_is_tried_again() -> None:
    audit = FakeAudit()
    calls: list[int] = []

    def flaky() -> pd.DataFrame:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return spy_bars()

    rec = recorder(audit, spy=flaky)

    rec.record([signal("AAA")], {"AAA": bars_15m()}, WHEN)
    rec.record([signal("BBB")], {"BBB": bars_15m()}, WHEN)
    rec.record([signal("CCC")], {"CCC": bars_15m()}, WHEN)

    first, second, third = (e[1] for e in audit.events)
    assert first["spy_above_200d"] is None and first["spy_asof"] is None
    assert second["spy_asof"] == "2026-11-19" and third["spy_asof"] == "2026-11-19"
    assert len(calls) == 2  # a good read is kept for the day, a failed one is not


def test_one_signal_that_breaks_does_not_stop_the_others_and_nothing_raises() -> None:
    audit = FakeAudit()
    rec = recorder(audit)
    bad = signal("BAD")
    object.__setattr__(bad, "closed_bar_timestamp", "not a time")

    written = rec.record([bad, signal("AAA")], {"AAA": bars_15m()}, WHEN)

    assert written == 1 and [e[1]["symbol"] for e in audit.events] == ["AAA"]
    assert rec.errors == 1
