import json
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from tests.fixtures.bars import make_bars
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.oracle.inputs import RECENT_BARS, build_oracle_request
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.trend import calculate_indicators

NOW = datetime(2026, 11, 2, 16, 10, tzinfo=timezone.utc)
CONTEXT = {"equity": 100000.0, "open_positions": 1}


def _signal() -> StrategySignal:
    return StrategySignal(
        version="1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=datetime(2026, 11, 2, 15, 45, tzinfo=timezone.utc),
        direction=SignalDirection.LONG,
    )


def _request(bars_15m: pd.DataFrame | None = None, live: str = "101.50"):
    m15, h1 = make_bars(NOW)
    return build_oracle_request(
        _signal(),
        m15 if bars_15m is None else bars_15m,
        h1,
        RUNTIME_STRATEGY,
        live_price=Decimal(live),
        context=CONTEXT,
    )


def test_the_indicators_are_the_ones_the_strategy_computed() -> None:
    m15, _ = make_bars(NOW)
    last = calculate_indicators(m15, RUNTIME_STRATEGY).iloc[-1]

    technicals = _request().technicals

    assert technicals["close"] == round(float(last["close"]), 4)
    assert technicals["ema_fast"] == round(float(last["ema_fast"]), 4)
    assert technicals["ema_slow"] == round(float(last["ema_slow"]), 4)
    assert technicals["rsi"] == round(float(last["rsi"]), 2)
    assert technicals["atr"] == round(float(last["atr"]), 4)  # which the strategy does not use
    # The cross that produced the signal is visible in the numbers: below before, above now.
    assert technicals["ema_fast_previous"] <= technicals["ema_slow_previous"]
    assert technicals["ema_fast"] > technicals["ema_slow"]


def test_the_hourly_trend_comes_from_the_hour_closed_by_then() -> None:
    technicals = _request().technicals

    assert technicals["hourly_trend_up"] is True  # make_bars builds an uptrend
    assert technicals["hourly_ema_fast"] > technicals["hourly_ema_slow"]


def test_the_live_price_and_how_far_it_is_from_the_last_close_are_given() -> None:
    request = _request(live="101.50")

    assert request.live_price == Decimal("101.50")
    close = request.technicals["close"]
    expected = round((101.50 - close) / close * 100, 3)
    assert request.technicals["live_price_vs_close_pct"] == expected


def test_the_last_bars_come_oldest_first_and_end_on_the_signal_bar() -> None:
    m15, _ = make_bars(NOW)

    bars = _request().recent_bars

    assert len(bars) == RECENT_BARS
    assert [b["start"] for b in bars] == sorted(b["start"] for b in bars)
    assert bars[-1]["close"] == round(float(m15["close"].iloc[-1]), 4)
    assert set(bars[0]) == {"start", "open", "high", "low", "close", "volume"}


def test_volume_is_compared_with_the_bars_before_it() -> None:
    m15, _ = make_bars(NOW)
    assert _request().volume_ratio == 1.0  # the fixture's volume is constant

    busy = m15.copy()
    busy.iloc[-1, busy.columns.get_loc("volume")] = 3000.0
    assert _request(busy).volume_ratio == 3.0


def test_what_the_history_cannot_support_is_none_not_nan() -> None:
    """A short history has no ATR yet, and JSON has no NaN."""
    m15, _ = make_bars(NOW)

    request = _request(m15.tail(3))

    assert request.technicals["atr"] is None  # needs 14 bars; three were given
    assert request.volume_ratio is None
    json.loads(request.model_dump_json())  # serialisable: this is what a provider sends


def test_the_request_is_numbers_and_fixed_labels_only() -> None:
    """Nothing in it comes from the outside world, so a prompt injection has nowhere to ride."""
    request = _request()

    def leaves(value: object):
        if isinstance(value, dict):
            for v in value.values():
                yield from leaves(v)
        elif isinstance(value, list):
            for v in value:
                yield from leaves(v)
        else:
            yield value

    payload = json.loads(request.model_dump_json(exclude={"signal"}))
    texts = [leaf for leaf in leaves(payload) if isinstance(leaf, str)]
    # Only the fixed rationale, the bar start times, and the decimal live price (a string in JSON).
    assert request.rationale in texts
    assert all(
        t == request.rationale or t.startswith("20") or t.replace(".", "").isdigit() for t in texts
    )
    assert request.context == CONTEXT
