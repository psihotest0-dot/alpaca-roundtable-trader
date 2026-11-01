"""C1: the live data path is broken in three stacked places. These tests pin the two
that live in AlpacaData and the strategy; the third (the invalid TimeFrame in the
orchestrator) is in test_runtime_cycle.py.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from tests.fixtures.bars import RUNTIME_STRATEGY, alpaca_shaped, make_bars, with_timestamp_column
from tests.fixtures.fakes import FakeStockClient
from trading_system.data.alpaca_data import AlpacaData
from trading_system.strategy.trend import generate_signals

FIFTEEN_MIN = TimeFrame(15, TimeFrameUnit.Minute)


def test_control_engineered_bars_produce_one_long_signal() -> None:
    """Sanity check of the fixture itself: with `timestamp` columns (the shape the
    existing unit tests use) the engineered series yields exactly one LONG signal."""
    now = datetime.now(timezone.utc)
    bars_15m, bars_1h = make_bars(now)

    signals = generate_signals(
        {"AAPL": with_timestamp_column(bars_15m)},
        {"AAPL": with_timestamp_column(bars_1h)},
        RUNTIME_STRATEGY,
        now,
    )

    assert [s.symbol for s in signals] == ["AAPL"]


@pytest.mark.xfail(
    strict=True,
    raises=UnboundLocalError,
    reason="C1c: fetch_bars imports pandas inside some branches, which makes `pd` a local "
    "variable for the whole function; the normal path reads it before it is assigned. "
    "The retry decorator then repeats this deterministic bug 5 times with growing sleeps.",
)
def test_fetch_bars_returns_the_bars_alpaca_sent(
    settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)  # skip the retry backoff
    now = datetime.now(timezone.utc)
    bars_15m, bars_1h = make_bars(now)
    data = AlpacaData(settings)
    data.client = FakeStockClient({"AAPL": bars_15m}, {"AAPL": bars_1h})

    frame = data.fetch_bars("AAPL", FIFTEEN_MIN, now - timedelta(days=30), now)

    assert len(frame) == len(bars_15m)


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="C1: fetch_bars is meant to return the timestamp as the index (it drops the symbol "
    "level of alpaca-py's MultiIndex), but generate_signals reads df['timestamp'].",
)
def test_bars_shaped_like_fetch_bars_output_flow_into_generate_signals(settings: Any) -> None:
    now = datetime.now(timezone.utc)
    bars_15m, bars_1h = make_bars(now)
    data = AlpacaData(settings)

    # What fetch_bars returns once C1c is fixed: its own `reset_index(level=0, drop=True)`.
    df_15m = alpaca_shaped("AAPL", bars_15m).reset_index(level=0, drop=True)
    df_1h = alpaca_shaped("AAPL", bars_1h).reset_index(level=0, drop=True)
    df_15m, df_1h = data.align_and_filter(
        data.clean_and_deduplicate(df_15m), data.clean_and_deduplicate(df_1h), now
    )

    signals = generate_signals({"AAPL": df_15m}, {"AAPL": df_1h}, RUNTIME_STRATEGY, now)

    assert [s.symbol for s in signals] == ["AAPL"]
