"""C1, C1c: the live data path, from Alpaca's response to the strategy's input.

Before task 1.3 this path was broken in three stacked places (the third, the invalid
TimeFrame in the orchestrator, is covered in test_runtime_cycle.py).
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from tests.fixtures.bars import RUNTIME_STRATEGY, make_bars, with_timestamp_column
from tests.fixtures.fakes import FakeStockClient
from trading_system.data.alpaca_data import FIFTEEN_MINUTES, AlpacaData
from trading_system.strategy.trend import generate_signals


def _data_with(settings: Any, bars_15m: pd.DataFrame, bars_1h: pd.DataFrame) -> AlpacaData:
    data = AlpacaData(settings)
    data.client = FakeStockClient({"AAPL": bars_15m}, {"AAPL": bars_1h})
    return data


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


def test_fetch_bars_returns_the_bars_alpaca_sent(settings: Any) -> None:
    """C1c. fetch_bars raised UnboundLocalError on its normal path, because it imported
    pandas inside some branches and so made `pd` a local name for the whole function."""
    now = datetime.now(timezone.utc)
    bars_15m, bars_1h = make_bars(now)
    data = _data_with(settings, bars_15m, bars_1h)

    frame = data.fetch_bars("AAPL", FIFTEEN_MINUTES, now - timedelta(days=30), now)

    assert len(frame) == len(bars_15m)
    assert str(frame.index.tz) == "UTC"
    assert list(frame["timestamp"]) == list(frame.index)
    assert {"open", "high", "low", "close", "volume"} <= set(frame.columns)


def test_bars_loaded_through_the_data_layer_flow_into_generate_signals(settings: Any) -> None:
    """C1. The data layer used to return the timestamp as the index only, and
    generate_signals read df['timestamp'] and raised KeyError. Same series, but through
    the real AlpacaData code with a client that answers in alpaca-py's real shape."""
    now = datetime.now(timezone.utc)
    bars_15m, bars_1h = make_bars(now)
    data = _data_with(settings, bars_15m, bars_1h)

    df_15m, df_1h = data.load_universe(["AAPL"], now, lookback_days=30, htf_lookback_days=90)
    signals = generate_signals(df_15m, df_1h, RUNTIME_STRATEGY, now)

    assert [s.symbol for s in signals] == ["AAPL"]


def test_generate_signals_accepts_bars_with_the_timestamp_only_as_index() -> None:
    """The strategy no longer depends on which of the two shapes it is handed."""
    now = datetime.now(timezone.utc)
    bars_15m, bars_1h = make_bars(now)  # DatetimeIndex, no timestamp column

    signals = generate_signals({"AAPL": bars_15m}, {"AAPL": bars_1h}, RUNTIME_STRATEGY, now)

    assert [s.symbol for s in signals] == ["AAPL"]
