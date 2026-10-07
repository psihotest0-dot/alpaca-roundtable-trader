import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch

from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import calculate_indicators, generate_signals, filter_signals
from trading_system.domain.models import StrategySignal


@pytest.fixture
def config():
    return StrategyConfig(min_warmup_bars=55)


def create_mock_data(size=60, start_time=None, freq="15min"):
    if start_time is None:
        start_time = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)

    timestamps = pd.date_range(start=start_time, periods=size, freq=freq)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        }
    )
    return df


def test_indicator_formulas(config):
    df = create_mock_data(60)
    res = calculate_indicators(df, config)
    assert "ema_fast" in res.columns
    assert "ema_slow" in res.columns
    assert "rsi" in res.columns
    assert "atr" in res.columns


@patch("trading_system.strategy.trend.calculate_indicators")
def test_valid_crossover(mock_calc, config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)

    # 15m data: ends at 12:00. 12:00 bar is open. 11:45 bar is closed.
    df_15 = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    # 1h data: ends at 12:00. 12:00 bar is open. 11:00 bar is closed.
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    df_15_mock = df_15.copy()
    df_15_mock["ema_fast"] = 100.0
    df_15_mock["ema_slow"] = 100.0
    df_15_mock["rsi"] = 60.0

    # In df_15, indices are 15m apart from 00:00 to 14:45.
    # We want eval_time = 12:05. So last closed is 11:45.
    # 00:00 is index 0. 11:45 is index 47. 12:00 is 48.
    df_15_mock.loc[46, "ema_fast"] = 99.0
    df_15_mock.loc[46, "ema_slow"] = 100.0
    df_15_mock.loc[47, "ema_fast"] = 101.0
    df_15_mock.loc[47, "ema_slow"] = 100.0

    df_1h_mock = df_1h.copy()
    df_1h_mock["ema_fast"] = 101.0
    df_1h_mock["ema_slow"] = 100.0

    def side_effect(df, conf, is_htf=False):
        return df_1h_mock if is_htf else df_15_mock

    mock_calc.side_effect = side_effect

    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].direction == "LONG"
    assert signals[0].closed_bar_timestamp == datetime(2023, 1, 1, 11, 45, tzinfo=timezone.utc)


@patch("trading_system.strategy.trend.calculate_indicators")
def test_no_crossover(mock_calc, config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
    df_15 = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    df_15_mock = df_15.copy()
    df_15_mock["ema_fast"] = 100.0
    df_15_mock["ema_slow"] = 100.0
    df_15_mock["rsi"] = 60.0

    df_1h_mock = df_1h.copy()
    df_1h_mock["ema_fast"] = 101.0
    df_1h_mock["ema_slow"] = 100.0

    mock_calc.side_effect = lambda df, conf, is_htf=False: df_1h_mock if is_htf else df_15_mock
    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0


@patch("trading_system.strategy.trend.calculate_indicators")
def test_rsi_filter(mock_calc, config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
    df_15 = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    df_15_mock = df_15.copy()
    df_15_mock["ema_fast"] = 100.0
    df_15_mock["ema_slow"] = 100.0
    df_15_mock.loc[46, "ema_fast"] = 99.0
    df_15_mock.loc[47, "ema_fast"] = 101.0

    df_1h_mock = df_1h.copy()
    df_1h_mock["ema_fast"] = 101.0
    df_1h_mock["ema_slow"] = 100.0

    mock_calc.side_effect = lambda df, conf, is_htf=False: df_1h_mock if is_htf else df_15_mock

    # Test RSI < 50
    df_15_mock["rsi"] = 40.0
    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0

    # Test RSI > 70
    df_15_mock["rsi"] = 80.0
    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0


@patch("trading_system.strategy.trend.calculate_indicators")
def test_hourly_confirmation_failure(mock_calc, config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
    df_15 = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    df_15_mock = df_15.copy()
    df_15_mock["ema_fast"] = 100.0
    df_15_mock["ema_slow"] = 100.0
    df_15_mock.loc[46, "ema_fast"] = 99.0
    df_15_mock.loc[47, "ema_fast"] = 101.0
    df_15_mock["rsi"] = 60.0

    df_1h_mock = df_1h.copy()
    df_1h_mock["ema_fast"] = 99.0  # Failed
    df_1h_mock["ema_slow"] = 100.0

    mock_calc.side_effect = lambda df, conf, is_htf=False: df_1h_mock if is_htf else df_15_mock

    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0


@patch("trading_system.strategy.trend.calculate_indicators")
def test_lookahead_prevention(mock_calc, config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
    df_15 = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    df_15_mock = df_15.copy()
    df_15_mock["ema_fast"] = 100.0
    df_15_mock["ema_slow"] = 100.0
    df_15_mock.loc[46, "ema_fast"] = 99.0
    df_15_mock.loc[47, "ema_fast"] = 101.0
    df_15_mock["rsi"] = 60.0

    df_1h_mock = df_1h.copy()
    df_1h_mock["ema_fast"] = 100.0
    df_1h_mock["ema_slow"] = 100.0

    # 15m closed is 11:45. Its close time is 12:00.
    # The valid 1h bar MUST close at or before 12:00.
    # 1h bar starting at 10:00 closes at 11:00 (index 10).
    # 1h bar starting at 11:00 closes at 12:00 (index 11).

    # Make 10:00 bar fail, 11:00 bar fail, 12:00 bar pass.
    # The 12:00 bar starts at 12:00 and closes at 13:00.
    # Because 1h bar starting at 11:00 closes at 12:00, it SHOULD be valid!
    # Wait, in the actual implementation:
    # `df_1h_closed = df_1h[df_1h['timestamp'] + pd.Timedelta(hours=1) <= eval_timestamp]`
    # 11:00 bar closes at 12:00. eval_timestamp = 12:05. So 11:00 bar is included in df_1h_closed.
    # `valid_1h_bars = df_1h_closed[df_1h_closed['timestamp'] + pd.Timedelta(hours=1) <= last_bar_close_time]`
    # last_bar_close_time = 11:45 + 15m = 12:00.
    # 11:00 bar closes at 12:00. 12:00 <= 12:00. So 11:00 bar IS included.
    # Therefore, we must make the 11:00 bar fail to show lookahead prevention for the 12:00 bar (which is open).
    df_1h_mock.loc[11, "ema_fast"] = 99.0  # 11:00 bar fails
    df_1h_mock.loc[12, "ema_fast"] = 101.0  # 12:00 bar passes (but shouldn't be seen)

    mock_calc.side_effect = lambda df, conf, is_htf=False: df_1h_mock if is_htf else df_15_mock

    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0


def test_incomplete_bars_handling(config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
    df_15 = create_mock_data(
        2, start_time=datetime(2023, 1, 1, 9, 45, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        2, start_time=datetime(2023, 1, 1, 3, 0, tzinfo=timezone.utc), freq="1h"
    )

    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0


def test_filter_signals():
    sig1 = StrategySignal(
        version="v1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=datetime.now(timezone.utc),
        direction="LONG",
    )
    sig2 = StrategySignal(
        version="v1",
        symbol="MSFT",
        timeframe="15m",
        closed_bar_timestamp=datetime.now(timezone.utc),
        direction="LONG",
    )

    active = {"AAPL"}
    filtered = filter_signals([sig1, sig2], active)

    assert len(filtered) == 1
    assert filtered[0].symbol == "MSFT"


def test_insufficient_warmup(config):
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)
    # Warmup requires 55 bars. Provide only 50.
    df_15 = create_mock_data(
        50, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    signals = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
    assert len(signals) == 0


def test_reprocessing_same_bar(config):
    # Tests that generating signals twice for the exact same data produces deterministic identical signal IDs.
    eval_time = datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc)

    df_15 = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="15min"
    )
    df_1h = create_mock_data(
        60, start_time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), freq="1h"
    )

    # Mock indicators directly to force a valid cross
    df_15["ema_fast"] = 100.0
    df_15["ema_slow"] = 100.0
    df_15["rsi"] = 60.0

    df_15.loc[46, "ema_fast"] = 99.0
    df_15.loc[46, "ema_slow"] = 100.0
    df_15.loc[47, "ema_fast"] = 101.0
    df_15.loc[47, "ema_slow"] = 100.0

    df_1h["ema_fast"] = 101.0
    df_1h["ema_slow"] = 100.0

    with patch("trading_system.strategy.trend.calculate_indicators") as mock_calc:

        def side_effect(df, conf, is_htf=False):
            return df_1h.copy() if is_htf else df_15.copy()

        mock_calc.side_effect = side_effect

        signals1 = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)
        signals2 = generate_signals({"AAPL": df_15}, {"AAPL": df_1h}, config, eval_time)

        assert len(signals1) == 1
        assert len(signals2) == 1
        assert signals1[0].signal_id == signals2[0].signal_id
