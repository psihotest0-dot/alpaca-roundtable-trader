import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import os

from trading_system.config import Settings
from trading_system.backtest.models import BacktestConfig
from trading_system.backtest.engine import BacktestEngine
from trading_system.domain.models import StrategySignal
from trading_system.domain.enums import SignalDirection


@pytest.fixture
def dummy_settings():
    os.environ["ALPACA_API_KEY"] = "dummy"
    os.environ["ALPACA_SECRET_KEY"] = "dummy"
    return Settings(
        alpaca_api_key="dummy", alpaca_secret_key="dummy", is_paper=True, oracle_enabled=False
    )


def _create_mock_bar(
    timestamp: datetime, o: float, h: float, low_val: float, c: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            "open": [o],
            "high": [h],
            "low": [low_val],
            "close": [c],
            "volume": [1000],
        }
    )


def test_backtest_pending_order_fill_and_slippage(dummy_settings):
    # Test that a pending limit order is filled correctly when the price drops below the limit
    bt_config = BacktestConfig(
        initial_capital=Decimal("10000.0"), slippage_pct=0.01
    )  # 1% slippage for easy math

    t0 = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)

    # Empty dataframes since we will inject state manually
    engine = BacktestEngine(
        dummy_settings, bt_config, {"AAPL": pd.DataFrame()}, {"AAPL": pd.DataFrame()}
    )

    # Inject a pending limit order for AAPL at $100
    sig = StrategySignal(
        version="1.0",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=t0,
        direction=SignalDirection.LONG,
    )
    sig._suggested_limit = Decimal("100.0")
    sig._suggested_shares = Decimal("10")
    engine.pending_orders["AAPL"] = sig

    # Create the current bar where price drops to $90
    bar_df = _create_mock_bar(t1, 95.0, 95.0, 90.0, 92.0)
    engine.df_15m = {"AAPL": bar_df}

    # Process
    engine._process_pending_orders(t1)

    # Check that order was filled.
    # Open is 95, which is < limit 100. So we get filled at open price (95).
    # Slippage is 1%, so fill price = 95 * 1.01 = 95.95
    assert "AAPL" not in engine.pending_orders
    assert "AAPL" in engine.positions

    pos = engine.positions["AAPL"]
    assert pos.quantity == Decimal("10")
    assert pos.average_entry_price == Decimal("95.95")

    # Cash should be 10000 - (10 * 95.95) - fees. Let's just assert that it's correctly calculated
    # 95.95 * 10 = 959.50. Fee (0.0001) = 0.09595. Total cost = 959.59595
    # Cash = 10000 - 959.59595 = 9040.40405
    assert round(engine.cash, 5) == Decimal("9040.40405")


def test_backtest_gap_aware_stop_loss(dummy_settings):
    # Test that if a bar opens BELOW the stop loss, we get filled at the open price (gap down)
    bt_config = BacktestConfig(initial_capital=Decimal("10000.0"), slippage_pct=0.0)

    t0 = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)

    engine = BacktestEngine(dummy_settings, bt_config, {}, {})

    # Inject active position with SL at 90
    from trading_system.backtest.models import BacktestPosition

    engine.positions["AAPL"] = BacktestPosition(
        symbol="AAPL",
        quantity=Decimal("10"),
        average_entry_price=Decimal("100.0"),
        entry_time=t0,
        stop_loss=Decimal("90.0"),
        take_profit=Decimal("110.0"),
    )

    # Create the current bar where it opens at 80 (gap down past 90)
    bar_df = _create_mock_bar(t1, 80.0, 85.0, 75.0, 82.0)
    engine.df_15m = {"AAPL": bar_df}

    engine._process_bar_executions(t1)

    assert "AAPL" not in engine.positions
    assert len(engine.trades) == 1

    trade = engine.trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == Decimal("80.0")  # Filled at open


def test_backtest_conservative_intrabar_collision(dummy_settings):
    # Test that if BOTH SL and TP are hit in the same bar, it defaults to STOP_LOSS
    bt_config = BacktestConfig(initial_capital=Decimal("10000.0"), slippage_pct=0.0)

    t0 = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=15)

    engine = BacktestEngine(dummy_settings, bt_config, {}, {})

    # Inject active position with SL at 90, TP at 110
    from trading_system.backtest.models import BacktestPosition

    engine.positions["AAPL"] = BacktestPosition(
        symbol="AAPL",
        quantity=Decimal("10"),
        average_entry_price=Decimal("100.0"),
        entry_time=t0,
        stop_loss=Decimal("90.0"),
        take_profit=Decimal("110.0"),
    )

    # Create the current bar where high > 110 AND low < 90
    bar_df = _create_mock_bar(t1, 100.0, 115.0, 85.0, 95.0)
    engine.df_15m = {"AAPL": bar_df}

    engine._process_bar_executions(t1)

    assert "AAPL" not in engine.positions
    assert len(engine.trades) == 1

    trade = engine.trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == Decimal("90.0")  # SL was 90


def test_backtest_e2e_engine_run(dummy_settings):
    # Run the full run() loop on mock data to ensure no crashes
    bt_config = BacktestConfig(initial_capital=Decimal("100000.0"), slippage_pct=0.0)

    now = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Give it enough data to pass the warmup phase
    idx_15m = pd.date_range(end=now, periods=1000, freq="15min", tz="UTC")
    df_15m = pd.DataFrame(index=idx_15m)
    df_15m["timestamp"] = df_15m.index
    df_15m["open"] = np.linspace(100, 150, 1000)
    df_15m["close"] = df_15m["open"] + np.random.normal(0, 0.5, 1000)
    df_15m["high"] = df_15m[["open", "close"]].max(axis=1) + 1.0
    df_15m["low"] = df_15m[["open", "close"]].min(axis=1) - 1.0
    df_15m["volume"] = 1000

    idx_1h = pd.date_range(end=now, periods=300, freq="1h", tz="UTC")
    df_1h = pd.DataFrame(index=idx_1h)
    df_1h["timestamp"] = df_1h.index
    df_1h["open"] = np.linspace(100, 150, 300)
    df_1h["close"] = df_1h["open"] + np.random.normal(0, 1.0, 300)
    df_1h["high"] = df_1h[["open", "close"]].max(axis=1) + 2.0
    df_1h["low"] = df_1h[["open", "close"]].min(axis=1) - 2.0
    df_1h["volume"] = 4000

    # We create an artificial strong upward trend in the last few bars to force a signal
    # EMA fast > EMA slow and RSI up
    df_15m.loc[df_15m.index[-10:], "close"] = np.linspace(150, 200, 10)
    df_1h.loc[df_1h.index[-5:], "close"] = np.linspace(150, 200, 5)

    engine = BacktestEngine(
        dummy_settings,
        bt_config,
        {"AAPL": df_15m.reset_index(drop=True)},
        {"AAPL": df_1h.reset_index(drop=True)},
    )

    result = engine.run()

    # We don't strictly care if it made a trade, just that it ran through without error
    # and populated the result object.
    assert result.initial_capital == Decimal("100000.0")
    assert result.final_capital > Decimal("0")
    assert isinstance(result.assumptions_and_limitations, list)
    assert len(result.assumptions_and_limitations) > 0
