"""The backtest follows the live loop's rules (Phase 3.2): its parameters and its hours."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from tests.fixtures.bars import make_bars, with_timestamp_column
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig
from trading_system.config import Settings
from trading_system.strategy.config import RUNTIME_STRATEGY

UTC = timezone.utc


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")


def _with_follow_up_bars(now: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """The cross-up fixture plus three bars after it: one that lets the signal be seen, one
    that dips to the limit (a fill), one that reaches the take-profit. Returns the frames and
    the start of the bar that fills."""
    bars_15m, bars_1h = make_bars(now)
    cross = float(bars_15m["close"].iloc[-1])
    follow = pd.DataFrame(
        {
            "open": [cross, cross, cross * 1.06],
            "high": [cross + 0.1, cross + 0.1, cross * 1.07],
            "low": [cross - 0.1, cross - 0.5, cross],
            "close": [cross, cross, cross * 1.06],
            "volume": 1000.0,
        },
        index=pd.date_range(
            bars_15m.index[-1] + pd.Timedelta(minutes=15), periods=3, freq="15min", tz="UTC"
        ),
    )
    frame = with_timestamp_column(pd.concat([bars_15m, follow]))
    return frame, with_timestamp_column(bars_1h), follow.index[1]


def _run(settings: Settings, now: datetime) -> tuple[Any, pd.Timestamp]:
    frame, hourly, fill_bar = _with_follow_up_bars(now)
    engine = BacktestEngine(
        settings,
        BacktestConfig(initial_capital=Decimal("100000")),
        {"AAPL": frame},
        {"AAPL": hourly},
    )
    return engine.run(), fill_bar


def test_the_backtest_runs_the_parameters_of_the_live_loop(settings: Settings) -> None:
    engine = BacktestEngine(settings, BacktestConfig(), {}, {})

    assert engine.strategy_config == RUNTIME_STRATEGY


def test_a_signal_inside_the_session_becomes_a_trade(settings: Settings) -> None:
    # 16:07 UTC is 11:07 in New York; the cross-up bar closes at 16:00 and the next ones follow.
    result, fill_bar = _run(settings, datetime(2026, 11, 2, 16, 7, tzinfo=UTC))

    entered = [t for t in result.trades if t.entry_time == fill_bar]
    assert len(entered) == 1
    assert entered[0].symbol == "AAPL" and entered[0].exit_reason == "TAKE_PROFIT"


def test_the_same_signal_before_the_open_is_not_traded(settings: Settings) -> None:
    # 12:07 UTC is 07:07 in New York: pre-market. The live loop does not look for entries then.
    result, fill_bar = _run(settings, datetime(2026, 11, 2, 12, 7, tzinfo=UTC))

    assert [t for t in result.trades if t.entry_time == fill_bar] == []


def test_signals_are_looked_for_only_at_bars_inside_the_session(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[pd.Timestamp] = []

    def spy(*args: Any, **kwargs: Any) -> list[Any]:
        asked.append(pd.Timestamp(args[3]))
        return []

    monkeypatch.setattr("trading_system.backtest.engine.generate_signals", spy)
    index = pd.date_range(
        "2026-11-02 13:00", "2026-11-02 22:00", freq="15min", tz="UTC"
    )  # 8:00-17:00 ET
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index
    )
    frame["timestamp"] = frame.index
    frame = frame.reset_index(drop=True)

    BacktestEngine(settings, BacktestConfig(), {"AAPL": frame}, {"AAPL": frame}).run()

    assert asked[0] == pd.Timestamp("2026-11-02 14:30", tz="UTC")  # 9:30 ET
    assert asked[-1] == pd.Timestamp("2026-11-02 20:45", tz="UTC")  # the last bar before 16:00 ET
    assert len(asked) == 26  # 6.5 hours of 15-minute bars


def test_an_order_left_at_the_close_does_not_fill_after_hours(settings: Settings) -> None:
    """The signal is acted on in the last bar of the session; the order lives one bar more and
    expires at the close. A dip to the limit in the after-hours bar after that must not fill it."""
    now = datetime(2026, 11, 2, 20, 52, tzinfo=UTC)  # the cross-up bar closes at 20:45 UTC
    bars_15m, bars_1h = make_bars(now)
    cross = float(bars_15m["close"].iloc[-1])
    follow = pd.DataFrame(
        {  # 20:45 (the signal), 21:00 (no dip), 21:15 (a dip to the limit), 21:30 (a take-profit)
            "open": [cross, cross + 0.2, cross, cross * 1.06],
            "high": [cross + 0.1, cross + 0.3, cross + 0.1, cross * 1.07],
            "low": [cross - 0.1, cross + 0.1, cross - 0.5, cross],
            "close": [cross, cross + 0.2, cross, cross * 1.06],
            "volume": 1000.0,
        },
        index=pd.date_range(
            bars_15m.index[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min", tz="UTC"
        ),
    )
    frame = with_timestamp_column(pd.concat([bars_15m, follow]))
    engine = BacktestEngine(
        settings,
        BacktestConfig(initial_capital=Decimal("100000")),
        {"AAPL": frame},
        {"AAPL": with_timestamp_column(bars_1h)},
    )

    result = engine.run()

    late_fill = pd.Timestamp("2026-11-02 21:15", tz="UTC")
    assert [t for t in result.trades if t.entry_time == late_fill] == []
    assert "AAPL" not in engine.positions and "AAPL" not in engine.pending_orders
