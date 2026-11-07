"""The ATR exit candidate (research plan, section 13.1), the parts that need no database: the
arithmetic of the exit, the ATR of a window, and the backtest engine with the option on and off."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.bars import QUIET, make_bars_with_follow_up
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig
from trading_system.cli import main
from trading_system.config import Settings
from trading_system.risk.entry_plan import (
    AtrExit,
    InvalidEntryPlan,
    atr_exit_percentages,
    build_entry_plan,
)
from trading_system.strategy.atr import atr_and_close
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.trend import calculate_indicators

UTC = timezone.utc
RULE = AtrExit(Decimal("6"), Decimal("7.5"))


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s", **overrides)


def _window(rows: int, *, half_range: float = 0.5, close: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2026-11-02 15:00", periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


# ------------------------------------------------------------------------------ the arithmetic


def test_an_atr_exit_is_the_multiples_of_the_atr_as_a_share_of_the_close() -> None:
    stop, target = atr_exit_percentages(1.0, 100.0, RULE)  # an ATR of 1 % of the price

    assert (stop, target) == (Decimal("0.06"), Decimal("0.075"))


@pytest.mark.parametrize(
    "atr, close",
    [(0.0, 100.0), (-1.0, 100.0), (float("nan"), 100.0), (1.0, 0.0), (1.0, float("inf"))],
)
def test_an_atr_that_is_not_a_positive_number_is_not_an_exit(atr: float, close: float) -> None:
    with pytest.raises(InvalidEntryPlan):
        atr_exit_percentages(atr, close, RULE)


def test_a_stop_of_the_whole_price_or_more_is_not_a_stop() -> None:
    with pytest.raises(InvalidEntryPlan, match="not a stop"):
        atr_exit_percentages(20.0, 100.0, RULE)  # 6 x 20 % = 120 %


def test_the_atr_of_a_window_is_the_one_the_live_code_computes() -> None:
    window = _window(80)

    atr, close = atr_and_close(window) or (0.0, 0.0)

    assert close == 100.0 and atr == pytest.approx(1.0)  # a range of 1.0 on every bar
    live = calculate_indicators(window, RUNTIME_STRATEGY)
    assert atr == pytest.approx(float(live["atr"].iloc[-1]))


def test_without_enough_bars_or_without_a_range_there_is_no_atr() -> None:
    assert atr_and_close(_window(RUNTIME_STRATEGY.atr_period)) is None  # the 14th bar is the first
    assert atr_and_close(_window(80, half_range=0.0)) is None  # an ATR of zero is no ATR


# ------------------------------------------------------------------------- the backtest engine


def _run(
    follow: list[tuple[float, float, float, float]], *, atr_exit: AtrExit | None, **settings: Any
) -> tuple[Any, pd.DataFrame]:
    """The result, and the bars up to and including the signal bar (the follow-up rows cut off)."""
    frame, hourly, _ = make_bars_with_follow_up(datetime(2026, 11, 2, 16, 7, tzinfo=UTC), follow)
    engine = BacktestEngine(
        _settings(**settings),
        BacktestConfig(initial_capital=Decimal("100000")),
        {"AAPL": frame},
        {"AAPL": hourly},
        atr_exit=atr_exit,
    )
    return engine.run(), frame.iloc[: -len(follow)]


def _atr_percentages(signal_window: pd.DataFrame) -> tuple[Decimal, Decimal]:
    """The ATR exit of the signal bar, worked out here from the bars up to it."""
    atr, close = atr_and_close(signal_window) or (0.0, 0.0)
    return atr_exit_percentages(atr, close, RULE)


UP_3_5_PCT = (1.0, 1.035, 1.0, 1.03)  # rises 3.5 %: short of the live 5 % target
DOWN_3_PCT = (1.0, 1.0, 0.97, 0.975)  # dips 3 %: short of the live 4 % stop


def test_with_the_exit_off_the_engine_is_what_it_was() -> None:
    plain, _ = _run([QUIET, QUIET, UP_3_5_PCT], atr_exit=None)

    assert plain.trades == [] and plain.signals >= 1  # 3.5 % reaches neither live level


def test_with_an_atr_exit_the_target_is_where_the_atr_puts_it() -> None:
    result, signal_window = _run([QUIET, QUIET, UP_3_5_PCT, QUIET], atr_exit=RULE)

    stop, target = _atr_percentages(signal_window)
    assert target < Decimal("0.035")  # the fixture's ATR is small: 7.5 ATR is under 3.5 %
    (trade,) = result.trades
    assert trade.exit_reason == "TAKE_PROFIT"
    expected = build_entry_plan(
        Decimal(str(signal_window.iloc[-1]["close"])),  # the first follow-up bar opens at it
        limit_buffer_pct=Decimal(str(_settings().entry_limit_buffer_pct)),
        stop_loss_pct=stop,
        take_profit_pct=target,
    )
    slippage = Decimal(str(BacktestConfig().slippage_pct))
    assert trade.exit_price == expected.take_profit_price * (1 - slippage)


def test_with_an_atr_exit_the_stop_is_where_the_atr_puts_it() -> None:
    result, signal_window = _run([QUIET, QUIET, DOWN_3_PCT, QUIET], atr_exit=RULE)

    stop, _ = _atr_percentages(signal_window)
    assert stop < Decimal("0.03")
    (trade,) = result.trades
    assert trade.exit_reason == "STOP_LOSS" and trade.pnl_pct is not None and trade.pnl_pct < 0


def test_a_signal_with_no_atr_is_not_traded_and_is_counted() -> None:
    frame, hourly, _ = make_bars_with_follow_up(
        datetime(2026, 11, 2, 16, 7, tzinfo=UTC), [QUIET, QUIET, UP_3_5_PCT]
    )
    flat = frame.copy()
    for column in ("open", "high", "low", "close"):
        flat[column] = 100.0  # no range anywhere: an ATR of zero
    engine = BacktestEngine(
        _settings(), BacktestConfig(), {"AAPL": flat}, {"AAPL": hourly}, atr_exit=RULE
    )

    result = engine.run()

    assert result.trades == [] and result.not_traded.get("No ATR", 0) == result.signals


def test_the_live_settings_are_not_touched_by_an_atr_run() -> None:
    settings = _settings()
    frame, hourly, _ = make_bars_with_follow_up(
        datetime(2026, 11, 2, 16, 7, tzinfo=UTC), [QUIET, QUIET, UP_3_5_PCT]
    )

    BacktestEngine(
        settings, BacktestConfig(), {"AAPL": frame}, {"AAPL": hourly}, atr_exit=RULE
    ).run()

    assert (settings.stop_loss_pct, settings.take_profit_pct) == (0.04, 0.05)


# --------------------------------------------------------------------------------- the command


def test_the_two_flags_of_the_command_go_together(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("trading_system.cli.Settings", return_value=_settings()):
        assert main(["backtest", "--atr-stop", "6"]) == 1
    assert "go together" in capsys.readouterr().out


def test_the_default_exit_of_the_live_path_stays_the_percentages() -> None:
    assert np.isclose(_settings().stop_loss_pct, 0.04) and np.isclose(
        _settings().take_profit_pct, 0.05
    )
