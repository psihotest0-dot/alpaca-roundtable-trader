"""What the backtest reports besides the trades (Phase 3.3): slippage on its own and the
equity day by day."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from tests.fixtures.bars import GAP_UP_6_PCT, QUIET, make_bars_with_follow_up
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig, BacktestResult
from trading_system.config import Settings

UTC = timezone.utc
IN_SESSION = datetime(2026, 11, 2, 16, 7, tzinfo=UTC)  # 11:07 in New York, a cross-up just closed


def _run(config: BacktestConfig) -> BacktestResult:
    settings = Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")
    frame, hourly, _ = make_bars_with_follow_up(IN_SESSION, [QUIET, QUIET, GAP_UP_6_PCT])
    return BacktestEngine(settings, config, {"AAPL": frame}, {"AAPL": hourly}).run()


def test_slippage_is_reported_as_what_the_fills_cost_against_the_price_seen() -> None:
    result = _run(BacktestConfig(slippage_pct=0.0005, fee_pct_per_trade=0.0))

    trade = result.trades[0]
    assert trade.exit_price is not None
    first_print = trade.entry_price / Decimal("1.0005")  # the fill is the price plus 0.05 %
    price_at_exit = trade.exit_price / Decimal("0.9995")  # and the sale is the price minus 0.05 %
    expected = trade.quantity * (trade.entry_price - first_print) + trade.quantity * (
        price_at_exit - trade.exit_price
    )
    assert trade.slippage_paid > 0
    assert abs(trade.slippage_paid - expected) < Decimal("0.000001")
    assert result.total_slippage == trade.slippage_paid


def test_slippage_is_a_breakdown_not_an_extra_cost() -> None:
    free = _run(BacktestConfig(slippage_pct=0.0, fee_pct_per_trade=0.0))
    costly = _run(BacktestConfig(slippage_pct=0.0005, fee_pct_per_trade=0.0))

    assert free.total_slippage == 0
    # The result already contains the slippage: the profit falls by about what is reported.
    free_pnl, costly_pnl = free.trades[0].pnl, costly.trades[0].pnl
    assert free_pnl is not None and costly_pnl is not None
    assert 0 < free_pnl - costly_pnl < costly.total_slippage * Decimal("1.1")


def test_the_equity_is_kept_for_every_new_york_day() -> None:
    result = _run(BacktestConfig(slippage_pct=0.0))

    days = list(result.daily_equity)
    assert len(days) >= 2 and days == sorted(days)
    assert days[-1] == "2026-11-02"  # the day of the signal and the trade
    # That day ends with the position closed, so it ends at the final capital.
    assert result.daily_equity[days[-1]] == pytest.approx(float(result.final_capital))


def test_daily_returns_start_from_the_initial_capital() -> None:
    result = BacktestResult(
        initial_capital=Decimal("100000"),
        final_capital=Decimal("100495"),
        total_return_pct=0.00495,
        max_drawdown_pct=0.0,
        trade_count=0,
        win_rate=0.0,
        profit_factor=0.0,
        total_fees=Decimal("0"),
        total_slippage=Decimal("0"),
        max_exposure_pct=0.0,
        daily_equity={"2026-11-03": 100495.0, "2026-11-02": 101000.0},  # out of order on purpose
    )

    returns = result.daily_returns()

    assert [day for day, _ in returns] == ["2026-11-02", "2026-11-03"]
    assert returns[0][1] == pytest.approx(0.01)
    assert returns[1][1] == pytest.approx(100495 / 101000 - 1)
