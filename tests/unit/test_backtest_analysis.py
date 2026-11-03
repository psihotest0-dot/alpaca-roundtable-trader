"""Reading a backtest result: expectancy, steadiness, costs, and the value of the Oracle."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pytest

from tests.fixtures.bars import GAP_UP_6_PCT, QUIET, make_bars_with_follow_up
from trading_system.backtest.analysis import (
    by_period,
    by_symbol,
    cost_sweep,
    expectancy,
    format_analysis,
    format_cost_sweep,
    format_oracle_value,
    oracle_value,
    sharpe_ratio,
    t_statistic,
    welch_t,
)
from trading_system.backtest.models import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    ShadowTrade,
)
from trading_system.config import Settings
from trading_system.domain.enums import OrderSide
from trading_system.strategy.trend_cross import TrendCrossStrategy

UTC = timezone.utc


def _trade(symbol: str, pnl: float, pct: float, exit_day: int = 2) -> BacktestTrade:
    """A closed trade; `exit_day` is the day of November 2026 it closed on (15:00 UTC)."""
    when = datetime(2026, 11, exit_day, 15, 0, tzinfo=UTC)
    return BacktestTrade(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal(10),
        entry_price=Decimal(100),
        entry_time=when - timedelta(days=1),  # a trade belongs to the day it CLOSED
        exit_price=Decimal(100),
        exit_time=when,
        exit_reason="X",
        pnl=Decimal(str(pnl)),
        pnl_pct=pct,
    )


def _shadow(pct: float, reason: str = "STOP_LOSS", verdict: str = "VETO") -> ShadowTrade:
    when = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
    return ShadowTrade(
        symbol="AAPL",
        signal_id="s",
        verdict=verdict,
        entry_time=when,
        entry_price=Decimal(100),
        exit_time=when,
        exit_price=Decimal(100),
        exit_reason=reason,
        pnl_pct=pct,
    )


def _result(
    trades: Optional[list[BacktestTrade]] = None,
    daily: Optional[dict[str, float]] = None,
    shadows: Optional[list[ShadowTrade]] = None,
) -> BacktestResult:
    return BacktestResult(
        initial_capital=Decimal(100000),
        final_capital=Decimal(100000),
        total_return_pct=0.0,
        max_drawdown_pct=0.0,
        trade_count=len(trades or []),
        win_rate=0.0,
        profit_factor=0.0,
        total_fees=Decimal(0),
        total_slippage=Decimal(0),
        max_exposure_pct=0.0,
        trades=trades or [],
        daily_equity=daily or {},
        shadow_trades=shadows or [],
    )


# --- the average trade ----------------------------------------------------------------------


def test_expectancy_of_a_mixed_set_of_trades() -> None:
    trades = [
        _trade("A", 50, 0.05),
        _trade("A", -40, -0.04),
        _trade("B", 50, 0.05),
        _trade("B", -40, -0.04),
    ]

    e = expectancy(trades)

    assert e.trades == 4 and e.avg_pnl == Decimal(5)
    assert e.avg_pct == pytest.approx(0.005)
    assert (e.avg_win_pct, e.avg_loss_pct) == (pytest.approx(0.05), pytest.approx(-0.04))
    assert e.payoff == pytest.approx(1.25)
    assert e.t_stat == pytest.approx(0.19245, abs=1e-4)  # 0.005 / (0.05196 / 2)


def test_expectancy_with_no_trades_is_zero_not_an_error() -> None:
    e = expectancy([])

    assert (e.trades, e.avg_pnl, e.avg_pct, e.payoff, e.t_stat) == (0, 0, 0.0, 0.0, 0.0)


def test_t_statistic_needs_spread_and_at_least_two_values() -> None:
    assert t_statistic([1.0, 2.0, 3.0]) == pytest.approx(2.0 / (1.0 / 3**0.5))
    assert t_statistic([0.5, 0.5]) == 0.0 and t_statistic([1.0]) == 0.0 and t_statistic([]) == 0.0


def test_sharpe_is_the_mean_over_the_spread_of_daily_returns_scaled_to_a_year() -> None:
    # Daily returns 2 %, 0, 2 %, 0: mean 1 %, spread 1.1547 %.
    result = _result(
        daily={
            "2026-11-02": 102000.0,
            "2026-11-03": 102000.0,
            "2026-11-04": 104040.0,
            "2026-11-05": 104040.0,
        }
    )

    assert sharpe_ratio(result) == pytest.approx(0.01 / 0.011547005 * 252**0.5, rel=1e-4)


def test_sharpe_of_a_flat_account_or_a_single_day_is_zero() -> None:
    assert sharpe_ratio(_result()) == 0.0
    assert sharpe_ratio(_result(daily={"2026-11-02": 101000.0})) == 0.0
    assert sharpe_ratio(_result(daily={"2026-11-02": 100000.0, "2026-11-03": 100000.0})) == 0.0


# --- steadiness -----------------------------------------------------------------------------


def test_by_symbol_ranks_by_money_and_counts_wins() -> None:
    result = _result(
        [_trade("MSFT", -50, -0.02), _trade("AAPL", 100, 0.04), _trade("AAPL", -20, -0.01)]
    )

    rows = by_symbol(result)

    assert [r.symbol for r in rows] == ["AAPL", "MSFT"]
    assert (rows[0].trades, rows[0].pnl, rows[0].win_rate) == (2, Decimal(80), 0.5)
    assert rows[0].avg_pct == pytest.approx(0.015) and rows[1].win_rate == 0.0


def _eight_days() -> dict[str, float]:
    # 8 trading days; the account goes up 2.01 %, stays, falls back to 100000, stays.
    values = [101000, 102010, 102010, 102010, 100000, 100000, 100000, 100000]
    return {f"2026-11-{day:02d}": float(v) for day, v in zip(range(2, 10), values)}


def test_by_period_cuts_the_replay_into_equal_stretches_with_their_own_result() -> None:
    result = _result(
        [_trade("A", 10, 0.01, exit_day=3), _trade("A", -5, -0.01, exit_day=6)],
        daily=_eight_days(),
    )

    rows = by_period(result, 4)

    assert [(r.first_day, r.last_day, r.days) for r in rows] == [
        ("2026-11-02", "2026-11-03", 2),
        ("2026-11-04", "2026-11-05", 2),
        ("2026-11-06", "2026-11-07", 2),
        ("2026-11-08", "2026-11-09", 2),
    ]
    assert [r.return_pct for r in rows] == pytest.approx([0.0201, 0.0, 100000 / 102010 - 1, 0.0])
    assert [r.trades for r in rows] == [1, 0, 1, 0]  # a trade belongs to the period it closed in
    assert [r.win_rate for r in rows] == [1.0, 0.0, 0.0, 0.0]


def test_by_period_gives_the_longer_stretches_to_the_first_ones() -> None:
    daily = {f"2026-11-{day:02d}": 100000.0 for day in range(2, 7)}  # five days

    rows = by_period(_result(daily=daily), 2)

    assert [r.days for r in rows] == [3, 2]


def test_by_period_never_makes_more_periods_than_days_and_none_without_days() -> None:
    assert len(by_period(_result(daily={"2026-11-02": 100000.0}), 4)) == 1
    assert by_period(_result(), 4) == []


# --- the Oracle -----------------------------------------------------------------------------


def test_the_oracle_value_is_what_the_approved_trades_made_over_what_the_vetoed_would_have() -> (
    None
):
    result = _result(
        [_trade("A", 50, 0.05), _trade("A", -40, -0.04)],
        shadows=[_shadow(-0.04), _shadow(-0.04), _shadow(0.05, "TAKE_PROFIT")],
    )

    value = oracle_value(result)

    assert value is not None
    assert (value.approved.trades, value.not_approved.trades) == (2, 3)
    assert value.approved.avg_pct == pytest.approx(0.005)
    assert value.not_approved.avg_pct == pytest.approx(-0.01)
    assert value.edge_pct == pytest.approx(0.015)  # approved did better...
    assert value.edge_t_stat == pytest.approx(0.27735, abs=1e-4)  # ...but by less than the noise
    assert value.not_approved.win_rate == pytest.approx(1 / 3)


def test_there_is_no_oracle_value_without_shadow_trades() -> None:
    assert oracle_value(_result([_trade("A", 50, 0.05)])) is None


def test_shadow_trades_still_open_are_counted() -> None:
    value = oracle_value(_result(shadows=[_shadow(0.0, "OPEN"), _shadow(0.05, "TAKE_PROFIT")]))

    assert value is not None and value.still_open == 1


# --- costs ----------------------------------------------------------------------------------


def test_a_higher_cost_lowers_the_result_of_the_same_replay() -> None:
    settings = Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")
    frame, hourly, _ = make_bars_with_follow_up(
        datetime(2026, 11, 2, 16, 7, tzinfo=UTC), [QUIET, QUIET, GAP_UP_6_PCT]
    )

    rows = cost_sweep(
        settings,
        BacktestConfig(),
        {"AAPL": frame},
        {"AAPL": hourly},
        TrendCrossStrategy(),
        slippage_levels=(0.0, 0.002, 0.02),
    )

    assert [r.slippage_pct for r in rows] == [0.0, 0.002, 0.02]
    assert [r.trades for r in rows] == [1, 1, 1]
    assert rows[0].total_return_pct > rows[1].total_return_pct > rows[2].total_return_pct


# --- text -----------------------------------------------------------------------------------


def test_the_report_has_its_sections_and_numbers() -> None:
    result = _result(
        [_trade("AAPL", 100, 0.04, exit_day=3), _trade("MSFT", -50, -0.02, exit_day=6)],
        daily=_eight_days(),
    )

    text = format_analysis(result, periods=4)

    assert "=== Per trade ===" in text and "Expectancy:" in text and "Sharpe (daily):" in text
    assert "4 consecutive periods" in text
    assert "1 of 4 periods ended up" in text  # +2.01 %, 0, -1.97 %, 0
    assert "=== By symbol (2 traded" in text and "AAPL" in text and "MSFT" in text


def test_the_cost_sweep_and_oracle_texts_show_every_row() -> None:
    settings = Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")
    frame, hourly, _ = make_bars_with_follow_up(
        datetime(2026, 11, 2, 16, 7, tzinfo=UTC), [QUIET, QUIET, GAP_UP_6_PCT]
    )
    rows = cost_sweep(
        settings, BacktestConfig(), {"AAPL": frame}, {"AAPL": hourly}, TrendCrossStrategy()
    )

    sweep = format_cost_sweep(rows).splitlines()

    assert len(sweep) == 2 + len(rows) and "0.20%" in sweep[-1]


def _verdict_text(approved: list[float], shadows: list[float]) -> str:
    result = _result(
        [_trade("A", 1, pct) for pct in approved], shadows=[_shadow(pct) for pct in shadows]
    )
    value = oracle_value(result)
    assert value is not None
    return format_oracle_value(value)


def test_the_oracle_is_said_to_have_helped_only_when_the_difference_is_clear() -> None:
    clear = _verdict_text([0.04, 0.06] * 15, [-0.03, -0.05] * 15)
    luck = _verdict_text([0.05, -0.04], [-0.04, -0.04, 0.05])
    coin_flip_like = _verdict_text([0.05, -0.05, 0.04, -0.04] * 5, [-0.05, 0.05, 0.04, -0.03] * 5)

    assert "the Oracle helped" in clear
    assert "no difference that can be told from luck" in luck
    assert (
        "no difference that can be told from luck" in coin_flip_like
    )  # a positive edge, but noise
    assert "Approved (traded)" in clear and "Not approved (shadow)" in clear


def test_the_oracle_is_said_to_have_hurt_when_it_kept_us_out_of_winners() -> None:
    assert "the Oracle hurt" in _verdict_text([-0.03, -0.05] * 15, [0.04, 0.06] * 15)


def test_welch_t_compares_two_groups_with_their_own_spread() -> None:
    assert welch_t([2.0, 4.0, 6.0], [1.0, 3.0, 5.0]) == pytest.approx(0.61237, abs=1e-4)
    assert welch_t([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert welch_t([1.0, 3.0], [2.0, 4.0]) == -welch_t([2.0, 4.0], [1.0, 3.0])


def test_welch_t_with_no_spread_or_too_few_values() -> None:
    assert welch_t([1.0, 1.0], [1.0, 1.0]) == 0.0  # the same, for certain
    assert welch_t([1.0, 1.0], [2.0, 2.0]) == float("-inf")  # different, for certain
    assert welch_t([1.0], [2.0, 3.0]) == 0.0 and welch_t([], []) == 0.0
