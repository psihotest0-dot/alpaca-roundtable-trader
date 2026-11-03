"""The backtest follows the live loop's rules: its parameters, its hours (Phase 3.2) and its
risk engine and entry (Phase 3.1)."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from tests.fixtures.bars import GAP_UP_6_PCT, QUIET, make_bars_with_follow_up
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig, BacktestResult
from trading_system.config import Settings
from trading_system.risk.entry_plan import build_entry_plan
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.trend_cross import TrendCrossStrategy

UTC = timezone.utc


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s", **overrides)


@pytest.fixture
def settings() -> Settings:
    return _settings()


def _run(
    settings: Settings,
    now: datetime,
    follow: list[tuple[float, float, float, float]],
    *,
    symbols: tuple[str, ...] = ("AAPL",),
    config: BacktestConfig | None = None,
) -> tuple[BacktestResult, pd.DatetimeIndex, pd.DataFrame]:
    """Runs the cross-up fixture followed by the `follow` bars; every symbol gets the same bars."""
    frame, hourly, follow_index = make_bars_with_follow_up(now, follow)
    engine = BacktestEngine(
        settings,
        config or BacktestConfig(initial_capital=Decimal("100000")),
        {s: frame for s in symbols},
        {s: hourly for s in symbols},
    )
    return engine.run(), follow_index, frame


def test_the_backtest_runs_the_strategy_of_the_live_loop_unless_told_otherwise(
    settings: Settings,
) -> None:
    engine = BacktestEngine(settings, BacktestConfig(), {}, {})

    assert isinstance(engine.strategy, TrendCrossStrategy)
    assert engine.strategy.config == RUNTIME_STRATEGY


# --- hours ---------------------------------------------------------------------------------


def test_a_signal_inside_the_session_becomes_a_trade(settings: Settings) -> None:
    # 16:07 UTC is 11:07 in New York; the cross-up bar closes at 16:00 and the next ones follow.
    result, follow_index, _ = _run(
        settings, datetime(2026, 11, 2, 16, 7, tzinfo=UTC), [QUIET, QUIET, GAP_UP_6_PCT]
    )

    assert [t.symbol for t in result.trades] == ["AAPL"]
    assert result.trades[0].entry_time == follow_index[0]  # acted on at once, not a bar later
    assert result.trades[0].exit_reason == "TAKE_PROFIT"


def test_the_same_signal_before_the_open_is_not_traded(settings: Settings) -> None:
    # 12:07 UTC is 07:07 in New York: pre-market. The live loop does not look for entries then.
    result, _, _ = _run(
        settings, datetime(2026, 11, 2, 12, 7, tzinfo=UTC), [QUIET, QUIET, GAP_UP_6_PCT]
    )

    assert result.trades == []


def test_a_signal_on_the_last_bar_of_the_session_is_acted_on(settings: Settings) -> None:
    # The cross-up bar closes at 20:45 UTC = 15:45 ET, a bar before the close: still in the session.
    result, follow_index, _ = _run(
        settings, datetime(2026, 11, 2, 20, 52, tzinfo=UTC), [QUIET, QUIET, GAP_UP_6_PCT]
    )

    assert [t.entry_time for t in result.trades] == [follow_index[0]]


def test_a_signal_that_arrives_at_the_close_is_not_acted_on(settings: Settings) -> None:
    # The cross-up bar closes at 21:00 UTC = 16:00 ET: the live loop sees a closed market.
    result, _, _ = _run(
        settings, datetime(2026, 11, 2, 21, 7, tzinfo=UTC), [QUIET, QUIET, GAP_UP_6_PCT]
    )

    assert result.trades == []


def test_signals_are_looked_for_only_at_bars_inside_the_session(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[pd.Timestamp] = []

    def spy(self: Any, data_15m: Any, data_1h: Any, eval_time: Any) -> list[Any]:
        asked.append(pd.Timestamp(eval_time))
        return []

    monkeypatch.setattr(TrendCrossStrategy, "generate_signals", spy)
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


# --- the real risk engine and the live entry (Phase 3.1) -----------------------------------

IN_SESSION = datetime(2026, 11, 2, 16, 7, tzinfo=UTC)


def test_the_entry_is_sized_and_placed_by_the_entry_plan_of_the_live_system(
    settings: Settings,
) -> None:
    result, follow_index, frame = _run(
        settings, IN_SESSION, [QUIET, QUIET, GAP_UP_6_PCT], config=BacktestConfig(slippage_pct=0.0)
    )

    first_print = Decimal(str(frame.loc[frame["timestamp"] == follow_index[0], "open"].iloc[0]))
    plan = build_entry_plan(
        first_print,
        limit_buffer_pct=Decimal(str(settings.entry_limit_buffer_pct)),
        stop_loss_pct=Decimal(str(settings.stop_loss_pct)),
        take_profit_pct=Decimal(str(settings.take_profit_pct)),
    )
    trade = result.trades[0]
    # Sized with the limit (the most the entry can cost), like the risk engine does it live:
    # 4 % of the account, whole shares.
    assert trade.quantity == (
        Decimal("100000") * Decimal("0.04") / plan.limit_price
    ).to_integral_value(rounding="ROUND_FLOOR")
    # A marketable limit fills at once at the price; with no slippage that is the price itself.
    assert trade.entry_price == first_print
    assert plan.limit_price > first_print


def test_the_fill_costs_the_slippage_but_never_more_than_the_limit(settings: Settings) -> None:
    follow = [QUIET, QUIET, GAP_UP_6_PCT]
    small, follow_index, frame = _run(
        settings, IN_SESSION, follow, config=BacktestConfig(slippage_pct=0.0005)
    )
    large, _, _ = _run(settings, IN_SESSION, follow, config=BacktestConfig(slippage_pct=0.02))

    first_print = Decimal(str(frame.loc[frame["timestamp"] == follow_index[0], "open"].iloc[0]))
    limit = build_entry_plan(
        first_print,
        limit_buffer_pct=Decimal(str(settings.entry_limit_buffer_pct)),
        stop_loss_pct=Decimal(str(settings.stop_loss_pct)),
        take_profit_pct=Decimal(str(settings.take_profit_pct)),
    ).limit_price
    assert small.trades[0].entry_price == first_print * Decimal("1.0005")
    assert large.trades[0].entry_price == limit  # 2 % of slippage cannot get past a 0.1 % limit


def test_the_stop_and_the_target_are_the_ones_of_the_entry_plan_not_of_the_fill(
    settings: Settings,
) -> None:
    # The first bar after entry falls to the plan's stop (4 % under the limit) and not below it.
    result, follow_index, frame = _run(
        settings,
        IN_SESSION,
        [(1.0, 1.001, 0.95, 0.96), QUIET],
        config=BacktestConfig(slippage_pct=0.0),
    )

    first_print = Decimal(str(frame.loc[frame["timestamp"] == follow_index[0], "open"].iloc[0]))
    plan = build_entry_plan(
        first_print,
        limit_buffer_pct=Decimal(str(settings.entry_limit_buffer_pct)),
        stop_loss_pct=Decimal(str(settings.stop_loss_pct)),
        take_profit_pct=Decimal(str(settings.take_profit_pct)),
    )
    trade = result.trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_time == follow_index[0]  # a new position is watched from its own bar
    assert trade.exit_price == plan.stop_price


def test_a_price_that_ran_away_from_the_signal_bar_is_not_chased(settings: Settings) -> None:
    # The next bar opens 6 % above the signal bar's close: the live price check allows 5 %.
    result, _, _ = _run(settings, IN_SESSION, [GAP_UP_6_PCT, GAP_UP_6_PCT])

    assert result.trades == []
    assert result.signals == 1
    assert result.not_traded == {"No usable price": 1}


def test_the_risk_engines_position_limit_applies(settings: Settings) -> None:
    """Two signals on the same bar and room for one position: the live limit lets one through.
    The simplified sizing the backtest used to have knew nothing of this limit."""
    one_slot = _settings(max_open_positions=1)

    result, _, _ = _run(
        one_slot, IN_SESSION, [QUIET, QUIET, GAP_UP_6_PCT], symbols=("AAPL", "MSFT")
    )

    assert result.signals == 2
    assert len(result.trades) == 1
    assert result.not_traded == {"Risk: Max open positions limit reached.": 1}


def test_the_daily_loss_guard_judges_the_replayed_day(settings: Settings) -> None:
    # A zero allowance stops every entry. That it is the guard speaking (and not the expiry
    # of an old signal or a missing baseline) shows in the reason.
    no_loss_allowed = _settings(max_daily_loss_pct=0.0)

    result, _, _ = _run(no_loss_allowed, IN_SESSION, [QUIET, QUIET, GAP_UP_6_PCT])

    assert result.trades == []
    assert list(result.not_traded) == ["Risk: Daily loss guard breached"]


def test_the_entry_fee_is_in_the_trade(settings: Settings) -> None:
    result, _, _ = _run(
        settings,
        IN_SESSION,
        [QUIET, QUIET, GAP_UP_6_PCT],
        config=BacktestConfig(slippage_pct=0.0, fee_pct_per_trade=0.001),
    )

    trade = result.trades[0]
    assert trade.exit_price is not None and trade.pnl is not None
    entry_fee = trade.quantity * trade.entry_price * Decimal("0.001")
    exit_fee = trade.quantity * trade.exit_price * Decimal("0.001")
    assert trade.fees_paid == entry_fee + exit_fee
    assert trade.pnl == trade.quantity * (trade.exit_price - trade.entry_price) - trade.fees_paid
    # and the cash agrees with the trade
    assert result.final_capital == Decimal("100000") + trade.pnl
