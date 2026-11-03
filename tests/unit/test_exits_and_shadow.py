"""How a bar closes a position, and a signal played out without being traded (Phase 3.5)."""

from decimal import Decimal as D

import pandas as pd
import pytest

from trading_system.backtest.exits import resolve_exit
from trading_system.backtest.shadow import simulate_shadow_trade

T0 = pd.Timestamp("2026-11-02 15:00", tz="UTC")


# --- resolve_exit ---------------------------------------------------------------------------


def test_a_bar_inside_the_levels_exits_nothing() -> None:
    assert resolve_exit(D(100), D(103), D(98), D(95), D(105)) is None


def test_the_stop_is_filled_at_its_level_when_the_bar_trades_through_it() -> None:
    assert resolve_exit(D(100), D(101), D(94), D(95), D(105)) == ("STOP_LOSS", D(95))


def test_a_gap_down_through_the_stop_is_filled_at_the_worse_open() -> None:
    assert resolve_exit(D(90), D(92), D(88), D(95), D(105)) == ("STOP_LOSS", D(90))


def test_the_target_is_filled_at_its_level_when_the_bar_trades_through_it() -> None:
    assert resolve_exit(D(100), D(106), D(99), D(95), D(105)) == ("TAKE_PROFIT", D(105))


def test_a_gap_up_through_the_target_is_filled_at_the_better_open() -> None:
    assert resolve_exit(D(108), D(110), D(107), D(95), D(105)) == ("TAKE_PROFIT", D(108))


def test_a_bar_that_reaches_both_levels_is_a_loss() -> None:
    assert resolve_exit(D(100), D(110), D(90), D(95), D(105)) == ("STOP_LOSS", D(95))


def test_missing_levels_are_not_hit() -> None:
    assert resolve_exit(D(100), D(200), D(1), None, None) is None
    assert resolve_exit(D(100), D(200), D(99), None, D(105)) == ("TAKE_PROFIT", D(105))


# --- simulate_shadow_trade ------------------------------------------------------------------


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range(T0, periods=len(rows), freq="15min")
    o, h, low, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c}, index=index)


def _shadow(rows, *, slippage=0.0, fee=0.0, entry_time=T0, entry=D(100), stop=D(96), target=D(105)):
    return simulate_shadow_trade(
        symbol="AAPL",
        signal_id="abc",
        verdict="VETO",
        bars=_bars(rows),
        entry_time=entry_time.to_pydatetime(),
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        slippage_pct=slippage,
        fee_pct=fee,
    )


QUIET = (100.0, 101.0, 99.0, 100.0)


def test_a_shadow_position_that_reaches_the_target_wins() -> None:
    trade = _shadow([QUIET, QUIET, (101.0, 106.0, 100.0, 105.5)])

    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.exit_time == (T0 + pd.Timedelta(minutes=30)).to_pydatetime()
    assert trade.pnl_pct == pytest.approx(0.05)


def test_a_shadow_position_that_reaches_the_stop_loses() -> None:
    trade = _shadow([QUIET, (99.0, 100.0, 95.0, 96.0)])

    assert trade.exit_reason == "STOP_LOSS"
    assert trade.pnl_pct == pytest.approx(-0.04)


def test_the_bar_of_the_entry_is_watched_like_for_a_real_position() -> None:
    trade = _shadow([(100.0, 100.5, 94.0, 95.0), QUIET])

    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_time == T0.to_pydatetime()


def test_bars_before_the_entry_do_not_count() -> None:
    # The first bar would have hit the stop, but the entry is on the second.
    trade = _shadow(
        [(99.0, 99.5, 90.0, 92.0), QUIET, (100.0, 106.0, 99.5, 105.0)],
        entry_time=T0 + pd.Timedelta(minutes=15),
    )

    assert trade.exit_reason == "TAKE_PROFIT"


def test_a_position_neither_level_reaches_is_closed_at_the_last_close_and_marked_open() -> None:
    trade = _shadow([QUIET, (100.0, 101.0, 99.0, 102.0)], slippage=0.001)

    assert trade.exit_reason == "OPEN"
    assert trade.exit_price == D("102.0")
    assert trade.pnl_pct == pytest.approx(0.02)  # no slippage on a mark, only on a real exit


def test_slippage_and_fees_are_in_the_result_like_for_a_real_trade() -> None:
    clean = _shadow([QUIET, (101.0, 106.0, 100.0, 105.5)])
    costly = _shadow([QUIET, (101.0, 106.0, 100.0, 105.5)], slippage=0.001, fee=0.001)

    # sold at 105 x 0.999, after a 0.1 % fee; bought at 100 plus a 0.1 % fee
    expected = (105 * 0.999 * 0.999) / (100 * 1.001) - 1
    assert costly.pnl_pct == pytest.approx(expected)
    assert costly.pnl_pct < clean.pnl_pct
    assert costly.exit_price == D("105") * D("0.999")


def test_a_gap_is_filled_at_the_open() -> None:
    trade = _shadow([QUIET, (90.0, 92.0, 88.0, 91.0)])

    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == D("90.0")
