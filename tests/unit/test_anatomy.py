"""The anatomy of closed trades (research/anatomy.py, plan section 16.2 and 16.3)."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_system.research.anatomy import (
    MIN_CLOSED_TRADES,
    SPLITS,
    STOP,
    TARGET,
    Report,
    Trade,
    format_report,
    leg_split,
    pair_round_trips,
    r_atr,
    r_plan,
    run_split,
    trades_from_trips,
)
from trading_system.research.paper_report import exit_reason_of


def fill(
    symbol: str, side: str, price: float, qty: float, when: str, order: str = "o"
) -> dict[str, Any]:
    return {
        "type": "fill",
        "symbol": symbol,
        "side": side,
        "price": str(price),
        "qty": str(qty),
        "transaction_time": when,
        "order_id": order,
    }


T0 = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc)


def trade(
    entry: float = 100.0,
    exit: float = 105.0,
    day: int = 0,
    *,
    held_days: int = 1,
    limit: float | None = 100.1,
    stop: float | None = 96.0,
    context: dict[str, Any] | None = None,
    reason: str = TARGET,
    symbol: str = "AAA",
) -> Trade:
    start = T0 + timedelta(days=day)
    return Trade(
        symbol=symbol,
        entry_time=start,
        entry_price=entry,
        exit_time=start + timedelta(days=held_days),
        exit_price=exit,
        exit_reason=reason,
        quantity=10,
        limit_price=limit,
        stop_price=stop,
        context=context or {},
    )


# --------------------------------------------------------------------------- round trips


def test_buys_and_sells_make_round_trips_with_volume_weighted_prices() -> None:
    fills = [
        fill("AAA", "buy", 10.0, 6, "2026-11-20T15:00:00Z", "e1"),
        fill("AAA", "buy", 11.0, 4, "2026-11-20T15:00:05Z", "e1"),  # a partial fill of the entry
        fill("AAA", "sell", 12.0, 10, "2026-11-23T16:00:00Z", "x1"),
        fill("BBB", "buy", 50.0, 1, "2026-11-20T15:30:00Z", "e2"),  # still open
        fill("AAA", "buy", 20.0, 1, "2026-11-24T15:00:00Z", "e3"),
        fill("AAA", "sell", 19.0, 1, "2026-11-24T16:00:00Z", "x3"),
    ]

    trips = pair_round_trips(fills)

    assert [t["symbol"] for t in trips] == ["AAA", "AAA"]  # BBB is open and is left out
    trades = trades_from_trips(
        trips, lambda order: STOP if order == "x3" else TARGET, lambda order: {"limit_price": 11.5}
    )
    first, second = trades
    assert first.entry_price == pytest.approx(10.4) and first.exit_price == 12.0
    assert first.quantity == 10 and first.exit_reason == TARGET and first.limit_price == 11.5
    assert second.exit_reason == STOP and second.entry_time < second.exit_time


def test_other_activity_types_are_ignored() -> None:
    odd = fill("AAA", "buy", 1.0, 1, "2026-11-20T15:00:00Z")
    odd["type"] = "dividend"

    assert pair_round_trips([odd]) == []


# --------------------------------------------------------------------------- the numbers


def test_r_plan_is_the_move_over_the_planned_risk() -> None:
    t = trade(entry=100.0, exit=105.0, limit=100.0, stop=96.0)

    assert r_plan(t) == pytest.approx(1.25)
    assert r_plan(trade(entry=100.0, exit=96.0, limit=100.0, stop=96.0)) == pytest.approx(-1.0)
    assert r_plan(trade(limit=None)) is None and r_plan(trade(stop=None)) is None
    assert r_plan(trade(limit=96.0, stop=96.0)) is None


def test_r_in_atr_units_needs_the_context() -> None:
    assert r_atr(trade(entry=100.0, exit=105.0, context={"atr_pct": 2.5})) == pytest.approx(2.0)
    assert r_atr(trade(context={})) is None


def daily(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [o for o, _ in rows.values()], "close": [c for _, c in rows.values()]},
        index=pd.DatetimeIndex(list(rows)),
    )


def test_the_nights_and_the_days_add_up_to_the_trade() -> None:
    bars = daily(
        {
            "2026-11-20": (100.0, 102.0),  # Friday
            "2026-11-23": (103.0, 104.0),  # Monday: one night over the weekend
            "2026-11-24": (104.5, 106.0),
        }
    )
    t = trade(entry=101.0, exit=105.0, held_days=4)  # in on Friday, out on Tuesday

    nights, days = leg_split(t, bars)  # type: ignore[misc]

    expected_nights = math.log(103.0 / 102.0) + math.log(104.5 / 104.0)
    assert nights == pytest.approx(expected_nights)
    assert nights + days == pytest.approx(math.log(105.0 / 101.0))


def test_a_trade_inside_one_day_has_no_night_and_a_missing_day_gives_nothing() -> None:
    same_day = trade(entry=100.0, exit=101.0, held_days=0)
    assert leg_split(same_day, daily({"2026-11-20": (100.0, 101.0)})) == (
        0.0,
        pytest.approx(math.log(1.01)),
    )
    gap = daily({"2026-11-20": (100.0, 101.0), "2026-11-24": (101.0, 102.0)})
    assert leg_split(trade(held_days=4), gap) is not None  # two edge days are enough for one night
    assert leg_split(trade(held_days=4), daily({"2026-11-23": (1.0, 1.0)})) is None


def test_the_exit_order_type_names_the_reason() -> None:
    assert exit_reason_of("stop") == STOP and exit_reason_of("limit") == TARGET
    assert exit_reason_of("market") == "other"


# --------------------------------------------------------------------------- the splits


def many(n_days: int, per_day: int, effect: float, *, seed: int = 1) -> list[Trade]:
    """Trades whose R is `effect` higher when the hour split's side is True, plus noise."""
    rng = np.random.default_rng(seed)
    out: list[Trade] = []
    for day in range(n_days):
        for k in range(per_day):
            early = k % 2 == 0
            r = rng.normal(0, 1) + (effect if early else 0.0)
            out.append(
                trade(
                    entry=100.0,
                    exit=100.0 + r * 4.1,
                    limit=100.0,
                    stop=95.9,
                    day=day,
                    context={"minutes_since_open": 0 if early else 120, "group": "risky"},
                )
            )
    return out


def test_a_real_difference_shows_and_noise_does_not() -> None:
    name = "hour: first 30 minutes"
    strong = run_split(name, many(60, 6, effect=1.0))
    noise = run_split(name, many(60, 6, effect=0.0, seed=7))

    assert strong.shows_something and strong.t > 2.6 and strong.same_sign_in_halves
    assert not noise.shows_something


def test_a_split_needs_fifty_trades_a_side_and_the_same_sign_in_both_halves() -> None:
    few = run_split("hour: first 30 minutes", many(6, 6, effect=3.0))  # 18 a side
    assert few.n_true == 18 and not few.shows_something

    first = many(30, 6, effect=2.0)
    second = [
        trade(
            exit=100.0 - (4.1 if early else 0.0) * 0.5,
            limit=100.0,
            stop=95.9,
            day=40 + d,
            context={"minutes_since_open": 0 if early else 120, "group": "risky"},
        )
        for d in range(30)
        for early in (True, False, True, False, True, False)
    ]
    flipped = run_split("hour: first 30 minutes", first + second)
    assert not flipped.same_sign_in_halves and not flipped.shows_something


def test_a_trade_without_the_field_is_on_neither_side() -> None:
    result = run_split(
        "earnings within 2 sessions", [trade(), trade(context={"earnings_known": False})]
    )

    assert result.n_true == 0 and result.n_false == 0 and not result.shows_something
    assert set(SPLITS) == {
        "group: risky vs normal",
        "market: SPY above its 200-day average",
        "stress: SPY volatility percentile 67 or more",
        "hour: first 30 minutes",
        "earnings within 2 sessions",
    }


# --------------------------------------------------------------------------- the report


def test_before_300_closed_trades_only_the_operational_numbers_are_shown() -> None:
    trades = [trade(context={"group": "risky", "atr_pct": 2.0}) for _ in range(12)]

    text = format_report(Report(trades=trades, open_trades=3, entries_filled=15))

    assert "closed trades: 12" in text and "risky: 12 closed" in text
    assert "not shown until 300 closed trades" in text
    assert "mean R" not in text and "nights" not in text and "SHOWS" not in text
    assert "t +" not in text


def test_from_300_the_result_and_the_splits_are_shown() -> None:
    trades = many(50, 6, effect=0.0)
    assert len(trades) == MIN_CLOSED_TRADES
    bars = {"AAA": daily({"2026-11-20": (100.0, 100.0), "2026-12-30": (100.0, 100.0)})}

    text = format_report(Report(trades=trades, open_trades=0, entries_filled=300, daily=bars))

    assert "mean R of the plan" in text and "the five splits" in text
    assert "hour: first 30 minutes" in text and "not shown until" not in text


def test_an_empty_report_says_so() -> None:
    assert "closed trades: 0" in format_report(Report(trades=[]))


def test_a_difference_below_the_bar_of_2_6_is_not_shown_even_with_many_trades() -> None:
    """t between 1 and 2.6, 150+ trades a side, the same sign in both halves: still "not shown"."""
    middling = run_split("hour: first 30 minutes", many(60, 6, effect=0.18, seed=4))

    assert middling.n_true >= 50 and middling.n_false >= 50 and middling.same_sign_in_halves
    assert 1.0 < abs(middling.t) < 2.6
    assert not middling.shows_something


def test_a_side_with_fewer_than_fifty_trades_does_not_show_whatever_the_t() -> None:
    everything = many(60, 6, effect=1.5, seed=2)
    early = [t for t in everything if t.context["minutes_since_open"] == 0]
    late = [t for t in everything if t.context["minutes_since_open"] == 120][::6]  # 30, spread out
    lopsided = early + late

    result = run_split("hour: first 30 minutes", lopsided)

    assert result.n_false == 30 and result.n_true >= 50
    assert abs(result.t) >= 2.6 and result.same_sign_in_halves
    assert not result.shows_something


def test_the_reasons_are_the_words_the_report_and_the_tests_use() -> None:
    assert (STOP, TARGET) == ("stop", "target")
    assert exit_reason_of("stop") == "stop" and exit_reason_of("limit") == "target"
