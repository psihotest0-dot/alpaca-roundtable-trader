"""The hours research (research/intraday.py): the opening-range breakout and its control, measured
on made-up 15-minute sessions whose truth is known, with the hold-out kept closed."""

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.research import stock_names
from trading_system.research.intraday import (
    EXIT_SLOT,
    FIRST_SIGNAL_SLOT,
    LAST_SIGNAL_SLOT,
    OPENING_SLOTS,
    INTRADAY_HOLDOUT_START,
    SLOTS,
    Session,
    build_session,
    forward_to_close,
    format_intraday,
    load_bars,
    measure_intraday,
    orb_30m_long,
    random_entry,
)
from trading_system.storage.research import open_observe_store

NY = "America/New_York"


def _slot_times(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{day.date()} 09:30", tz=NY).tz_convert("UTC")
    return pd.DatetimeIndex([start + pd.Timedelta(minutes=15 * k) for k in range(SLOTS)])


def _frame(day: pd.Timestamp, opens: np.ndarray, closes: np.ndarray) -> pd.DataFrame:
    highs = np.maximum(opens, closes) + 0.01
    lows = np.minimum(opens, closes) - 0.01
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=_slot_times(day).rename("timestamp"),
    )


def planted_sessions(
    days: int = 40,
    stocks: int = 80,
    *,
    planted: float = 0.05,
    fraction: float = 0.4,
    seed: int = 3,
) -> dict[str, pd.DataFrame]:
    """Every stock opens at 100 with an opening range whose high is 100.2. A stock-day is either
    quiet (never closes above 100.2 again) or breaks out at 10:00 and then drifts by `planted` a bar.
    So the signal fires exactly on the breakout days, and the truth after it is `planted`."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[pd.DataFrame]] = {s: [] for s in stock_names(stocks)}
    for day in pd.bdate_range("2026-07-06", periods=days):
        for symbol in out:
            closes = np.full(SLOTS, 100.0)
            if rng.random() < fraction:
                closes[2] = 100.5
                closes[3:] = (
                    100.5 + planted * np.arange(1, SLOTS - 2) + rng.normal(0, 0.03, SLOTS - 3)
                )
            else:
                closes[2:] = 100.0 + rng.normal(0, 0.03, SLOTS - 2)
            opens = np.concatenate([[100.0], closes[:-1]])
            frame = _frame(day, opens, closes)
            frame.iloc[:2, frame.columns.get_loc("high")] = 100.2  # the opening range
            out[symbol].append(frame)
    return {s: pd.concat(parts) for s, parts in out.items()}


def walking_sessions(days: int = 40, stocks: int = 80, seed: int = 5) -> dict[str, pd.DataFrame]:
    """Plain random walks, no effect anywhere."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[pd.DataFrame]] = {s: [] for s in stock_names(stocks)}
    for day in pd.bdate_range("2026-07-06", periods=days):
        for symbol in out:
            closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, SLOTS)))
            opens = np.concatenate([[100.0], closes[:-1]])
            out[symbol].append(_frame(day, opens, closes))
    return {s: pd.concat(parts) for s, parts in out.items()}


def _one(closes: dict[int, float], *, highs: dict[int, float] | None = None) -> Session:
    """One stock, one day: quiet at 100 except for the closes given (and the highs given)."""
    shape = (1, SLOTS, 1)
    opens, high, close = (np.full(shape, 100.0) for _ in range(3))
    high[:, :, :] = 100.0
    high[0, 0, 0] = high[0, 1, 0] = 100.2
    for slot, value in closes.items():
        close[0, slot, 0] = value
    for slot, value in (highs or {}).items():
        high[0, slot, 0] = value
    return Session(pd.DatetimeIndex(["2026-07-06"]), ["AAA"], opens, high, close)


# ------------------------------------------------------------------------------ the signal


def test_the_first_close_above_the_opening_range_is_the_signal_and_the_entry_is_the_next_bar() -> (
    None
):
    fired, entry = orb_30m_long(_one({2: 100.5}))

    assert bool(fired[0, 0]) and int(entry[0, 0]) == 3


def test_a_close_equal_to_the_range_high_is_not_a_breakout() -> None:
    fired, _ = orb_30m_long(_one({2: 100.2, 5: 100.2}))

    assert not fired[0, 0]


def test_only_the_first_breakout_counts() -> None:
    fired, entry = orb_30m_long(_one({4: 100.5, 8: 101.0}))

    assert bool(fired[0, 0]) and int(entry[0, 0]) == 5  # in after the first, not the second


def test_the_signal_window_is_ten_to_a_quarter_to_two() -> None:
    last_fired, last_entry = orb_30m_long(_one({17: 100.5}))  # 13:45 closes at 14:00
    too_late, _ = orb_30m_long(_one({18: 100.5}))  # 14:00 is out
    opening, _ = orb_30m_long(_one({1: 100.5}))  # the range itself is not a signal bar

    assert bool(last_fired[0, 0]) and int(last_entry[0, 0]) == 18
    assert not too_late[0, 0] and not opening[0, 0]


def test_no_opening_range_no_signal() -> None:
    session = _one({2: 100.5})
    session.high[0, 1, 0] = np.nan  # the 09:45 bar is missing

    fired, _ = orb_30m_long(session)

    assert not fired[0, 0]


def test_a_breakout_without_a_bar_to_enter_on_or_without_an_exit_is_no_event_and_no_second_try() -> (
    None
):
    no_entry = _one({2: 100.5, 6: 101.0})
    no_entry.open[0, 3, 0] = np.nan  # nothing to enter on after the first breakout
    no_exit = _one({2: 100.5})
    no_exit.close[0, EXIT_SLOT, 0] = np.nan

    fired_a, _ = orb_30m_long(no_entry)
    fired_b, _ = orb_30m_long(no_exit)

    assert not fired_a[0, 0]  # the later breakout at 6 is not used instead
    assert not fired_b[0, 0]


def test_the_constants_are_the_sessions_clock_as_registered() -> None:
    def slot(hour: int, minute: int) -> int:
        return (hour * 60 + minute - (9 * 60 + 30)) // 15

    assert SLOTS == 26  # 09:30 to 15:45, 26 bars of 15 minutes
    assert EXIT_SLOT == slot(15, 45) == 25  # the last bar's close is the end of the session
    assert (FIRST_SIGNAL_SLOT, LAST_SIGNAL_SLOT) == (slot(10, 0), slot(13, 45))
    assert OPENING_SLOTS == slot(10, 0)  # the opening range is the first 30 minutes


def test_the_return_runs_from_the_open_of_the_entry_bar_to_the_close_of_the_last_bar() -> None:
    session = _one({3: 101.0, 21: 100.0, EXIT_SLOT: 102.0})
    session.open[0, 3, 0] = 100.0  # the bar it enters at opens at 100 and closes at 101

    forward = forward_to_close(session)

    assert forward[0, 3, 0] == pytest.approx(
        0.02
    )  # 102 / 100: not from the 101 close, not to slot 21
    assert forward[0, EXIT_SLOT, 0] == pytest.approx(102.0 / 100.0 - 1)  # the last bar's own open


def test_the_control_draws_other_stocks_on_other_days() -> None:
    session = build_session(walking_sessions(days=6, stocks=80))

    fired, _ = random_entry(session)

    rows = {tuple(row) for row in fired}
    assert len(rows) == len(fired)  # no two days drew the same tenth


def test_the_control_draws_the_same_stocks_for_the_same_day() -> None:
    session = build_session(walking_sessions(days=5))

    first, _ = random_entry(session)
    second, _ = random_entry(session)

    assert (first == second).all()
    assert 0.05 < first.mean() < 0.16  # about a tenth


# ------------------------------------------------------------------------------ the bars


def test_only_the_regular_session_counts_in_winter_and_in_summer() -> None:
    def day_bars(day: str) -> pd.DataFrame:
        times = pd.date_range(f"{day} 08:00", f"{day} 16:30", freq="15min", tz=NY)
        n = len(times)
        return pd.DataFrame(
            {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0},
            index=times.tz_convert("UTC"),
        ).iloc[:n]

    for day in ("2026-01-12", "2026-07-13"):  # EST and EDT: 09:30 is 14:30 or 13:30 UTC
        session = build_session({"AAA": day_bars(day)})

        assert session.open.shape == (1, SLOTS, 1)
        assert np.isfinite(session.open[0, :, 0]).all()  # all 26 slots, nothing from 08:00-09:15
        assert session.days[0] == pd.Timestamp(day)


def test_a_bar_off_the_quarter_hour_is_left_out() -> None:
    bars = walking_sessions(days=1, stocks=1)
    name = next(iter(bars))
    odd = bars[name].copy()
    odd.index = odd.index + pd.Timedelta(minutes=7)

    session = build_session({name: odd})

    assert np.isnan(session.open).all()


# ------------------------------------------------------------------------------ the measurement


def test_an_effect_planted_after_the_breakout_is_found_and_costs_are_taken_off() -> None:
    session = build_session(planted_sessions(planted=0.05))

    m = measure_intraday(session, "orb_30m_long")

    # ~22 bars of +0.05 on 100.5, minus the universe's own drift (40 % of stocks): about +0.65 %
    assert m.vs_universe.mean == pytest.approx(0.0065, abs=0.0015)
    assert m.net.mean == pytest.approx(m.vs_universe.mean - 0.001)
    assert m.net.t > 10 and m.events >= 1000
    assert m.criteria.all_met and m.days_with_signal == 40
    assert "meets them" in format_intraday(m)


def test_the_control_stays_quiet_where_the_signal_finds_the_effect() -> None:
    session = build_session(planted_sessions(planted=0.05))

    control = measure_intraday(session, "random_entry")

    assert abs(control.vs_universe.t) < 3 and abs(control.vs_universe.mean) < 0.003


def test_pure_noise_gives_neither_the_signal_nor_the_control_anything() -> None:
    session = build_session(walking_sessions())

    orb = measure_intraday(session, "orb_30m_long")
    control = measure_intraday(session, "random_entry")

    assert abs(orb.vs_universe.t) < 3.5 and abs(control.vs_universe.t) < 3.5
    assert not orb.criteria.all_met


def test_a_negative_effect_is_found_with_the_opposite_sign() -> None:
    session = build_session(planted_sessions(planted=-0.05))

    m = measure_intraday(session, "orb_30m_long")

    assert m.vs_universe.mean < -0.003 and m.net.t < -3 and not m.criteria.all_met


def test_days_with_too_few_signals_are_not_observations() -> None:
    session = build_session(planted_sessions(fraction=0.01, stocks=80, days=10, seed=1))

    m = measure_intraday(session, "orb_30m_long")  # ~1 of 80 a day: under the 5-stock minimum

    assert m.days_with_signal <= 1 and not m.criteria.enough_events


def test_the_results_come_by_month() -> None:
    session = build_session(planted_sessions(days=60))

    m = measure_intraday(session, "orb_30m_long")

    assert [s.label for s in m.by_month] == ["2026-07", "2026-08", "2026-09"]
    assert sum(s.dates for s in m.by_month) == m.days_with_signal


def test_an_unknown_signal_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown intraday signal"):
        measure_intraday(build_session(walking_sessions(days=3)), "gap_and_go")


# ----------------------------------------------------------------------------- the hold-out


def _store(tmp_path: Any, bars: dict[str, pd.DataFrame]) -> Any:
    trading = f"sqlite:///{(tmp_path / 'trading.db').as_posix()}"
    store = open_observe_store(f"sqlite:///{(tmp_path / 'observe.db').as_posix()}", trading)
    for symbol, frame in bars.items():
        store.replace(
            symbol,
            "15m",
            frame,
            adjustment="all",
            fetched_at=datetime(2026, 11, 6, tzinfo=timezone.utc),
        )
    return store


def test_the_hold_out_is_never_loaded(tmp_path: Any) -> None:
    days = pd.bdate_range("2026-08-24", "2026-09-30")  # runs well past 2026-09-08
    bars: dict[str, list[pd.DataFrame]] = {s: [] for s in stock_names(3)}
    for day in days:
        for symbol in bars:
            closes = np.full(SLOTS, 100.0)
            bars[symbol].append(_frame(day, np.full(SLOTS, 100.0), closes))
    store = _store(tmp_path, {s: pd.concat(p) for s, p in bars.items()})

    session = load_bars(store, list(bars))

    assert session.days[-1].date() < INTRADAY_HOLDOUT_START
    assert session.days[-1] >= pd.Timestamp(INTRADAY_HOLDOUT_START) - pd.Timedelta(days=5)
    with pytest.raises(ValueError, match="hold-out"):
        load_bars(store, list(bars), end=date(2026, 10, 1))


def test_the_spy_market_symbol_is_not_one_of_the_stocks(tmp_path: Any) -> None:
    bars = walking_sessions(days=3, stocks=3)
    bars["SPY"] = next(iter(bars.values()))
    store = _store(tmp_path, bars)

    session = load_bars(store, [*bars])

    assert "SPY" not in session.symbols and len(session.symbols) == 3
