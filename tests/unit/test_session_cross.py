"""The two candidates that repair the live strategy (research plan, section 12).

They are `trend_cross` shown cleaner data: only bars of the regular session reach the indicators,
and the "late" one also takes no entry on a signal bar that starts in the first 30 minutes. The
proofs here are about those two changes and nothing else: on data that was clean already they must
signal exactly where `trend_cross` does; on data with stray prints outside the session they must
not move, while `trend_cross` does; and the hint the backtest uses to skip bars must never say
"impossible" where the strategy signals."""

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.bars import alpaca_shaped, random_walk_session_bars, with_timestamp_column
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.registry import get_strategy, strategy_names
from trading_system.strategy.session_cross import (
    SessionTrendCrossStrategy,
    session_mask_1h,
    session_mask_15m,
)
from trading_system.strategy.trend_cross import TrendCrossStrategy

SEED = 3  # 15 weekdays from 2026-11-02, drift 0.0004: ten signals for the live parameters
EARLY_SEED = 1  # 20 weekdays: a cross-up on the 09:30 bar of 2026-11-04
EARLY_SEED_SECOND_BAR = 5  # 20 weekdays: a cross-up on the 09:45 bar of 2026-11-19
DAYS = 15
DRIFT = 0.0004

NAMES = ("trend_cross", "trend_cross_session", "trend_cross_session_late")


def _clean(seed: int = SEED, days: int = DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    return random_walk_session_bars(seed, "2026-11-02", days, drift=DRIFT)


def _with_stray_prints(
    bars_15m: pd.DataFrame, bars_1h: pd.DataFrame, seed: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same bars plus what the stored data really has outside the session: a few thin bars
    before the open and after the close of every day, each at a price up to a few percent off."""
    rng = np.random.default_rng(seed)
    days = sorted({t.normalize() for t in bars_15m["timestamp"]})
    rows_15m: list[tuple[pd.Timestamp, float]] = []
    rows_1h: list[tuple[pd.Timestamp, float]] = []
    for day in days:
        reference = float(bars_15m["open"][bars_15m["timestamp"] >= day].iloc[0])
        for hours, minutes in (
            (13, 0),
            (13, 15),
            (13, 30),
            (13, 45),
            (14, 0),
            (14, 15),
            (21, 0),
            (21, 15),
        ):
            stray = reference * (1 + rng.normal(0, 0.03))
            rows_15m.append((day + pd.Timedelta(hours=hours, minutes=minutes), stray))
        for hours in (13, 21):  # 08:00 and 16:00 New York
            rows_1h.append((day + pd.Timedelta(hours=hours), reference * (1 + rng.normal(0, 0.03))))

    def extra(rows: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "open": [p for _, p in rows],
                "high": [p for _, p in rows],
                "low": [p for _, p in rows],
                "close": [p for _, p in rows],
                "volume": 1.0,
            },
            index=pd.DatetimeIndex([t for t, _ in rows]),
        )
        return frame

    out = []
    for frame, rows in ((bars_15m, rows_15m), (bars_1h, rows_1h)):
        merged = pd.concat([frame.drop(columns="timestamp"), extra(rows)]).sort_index()
        out.append(with_timestamp_column(merged))
    return out[0], out[1]


@cache
def _data(polluted: bool, seed: int = SEED, days: int = DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = _clean(seed, days)
    return _with_stray_prints(*clean) if polluted else clean


@cache
def _signals(name: str, polluted: bool) -> tuple[pd.Timestamp, ...]:
    """The bars (by start time) the strategy signalled on, one evaluation at every bar's close."""
    bars_15m, bars_1h = _data(polluted)
    strategy = get_strategy(name)
    found: list[pd.Timestamp] = []
    for stamp in bars_15m["timestamp"]:
        when = stamp + pd.Timedelta(minutes=15)
        for signal in strategy.generate_signals({"X": bars_15m}, {"X": bars_1h}, when):
            found.append(pd.Timestamp(signal.closed_bar_timestamp))
    return tuple(found)


def _new_york_minute(stamp: pd.Timestamp) -> int:
    local = stamp.tz_convert("America/New_York")
    return int(local.hour * 60 + local.minute)


# ---- which bars are the regular session --------------------------------------------------------


@pytest.mark.parametrize(
    "day, expected_open_utc",
    [("2026-11-04", "14:30"), ("2026-07-08", "13:30")],  # winter time and summer time
)
def test_the_session_bars_are_those_from_the_open_to_the_last_quarter_hour(
    day: str, expected_open_utc: str
) -> None:
    open_utc = pd.Timestamp(f"{day} {expected_open_utc}", tz="UTC")
    # minutes from the open: 08:45, 09:15, 09:30, 09:45, 15:45, 16:00, 16:15, 16:30
    offsets = [-45, -15, 0, 15, 375, 390, 405, 420]
    instants = pd.DatetimeIndex([open_utc + pd.Timedelta(minutes=m) for m in offsets])

    kept = session_mask_15m(instants)

    assert list(kept) == [False, False, True, True, True, False, False, False]


def test_the_hourly_bars_kept_are_those_from_09_00_to_15_00():
    open_utc = pd.Timestamp("2026-11-04 13:00", tz="UTC")  # 08:00 New York
    instants = pd.DatetimeIndex([open_utc + pd.Timedelta(hours=h) for h in range(0, 9)])

    kept = session_mask_1h(instants)

    # 08:00 out; 09:00, 10:00 ... 15:00 in; 16:00 out
    assert list(kept) == [False, True, True, True, True, True, True, True, False]


# ---- the two changes, and only those -----------------------------------------------------------


def test_on_data_that_was_clean_already_it_signals_exactly_where_trend_cross_does():
    live = _signals("trend_cross", polluted=False)

    assert len(live) >= 5  # the comparison must have something to compare
    assert _signals("trend_cross_session", polluted=False) == live


def test_stray_prints_outside_the_session_move_trend_cross_but_not_the_session_version():
    live_clean = _signals("trend_cross", polluted=False)
    live_dirty = _signals("trend_cross", polluted=True)
    session_clean = _signals("trend_cross_session", polluted=False)
    session_dirty = _signals("trend_cross_session", polluted=True)

    assert live_dirty != live_clean, "the stray prints must change what trend_cross sees"
    assert session_dirty == session_clean


@pytest.mark.parametrize(
    "seed, bar_minute",
    [(EARLY_SEED, 9 * 60 + 30), (EARLY_SEED_SECOND_BAR, 9 * 60 + 45)],
    ids=["cross-up on the 09:30 bar", "cross-up on the 09:45 bar"],
)
def test_the_late_version_drops_exactly_the_signals_of_the_first_half_hour(
    seed: int, bar_minute: int
) -> None:
    bars_15m, bars_1h = _clean(seed, 20)
    session = get_strategy("trend_cross_session")
    late = get_strategy("trend_cross_session_late")

    session_early, late_early = [], []
    for stamp in bars_15m["timestamp"]:
        if _new_york_minute(stamp) >= 10 * 60:
            continue  # only the evaluations at the first two bars of a day matter here
        when = stamp + pd.Timedelta(minutes=15)
        session_early += session.generate_signals({"X": bars_15m}, {"X": bars_1h}, when)
        late_early += late.generate_signals({"X": bars_15m}, {"X": bars_1h}, when)

    started = {_new_york_minute(pd.Timestamp(s.closed_bar_timestamp)) for s in session_early}
    assert bar_minute in started, "the fixture must have a cross-up on that bar"
    assert late_early == []


def test_the_late_version_keeps_every_signal_that_starts_at_10_00_or_later():
    session = _signals("trend_cross_session", polluted=False)
    late = _signals("trend_cross_session_late", polluted=False)

    assert late == tuple(t for t in session if _new_york_minute(t) >= 10 * 60)
    assert len(late) >= 5


# ---- the hint the backtest uses ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["trend_cross_session", "trend_cross_session_late"])
def test_the_hint_never_says_impossible_where_the_strategy_signals(name: str) -> None:
    bars_15m, bars_1h = _data(polluted=True)
    strategy = get_strategy(name)

    flags = strategy.candidate_bars(bars_15m, bars_1h)

    assert flags is not None
    outside = ~session_mask_15m(pd.DatetimeIndex(bars_15m["timestamp"]))
    assert not flags[outside].any(), "a bar outside the session can never carry a signal"
    fired = _signals(name, polluted=True)
    assert fired
    starts = list(bars_15m["timestamp"])
    for stamp in fired:
        assert flags[starts.index(stamp)], f"the hint said impossible where {name} signalled"


# ---- shape, versions, registry -----------------------------------------------------------------


def test_the_three_strategies_have_versions_of_their_own_so_the_records_do_not_mix():
    versions = {get_strategy(name).version for name in NAMES}

    assert len(versions) == 3
    assert get_strategy("trend_cross").version == RUNTIME_STRATEGY.strategy_version


def test_the_signal_ids_differ_for_the_same_bar():
    bars_15m, bars_1h = _clean()
    stamp = _signals("trend_cross", polluted=False)[0]
    when = stamp + pd.Timedelta(minutes=15)

    ids = {
        get_strategy(name).generate_signals({"X": bars_15m}, {"X": bars_1h}, when)[0].signal_id
        for name in ("trend_cross", "trend_cross_session")
    }

    assert len(ids) == 2


def test_it_reads_alpacas_own_frame_shape_and_empty_frames():
    bars_15m, bars_1h = _clean()
    stamp = _signals("trend_cross_session", polluted=False)[0]
    when = stamp + pd.Timedelta(minutes=15)
    strategy = get_strategy("trend_cross_session")

    plain = strategy.generate_signals({"X": bars_15m}, {"X": bars_1h}, when)
    alpaca = strategy.generate_signals(
        {"X": alpaca_shaped("X", bars_15m.drop(columns="timestamp"))},
        {"X": alpaca_shaped("X", bars_1h.drop(columns="timestamp"))},
        when,
    )

    assert [s.signal_id for s in alpaca] == [s.signal_id for s in plain] != []
    assert strategy.generate_signals({"X": bars_15m.iloc[0:0]}, {"X": bars_1h}, when) == []


def test_the_live_strategy_is_what_it_was_and_the_candidates_are_listed():
    names: list[Any] = strategy_names()

    assert names[0] == "trend_cross"
    assert {"trend_cross_session", "trend_cross_session_late"} <= set(names)
    live = get_strategy("trend_cross")
    assert isinstance(live, TrendCrossStrategy) and live.config is RUNTIME_STRATEGY
    assert isinstance(get_strategy("trend_cross_session"), SessionTrendCrossStrategy)


def test_a_negative_late_start_is_refused():
    with pytest.raises(ValueError, match="negative"):
        SessionTrendCrossStrategy("x", skip_open_minutes=-1)
