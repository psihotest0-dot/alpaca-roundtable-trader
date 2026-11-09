"""`dip_in_uptrend` (research plan, section 14, variant 19): a sharp dip inside a rising hourly trend.

The rule is three conditions and a start time: the last closed hourly bar has EMA(12) above EMA(26),
the last closed 15-minute bar has RSI(14) below 30, and the signal bar does not start before 10:00
New York time; only bars of the regular session reach the indicators. The proofs here are about
those and about the contract every strategy keeps: nothing from after the evaluation time, nothing
from outside the session, and a backtest hint that never says "impossible" where it signals."""

from functools import cache

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.bars import random_walk_session_bars, with_timestamp_column
from tests.unit.test_session_cross import _with_stray_prints
from trading_system.domain.enums import SignalDirection
from trading_system.strategy import dip_cross
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.dip_cross import DipInUptrendStrategy
from trading_system.strategy.registry import get_strategy
from trading_system.strategy.session_cross import session_mask_1h, session_mask_15m
from trading_system.strategy.trend import calculate_indicators

SEED = 5  # 15 weekdays from 2026-11-02, drift 0.0004: eleven signals
DAYS = 15
DRIFT = 0.0004


def _new_york_minute(stamp: pd.Timestamp) -> int:
    local = stamp.tz_convert("America/New_York")
    return int(local.hour * 60 + local.minute)


@cache
def _walk(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    return random_walk_session_bars(seed, "2026-11-02", DAYS, drift=DRIFT)


def _found(
    bars_15m: pd.DataFrame, bars_1h: pd.DataFrame, strategy: DipInUptrendStrategy | None = None
) -> list[pd.Timestamp]:
    """The bars (by start time) the strategy signalled on, one evaluation at every bar's close."""
    strategy = strategy or DipInUptrendStrategy()
    found: list[pd.Timestamp] = []
    for stamp in bars_15m["timestamp"]:
        when = stamp + pd.Timedelta(minutes=15)
        for signal in strategy.generate_signals({"X": bars_15m}, {"X": bars_1h}, when):
            found.append(pd.Timestamp(signal.closed_bar_timestamp))
    return found


def _bars_from_closes(closes: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(15m, 1h) session bars of New York winter days from the given 15-minute closes (26 a day,
    first day 2026-11-02); the hourly bars are made from the 15-minute ones."""
    days = len(closes) // 26
    index = pd.DatetimeIndex(
        [
            day + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(minutes=15 * i)
            for day in pd.bdate_range("2026-11-02", periods=days, tz="UTC")
            for i in range(26)
        ]
    )
    opens = np.concatenate([[closes[0]], closes[:-1]])
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0005,
            "low": np.minimum(opens, closes) * 0.9995,
            "close": closes,
            "volume": 1000.0,
        },
        index=index,
    )
    hourly = frame.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return with_timestamp_column(frame), with_timestamp_column(hourly.dropna())


def _rise_then_dip(
    dip_slot: int, dip_bars: int = 6, day: int = 11, step: float = 0.992
) -> np.ndarray:
    """Eleven days of a steady rise (0.15 % a bar), then a fall by the factor `step` a bar for
    `dip_bars` bars from slot `dip_slot` of day `day`'s session (slot 0 is the 09:30 bar), then
    flat."""
    closes = 100.0 * np.cumprod(np.full(26 * (day + 1), 1.0015))
    start = 26 * day + dip_slot
    for i in range(dip_bars):
        closes[start + i] = closes[start + i - 1] * step
    closes[start + dip_bars :] = closes[start + dip_bars - 1]
    return closes


# ---- the strategy is in the list and is its own strategy --------------------------------------


def test_it_is_chosen_by_name_and_is_a_different_strategy_from_the_live_one() -> None:
    strategy = get_strategy("dip_in_uptrend")

    assert isinstance(strategy, DipInUptrendStrategy)
    assert strategy.name == "dip_in_uptrend"
    # The version goes into every signal id: a different rule must not share the live one's ids.
    assert strategy.version == RUNTIME_STRATEGY.strategy_version + "-dip"
    # ...and it reads the live strategy's own indicator settings, nothing tuned.
    assert strategy.config.fast_ema_period == RUNTIME_STRATEGY.fast_ema_period == 12
    assert strategy.config.slow_ema_period == RUNTIME_STRATEGY.slow_ema_period == 26
    assert strategy.config.rsi_period == RUNTIME_STRATEGY.rsi_period == 14
    assert dip_cross.RSI_BELOW == 30.0
    assert dip_cross.SKIP_OPEN_MINUTES == 30


# ---- what a signal is --------------------------------------------------------------------------


def test_every_signal_is_a_long_on_a_session_bar_from_ten_o_clock_with_rsi_below_30_in_an_uptrend() -> (
    None
):
    bars_15m, bars_1h = _walk()
    strategy = DipInUptrendStrategy()
    config = strategy.config
    ind_15 = calculate_indicators(
        bars_15m[session_mask_15m(pd.DatetimeIndex(bars_15m["timestamp"]))], config
    ).set_index("timestamp")
    ind_1h = calculate_indicators(
        bars_1h[session_mask_1h(pd.DatetimeIndex(bars_1h["timestamp"]))], config, is_htf=True
    )
    seen = 0

    for stamp in bars_15m["timestamp"]:
        when = stamp + pd.Timedelta(minutes=15)
        for signal in strategy.generate_signals({"X": bars_15m}, {"X": bars_1h}, when):
            seen += 1
            bar = pd.Timestamp(signal.closed_bar_timestamp)
            assert signal.direction == SignalDirection.LONG
            assert signal.timeframe == "15m"
            assert bar == stamp  # the bar that had just closed, not an older one
            assert 10 * 60 <= _new_york_minute(bar) < 16 * 60
            assert ind_15.loc[bar, "rsi"] < 30
            closed_hours = ind_1h[ind_1h["timestamp"] + pd.Timedelta(hours=1) <= when]
            assert closed_hours.iloc[-1]["ema_fast"] > closed_hours.iloc[-1]["ema_slow"]
            assert pd.Timestamp(signal.expires_at) == when + pd.Timedelta(minutes=15)
    assert seen == 11  # the fixture does contain dips; a quiet strategy would pass the loop above


def test_a_dip_inside_a_steady_rise_signals_and_a_steady_fall_does_not() -> None:
    dip_15m, dip_1h = _bars_from_closes(_rise_then_dip(dip_slot=10))
    fall_15m, fall_1h = _bars_from_closes(100.0 * np.cumprod(np.full(26 * 12, 0.998)))

    dip = _found(dip_15m, dip_1h)
    fall = _found(fall_15m, fall_1h)

    assert dip, "the dip after eleven days of rise is the case the strategy is for"
    assert all(_new_york_minute(t) >= 10 * 60 for t in dip)
    # The RSI is far below 30 on a steady fall (the hint says so), but the hourly trend is down.
    assert DipInUptrendStrategy().candidate_bars(fall_15m, fall_1h).sum() > 100  # type: ignore[union-attr]
    assert fall == []


def test_no_entry_when_the_signal_bar_starts_before_ten_even_if_the_dip_is_there() -> None:
    # Two falls of 4 % in the first two bars of the day (09:30 and 09:45): the RSI is far below 30
    # on the second, while the hourly trend is still up.
    bars_15m, bars_1h = _bars_from_closes(_rise_then_dip(dip_slot=0, dip_bars=2, step=0.96))

    found = _found(bars_15m, bars_1h)

    assert not any(_new_york_minute(t) < 10 * 60 for t in found)

    # The same data with the start rule switched off does signal on those two bars, so it was the
    # rule that held them back and not the data.
    original = dip_cross.SKIP_OPEN_MINUTES
    dip_cross.SKIP_OPEN_MINUTES = 0
    try:
        unrestricted = _found(bars_15m, bars_1h)
    finally:
        dip_cross.SKIP_OPEN_MINUTES = original
    assert any(_new_york_minute(t) < 10 * 60 for t in unrestricted)


def test_an_hourly_bar_that_closes_after_the_signal_bar_is_not_used() -> None:
    bars_15m, bars_1h = _bars_from_closes(_rise_then_dip(dip_slot=10))
    # a signal bar that does not end on the hour, so the hour it sits in is still open at its close
    bar = next(t for t in _found(bars_15m, bars_1h) if (t + pd.Timedelta(minutes=15)).minute != 0)
    closes_at = bar + pd.Timedelta(minutes=15)
    open_hour = bars_1h["timestamp"][
        (bars_1h["timestamp"] <= bar) & (bars_1h["timestamp"] + pd.Timedelta(hours=1) > closes_at)
    ].iloc[0]
    crashed = bars_1h.copy()
    crashed.loc[crashed["timestamp"] >= open_hour, "close"] *= 0.3  # the hour that is not over yet

    # evaluated two hours later, with the 15-minute bars known only up to the signal bar
    found = DipInUptrendStrategy().generate_signals(
        {"X": bars_15m[bars_15m["timestamp"] <= bar]},
        {"X": crashed},
        closes_at + pd.Timedelta(hours=2),
    )

    assert [pd.Timestamp(s.closed_bar_timestamp) for s in found] == [bar]


# ---- the contract every strategy keeps --------------------------------------------------------


def test_stray_prints_outside_the_session_do_not_move_the_signals() -> None:
    clean = _walk()
    polluted = _with_stray_prints(*clean)

    assert _found(*polluted) == _found(*clean)


def test_a_bar_that_comes_after_the_evaluation_time_changes_nothing() -> None:
    bars_15m, bars_1h = _walk()
    strategy = DipInUptrendStrategy()

    for stamp in bars_15m["timestamp"].iloc[60::5]:
        when = stamp + pd.Timedelta(minutes=15)
        everything = strategy.generate_signals({"X": bars_15m}, {"X": bars_1h}, when)
        known = strategy.generate_signals(
            {"X": bars_15m[bars_15m["timestamp"] < when]},
            {"X": bars_1h[bars_1h["timestamp"] < when]},
            when,
        )
        assert [s.signal_id for s in known] == [s.signal_id for s in everything]


def test_without_enough_bars_or_without_hourly_bars_it_says_nothing() -> None:
    bars_15m, bars_1h = _walk()
    when = bars_15m["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
    strategy = DipInUptrendStrategy()

    assert strategy.generate_signals({"X": bars_15m.iloc[:20]}, {"X": bars_1h}, when) == []
    assert strategy.generate_signals({"X": bars_15m}, {}, when) == []
    assert strategy.generate_signals({"X": bars_15m.iloc[:0]}, {"X": bars_1h}, when) == []


@pytest.mark.parametrize("seed", [2, 5, 8])
def test_the_backtest_hint_never_says_impossible_where_the_strategy_signals(seed: int) -> None:
    bars_15m, bars_1h = _walk(seed)
    strategy = DipInUptrendStrategy()
    hint = strategy.candidate_bars(bars_15m, bars_1h)
    assert hint is not None
    starts = list(bars_15m["timestamp"])

    for bar in _found(bars_15m, bars_1h):
        assert hint[starts.index(bar)], f"hint says no signal is possible on {bar}"
    assert hint.sum() >= len(_found(bars_15m, bars_1h))
