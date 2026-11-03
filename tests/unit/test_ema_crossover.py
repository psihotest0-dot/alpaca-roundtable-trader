"""The ported strategies of the original Money_bot (Phase 3.3a): what they decide, case by case.

The numbers are small on purpose (EMA 3/6, a 3-period hourly average) so each case can be read:
a gentle uptrend puts the close a little above the fast EMA (0.2 %), which is within 0.5 %."""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.bars import random_walk_session_bars, with_timestamp_column
from trading_system.domain.enums import SignalDirection
from trading_system.strategy.ema_crossover import (
    STRICT_EMA_CROSSOVER,
    EmaCrossoverConfig,
    EmaCrossoverStrategy,
)

UTC = timezone.utc
START = pd.Timestamp("2026-11-02 14:30", tz="UTC")


def _strategy(pullback: float = 0.005, volume_window: Optional[int] = None, htf: int = 3):
    return EmaCrossoverStrategy(
        EmaCrossoverConfig(
            strategy_version="t",
            short_window=3,
            long_window=6,
            pullback_pct=pullback,
            volume_window=volume_window,
            htf_ema_period=htf,
        )
    )


def _bars_15m(closes: list[float], volumes: Optional[list[float]] = None) -> pd.DataFrame:
    index = pd.date_range(START, periods=len(closes), freq="15min", tz="UTC")
    closes_a = np.array(closes)
    frame = pd.DataFrame(
        {
            "open": closes_a,
            "high": closes_a * 1.001,
            "low": closes_a * 0.999,
            "close": closes_a,
            "volume": volumes if volumes is not None else 1000.0,
        },
        index=index,
    )
    return with_timestamp_column(frame)


def _bars_1h(closes: list[float], first: str = "2026-11-02 10:00") -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(first, tz="UTC"), periods=len(closes), freq="1h")
    closes_a = np.array(closes)
    frame = pd.DataFrame(
        {"open": closes_a, "high": closes_a, "low": closes_a, "close": closes_a, "volume": 4000.0},
        index=index,
    )
    return with_timestamp_column(frame)


# A gentle uptrend: 20 bars, 14:30 .. 19:15, the last one closing at 19:30.
UPTREND = [100 + 0.1 * i for i in range(20)]
LAST_BAR = pd.Timestamp("2026-11-02 19:15", tz="UTC")
AFTER_LAST_BAR = datetime(2026, 11, 2, 19, 30, tzinfo=UTC)
HOURLY_BELOW = [90.0] * 12  # the hourly average is under the price: allowed
HOURLY_ABOVE = [130.0] * 12  # the hourly average is over the price: blocked


def _signals(strategy, bars, hourly, when=AFTER_LAST_BAR):
    return strategy.generate_signals({"AAPL": bars}, {"AAPL": hourly}, when)


def test_a_pullback_in_an_uptrend_above_the_hourly_average_is_a_long_signal() -> None:
    [signal] = _signals(_strategy(), _bars_15m(UPTREND), _bars_1h(HOURLY_BELOW))

    assert signal.symbol == "AAPL" and signal.direction is SignalDirection.LONG
    assert signal.closed_bar_timestamp == LAST_BAR
    assert signal.expires_at == LAST_BAR + pd.Timedelta(minutes=30)  # the bar after it
    assert signal.version == "t" and signal.timeframe == "15m"


def test_the_close_must_be_above_the_hourly_average() -> None:
    assert _signals(_strategy(), _bars_15m(UPTREND), _bars_1h(HOURLY_ABOVE)) == []


def test_a_downtrend_gives_no_signal_because_the_ported_strategy_is_long_only() -> None:
    falling = [120 - 0.1 * i for i in range(20)]  # fast EMA below slow, price at the EMA
    assert _signals(_strategy(), _bars_15m(falling), _bars_1h([100.0] * 12)) == []


def test_a_flat_price_is_not_an_uptrend() -> None:
    assert _signals(_strategy(), _bars_15m([100.0] * 20), _bars_1h(HOURLY_BELOW)) == []


def test_a_close_far_from_the_fast_ema_is_not_a_pullback_until_the_band_is_widened() -> None:
    jumped = UPTREND[:-1] + [UPTREND[-2] * 1.03]  # 3 % up on the last bar: 1.5 % over the EMA

    narrow = _signals(_strategy(pullback=0.005), _bars_15m(jumped), _bars_1h(HOURLY_BELOW))
    wide = _signals(_strategy(pullback=0.05), _bars_15m(jumped), _bars_1h(HOURLY_BELOW))

    assert narrow == []
    assert len(wide) == 1  # the 1 % -> 5 % wish: the same bar now qualifies


def _ema(values: list[float], span: int) -> float:
    """An independent calculation, to state the premise of a case without the strategy."""
    return float(pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1])


# A steeper trend (0.4 a bar), where the fast and the slow EMA are further apart.
STEEP = [100 + 0.4 * i for i in range(20)]


def test_the_band_is_measured_from_the_fast_ema_not_the_slow_one() -> None:
    fast, slow, close = _ema(STEEP, 3), _ema(STEEP, 6), STEEP[-1]
    assert abs(close - fast) / fast < 0.005 < abs(close - slow) / slow  # 0.37 % and 0.94 %

    assert len(_signals(_strategy(), _bars_15m(STEEP), _bars_1h(HOURLY_BELOW))) == 1


def test_a_dip_under_the_fast_ema_by_more_than_the_band_is_not_a_pullback() -> None:
    dipped = STEEP[:-1] + [STEEP[-2] - 1.6]
    fast, slow, close = _ema(dipped, 3), _ema(dipped, 6), dipped[-1]
    assert fast > slow  # the uptrend is intact
    assert (fast - close) / fast > 0.005  # 0.57 % under the fast EMA: outside the band
    assert (slow - close) / slow < 0.005  # but only 0.41 % under the slow one

    assert _signals(_strategy(), _bars_15m(dipped), _bars_1h(HOURLY_BELOW)) == []


def test_a_dip_just_under_the_fast_ema_is_still_a_pullback() -> None:
    slightly_under = UPTREND[:-1] + [UPTREND[-2] - 0.15]

    assert len(_signals(_strategy(), _bars_15m(slightly_under), _bars_1h(HOURLY_BELOW))) == 1


def test_the_backtest_hint_follows_the_hourly_filter() -> None:
    """Where the hourly average decides the case, the hint must say the same, and must read the
    same hourly bar (not the hour that is still forming)."""
    strategy = _strategy(htf=1)
    bars = _bars_15m(UPTREND)
    forming_hour_is_high = _bars_1h([90.0] * 9 + [1000.0])  # 10:00 .. 19:00, the last forming

    below, above, forming = (
        strategy.candidate_bars(bars, hourly)
        for hourly in (_bars_1h(HOURLY_BELOW), _bars_1h(HOURLY_ABOVE), forming_hour_is_high)
    )

    assert below is not None and above is not None and forming is not None
    assert below[-1] and not above[-1]
    assert forming[-1]  # the forming hour's 1000 is not what the 19:15 bar is compared with


def test_a_bar_still_forming_is_not_judged() -> None:
    forming = datetime(2026, 11, 2, 19, 25, tzinfo=UTC)  # the 19:15 bar closes at 19:30

    [signal] = _signals(_strategy(), _bars_15m(UPTREND), _bars_1h(HOURLY_BELOW), when=forming)

    assert signal.closed_bar_timestamp == LAST_BAR - pd.Timedelta(minutes=15)  # the 19:00 bar


def test_the_hour_that_is_forming_is_not_used_for_the_hourly_average() -> None:
    # htf_ema_period=1: the average is the close of the last hourly bar that may be used.
    # The 19:15 bar belongs to the 19:00 hour, which is still forming; its close (1000) must
    # not count, the 18:00 bar's (90) does.
    hourly = _bars_1h([90.0] * 9 + [1000.0], first="2026-11-02 10:00")  # 10:00 .. 19:00
    assert hourly.index[-1] == pd.Timestamp("2026-11-02 19:00", tz="UTC")

    allowed = _signals(_strategy(htf=1), _bars_15m(UPTREND), hourly)
    included = _signals(
        _strategy(htf=1),
        _bars_15m(UPTREND),
        _bars_1h([90.0] * 8 + [1000.0, 90.0]),  # 18:00 = 1000
    )

    assert len(allowed) == 1
    assert included == []


def test_a_signal_is_given_on_every_bar_where_the_conditions_hold_not_only_the_first() -> None:
    bars, hourly = _bars_15m(UPTREND), _bars_1h(HOURLY_BELOW)
    last = _signals(_strategy(), bars, hourly)
    one_earlier = _signals(
        _strategy(), bars, hourly, when=datetime(2026, 11, 2, 19, 15, tzinfo=UTC)
    )

    assert [s.closed_bar_timestamp for s in last] == [LAST_BAR]
    assert [s.closed_bar_timestamp for s in one_earlier] == [LAST_BAR - pd.Timedelta(minutes=15)]


def test_too_few_bars_give_no_signal() -> None:
    assert _signals(_strategy(), _bars_15m(UPTREND[:6]), _bars_1h(HOURLY_BELOW)) == []


def test_a_symbol_without_hourly_bars_is_skipped() -> None:
    strategy = _strategy()
    assert strategy.generate_signals({"AAPL": _bars_15m(UPTREND)}, {}, AFTER_LAST_BAR) == []


def test_the_strict_bot_wants_the_volume_to_be_at_least_80_percent_of_its_average() -> None:
    strategy = _strategy(volume_window=4)

    def run(last_volume: float):
        volumes = [1000.0] * 19 + [last_volume]  # the average of the last 4 includes this bar
        return _signals(strategy, _bars_15m(UPTREND, volumes), _bars_1h(HOURLY_BELOW))

    assert run(700.0) == []  # 700 < 0.8 x 925
    assert len(run(800.0)) == 1  # 800 >= 0.8 x 950


def test_the_stable_bot_has_no_volume_condition() -> None:
    volumes = [1000.0] * 19 + [1.0]
    assert len(_signals(_strategy(), _bars_15m(UPTREND, volumes), _bars_1h(HOURLY_BELOW))) == 1


def test_the_originals_parameters_are_the_recorded_ones() -> None:
    strict = STRICT_EMA_CROSSOVER
    assert (strict.short_window, strict.long_window, strict.pullback_pct) == (20, 50, 0.01)
    assert (strict.volume_window, strict.min_relative_volume, strict.htf_ema_period) == (
        20,
        0.8,
        200,
    )


@pytest.mark.parametrize("volume_window", [None, 10])
def test_the_backtest_hint_never_misses_a_signal(volume_window: Optional[int]) -> None:
    """candidate_bars may say "possible" where nothing fires, never "impossible" where
    something does. Checked on a random walk, one evaluation per bar."""
    strategy = EmaCrossoverStrategy(
        EmaCrossoverConfig(
            strategy_version="t",
            short_window=5,
            long_window=12,
            pullback_pct=0.02,
            volume_window=volume_window,
            htf_ema_period=20,
        )
    )
    bars_15m, bars_1h = random_walk_session_bars(21, "2026-10-05", 15, drift=0.0003)
    flags = strategy.candidate_bars(
        bars_15m.set_index("timestamp", drop=False).rename_axis(None),
        bars_1h.set_index("timestamp", drop=False).rename_axis(None),
    )
    assert flags is not None

    fired = 0
    for i in range(len(bars_15m)):
        when = bars_15m["timestamp"].iloc[i] + pd.Timedelta(minutes=15)
        signals = strategy.generate_signals({"X": bars_15m}, {"X": bars_1h}, when)
        if signals:
            fired += 1
            assert flags[i], f"the hint said impossible at bar {i} where the strategy signals"

    assert fired >= 5, "the data has to make the strategy fire for this to prove anything"
    assert flags.sum() < len(flags)  # and the hint does rule bars out
