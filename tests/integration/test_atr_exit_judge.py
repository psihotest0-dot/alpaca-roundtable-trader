"""The judge of the ATR exit candidate (research/atr_candidate.py) and the play-out with the exit rule
(backtest/shadow.py), on signals and bars made up with a known truth: every symbol has 60 bars with a
range of `rng` % around 100 (so its ATR is `rng` % of the price) and then a path chosen for the case."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from trading_system.backtest.shadow import SignalToPlay, play_out
from trading_system.cli import main
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.research import atr_candidate as module
from trading_system.research import candidates as candidates_module
from trading_system.research.atr_candidate import (
    ATR_CANDIDATE,
    ATR_EXIT,
    FIRST_LOOK_T,
    PAIRED_T,
    SECOND_LOOK_T,
    format_atr_exit,
    judge_atr_exit,
)
from trading_system.research.candidates import (
    CONTROL,
    CYCLE_EVENT,
    JUDGEMENT_EVENT,
    MIN_TRADES,
    journal_signal,
)
from trading_system.storage.repositories import AuditRepository
from trading_system.storage.research import open_observe_store

UTC = timezone.utc
BASE = datetime(2026, 11, 9, 5, 0, tzinfo=UTC)  # a Monday, midnight in New York
NOW = datetime(2027, 1, 15, tzinfo=UTC)  # long after the last signal (maturity is 14 days)
DAYS = 45
Path_ = list[tuple[float, float]]  # (open, close) of each bar after the signal, % above 100
RANGES = [0.8, 0.9, 1.0, 1.1, 1.2]  # the ATR, % of the price: stop 4.8-7.2 %, target 6-9 %


def _from_closes(closes: list[float]) -> Path_:
    out, previous = [], 0.0
    for close in closes:
        out.append((previous, close))
        previous = close
    return out


# The live exit (4 % / 5 %) takes +5 % and goes; the candidate (6 / 7.5 ATR, 6-9 %) rides on.
GOOD: Path_ = _from_closes([0.6 * k for k in range(1, 26)])
# +5.4 % (the live target) and then it falls away: the live exit has its +5 %, the candidate its stop.
REVERSAL: Path_ = _from_closes(
    [0.6 * k for k in range(1, 10)] + [5.4 - 1.2 * j for j in range(1, 16)]
)
# Quiet on the entry bar, then it opens 12 % up: both exits fill at that open, the same price.
GAP: Path_ = [(0.0, 0.2), (12.0, 12.0), (12.0, 12.1), (12.1, 12.2)]


@pytest.fixture
def observe_store(tmp_path: Path, db_url: str) -> Any:
    return open_observe_store(f"sqlite:///{(tmp_path / 'observe.db').as_posix()}", db_url)


class World:
    def __init__(self, session_factory: Any, observe_store: Any, settings: Any) -> None:
        self.audit = AuditRepository(session_factory)
        self.store, self.settings = observe_store, settings
        self.count = 0

    def signal(
        self,
        path: Path_,
        *,
        day: int,
        rng: float = 1.0,
        slot: int = 0,
        symbol: str | None = None,
        bars_before: int = 60,
        journal: bool = True,
    ) -> str:
        self.count += 1
        symbol = symbol or f"S{self.count:04d}"
        closed = BASE + timedelta(days=day, minutes=15 * slot)
        if journal:
            signal = StrategySignal(
                version=f"{CONTROL}-1",
                symbol=symbol,
                timeframe="15m",
                closed_bar_timestamp=closed,
                direction=SignalDirection.LONG,
            )
            journal_signal(self.audit, CONTROL, signal, closed + timedelta(minutes=1))
        self._bars(symbol, closed, rng, path, bars_before)
        return symbol

    def _bars(self, symbol: str, closed: datetime, rng: float, path: Path_, before: int) -> None:
        signal_bar = pd.Timestamp(closed)
        past = pd.date_range(end=signal_bar, periods=before, freq="15min", tz="UTC")
        rows = [(100.0, 100.0 + rng / 2, 100.0 - rng / 2, 100.0)] * before
        index = list(past)
        for i, (o, c) in enumerate(path):
            op, cl = 100 * (1 + o / 100), 100 * (1 + c / 100)
            rows.append((op, max(op, cl) * 1.0003, min(op, cl) * 0.9997, cl))
            index.append(signal_bar + pd.Timedelta(minutes=15 * (i + 1)))
        frame = pd.DataFrame(
            rows,
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex(index, name="timestamp"),
        ).assign(volume=1000.0)
        self.store.replace(symbol, "15m", frame, adjustment="all", fetched_at=NOW)

    def run_days(self, days: int = DAYS) -> None:
        for day in range(days):
            self.audit.record(
                CYCLE_EVENT, {"bar_start": "x", "errors": 0}, when=BASE + timedelta(days=day)
            )

    def judge(self) -> Any:
        return judge_atr_exit(self.audit, self.store, self.settings, NOW)

    def looks(self) -> list[dict[str, Any]]:
        return [
            d
            for _, d in self.audit.events(
                JUDGEMENT_EVENT, datetime(2000, 1, 1, tzinfo=UTC), NOW + timedelta(days=1)
            )
            if d.get("candidate") == ATR_CANDIDATE
        ]

    def fill(self, path: Path_, count: int, *, first_day: int = 0, days: int = DAYS) -> None:
        for i in range(count):
            self.signal(path, day=first_day + i % days, rng=RANGES[i % 5], slot=i // days)


@pytest.fixture
def world(session_factory: Any, observe_store: Any, make_settings: Any) -> World:
    return World(session_factory, observe_store, make_settings())


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> None:
    """The logic of the looks on a dozen trades; one test below uses the registered numbers."""
    for name in ("MIN_TRADES",):
        monkeypatch.setattr(module, name, 12)
    monkeypatch.setattr(module, "MIN_RUN_DAYS", 5)
    monkeypatch.setattr(candidates_module, "MIN_TRADES", 12)


# ------------------------------------------------------------------- what was registered


def test_the_numbers_in_the_code_are_the_ones_registered_before_any_code() -> None:
    assert (ATR_EXIT.stop_atr, ATR_EXIT.target_atr) == (Decimal("6"), Decimal("7.5"))
    assert (FIRST_LOOK_T, SECOND_LOOK_T, PAIRED_T) == (2.4, 2.7, 1.0)
    assert ATR_CANDIDATE == "trend_cross_atr_exit"


# --------------------------------------------------------------------------- the play-out


def test_the_play_out_with_the_exit_rule_uses_the_atr_of_the_window(
    world: World, make_settings: Any
) -> None:
    # +6 %: past the live 5 % target, short of the 7.5 ATR (7.5 % at a range of 1 %)
    world.signal(_from_closes([6.0, 6.1, 6.2]), day=0, rng=1.0, symbol="AAA")
    sig = SignalToPlay("s", "AAA", pd.Timestamp(BASE), "x")

    live = play_out([sig], world.store, world.settings)
    candidate = play_out([sig], world.store, world.settings, exit_rule=ATR_EXIT)

    assert live.trades[0].exit_reason == "TAKE_PROFIT"
    assert candidate.trades[0].exit_reason == "OPEN" and candidate.no_atr == 0


def test_the_atr_of_a_signal_does_not_see_the_bars_after_it(world: World) -> None:
    # Quiet entry bar, then a drop of 8 %. The stop sits 6 ATR = 6 % below (the ATR of the 60 bars
    # BEFORE the signal is 1 %): the drop hits it. If the bars after the signal leaked into the ATR,
    # the 8 % bar would widen the stop to about 9 % and the drop would not reach it.
    world.signal(_from_closes([0.0, -8.0, -8.1]), day=0, rng=1.0, symbol="AAA")
    sig = SignalToPlay("s", "AAA", pd.Timestamp(BASE), "x")

    played = play_out([sig], world.store, world.settings, exit_rule=ATR_EXIT)

    (trade,) = played.trades
    assert trade.exit_reason == "STOP_LOSS" and -0.07 < trade.pnl_pct < -0.055


def test_a_signal_without_enough_bars_for_an_atr_is_counted_not_played(world: World) -> None:
    world.signal(GOOD, day=0, symbol="AAA", bars_before=8)
    sig = SignalToPlay("s", "AAA", pd.Timestamp(BASE), "x")

    played = play_out([sig], world.store, world.settings, exit_rule=ATR_EXIT)

    assert played.trades == [] and played.no_atr == 1
    assert len(play_out([sig], world.store, world.settings).trades) == 1  # the live exit needs none


# --------------------------------------------------------------------------- no peeking


def test_below_the_minimum_only_counts_are_shown_and_no_return_is_computed(world: World) -> None:
    world.fill(GOOD, MIN_TRADES - 1)
    world.run_days()

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        j = world.judge()

    assert (j.pairs, j.verdict, j.detail) == (MIN_TRADES - 1, "no verdict", None)
    assert "Nothing is computed below the minimum" in j.reason and world.looks() == []


def test_too_few_days_run_is_counts_only_however_many_trades(world: World) -> None:
    world.fill(GOOD, MIN_TRADES)
    world.run_days(10)

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        j = world.judge()

    assert j.verdict == "no verdict" and j.detail is None and "over 10 trading days" in j.reason


# ------------------------------------------------------------------------------ the verdict


def test_an_exit_that_rides_the_gain_helps_and_the_look_is_journalled_once(world: World) -> None:
    world.fill(GOOD, MIN_TRADES)  # the registered numbers: 150 trades, 45 days
    world.run_days()

    j = world.judge()

    assert j.verdict == "helps" and j.detail is not None
    d = j.detail
    assert d.difference > 0.01 and d.mean > d.fixed_mean > 0 and d.t >= FIRST_LOOK_T
    assert d.spread[0] < d.spread[1]  # printed: the candidate's trades are more spread out
    (look,) = world.looks()
    assert look["look"] == 1 and look["verdict"] == "helps" and look["trades"] == MIN_TRADES
    assert "LOOK" in format_atr_exit(j) and "not criteria" in format_atr_exit(j)

    again = world.judge()  # the same sample cannot be looked at twice
    assert again.looks_before == 1 and again.verdict == "no verdict" and again.detail is None
    assert "need 300" in again.reason and len(world.looks()) == 1


def test_an_exit_that_gives_the_gain_back_is_not_shown_to_help(world: World, small: None) -> None:
    world.fill(REVERSAL, 12, days=12)
    world.run_days()

    j = world.judge()

    assert j.verdict == "not shown to help" and j.detail is not None
    assert j.detail.difference < 0
    assert "difference to the live exit" in j.reason and "mean net trade" in j.reason


def test_a_gain_that_is_only_in_the_first_half_is_not_enough(world: World, small: None) -> None:
    world.fill(GOOD, 6, days=6)  # earlier days: the candidate rides on
    world.fill(GAP, 6, first_day=20, days=6)  # later days: both exits fill at the same open
    world.run_days()

    j = world.judge()

    assert j.verdict == "not shown to help" and "both halves" in j.reason
    assert j.detail is not None and j.detail.difference > 0 and j.detail.half_differences[1] == 0


def test_trades_of_one_day_are_not_independent_the_daily_means_decide(
    world: World, small: None
) -> None:
    # six trades on one day that the candidate rides to about +7 %, six on another where the live exit
    # is stopped out at -4 % and the candidate sits through the dip to about +0.4 %: at the level of
    # trades everything holds, but two daily means cannot carry a t of 1.5
    dip_and_recover = _from_closes([-1.0, -2.0, -3.0, -4.5, -3.0, -1.5, 0.0, 0.5])
    for i in range(6):
        world.signal(GOOD, day=0, rng=RANGES[i % 5], slot=i)
        world.signal(dip_and_recover, day=1, rng=1.0, slot=i)
    world.run_days()

    j = world.judge()

    assert j.verdict == "not shown to help" and "daily means" in j.reason
    assert "mean net trade" not in j.reason and "both halves" not in j.reason
    assert "difference to the live exit" not in j.reason
    assert j.detail is not None and j.detail.days_with_trades == 2


def test_the_trades_are_those_of_the_live_exit_one_open_trade_per_symbol(
    world: World, small: None
) -> None:
    # two signals in a row on a symbol whose trade stays open under the live exit: one trade
    world.signal(_from_closes([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]), day=0, symbol="AAA")
    world.signal(_from_closes([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]), day=0, slot=1, symbol="AAA")

    j = world.judge()

    assert j.signals == 2 and j.pairs == 1


def test_signals_younger_than_two_weeks_are_left_out(world: World, small: None) -> None:
    world.signal(GOOD, day=0, symbol="OLD")
    young = NOW - timedelta(days=5)
    world.signal(GOOD, day=(young - BASE).days, symbol="NEW")

    assert world.judge().signals == 1


def test_a_signal_with_no_atr_is_counted(world: World, small: None) -> None:
    world.signal(GOOD, day=0, symbol="AAA", bars_before=8)
    world.signal(GOOD, day=1, symbol="BBB")

    j = world.judge()

    assert (j.signals, j.pairs, j.no_atr) == (2, 1, 1)


# --------------------------------------------------------------------------------- the looks


def test_the_second_look_wants_twice_the_first_and_a_higher_bar(
    world: World, small: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    bars: list[float] = []
    real = module._verdict

    def spy(detail: Any, bar: float) -> Any:
        bars.append(bar)
        return real(detail, bar)

    monkeypatch.setattr(module, "_verdict", spy)
    world.fill(GOOD, 15, days=15)  # more than the minimum of 12
    world.run_days()
    assert world.judge().detail is not None  # the first look, with 15 trades

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        again = world.judge()  # 15 is over twice the minimum's half, but not twice 15
    assert again.verdict == "no verdict" and "need 30" in again.reason

    world.fill(GOOD, 15, first_day=15, days=15)
    assert world.judge().detail is not None  # 30 trades: the second look
    third = world.judge()
    assert third.verdict == "no more looks" and third.detail is None
    assert bars == [2.4, 2.7] and [look["look"] for look in world.looks()] == [1, 2]


# --------------------------------------------------------------------------------- the command


def test_the_command_prints_both_judges(
    world: World, make_settings: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real_now = datetime.now(UTC)
    for i in range(3):
        closed = real_now - timedelta(days=20 + i)
        signal = StrategySignal(
            version=f"{CONTROL}-1",
            symbol=f"S{i:04d}",
            timeframe="15m",
            closed_bar_timestamp=closed,
            direction=SignalDirection.LONG,
        )
        journal_signal(world.audit, CONTROL, signal, closed + timedelta(minutes=1))
        world._bars(f"S{i:04d}", closed, 1.0, GOOD, 60)
    settings = make_settings(
        observe_database_url=f"sqlite:///{(tmp_path / 'observe.db').as_posix()}"
    )

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["judge-candidates"]) == 0

    out = capsys.readouterr().out
    assert "The candidates run forward in time" in out
    assert "The ATR exit candidate (trend_cross_atr_exit: stop 6 ATR, target 7.5 ATR)" in out
    assert "3 mature control signals, 3 paired trades" in out
    # three candidates of 12.1 and section 14, and the ATR block
    assert out.count("Nothing is computed below the minimum") == 4
