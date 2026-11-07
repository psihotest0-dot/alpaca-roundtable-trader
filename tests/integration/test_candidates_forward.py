"""The judge of the strategy candidates (research/candidates.py): counts only until the minimum, one
look per sample, the four conditions of docs/strategy-research-plan.md section 12.1 -- on signals and
bars made up with a known truth. The recording side is tested with the observer (test_oracle_observer.py)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from trading_system.backtest.shadow import SignalToPlay, play_out
from trading_system.cli import main
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.research import candidates as module
from trading_system.research.candidates import (
    CANDIDATES,
    CONTROL,
    CYCLE_EVENT,
    JUDGEMENT_EVENT,
    MIN_RUN_DAYS,
    MIN_TRADES,
    format_candidates,
    journal_signal,
    judge_candidates,
)
from trading_system.storage.repositories import AuditRepository
from trading_system.storage.research import open_observe_store

UTC = timezone.utc
BASE = datetime(
    2026, 11, 9, 5, 0, tzinfo=UTC
)  # a Monday, midnight in New York: day d is one New York date
NOW = datetime(2027, 1, 15, tzinfo=UTC)  # long after the last signal (maturity is 14 days)
LATE, SESSION = CANDIDATES[1], CANDIDATES[0]
DAYS = 45


@pytest.fixture
def observe_store(tmp_path: Path, db_url: str) -> Any:
    return open_observe_store(f"sqlite:///{(tmp_path / 'observe.db').as_posix()}", db_url)


class World:
    """Signals journalled as the observer journals them, and the bars that follow them. Every signal
    has a symbol of its own unless one is given, so the one-trade-per-symbol rule stays out of the way."""

    def __init__(self, session_factory: Any, observe_store: Any, settings: Any) -> None:
        self.audit = AuditRepository(session_factory)
        self.store, self.settings = observe_store, settings
        self.count = 0

    def signal(
        self,
        strategies: tuple[str, ...],
        *,
        day: int,
        growth: float,
        symbol: str | None = None,
        slot: int = 0,
        days_ago: int | None = None,
        closed_at: datetime | None = None,
    ) -> str:
        """One signal under each strategy named, on one symbol. `growth`: the price moves by this much
        each bar after the entry; 0.0002 over the 120 bars given is about +2.4 %, short of the target, so
        the trade is still open at the end of the data and is valued at the last close."""
        self.count += 1
        symbol = symbol or f"S{self.count:04d}"
        closed = BASE + timedelta(days=day, minutes=15 * slot)
        if days_ago is not None:
            closed = NOW - timedelta(days=days_ago)
        if closed_at is not None:
            closed = closed_at
        for strategy in strategies:
            signal = StrategySignal(
                version=f"{strategy}-1",
                symbol=symbol,
                timeframe="15m",
                closed_bar_timestamp=closed,
                direction=SignalDirection.LONG,
            )
            journal_signal(self.audit, strategy, signal, closed + timedelta(minutes=1))
        self._bars(symbol, closed, growth)
        return symbol

    def _bars(self, symbol: str, closed: datetime, growth: float) -> None:
        entry = pd.Timestamp(closed) + pd.Timedelta(minutes=15)
        index = pd.date_range(
            entry - pd.Timedelta(hours=3), periods=12 + 120, freq="15min", tz="UTC"
        )
        closes = [100.0] * 12
        for _ in range(120):
            closes.append(closes[-1] * (1 + growth))
        series = pd.Series(closes, index=index)
        opens = series.shift(1).fillna(100.0)
        frame = pd.DataFrame(
            {
                "open": opens,
                "high": pd.concat([opens, series], axis=1).max(axis=1) * 1.0001,
                "low": pd.concat([opens, series], axis=1).min(axis=1) * 0.9999,
                "close": series,
                "volume": 1000.0,
            },
            index=index.rename("timestamp"),
        )
        self.store.replace(symbol, "15m", frame, adjustment="all", fetched_at=NOW)

    def run_days(self, days: int = DAYS, start: datetime = BASE) -> None:
        """A pass of the observer on each of `days` days (the marker, even without a signal)."""
        for day in range(days):
            self.audit.record(
                CYCLE_EVENT, {"bar_start": "x", "errors": 0}, when=start + timedelta(days=day)
            )

    def judge(self, when: datetime = NOW) -> Any:
        return judge_candidates(self.audit, self.store, self.settings, when)

    def looks(self) -> list[dict[str, Any]]:
        return [
            d
            for _, d in self.audit.events(
                JUDGEMENT_EVENT, datetime(2000, 1, 1, tzinfo=UTC), NOW + timedelta(days=1)
            )
        ]

    def fill(
        self,
        candidate: str = LATE,
        *,
        trades: int = MIN_TRADES,
        good: float = 0.0002,
        control_extra: int = MIN_TRADES,
        bad: float = -0.0002,
    ) -> None:
        """`trades` signals of the candidate (the control has the same ones) plus `control_extra`
        signals only the control has, going `bad`; spread over the 45 days. The candidate trades are
        slightly different from each other so that they have a spread."""
        for i in range(trades):
            jitter = 0.00005 * ((i % 5) - 2) / 2
            self.signal((candidate, CONTROL), day=i % DAYS, growth=good + jitter, slot=i // DAYS)
        for i in range(control_extra):
            jitter = 0.00005 * ((i % 5) - 2) / 2
            self.signal((CONTROL,), day=i % DAYS, growth=bad + jitter, slot=i // DAYS)


@pytest.fixture
def world(session_factory: Any, observe_store: Any, make_settings: Any) -> World:
    return World(session_factory, observe_store, make_settings())


def _result(j: Any, name: str = LATE) -> Any:
    return next(r for r in j.results if r.name == name)


# ------------------------------------------------------------------- what was registered


def test_the_numbers_in_the_code_are_the_ones_registered_before_the_first_signal() -> None:
    assert (module.MIN_TRADES, module.MIN_RUN_DAYS, module.MATURITY) == (
        150,
        40,
        timedelta(days=14),
    )
    assert (module.FIRST_LOOK_T, module.SECOND_LOOK_T) == (2.2, 2.5)
    assert (module.DIFFERENCE_T, module.DAILY_T, module.MAX_LOOKS) == (1.0, 1.5, 2)
    assert CANDIDATES == ("trend_cross_session", "trend_cross_session_late")
    assert CONTROL == "trend_cross"


# --------------------------------------------------------------------------- no peeking


def test_below_the_minimum_only_counts_are_shown_and_no_return_is_computed(world: World) -> None:
    world.fill(trades=MIN_TRADES - 1)
    world.run_days()

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        j = world.judge()

    r = _result(j)
    assert (r.signals, r.trades, r.verdict, r.detail) == (149, 149, "no verdict", None)
    assert "Nothing is computed below the minimum" in r.reason
    assert world.looks() == []


def test_too_few_days_run_is_counts_only_however_many_trades(world: World) -> None:
    world.fill()
    world.run_days(MIN_RUN_DAYS - 1)

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        r = _result(world.judge())

    assert r.verdict == "no verdict" and r.detail is None and r.trades == MIN_TRADES
    assert f"over {MIN_RUN_DAYS - 1} trading days" in r.reason


def test_each_candidate_is_judged_on_its_own_count(world: World) -> None:
    world.fill(LATE)  # enough
    for i in range(20):  # the other one: far too few
        world.signal((SESSION,), day=i, growth=0.0002)
    world.run_days()

    j = world.judge()

    assert _result(j, LATE).detail is not None
    assert _result(j, SESSION).verdict == "no verdict" and _result(j, SESSION).trades == 20


# ------------------------------------------------------------------------------ the verdict


def test_a_clear_improvement_helps_and_the_look_is_journalled_once(world: World) -> None:
    world.fill()
    world.run_days()

    j = world.judge()

    r = _result(j)
    assert r.verdict == "helps" and r.detail is not None
    assert r.detail.mean > 0.01 and r.detail.difference > 0.005 and r.detail.t >= 2.2
    assert (
        r.detail.open_trades == MIN_TRADES
    )  # no level was reached: valued at the last close, counted
    (look,) = world.looks()
    assert look["candidate"] == LATE and look["look"] == 1 and look["verdict"] == "helps"
    assert "LOOK" in format_candidates(j)
    # the other candidate had no signals: counts only, and no look of its own
    assert _result(j, SESSION).verdict == "no verdict"

    again = _result(world.judge())  # the same sample cannot be looked at twice
    assert again.looks_before == 1 and again.verdict == "no verdict" and again.detail is None
    assert len(world.looks()) == 1


def test_no_better_than_the_control_is_not_shown_to_help(world: World) -> None:
    world.fill(control_extra=0)  # the control has exactly the candidate's signals
    world.run_days()

    r = _result(world.judge())

    assert r.verdict == "not shown to help"
    assert "difference to the control" in r.reason and "mean net trade" not in r.reason


def test_a_candidate_that_loses_money_is_not_shown_to_help(world: World) -> None:
    world.fill(good=-0.0002, bad=-0.0004)  # better than the control, but below zero

    world.run_days()
    r = _result(world.judge())

    assert r.verdict == "not shown to help" and "mean net trade" in r.reason
    assert "difference to the control" not in r.reason  # that condition held


def test_a_gain_that_is_only_in_the_first_half_is_not_enough(world: World) -> None:
    for i in range(MIN_TRADES):
        early = i < MIN_TRADES // 2
        jitter = 0.00005 * ((i % 5) - 2) / 2
        world.signal(
            (LATE, CONTROL),
            day=(i % 22) if early else 23 + i % 22,
            growth=(0.0002 if early else -0.00005) + jitter,
        )
    for i in range(MIN_TRADES):
        jitter = 0.00005 * ((i % 5) - 2) / 2
        world.signal((CONTROL,), day=i % DAYS, growth=-0.0002 + jitter)
    world.run_days()

    r = _result(world.judge())

    assert r.verdict == "not shown to help" and "both halves" in r.reason
    assert r.detail is not None and r.detail.mean > 0 and r.detail.halves[1] < 0


def test_trades_of_one_day_are_not_independent_the_daily_means_decide(world: World) -> None:
    # 150 trades on two days: the trade-level numbers look fine, two daily means cannot carry t 1.5
    for i in range(MIN_TRADES):
        day = i % 2
        jitter = 0.00005 * ((i % 5) - 2) / 2
        world.signal(
            (LATE, CONTROL), day=day, growth=(0.0003 if day == 0 else 0.00003) + jitter, slot=i // 2
        )
    for i in range(MIN_TRADES):
        jitter = 0.00005 * ((i % 5) - 2) / 2
        world.signal((CONTROL,), day=i % 2, growth=-0.0002 + jitter, slot=i // 2)
    world.run_days()

    r = _result(world.judge())

    assert r.verdict == "not shown to help" and "daily means" in r.reason
    assert r.detail is not None and r.detail.days_with_trades == 2
    assert r.detail.mean > 0 and r.detail.t >= 2.2 and r.detail.difference_t >= 1.0


# -------------------------------------------------------------------------- what counts


def test_a_signal_that_arrived_twice_counts_once(world: World) -> None:
    symbol = world.signal((LATE,), day=0, growth=0.0002)
    signal = StrategySignal(
        version=f"{LATE}-1",
        symbol=symbol,
        timeframe="15m",
        closed_bar_timestamp=BASE,
        direction=SignalDirection.LONG,
    )
    journal_signal(world.audit, LATE, signal, BASE + timedelta(minutes=16))  # the same bar again

    r = _result(world.judge())

    assert r.signals == 1 and r.trades == 1


def test_a_signal_younger_than_two_weeks_is_left_out(world: World) -> None:
    world.signal((LATE,), day=0, growth=0.0002, days_ago=13)
    world.signal((LATE,), day=0, growth=0.0002, days_ago=15)

    r = _result(world.judge())

    assert r.signals == 1


def test_a_symbol_has_one_open_trade_at_a_time(world: World) -> None:
    # the first is still open at the end of the data, so the next signal on the symbol cannot enter
    world.signal((LATE,), day=0, growth=0.0002, symbol="AAA")
    world.signal((LATE,), day=0, growth=0.0002, symbol="AAA", slot=1)

    r = _result(world.judge())

    assert r.signals == 2 and r.trades == 1


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range(BASE, periods=len(rows), freq="15min", tz="UTC", name="timestamp")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(
        volume=1000.0
    )


def test_play_out_skips_a_signal_while_the_symbols_trade_is_open_and_not_after_it(
    observe_store: Any, make_settings: Any
) -> None:
    flat = (100.0, 100.1, 99.9, 100.0)
    crash = (100.0, 100.0, 90.0, 91.0)  # through the stop
    observe_store.replace(
        "AAA",
        "15m",
        _frame([flat, flat, flat, flat, crash, flat, flat, flat]),
        adjustment="all",
        fetched_at=NOW,
    )

    def signal(index: int) -> SignalToPlay:  # a signal on bar `index` enters on bar index + 1
        return SignalToPlay(
            f"s{index}", "AAA", pd.Timestamp(BASE) + pd.Timedelta(minutes=15 * index), "x"
        )

    settings = make_settings()
    played = play_out(
        [signal(0), signal(1), signal(5)], observe_store, settings, one_at_a_time=True
    )

    # signal 0 enters on bar 1 and is stopped out on bar 4; signal 1 enters on bar 2 (busy: skipped);
    # signal 5 enters on bar 6, after the exit
    assert [t.signal_id for t in played.trades] == ["s0", "s5"]
    assert played.skipped_busy == 1 and played.trades[0].exit_reason == "STOP_LOSS"

    everything = play_out([signal(0), signal(1), signal(5)], observe_store, settings)
    assert [t.signal_id for t in everything.trades] == ["s0", "s1", "s5"]  # the Oracle judge's way


# ------------------------------------------------------------------------------ the looks


def test_two_looks_at_most_and_the_second_wants_double_and_a_higher_bar(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MIN_TRADES", 10)
    monkeypatch.setattr(module, "MIN_RUN_DAYS", 5)
    bars: list[float] = []
    real_verdict = module._verdict

    def spy(detail: Any, bar: float) -> Any:
        bars.append(bar)
        return real_verdict(detail, bar)

    monkeypatch.setattr(module, "_verdict", spy)
    world.fill(trades=10, control_extra=10)
    world.run_days()

    first = _result(world.judge())
    assert first.detail is not None and first.looks_before == 0

    world.fill(trades=10, control_extra=10)  # 10 more of each: 20 in all
    second = _result(world.judge())
    assert second.detail is not None and second.looks_before == 1

    third = _result(world.judge())
    assert third.verdict == "no more looks" and third.detail is None
    assert bars == [2.2, 2.5]  # the first look's bar, then the higher one; no third look
    assert [look["look"] for look in world.looks()] == [1, 2]


def test_the_second_look_wants_twice_what_the_first_look_had(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    # at the real signal rate the first look has far more than the minimum; the second must not be
    # available at once on the same sample just because the minimum doubled is already there
    monkeypatch.setattr(module, "MIN_TRADES", 10)
    monkeypatch.setattr(module, "MIN_RUN_DAYS", 5)
    world.fill(trades=25, control_extra=25)
    world.run_days()
    assert _result(world.judge()).detail is not None  # the first look, with 25 trades

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        again = _result(world.judge())  # 25 is more than twice the minimum, but not twice 25

    assert again.verdict == "no verdict" and "need 50" in again.reason

    world.fill(trades=25, control_extra=25)
    assert _result(world.judge()).detail is not None  # 50 trades: the second look
    assert [look["look"] for look in world.looks()] == [1, 2]


def test_between_the_looks_there_are_only_counts(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MIN_TRADES", 10)
    monkeypatch.setattr(module, "MIN_RUN_DAYS", 5)
    world.fill(trades=10, control_extra=10)
    world.run_days()
    world.judge()  # the first look

    with patch.object(module, "_detail", side_effect=AssertionError("computed a return")):
        r = _result(world.judge())  # 10 trades, the second look wants 20

    assert r.verdict == "no verdict" and "need 20" in r.reason


# ------------------------------------------------------------------------------ the command


def test_the_command_prints_the_counts(
    world: World, make_settings: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # the command reads the real clock, so the data hangs on it (a week to a month back)
    real_now = datetime.now(UTC)
    for i in range(5):
        world.signal(
            (LATE, CONTROL), day=0, growth=0.0002, closed_at=real_now - timedelta(days=20 + i)
        )
    world.run_days(3, start=real_now - timedelta(days=25))
    settings = make_settings(
        observe_database_url=f"sqlite:///{(tmp_path / 'observe.db').as_posix()}"
    )

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["judge-candidates"]) == 0

    out = capsys.readouterr().out
    assert "trading days run: 3 (need 40)" in out
    assert f"{LATE}: 5 mature signals, 5 trades; looks taken 0 of 2" in out
    assert "verdict: no verdict" in out and "Nothing is computed below the minimum" in out
