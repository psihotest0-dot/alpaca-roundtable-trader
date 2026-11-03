"""The Runner's schedule, against a fake clock, a fake broker and a recording cycle."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from trading_system.runtime.cycle import EntriesReport
from trading_system.runtime.runner import (
    BARS_GIVE_UP,
    BARS_RETRY,
    CLOCK_RETRY,
    CLOSED_HEARTBEAT,
    ENTRY_DELAY,
    POLL,
    Runner,
)

UTC = timezone.utc
BOUNDARY = datetime(2026, 11, 2, 15, 30, tzinfo=UTC)  # 10:30 in New York


class FakeWorld:
    """A clock that only moves when the runner waits, a broker clock, and a cycle that records
    what it was asked to do."""

    def __init__(self, start: datetime, *, market_open: bool = True) -> None:
        self.now = start
        self.market_open = market_open
        self.clock_error: Exception | None = None
        self.waits: list[float] = []
        self.entries: list[tuple[datetime, datetime]] = []
        self.housekeeping: list[datetime] = []
        self.revisits: list[datetime] = []
        self.revisit_error: Exception | None = None
        self.entry_reports: list[EntriesReport | Exception] = []

    # broker
    def get_clock(self) -> SimpleNamespace:
        if self.clock_error:
            raise self.clock_error
        return SimpleNamespace(
            is_open=self.market_open,
            timestamp=self.now,
            next_open=self.now + timedelta(hours=15),
        )

    # cycle
    def run_housekeeping(self, now: datetime) -> None:
        self.housekeeping.append(now)

    def run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        self.entries.append((now, expected_bar_start))
        result = self.entry_reports.pop(0) if self.entry_reports else EntriesReport()
        if isinstance(result, Exception):
            raise result
        return result

    def revisit_held(self, now: datetime) -> int:
        self.revisits.append(now)
        if self.revisit_error:
            raise self.revisit_error
        return 0

    # runner hooks
    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += timedelta(seconds=seconds)

    def runner(self, heartbeat: Any = None) -> Runner:
        return Runner(
            self,
            self,
            now=lambda: self.now,
            wait=self.wait,
            heartbeat=heartbeat,  # type: ignore[arg-type]
        )


class RecordingHeartbeat:
    """Stands in for the file: keeps every beat."""

    def __init__(self) -> None:
        self.beats: list[tuple[dict[str, Any], datetime]] = []
        self.error: Exception | None = None

    def beat(self, fields: dict[str, Any], *, now: datetime) -> None:
        if self.error:
            raise self.error
        self.beats.append((fields, now))


def test_the_first_tick_after_the_bar_closed_looks_for_entries_once() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY + timedelta(seconds=5))
    runner = world.runner()

    runner.tick()
    runner.tick()  # same bar, a minute later

    assert len(world.entries) == 1
    assert world.entries[0][1] == BOUNDARY - timedelta(minutes=15)  # the bar that just closed
    assert len(world.housekeeping) == 2  # but housekeeping runs on every tick


def test_it_waits_for_the_delay_after_the_boundary_before_looking() -> None:
    world = FakeWorld(BOUNDARY + timedelta(seconds=5))
    runner = world.runner()

    runner.tick()

    assert world.entries == []
    assert world.waits == [pytest.approx(ENTRY_DELAY.total_seconds() - 5)]
    runner.tick()
    assert len(world.entries) == 1


def test_every_new_bar_gets_its_own_look() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    runner = world.runner()

    for _ in range(40):  # about 40 minutes of ticks
        runner.tick()

    looked_at = [expected for _, expected in world.entries]
    assert looked_at == [
        BOUNDARY - timedelta(minutes=15),
        BOUNDARY,
        BOUNDARY + timedelta(minutes=15),
    ]


def test_it_never_waits_longer_than_a_poll_while_the_market_is_open() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    runner = world.runner()

    for _ in range(10):
        runner.tick()

    assert max(world.waits) <= POLL.total_seconds()


def test_a_missing_bar_is_retried_soon_and_then_given_up() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    late = EntriesReport(skipped="bar not available yet", retry_later=True)
    world.entry_reports = [late] * 100
    runner = world.runner()

    while world.now < BOUNDARY + timedelta(minutes=16):
        runner.tick()

    this_bar, next_bar = BOUNDARY - timedelta(minutes=15), BOUNDARY
    first = [t for t, expected in world.entries if expected == this_bar]
    second = [t for t, expected in world.entries if expected == next_bar]
    assert first[1] - first[0] == BARS_RETRY  # retried quickly, not after a whole poll
    assert first[-1] == BOUNDARY + ENTRY_DELAY + BARS_GIVE_UP  # the last try, then it gives up
    assert second[0] == BOUNDARY + timedelta(minutes=15) + ENTRY_DELAY  # and moves on


def test_a_failing_entry_look_up_does_not_stop_the_runner_and_is_retried() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.entry_reports = [RuntimeError("boom"), EntriesReport()]
    runner = world.runner()

    runner.tick()
    runner.tick()
    runner.tick()

    assert len(world.entries) == 2  # failed once, succeeded on the retry, then done for this bar


def test_a_halted_pass_counts_as_done_for_the_bar() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.entry_reports = [EntriesReport(skipped="halted")]
    runner = world.runner()

    runner.tick()
    runner.tick()

    assert len(world.entries) == 1


def test_a_closed_market_gets_a_heartbeat_and_housekeeping_but_no_entries() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY, market_open=False)
    runner = world.runner()

    runner.tick()

    assert world.entries == []
    assert len(world.housekeeping) == 1  # stale orders are dealt with before the open
    assert world.waits == [CLOSED_HEARTBEAT.total_seconds()]  # 15 hours to the open, 5 minutes wait


def test_a_closed_market_wakes_for_the_open_when_it_is_closer_than_a_heartbeat() -> None:
    world = FakeWorld(BOUNDARY, market_open=False)
    world.get_clock = lambda: SimpleNamespace(  # type: ignore[method-assign]
        is_open=False, timestamp=world.now, next_open=world.now + timedelta(seconds=40)
    )

    world.runner().tick()

    assert world.waits == [40.0]


def test_an_unreadable_clock_is_retried_and_does_not_crash() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.clock_error = ConnectionError("down")
    runner = world.runner()

    runner.tick()

    assert world.waits == [CLOCK_RETRY.total_seconds()]
    assert world.entries == [] and world.housekeeping == []


def test_stop_ends_the_loop_and_the_wait() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    runner = Runner(world, world, now=lambda: world.now)  # type: ignore[arg-type]
    seen: list[Any] = []
    original = world.run_housekeeping

    def housekeeping_then_stop(now: datetime) -> None:
        original(now)
        seen.append(now)
        runner.stop()  # as a signal handler would

    world.run_housekeeping = housekeeping_then_stop  # type: ignore[method-assign]

    runner.run()  # the real Event.wait returns at once because stop() was called

    assert runner.stopped and len(seen) == 1


def test_the_default_wait_is_cut_short_by_stop_within_a_slice() -> None:
    """A long wait would hold a signal handler back for its whole length on Windows."""
    import threading
    import time

    runner = Runner(object(), object())  # type: ignore[arg-type]
    threading.Timer(0.3, runner.stop).start()

    started = time.monotonic()
    runner._sleep_until_stopped(30)

    assert time.monotonic() - started < 3  # not 30


def test_the_default_wait_waits_when_nobody_stops_it() -> None:
    import time

    runner = Runner(object(), object())  # type: ignore[arg-type]

    started = time.monotonic()
    runner._sleep_until_stopped(0.25)

    assert 0.2 <= time.monotonic() - started < 2


def test_held_signals_are_asked_again_on_the_ticks_that_do_not_look_for_entries() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    runner = world.runner()

    runner.tick()  # the entry look-up for this bar: it does the asking itself
    assert world.revisits == []
    runner.tick()
    runner.tick()

    assert len(world.revisits) == 2


def test_a_failing_revisit_does_not_stop_the_runner() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.revisit_error = RuntimeError("oracle exploded")
    runner = world.runner()

    runner.tick()
    runner.tick()
    runner.tick()  # still ticking

    assert len(world.revisits) == 2


def test_nothing_is_revisited_while_the_market_is_closed() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY, market_open=False)

    world.runner().tick()

    assert world.revisits == []


# --- the pulse (Phase 6.1b) ------------------------------------------------------------------


def test_a_tick_with_the_market_open_beats_with_the_wait_it_is_about_to_make() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY + timedelta(seconds=2))
    pulse = RecordingHeartbeat()
    runner = world.runner(pulse)

    runner.tick()

    [(beat, at)] = pulse.beats
    assert beat["market_open"] is True and beat["clock_error"] is False
    assert beat["next_within_seconds"] == world.waits[-1]  # what it says it will do, it does
    assert beat["market_open_since"] == (BOUNDARY + ENTRY_DELAY + timedelta(seconds=2)).isoformat()
    assert beat["last_entries_at"] == beat["market_open_since"] and beat["last_entries_errors"] == 0
    assert beat["started_at"] == (BOUNDARY + ENTRY_DELAY + timedelta(seconds=2)).isoformat()
    assert at == BOUNDARY + ENTRY_DELAY + timedelta(seconds=2)


def test_a_pass_with_broken_signals_is_in_the_beat_until_a_clean_one_replaces_it() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.entry_reports = [EntriesReport(signals=2, errors=2), EntriesReport(signals=1)]
    pulse = RecordingHeartbeat()
    runner = world.runner(pulse)

    runner.tick()
    world.now = BOUNDARY + timedelta(minutes=15) + ENTRY_DELAY
    runner.tick()

    assert [b["last_entries_errors"] for b, _ in pulse.beats] == [2, 0]


def test_a_pass_that_will_be_retried_does_not_count_as_a_look_up_that_completed() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.entry_reports = [EntriesReport(skipped="bar not available yet", retry_later=True)]
    pulse = RecordingHeartbeat()

    world.runner(pulse).tick()

    [(beat, _)] = pulse.beats
    assert beat["last_entries_at"] is None


def test_a_pass_that_completed_but_skipped_says_why() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.entry_reports = [EntriesReport(skipped="trading halted")]
    pulse = RecordingHeartbeat()

    world.runner(pulse).tick()

    [(beat, _)] = pulse.beats
    assert beat["last_entries_skipped"] == "trading halted" and beat["last_entries_at"]


def test_a_look_up_that_crashes_is_in_the_beat_as_an_error() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    world.entry_reports = [RuntimeError("boom")]
    pulse = RecordingHeartbeat()

    world.runner(pulse).tick()

    [(beat, _)] = pulse.beats
    assert beat["last_entries_errors"] == 1 and beat["last_entries_skipped"] == "crashed"


def test_a_closed_market_beats_with_its_longer_wait_and_forgets_that_it_was_open() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    pulse = RecordingHeartbeat()
    runner = world.runner(pulse)
    runner.tick()  # open
    world.market_open = False

    runner.tick()  # closed

    open_beat, closed_beat = (b for b, _ in pulse.beats)
    assert open_beat["market_open"] is True and open_beat["market_open_since"]
    assert closed_beat["market_open"] is False and closed_beat["market_open_since"] is None
    assert closed_beat["next_within_seconds"] == CLOSED_HEARTBEAT.total_seconds()


def test_the_market_opening_again_starts_a_new_stretch() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY, market_open=False)
    pulse = RecordingHeartbeat()
    runner = world.runner(pulse)
    runner.tick()
    world.market_open = True
    world.now += timedelta(hours=16)
    reopened_at = world.now

    runner.tick()

    assert pulse.beats[-1][0]["market_open_since"] == reopened_at.isoformat()


def test_an_unreadable_clock_beats_with_the_error_and_the_retry_wait() -> None:
    world = FakeWorld(BOUNDARY)
    world.clock_error = RuntimeError("no clock")
    pulse = RecordingHeartbeat()

    world.runner(pulse).tick()

    [(beat, _)] = pulse.beats
    assert beat["clock_error"] is True and beat["market_open"] is None
    assert beat["next_within_seconds"] == CLOCK_RETRY.total_seconds()


def test_a_pulse_that_cannot_be_written_does_not_stop_the_loop() -> None:
    world = FakeWorld(BOUNDARY + ENTRY_DELAY)
    pulse = RecordingHeartbeat()
    pulse.error = OSError("read-only file system")

    world.runner(pulse).tick()

    assert len(world.entries) == 1 and len(world.waits) == 1  # it looked, and it waited
