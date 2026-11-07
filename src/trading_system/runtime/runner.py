"""The schedule of the trading loop (task 1.7).

`Runner` decides when `TradingCycle` acts. Every tick it does the housekeeping (reconcile,
expire, check protection, record fills), and once per closed 15-minute bar, a little after the
bar closes, it looks for entries. It never sleeps for longer than a heartbeat: when the market
is closed it wakes up every few minutes to say so, and `stop()` (a SIGTERM, Ctrl+C) ends the
wait within a second instead of after the current sleep.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from trading_system.observability import metrics
from trading_system.runtime.cycle import TradingCycle
from trading_system.runtime.heartbeat import Heartbeat
from trading_system.runtime.schedule import BAR, bar_boundary, expected_bar_start

logger = logging.getLogger(__name__)

# Alpaca publishes a bar a few seconds after it closes; the 15-minute bars of 22 symbols
# were measured to take about 40 s to download, so the entry look-up starts a little late.
ENTRY_DELAY = timedelta(seconds=30)
# While the newest bar is missing, ask again this often ...
BARS_RETRY = timedelta(seconds=15)
# ... and give up on the bar after this long, rather than act on the one before it.
BARS_GIVE_UP = timedelta(minutes=3)
# The longest wait between two housekeeping rounds with the market open ...
POLL = timedelta(seconds=60)
# ... and with it closed (a heartbeat in the log, no more).
CLOSED_HEARTBEAT = timedelta(minutes=5)
# After the broker's clock could not be read.
CLOCK_RETRY = timedelta(seconds=30)
# The longest the runner stays inside one wait (see Runner._sleep_until_stopped).
WAIT_SLICE = 1.0


class Runner:
    def __init__(
        self,
        cycle: TradingCycle,
        broker: Any,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        wait: Optional[Callable[[float], Any]] = None,
        heartbeat: Optional[Heartbeat] = None,
    ) -> None:
        self.cycle = cycle
        self.broker = broker
        self._now = now
        self.heartbeat = heartbeat
        self._started_at = now()
        self._open_since: Optional[datetime] = None  # when this stretch of open market began
        self._last_entries_at: Optional[datetime] = None  # the last look-up that completed
        self._last_entries_errors = 0
        self._last_entries_skipped: Optional[str] = None
        self._stop = threading.Event()
        # Tests pass a fake clock instead.
        self._wait = wait if wait is not None else self._sleep_until_stopped
        self._last_bar: Optional[datetime] = None  # the boundary of the bar already handled

    def stop(self) -> None:
        self._stop.set()

    def _sleep_until_stopped(self, seconds: float) -> None:
        """Waits `seconds`, or until stop() is called, in slices of one second.

        One long Event.wait would not do: on Windows a signal handler only runs once the main
        thread leaves the wait (measured: a SIGTERM sent during Event.wait(30) was handled
        after 30 s), so Ctrl+C would take up to the heartbeat to be noticed."""
        deadline = time.monotonic() + seconds
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._stop.wait(min(WAIT_SLICE, remaining))

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run(self) -> None:
        logger.info("Runner started.")
        while not self._stop.is_set():
            self.tick()
        logger.info("Runner stopped.")

    def tick(self) -> None:
        """One round: housekeeping, the entry look-up if it is due, then wait until the next
        thing that needs doing."""
        now = self._now()
        try:
            clock = self.broker.get_clock()
        except Exception as e:
            logger.error(f"Could not read the market clock: {e}")
            self._pause(CLOCK_RETRY.total_seconds(), market_open=None, clock_error=True)
            return

        # Housekeeping does not depend on the market being open: a GTC entry left over from
        # yesterday has to be cancelled before the open, not after it.
        self.cycle.run_housekeeping(now)

        if not clock.is_open:
            until_open = max((clock.next_open - clock.timestamp).total_seconds(), 1.0)
            logger.info(f"Market is closed; next open in {until_open / 3600:.1f} h.")
            self._open_since = None
            self._pause(min(until_open, CLOSED_HEARTBEAT.total_seconds()), market_open=False)
            return

        if self._open_since is None:
            self._open_since = now
        if not self._look_for_entries(now):
            self._revisit_held(now)
        self._pause(self._seconds_until_next_event(self._now()), market_open=True)

    def _pause(
        self, seconds: float, *, market_open: Optional[bool], clock_error: bool = False
    ) -> None:
        """Says how the loop is (and when it will next say so), then waits."""
        self._beat(seconds, market_open, clock_error)
        self._wait(seconds)

    def _beat(self, next_within: float, market_open: Optional[bool], clock_error: bool) -> None:
        # Never raises. The heartbeat file below is the healthcheck's; this is Grafana's.
        metrics.beat(market_open)
        if self.heartbeat is None:
            return
        try:
            self.heartbeat.beat(
                {
                    "started_at": self._started_at.isoformat(),
                    "market_open": market_open,
                    "market_open_since": self._open_since.isoformat() if self._open_since else None,
                    "clock_error": clock_error,
                    "next_within_seconds": next_within,
                    "last_entries_at": (
                        self._last_entries_at.isoformat() if self._last_entries_at else None
                    ),
                    "last_entries_errors": self._last_entries_errors,
                    "last_entries_skipped": self._last_entries_skipped,
                },
                now=self._now(),
            )
        except Exception as e:  # a pulse that cannot be written is for the healthcheck to notice
            logger.warning(f"Could not write the heartbeat: {e}")

    def _revisit_held(self, now: datetime) -> None:
        try:
            self.cycle.revisit_held(now)
        except Exception:
            logger.exception("Re-examining held signals failed.")

    def _look_for_entries(self, now: datetime) -> bool:
        """True if the entry look-up ran on this tick (whatever its outcome)."""
        boundary = bar_boundary(now)
        if self._last_bar == boundary or now < boundary + ENTRY_DELAY:
            return False
        try:
            report = self.cycle.run_entries(now, expected_bar_start(now))
            bar_ready = not report.retry_later
            metrics.entries_pass(
                "retry_later" if report.retry_later else ("skipped" if report.skipped else "ok"),
                signals=report.signals,
                errors=report.errors,
            )
            if bar_ready:
                self._last_entries_at = now
                self._last_entries_errors = report.errors
                self._last_entries_skipped = report.skipped
        except Exception:
            logger.exception("The entry look-up failed.")
            bar_ready = False
            self._last_entries_errors = 1
            self._last_entries_skipped = "crashed"
            metrics.entries_pass("crashed", errors=1)
        if bar_ready:
            self._last_bar = boundary
        elif now >= boundary + ENTRY_DELAY + BARS_GIVE_UP:
            logger.error(
                f"Giving up on the bar that closed at {boundary:%H:%M}: it never became available."
            )
            self._last_bar = boundary
        return True

    def _seconds_until_next_event(self, now: datetime) -> float:
        boundary = bar_boundary(now)
        if self._last_bar == boundary:
            next_entries = boundary + BAR + ENTRY_DELAY
        elif now < boundary + ENTRY_DELAY:
            next_entries = boundary + ENTRY_DELAY
        else:
            next_entries = now + BARS_RETRY  # the bar was not ready; try again
        until_entries = (next_entries - now).total_seconds()
        return max(1.0, min(until_entries, POLL.total_seconds()))
