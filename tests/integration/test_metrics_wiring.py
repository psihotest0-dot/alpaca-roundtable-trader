"""Where the metrics are set: the Runner's tick and look for entries, the cycle's risk rejection and
account, and the orchestrator's /metrics server. The names and the helpers are in tests/unit/test_metrics.py."""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

from trading_system.runtime.cycle import EntriesReport
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.runtime.runner import Runner

UTC = timezone.utc
# 16:01 UTC: a minute after a bar closed, past the 30 s the Runner waits before it looks.
LATER = datetime(2026, 11, 9, 16, 1, tzinfo=UTC)
RunCycle = Callable[..., SimpleNamespace]


def _value(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


class StubCycle:
    def __init__(self, report: EntriesReport | Exception) -> None:
        self.report = report

    def run_housekeeping(self, now: datetime) -> None:
        pass

    def revisit_held(self, now: datetime) -> int:
        return 0

    def run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        if isinstance(self.report, Exception):
            raise self.report
        return self.report


def _runner(report: EntriesReport | Exception, *, market_open: bool = True) -> Runner:
    clock = SimpleNamespace(
        is_open=market_open, timestamp=LATER, next_open=LATER + timedelta(hours=1)
    )
    broker = SimpleNamespace(get_clock=lambda: clock)
    return Runner(StubCycle(report), broker, now=lambda: LATER, wait=lambda seconds: None)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ the Runner


def test_a_tick_stamps_the_heartbeat_and_the_market_state_for_grafana() -> None:
    metrics_before = _value("alpaca_heartbeat_timestamp_seconds")
    runner = _runner(EntriesReport(), market_open=False)

    runner.tick()

    assert _value("alpaca_heartbeat_timestamp_seconds") > metrics_before
    assert _value("alpaca_market_open") == 0


def test_the_runner_publishes_when_the_open_market_began() -> None:
    _runner(EntriesReport()).tick()
    assert _value("alpaca_market_open_since_timestamp_seconds") == LATER.timestamp()

    _runner(EntriesReport(), market_open=False).tick()
    assert _value("alpaca_market_open_since_timestamp_seconds") == 0.0


def test_a_look_for_entries_is_counted_with_its_signals_and_errors() -> None:
    ok = _value("alpaca_entries_passes_total", {"outcome": "ok"})
    signals, errors = _value("alpaca_signals_total"), _value("alpaca_entries_errors_total")
    runner = _runner(EntriesReport(signals=2, errors=1))

    runner.tick()

    assert _value("alpaca_market_open") == 1
    assert _value("alpaca_entries_passes_total", {"outcome": "ok"}) == ok + 1
    assert _value("alpaca_signals_total") == signals + 2
    assert _value("alpaca_entries_errors_total") == errors + 1
    assert _value("alpaca_last_entries_timestamp_seconds") > 0


def test_a_bar_that_was_not_there_yet_is_a_retry_not_a_pass_that_looked() -> None:
    retry = _value("alpaca_entries_passes_total", {"outcome": "retry_later"})
    runner = _runner(EntriesReport(skipped="bar not available yet", retry_later=True))

    runner.tick()

    assert _value("alpaca_entries_passes_total", {"outcome": "retry_later"}) == retry + 1


def test_a_look_that_crashes_is_counted_and_the_loop_goes_on() -> None:
    crashed = _value("alpaca_entries_passes_total", {"outcome": "crashed"})
    runner = _runner(RuntimeError("alpaca is down"))

    runner.tick()  # does not raise

    assert _value("alpaca_entries_passes_total", {"outcome": "crashed"}) == crashed + 1


# ----------------------------------------------------------------------------- the cycle


def test_a_risk_rejection_is_counted_by_its_reason(run_cycle: RunCycle) -> None:
    reason = "Max open positions limit reached."
    before = _value("alpaca_rejected_signals_total", {"reason": reason})

    cycle = run_cycle(tweak=lambda c: setattr(c.risk_engine.settings, "max_open_positions", 0))

    assert [d.approved for d in cycle.decisions] == [False]
    assert _value("alpaca_rejected_signals_total", {"reason": reason}) == before + 1


def test_the_account_is_published_after_a_look_for_entries(run_cycle: RunCycle) -> None:
    run_cycle(equity="123456")

    assert _value("alpaca_account_equity") == 123456.0
    assert _value("alpaca_active_positions") == 0 and _value("alpaca_gross_exposure") == 0.0


def test_the_housekeeping_publishes_the_account_at_most_once_a_minute(
    run_cycle: RunCycle,
) -> None:
    from trading_system.observability import metrics

    ran = run_cycle(equity="123456")
    metrics.ACCOUNT_EQUITY.set(0)

    ran.cycle.run_housekeeping(datetime.now(UTC))  # the look for entries published a moment ago
    assert _value("alpaca_account_equity") == 0

    ran.cycle._account_published = float("-inf")  # a minute later
    ran.cycle.run_housekeeping(datetime.now(UTC))
    assert _value("alpaca_account_equity") == 123456.0  # real also while the market is closed


def test_an_account_that_cannot_be_read_is_a_missing_number_and_nothing_more(
    run_cycle: RunCycle,
) -> None:
    ran = run_cycle(equity="123456")
    ran.cycle.broker.get_positions = _raises(ConnectionError("alpaca hiccup"))
    metrics_equity = _value("alpaca_account_equity")

    ran.cycle._publish_account()  # does not raise

    assert _value("alpaca_account_equity") == metrics_equity  # the number is simply not updated


def _raises(error: Exception) -> Callable[[], Any]:
    def call() -> Any:
        raise error

    return call


def test_a_reconciliation_that_refreshed_every_order_is_stamped(run_cycle: RunCycle) -> None:
    from trading_system.observability import metrics

    metrics.RECONCILE_TIMESTAMP.set(0)

    run_cycle()  # the housekeeping reconciles before it looks for entries

    assert _value("alpaca_reconcile_timestamp_seconds") > 0


# ------------------------------------------------------------------ the /metrics server


def test_the_server_starts_only_when_it_is_switched_on(make_settings: Any) -> None:
    with patch("trading_system.observability.metrics.start_metrics_server") as start:
        Orchestrator(make_settings(metrics_enabled=False))._start_metrics()
        start.assert_not_called()

        Orchestrator(
            make_settings(metrics_enabled=True, metrics_host="0.0.0.0", metrics_port=9100)
        )._start_metrics()
        start.assert_called_once_with(9100, "0.0.0.0")


def test_a_port_that_cannot_be_taken_is_an_error_in_the_log_and_nothing_more(
    make_settings: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch(
            "trading_system.observability.metrics.start_metrics_server",
            side_effect=OSError("address already in use"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        Orchestrator(make_settings(metrics_enabled=True))._start_metrics()  # does not raise

    assert "Could not start the metrics server" in caplog.text
    assert "The trading goes on without it" in caplog.text


def test_both_loops_start_the_server_before_they_run(make_settings: Any) -> None:
    class StopAtOnce:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self) -> None:
            pass

    cycle = SimpleNamespace(broker=object())
    settings = make_settings(
        oracle_enabled=True, oracle_provider="backend", oracle_backend_token="t" * 48
    )
    orchestrator = Orchestrator(settings)

    with (
        patch.object(Orchestrator, "_start_metrics") as start,
        patch("trading_system.runtime.orchestrator.Runner", StopAtOnce),
        patch.object(Orchestrator, "build_cycle", return_value=cycle),
        patch.object(Orchestrator, "build_observation_cycle", return_value=cycle),
    ):
        orchestrator.run(is_paper=False, handle_signals=False)
        orchestrator.run_observation(handle_signals=False)

    assert start.call_count == 2
