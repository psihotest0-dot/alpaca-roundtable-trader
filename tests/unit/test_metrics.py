"""The metrics module (observability/metrics.py): what is published and what is not, the helpers the
loops call (they never raise), and the /metrics page itself."""

import re
import socket
import time
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY
from prometheus_client.metrics import MetricWrapperBase

from trading_system.observability import metrics

SRC = Path(__file__).resolve().parents[2] / "src" / "trading_system"


def _value(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def _objects() -> dict[str, MetricWrapperBase]:
    return {k: v for k, v in vars(metrics).items() if isinstance(v, MetricWrapperBase)}


# ----------------------------------------------------------------------- what is published


def test_every_metric_has_the_alpaca_prefix_so_it_cannot_meet_the_backends() -> None:
    names = [m._name for m in _objects().values()]

    assert names and all(n.startswith("alpaca_") for n in names), names


def _published() -> set[str]:
    return {n for n, m in _objects().items() if m in REGISTRY._collector_to_names}


def _private() -> set[str]:
    return {n for n, m in _objects().items() if m in metrics.NOT_WIRED._collector_to_names}


def test_each_metric_is_either_published_or_private_never_both_never_neither() -> None:
    assert _published().isdisjoint(_private()) and _published() | _private() == set(_objects())
    assert {
        "HEARTBEAT_TIMESTAMP",
        "ORACLE_ANSWERS",
        "ORACLE_ERRORS",
        "UNPROTECTED_POSITIONS",
    } <= _published()
    assert {"RUNTIME_HEALTH", "UNRESOLVED_INTENTS", "DAILY_LOSS"} <= _private()


def test_a_metric_in_the_private_registry_is_set_by_nothing_in_the_code() -> None:
    """The reverse of the rule: a metric that is wired must be published. (ORACLE_ERRORS was set in
    oracle/recording.py while it still sat in the private registry; the first version of this change
    hid it, and a test that read it caught that.)"""
    code = {p: p.read_text(encoding="utf-8") for p in SRC.rglob("*.py") if p.name != "metrics.py"}
    used = {n: [p.name for p, text in code.items() if re.search(rf"{n}", text)] for n in _private()}

    assert _private() and {n: files for n, files in used.items() if files} == {}


def test_the_page_a_scraper_reads_has_the_published_metrics_and_not_the_hidden_ones() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    metrics.beat(True)
    metrics.protection_checked(0)

    metrics.start_metrics_server(port, "127.0.0.1")
    page = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()

    assert "alpaca_heartbeat_timestamp_seconds " in page
    assert "alpaca_unprotected_positions 0.0" in page
    assert "alpaca_unresolved_intents" not in page and "alpaca_daily_loss" not in page


# ---------------------------------------------------------------------------- the helpers


def test_beat_stamps_the_heartbeat_and_the_market_state() -> None:
    metrics.beat(True)
    assert abs(_value("alpaca_heartbeat_timestamp_seconds") - time.time()) < 5
    assert _value("alpaca_market_open") == 1
    metrics.beat(False, when=123.0)
    assert (
        _value("alpaca_heartbeat_timestamp_seconds") == 123.0 and _value("alpaca_market_open") == 0
    )


def test_a_clock_that_could_not_be_read_leaves_the_market_state_alone() -> None:
    metrics.beat(True)

    metrics.beat(None)

    assert _value("alpaca_market_open") == 1


def test_a_pass_is_counted_and_only_a_pass_that_looked_stamps_the_time() -> None:
    ok_before = _value("alpaca_entries_passes_total", {"outcome": "ok"})
    signals_before, errors_before = (
        _value("alpaca_signals_total"),
        _value("alpaca_entries_errors_total"),
    )
    metrics.LAST_ENTRIES_TIMESTAMP.set(0)

    metrics.entries_pass("retry_later")
    metrics.entries_pass("crashed", errors=1)
    assert _value("alpaca_last_entries_timestamp_seconds") == 0  # neither looked

    metrics.entries_pass("ok", signals=3, errors=1, when=555.0)

    assert _value("alpaca_entries_passes_total", {"outcome": "ok"}) == ok_before + 1
    assert _value("alpaca_signals_total") == signals_before + 3
    assert _value("alpaca_entries_errors_total") == errors_before + 2
    assert _value("alpaca_last_entries_timestamp_seconds") == 555.0


def test_a_rejection_reason_cannot_make_a_new_series_out_of_a_number() -> None:
    reason = "Planned risk 1234 exceeds the limit"
    before = _value("alpaca_rejected_signals_total", {"reason": reason})

    metrics.rejected(f"{reason}: 5.67 percent of equity")
    metrics.rejected("")
    metrics.rejected("x" * 200)

    assert _value("alpaca_rejected_signals_total", {"reason": reason}) == before + 1
    assert _value("alpaca_rejected_signals_total", {"reason": "unknown"}) >= 1
    assert _value("alpaca_rejected_signals_total", {"reason": "x" * 80}) >= 1


def test_the_account_the_protection_check_and_the_reconciliation_are_published() -> None:
    metrics.account(98765.5, 3, 0.12)
    metrics.protection_checked(2, when=777.0)
    metrics.reconciled(when=888.0)

    assert _value("alpaca_account_equity") == 98765.5 and _value("alpaca_active_positions") == 3
    assert _value("alpaca_gross_exposure") == 0.12
    assert _value("alpaca_unprotected_positions") == 2
    assert _value("alpaca_protection_check_timestamp_seconds") == 777.0
    assert _value("alpaca_reconcile_timestamp_seconds") == 888.0


def test_candidates_are_counted_by_strategy_and_a_broken_one_is_not_a_count() -> None:
    before = _value("alpaca_candidate_signals_total", {"strategy": "trend_cross_session"})
    late = _value("alpaca_candidate_signals_total", {"strategy": "trend_cross_session_late"})
    errors = _value("alpaca_candidate_errors_total")

    metrics.candidate_pass(
        {"trend_cross_session": 4, "trend_cross_session_late": -1, "trend_cross": 0}, 1
    )

    assert (
        _value("alpaca_candidate_signals_total", {"strategy": "trend_cross_session"}) == before + 4
    )
    # a strategy that broke (-1) or found nothing (0) adds nothing; the counter is global, so compare
    assert (
        _value("alpaca_candidate_signals_total", {"strategy": "trend_cross_session_late"}) == late
    )
    assert _value("alpaca_candidate_errors_total") == errors + 1


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.beat(True),
        lambda: metrics.entries_pass("ok", signals=1, errors=1),
        lambda: metrics.rejected("x"),
        lambda: metrics.account(1.0, 1, 1.0),
        lambda: metrics.protection_checked(1),
        lambda: metrics.reconciled(),
        lambda: metrics.candidate_pass({"a": 1}, 1),
    ],
)
def test_a_metric_that_breaks_never_breaks_the_caller(call: Any) -> None:
    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("the registry is broken")

    names = [
        "HEARTBEAT_TIMESTAMP",
        "MARKET_OPEN",
        "LAST_ENTRIES_TIMESTAMP",
        "ENTRIES_PASSES",
        "SIGNALS_FOUND",
        "ENTRIES_ERRORS",
        "REJECTED_SIGNALS",
        "ACCOUNT_EQUITY",
        "ACTIVE_POSITIONS",
        "GROSS_EXPOSURE",
        "UNPROTECTED_POSITIONS",
        "PROTECTION_CHECK_TIMESTAMP",
        "RECONCILE_TIMESTAMP",
        "CANDIDATE_SIGNALS",
        "CANDIDATE_ERRORS",
    ]
    with patch.multiple(metrics, **{n: Broken() for n in names}):
        assert call() is None  # no exception
