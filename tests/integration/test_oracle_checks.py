"""Looking at the backend Oracle from outside the loop (task 2.2, T7): `health`, `doctor
--network` and `oracle-probe`, against a fake backend on a real socket."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.oracle_backend import GOOD_TOKEN, FakeOracleBackend, Reply, answer, error
from trading_system.cli import main
from trading_system.oracle.checks import (
    PROBE_VERSION,
    health_findings,
    ping_backend,
    probe_signal,
)
from trading_system.runtime.heartbeat import Heartbeat
from trading_system.storage.models import (
    BrokerOrderModel,
    OracleDecisionModel,
    RiskDecisionModel,
    StrategySignalModel,
)

OK = Reply(200, {"status": "ok"})


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No `.env` of the repository; a fresh heartbeat so that only the Oracle can be wrong."""
    monkeypatch.chdir(tmp_path)
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.setenv(name, "test-" + name.lower())
    path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("HEARTBEAT_PATH", str(path))
    Heartbeat(path).beat(
        {"market_open": False, "next_within_seconds": 300.0, "last_entries_errors": 0},
        now=datetime.now(timezone.utc),
    )


def _backend_env(
    monkeypatch: pytest.MonkeyPatch, url: str, mode: str = "shadow", token: str = GOOD_TOKEN
) -> None:
    monkeypatch.setenv("ORACLE_ENABLED", "true")
    monkeypatch.setenv("ORACLE_PROVIDER", "backend")
    monkeypatch.setenv("ORACLE_MODE", mode)
    monkeypatch.setenv("ORACLE_BACKEND_URL", url)
    monkeypatch.setenv("ORACLE_BACKEND_TOKEN", token)
    monkeypatch.setenv("ORACLE_TIMEOUT_SECONDS", "1")


# --- health ------------------------------------------------------------------------------------


def test_with_the_oracle_off_health_does_not_look_at_the_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["health"]) == 0
    assert capsys.readouterr().out.strip() == "healthy"


def test_health_pings_the_backend_and_stays_quiet_when_it_answers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOracleBackend(OK) as backend:
        _backend_env(monkeypatch, backend.url)
        assert main(["health"]) == 0

    assert capsys.readouterr().out.strip() == "healthy"
    (received,) = backend.received
    assert (received.method, received.path) == ("GET", "/oracle/ping")
    assert received.authorization == f"Bearer {GOOD_TOKEN}"


@pytest.mark.parametrize(
    "reply, kind",
    [(error(401, "unauthorized"), "unauthorized"), (Reply(500, "x"), "unavailable")],
)
def test_in_shadow_mode_a_backend_that_is_wrong_is_a_warning_and_the_container_stays_healthy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reply: Reply,
    kind: str,
) -> None:
    with FakeOracleBackend(reply) as backend:
        _backend_env(monkeypatch, backend.url, "shadow")
        assert main(["health"]) == 0  # trading is not affected, so Docker must not say otherwise

    out = capsys.readouterr().out.strip()
    assert out.startswith(f"healthy (warning: Oracle backend {kind}:") and "shadow mode" in out
    assert GOOD_TOKEN not in out


def test_in_shadow_mode_a_backend_nobody_listens_on_is_only_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOracleBackend() as backend:
        pass
    _backend_env(monkeypatch, backend.url, "shadow")

    assert main(["health"]) == 0
    assert "healthy (warning: Oracle backend unavailable:" in capsys.readouterr().out


def test_in_active_mode_a_backend_that_is_wrong_makes_the_container_unhealthy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOracleBackend(error(401, "unauthorized")) as backend:
        _backend_env(monkeypatch, backend.url, "active")
        assert main(["health"]) == 1

    out = capsys.readouterr().out
    assert out.startswith("UNHEALTHY: Oracle backend unauthorized:") and "no entries" in out


def test_in_active_mode_a_backend_that_answers_is_healthy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOracleBackend(OK) as backend:
        _backend_env(monkeypatch, backend.url, "active")
        assert main(["health"]) == 0

    assert capsys.readouterr().out.strip() == "healthy"


def test_a_stale_loop_and_a_bad_backend_are_both_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Heartbeat(tmp_path / "heartbeat.json").beat(
        {"market_open": False, "next_within_seconds": 60.0},
        now=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    with FakeOracleBackend(error(401, "unauthorized")) as backend:
        _backend_env(monkeypatch, backend.url, "active")
        assert main(["health"]) == 1

    out = capsys.readouterr().out
    assert "the loop has not ticked" in out and "Oracle backend unauthorized" in out


def test_a_broken_ping_cannot_crash_the_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("trading_system.oracle.backend.BackendOracle.ping", broken)
    _backend_env(monkeypatch, "http://127.0.0.1:9", "shadow")

    assert main(["health"]) == 0  # shadow: a warning at most


def test_the_findings_are_empty_when_the_backend_is_not_in_use() -> None:
    class Off:
        oracle_enabled = False
        oracle_provider = "backend"
        oracle_mode = "active"

    assert ping_backend(Off()) is None
    assert health_findings(Off()) == ([], [])


# --- doctor --network -------------------------------------------------------------------------------


@pytest.fixture
def broker_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broker:
        def __init__(self, _settings: Any) -> None: ...

        def get_clock(self) -> Any:
            return type("Clock", (), {"is_open": False})()

    monkeypatch.setattr("trading_system.broker.alpaca_broker.AlpacaBroker", Broker)


def test_doctor_says_when_it_is_not_looking_at_the_backend(
    broker_ok: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor", "--network"]) == 0
    assert "Oracle: not using the backend" in capsys.readouterr().out


def test_doctor_reports_a_backend_that_accepts_the_token(
    broker_ok: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOracleBackend(OK) as backend:
        _backend_env(monkeypatch, backend.url, "shadow")
        assert main(["doctor", "--network"]) == 0

    out = capsys.readouterr().out
    assert "Oracle backend reachable" in out and "mode: shadow" in out and GOOD_TOKEN not in out


def test_doctor_fails_when_the_backend_refuses_the_token(
    broker_ok: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeOracleBackend(error(401, "unauthorized")) as backend:
        _backend_env(monkeypatch, backend.url)
        assert main(["doctor", "--network"]) == 1

    assert "Oracle backend check failed (unauthorized)" in capsys.readouterr().out


def test_doctor_without_network_does_not_touch_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with FakeOracleBackend() as backend:
        _backend_env(monkeypatch, backend.url)
        assert main(["doctor"]) == 0
    assert backend.received == []


# --- oracle-probe ---------------------------------------------------------------------------------------


@pytest.fixture
def probe_env(monkeypatch: pytest.MonkeyPatch, db_url: str) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)


def _count(session_factory: Any, model: Any) -> int:
    with session_factory() as session:
        return int(session.query(model).count())


def test_the_probe_asks_with_an_artificial_signal_and_shows_what_the_model_said(
    probe_env: None,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeOracleBackend(OK, answer("VETO", "Няма пазарни данни.", request_id="req-42")) as b:
        _backend_env(monkeypatch, b.url, "shadow")
        assert main(["oracle-probe", "--symbol", "msft"]) == 0

    out = capsys.readouterr().out
    # the model's word from the record, not the APPROVE the shadow wrapper hands to a loop
    assert "verdict:     VETO" in out and "mode:        shadow" in out
    assert "request_id:  req-42" in out and "Няма пазарни данни." in out
    assert "GET /oracle/decisions?symbol=MSFT" in out
    ping, evaluate = b.received
    assert (ping.path, evaluate.path) == ("/oracle/ping", "/oracle/evaluate")
    body = evaluate.json
    assert body["signal"]["version"] == PROBE_VERSION and body["signal"]["symbol"] == "MSFT"
    assert "ORACLE PROBE" in body["rationale"] and body["mode"] == "shadow"
    with session_factory() as session:
        (signal,) = session.query(StrategySignalModel).all()
        (row,) = session.query(OracleDecisionModel).all()
    assert signal.version == PROBE_VERSION
    assert row.signal_id == signal.signal_id == body["signal"]["signal_id"]


def test_the_probe_places_no_order_and_makes_no_risk_decision(
    probe_env: None, monkeypatch: pytest.MonkeyPatch, session_factory: Any
) -> None:
    with FakeOracleBackend(OK, answer("APPROVE")) as backend:
        _backend_env(monkeypatch, backend.url, "active")
        assert main(["oracle-probe"]) == 0

    assert _count(session_factory, BrokerOrderModel) == 0
    assert _count(session_factory, RiskDecisionModel) == 0


def test_every_probe_is_a_new_question(
    probe_env: None, monkeypatch: pytest.MonkeyPatch, session_factory: Any
) -> None:
    with FakeOracleBackend(OK, answer("APPROVE"), OK, answer("APPROVE")) as backend:
        _backend_env(monkeypatch, backend.url)
        assert main(["oracle-probe"]) == 0
        assert main(["oracle-probe"]) == 0

    posts = [r for r in backend.received if r.path == "/oracle/evaluate"]
    assert len(posts) == 2
    assert posts[0].json["signal"]["signal_id"] != posts[1].json["signal"]["signal_id"]
    assert _count(session_factory, StrategySignalModel) == 2


def test_the_probe_fails_with_the_cause_when_the_model_does_not_answer(
    probe_env: None,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeOracleBackend(OK, error(504, "model_timeout")) as backend:
        _backend_env(monkeypatch, backend.url)
        assert main(["oracle-probe"]) == 1

    out = capsys.readouterr().out
    assert "(no answer)" in out and "error_kind:  timeout" in out
    assert _count(session_factory, OracleDecisionModel) == 1  # the failure is recorded too


def test_the_probe_stops_at_a_failed_ping_and_writes_nothing(
    probe_env: None,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with FakeOracleBackend(error(401, "unauthorized")) as backend:
        _backend_env(monkeypatch, backend.url)
        assert main(["oracle-probe"]) == 1

    assert "The ping failed (unauthorized)" in capsys.readouterr().out
    assert [r.path for r in backend.received] == ["/oracle/ping"]
    assert _count(session_factory, StrategySignalModel) == 0


def test_the_probe_refuses_when_the_backend_oracle_is_off(
    probe_env: None, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["oracle-probe"]) == 1

    assert "The backend Oracle is off" in capsys.readouterr().out
    assert _count(session_factory, StrategySignalModel) == 0


def test_a_probe_signal_is_marked_and_unique_per_moment() -> None:
    now = datetime(2026, 11, 6, 12, 0, 0, 123456, tzinfo=timezone.utc)
    first, second = probe_signal("AAPL", now), probe_signal("AAPL", now + timedelta(microseconds=1))

    assert first.version == PROBE_VERSION and first.signal_id != second.signal_id
