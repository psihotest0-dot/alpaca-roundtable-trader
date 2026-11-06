"""The backend Oracle in the real cycle, in shadow and in active mode (task 2.2, T4 and T6).

A real TradingCycle, risk engine, execution engine and SQLite; only Alpaca and the Go backend are
fakes, and the backend is a real HTTP server on the loopback. What is shown:

* shadow mode never changes an order, whatever the backend says or fails to say, and the record
  keeps what the MODEL said (not the APPROVE the loop was handed),
* active mode obeys a VETO, and a backend that cannot be reached gives no order (fail closed),
* the record, the export for the backtest and the metrics all tell the real story.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import REGISTRY

from tests.fixtures.oracle_backend import GOOD_TOKEN, FakeOracleBackend, Reply, answer, error
from trading_system.config import Settings
from trading_system.domain.enums import OracleVerdict
from trading_system.oracle import OracleRequest, create_oracle
from trading_system.oracle.backend import BackendOracle
from trading_system.oracle.recording import RecordingOracle
from trading_system.oracle.shadow import ShadowOracle
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.storage.models import (
    BrokerOrderModel,
    OracleDecisionModel,
    RiskDecisionModel,
)
from trading_system.storage.repositories import OracleDecisionRepository, SignalRepository

RunCycle = Callable[..., SimpleNamespace]
MakeSettings = Callable[..., Settings]


def _settings(make_settings: MakeSettings, backend: FakeOracleBackend, **more: Any) -> Settings:
    return make_settings(
        oracle_enabled=True,
        oracle_provider="backend",
        oracle_backend_url=backend.url,
        oracle_backend_token=GOOD_TOKEN,
        oracle_timeout_seconds=1,
        **more,
    )


def _orders(session_factory: Any) -> int:
    with session_factory() as session:
        return int(session.query(BrokerOrderModel).count())


def _risk_records(session_factory: Any) -> int:
    with session_factory() as session:
        return int(session.query(RiskDecisionModel).count())


def _rows(session_factory: Any) -> list[OracleDecisionModel]:
    with session_factory() as session:
        return list(session.query(OracleDecisionModel).order_by(OracleDecisionModel.id).all())


def _answers(verdict: str, kind: str) -> float:
    return (
        REGISTRY.get_sample_value("oracle_answers_total", {"verdict": verdict, "kind": kind}) or 0.0
    )


@pytest.fixture
def run_with(
    run_cycle: RunCycle, make_settings: MakeSettings, session_factory: Any
) -> Callable[..., SimpleNamespace]:
    """`run_with(backend, mode=...)`: one cycle whose Oracle is built from the settings."""

    def _run(backend: FakeOracleBackend, mode: str = "shadow", **more: Any) -> SimpleNamespace:
        settings = _settings(make_settings, backend, oracle_mode=mode, **more)
        oracle = create_oracle(settings, OracleDecisionRepository(session_factory))
        return run_cycle(oracle=oracle)

    return _run


# --- how the wrappers are stacked ------------------------------------------------------------


def test_the_record_sits_inside_the_shadow_so_it_keeps_what_the_model_said(
    make_settings: MakeSettings, session_factory: Any
) -> None:
    """The trap of 2.2: Shadow(Recording(Backend)). The other way round the store would hold the
    APPROVE that the loop is handed, and the experiment would measure nothing."""
    repo = OracleDecisionRepository(session_factory)
    with FakeOracleBackend() as backend:
        shadow = create_oracle(_settings(make_settings, backend, oracle_mode="shadow"), repo)
        active = create_oracle(_settings(make_settings, backend, oracle_mode="active"), repo)
        bare = create_oracle(_settings(make_settings, backend, oracle_mode="shadow"))

    assert isinstance(shadow, ShadowOracle)
    assert isinstance(shadow.provider, RecordingOracle)
    assert isinstance(shadow.provider.provider, BackendOracle)
    assert isinstance(active, RecordingOracle) and isinstance(active.provider, BackendOracle)
    assert isinstance(bare, ShadowOracle) and isinstance(bare.provider, BackendOracle)


def test_shadow_is_the_default_mode(make_settings: MakeSettings, session_factory: Any) -> None:
    with FakeOracleBackend() as backend:
        settings = make_settings(
            oracle_enabled=True,
            oracle_provider="backend",
            oracle_backend_url=backend.url,
            oracle_backend_token=GOOD_TOKEN,
        )

    assert settings.oracle_mode == "shadow"
    assert isinstance(
        create_oracle(settings, OracleDecisionRepository(session_factory)), ShadowOracle
    )


def test_the_orchestrator_builds_the_stack_from_the_settings(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with FakeOracleBackend() as backend:
        settings = _settings(make_settings, backend)
    monkeypatch.setattr("trading_system.broker.alpaca_broker.AlpacaBroker", lambda _s: object())
    monkeypatch.setattr("trading_system.data.alpaca_data.AlpacaData", lambda _s: object())

    cycle = Orchestrator(settings).build_cycle(False)

    assert isinstance(cycle.oracle, ShadowOracle)
    assert isinstance(cycle.oracle.provider, RecordingOracle)
    assert isinstance(cycle.oracle.provider.provider, BackendOracle)


# --- shadow: nothing about the orders changes --------------------------------------------------


@pytest.mark.parametrize("verdict", ["APPROVE", "VETO", "HOLD"])
def test_in_shadow_mode_the_order_goes_whatever_the_model_says_and_the_record_keeps_its_word(
    run_with: Callable[..., SimpleNamespace], session_factory: Any, verdict: str
) -> None:
    with FakeOracleBackend(answer(verdict, "Моделът казва " + verdict)) as backend:
        cycle = run_with(backend, "shadow")

    assert (cycle.report.submitted, cycle.report.held) == (1, 0)
    assert _orders(session_factory) == 1
    (row,) = _rows(session_factory)
    assert row.verdict is OracleVerdict(verdict)  # what the model said, not what the loop heard
    assert (row.mode, row.failed, row.error_kind) == ("shadow", False, None)
    assert (row.request_id, row.model, row.provider) == ("req-1", "test-model", "backend")
    assert row.reasoning == "Моделът казва " + verdict
    (received,) = backend.received
    assert received.json["mode"] == "shadow" and received.json["signal"]["symbol"] == "AAPL"


@pytest.mark.parametrize(
    "reply, kind",
    [
        (error(504, "model_timeout"), "timeout"),
        (error(502, "model_unavailable"), "unavailable"),
        (error(401, "unauthorized"), "unauthorized"),
        (error(400, "invalid_request", detail="signal.symbol"), "bad_request"),
        (Reply(200, "<html>not json</html>"), "bad_reply"),
        (Reply(200, {"verdict": "BUY NOW"}), "bad_reply"),
    ],
)
def test_in_shadow_mode_a_backend_that_fails_does_not_stop_the_order(
    run_with: Callable[..., SimpleNamespace], session_factory: Any, reply: Reply, kind: str
) -> None:
    before = _answers("HOLD", kind)
    with FakeOracleBackend(reply) as backend:
        cycle = run_with(backend, "shadow")

    assert cycle.report.submitted == 1 and _orders(session_factory) == 1
    (row,) = _rows(session_factory)
    assert (row.verdict, row.failed, row.error_kind, row.mode) == (
        OracleVerdict.HOLD,
        True,
        kind,
        "shadow",
    )
    assert _answers("HOLD", kind) == before + 1  # the metric tells the causes apart


def test_in_shadow_mode_a_backend_nobody_listens_on_does_not_stop_the_order(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend() as backend:
        pass  # stopped: its port is closed now
    cycle = run_with(backend, "shadow")

    assert cycle.report.submitted == 1 and _orders(session_factory) == 1
    (row,) = _rows(session_factory)
    assert row.failed and row.error_kind == "unavailable"


def test_in_shadow_mode_the_loop_logs_what_the_model_said(
    run_with: Callable[..., SimpleNamespace], caplog: pytest.LogCaptureFixture
) -> None:
    with FakeOracleBackend(answer("VETO", "Новината е лоша.")) as backend:
        with caplog.at_level("INFO"):
            run_with(backend, "shadow")

    text = caplog.text
    assert "Shadow Oracle on AAPL: the model said VETO - Новината е лоша." in text
    assert "the signal goes on" in text


# --- active: the verdict counts -------------------------------------------------------------------


def test_in_active_mode_a_veto_ends_the_signal(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend(answer("VETO")) as backend:
        cycle = run_with(backend, "active")

    assert (cycle.report.submitted, cycle.report.held) == (0, 0)
    # Risk's first look approved it (so the model was asked), but nothing of that is recorded and
    # the final evaluation never happened: the record of risk decisions is empty.
    assert _orders(session_factory) == 0 and len(cycle.decisions) == 1
    assert _risk_records(session_factory) == 0
    (row,) = _rows(session_factory)
    assert (row.verdict, row.mode) == (OracleVerdict.VETO, "active")
    assert backend.received[0].json["mode"] == "active"


def test_in_active_mode_an_approve_lets_the_signal_through(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend(answer("APPROVE")) as backend:
        cycle = run_with(backend, "active")

    assert cycle.report.submitted == 1 and _orders(session_factory) == 1
    assert _rows(session_factory)[0].mode == "active"


def test_in_active_mode_a_hold_waits(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend(answer("HOLD")) as backend:
        cycle = run_with(backend, "active")

    assert (cycle.report.submitted, cycle.report.held) == (0, 1)
    assert _orders(session_factory) == 0


@pytest.mark.parametrize(
    "reply",
    [
        error(504, "model_timeout"),
        error(502, "model_unavailable"),
        error(401, "unauthorized"),
        Reply(200, "not json"),
        Reply(200, {"verdict": "approve"}),
    ],
)
def test_in_active_mode_a_backend_that_fails_gives_no_order_but_the_signal_waits(
    run_with: Callable[..., SimpleNamespace], session_factory: Any, reply: Reply
) -> None:
    with FakeOracleBackend(reply) as backend:
        cycle = run_with(backend, "active")

    assert cycle.report.submitted == 0 and _orders(session_factory) == 0
    assert cycle.report.held == 1  # a failure is "not now", never a final no and never a yes
    (row,) = _rows(session_factory)
    assert row.failed and row.verdict is OracleVerdict.HOLD and row.mode == "active"


def test_in_active_mode_a_backend_nobody_listens_on_gives_no_order(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend() as backend:
        pass
    cycle = run_with(backend, "active")

    assert cycle.report.submitted == 0 and _orders(session_factory) == 0
    assert cycle.report.held == 1 and _rows(session_factory)[0].error_kind == "unavailable"


def test_in_active_mode_a_backend_that_answers_late_gives_no_order_either(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend(Reply(200, answer("APPROVE").body, delay=2.0)) as backend:
        cycle = run_with(backend, "active")  # the timeout is 1 s

    assert cycle.report.submitted == 0 and _orders(session_factory) == 0
    assert _rows(session_factory)[0].error_kind == "timeout"


def test_a_hold_that_the_model_gave_is_asked_again_and_the_next_answer_counts(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    from datetime import datetime, timezone

    with FakeOracleBackend(answer("HOLD"), answer("APPROVE")) as backend:
        cycle = run_with(backend, "active")
        assert cycle.report.held == 1
        ordered = cycle.cycle.revisit_held(datetime.now(timezone.utc))

    assert ordered == 1 and _orders(session_factory) == 1
    assert [r.verdict for r in _rows(session_factory)] == [
        OracleVerdict.HOLD,
        OracleVerdict.APPROVE,
    ]
    assert len(backend.received) == 2


# --- the record, as the backtest will read it ---------------------------------------------------------


def test_the_export_for_the_backtest_carries_the_models_word_not_the_loops(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend(answer("VETO", "Слаб сигнал.")) as backend:
        cycle = run_with(backend, "shadow")
    assert cycle.report.submitted == 1  # the loop traded it all the same

    export = OracleDecisionRepository(session_factory).replay_export()

    (entry,) = export.values()
    assert (entry["approved"], entry["verdict"], entry["final"]) == (False, "VETO", True)
    assert (entry["mode"], entry["failed"], entry["model"]) == ("shadow", False, "test-model")
    assert entry["provider"] == "backend" and entry["reason"] == "Слаб сигнал."


def test_the_export_marks_an_outage_so_that_it_is_not_taken_for_a_veto(
    run_with: Callable[..., SimpleNamespace], session_factory: Any
) -> None:
    with FakeOracleBackend(error(504, "model_timeout")) as backend:
        run_with(backend, "shadow")

    (entry,) = OracleDecisionRepository(session_factory).replay_export().values()

    assert entry["failed"] is True and entry["final"] is False  # no word of the model's at all
    # The loop traded the signal anyway, so the replay must trade it too, not play it as a veto.
    assert entry["approved"] is True


def test_in_shadow_mode_a_hold_of_the_model_was_traded_too_and_a_veto_was_not_approved(
    run_cycle: RunCycle, make_settings: MakeSettings, session_factory: Any
) -> None:
    """`approved` is what the replay will do with the signal. Shadow HOLD: traded (the model gave
    no final word). Shadow VETO: the model's word stands, the replay plays it as refused."""
    repo = OracleDecisionRepository(session_factory)
    export: dict[str, dict[str, Any]] = {}
    for verdict in ("HOLD", "VETO"):
        with FakeOracleBackend(answer(verdict)) as backend:
            settings = _settings(make_settings, backend, oracle_mode="shadow")
            oracle = create_oracle(settings, repo)
            assert oracle is not None
            result = run_cycle(oracle=oracle)
            assert result.report.submitted == 1  # traded in both cases
        export[verdict] = next(iter(repo.replay_export().values()))
        with session_factory() as session:  # the next round starts from an empty record
            session.query(OracleDecisionModel).delete()
            session.commit()

    assert export["HOLD"]["approved"] is True and export["HOLD"]["final"] is False
    assert export["VETO"]["approved"] is False and export["VETO"]["final"] is True


@pytest.mark.parametrize("reply", [answer("HOLD"), error(504, "model_timeout")])
def test_in_active_mode_a_signal_without_a_final_word_was_not_traded_and_the_export_says_so(
    run_with: Callable[..., SimpleNamespace], session_factory: Any, reply: Reply
) -> None:
    with FakeOracleBackend(reply) as backend:
        cycle = run_with(backend, "active")
    assert cycle.report.submitted == 0

    (entry,) = OracleDecisionRepository(session_factory).replay_export().values()

    assert entry["approved"] is False and entry["mode"] == "active"


def test_a_final_answer_is_not_asked_of_the_backend_twice_and_shadow_still_says_yes(
    make_settings: MakeSettings, session_factory: Any, signal: Any
) -> None:
    SignalRepository(session_factory).save_signal(signal)  # decisions reference it (FK)
    with FakeOracleBackend(answer("VETO"), answer("APPROVE")) as backend:
        settings = _settings(make_settings, backend, oracle_mode="shadow")
        oracle = create_oracle(settings, OracleDecisionRepository(session_factory))
        assert oracle is not None

        first = oracle.evaluate(OracleRequest(signal=signal))
        again = oracle.evaluate(OracleRequest(signal=signal))  # a restart would do the same

    assert len(backend.received) == 1  # the model's final word was served from the record
    assert first.verdict is again.verdict is OracleVerdict.APPROVE  # what the loop is told
    (row,) = _rows(session_factory)
    assert row.verdict is OracleVerdict.VETO  # what was recorded
