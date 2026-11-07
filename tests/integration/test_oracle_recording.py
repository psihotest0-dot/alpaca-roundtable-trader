"""The Oracle's answers are recorded; final ones are not asked twice (Phase 2.3)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

from tests.fixtures.oracles import ScriptedOracle
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig
from trading_system.cli import main
from trading_system.domain.enums import OracleVerdict, SignalDirection
from trading_system.domain.models import OracleDecision, StrategySignal
from trading_system.oracle import NullOracle, OracleRequest, create_oracle
from trading_system.oracle.recording import RecordingOracle
from trading_system.storage.models import OracleDecisionModel
from trading_system.storage.repositories import OracleDecisionRepository, SignalRepository

APPROVE, VETO, HOLD = OracleVerdict.APPROVE, OracleVerdict.VETO, OracleVerdict.HOLD


def _signal(symbol: str = "AAPL", minute: int = 30) -> StrategySignal:
    return StrategySignal(
        version="1",
        symbol=symbol,
        timeframe="15m",
        closed_bar_timestamp=datetime(2026, 11, 2, 15, minute, tzinfo=timezone.utc),
        direction=SignalDirection.LONG,
    )


@pytest.fixture
def repo(session_factory: Any) -> OracleDecisionRepository:
    return OracleDecisionRepository(session_factory)


@pytest.fixture
def saved_signal(session_factory: Any) -> StrategySignal:
    signal = _signal()
    SignalRepository(session_factory).save_signal(signal)  # decisions reference it (FK)
    return signal


def _decision(signal: StrategySignal, verdict: OracleVerdict, **kw: Any) -> OracleDecision:
    return OracleDecision(signal_id=signal.signal_id or "", verdict=verdict, **kw)


def test_every_verdict_survives_a_database_round_trip(
    repo: OracleDecisionRepository, session_factory: Any
) -> None:
    """The same guard as for orders: a member must come back as itself."""
    for minute, verdict in zip((0, 15, 30), OracleVerdict):
        signal = _signal(minute=minute)
        SignalRepository(session_factory).save_signal(signal)
        repo.save(_decision(signal, verdict, confidence=0.7, provider="p", reason="r"))

    with session_factory() as fresh:
        rows = fresh.query(OracleDecisionModel).order_by(OracleDecisionModel.id).all()
        assert [r.verdict for r in rows] == list(OracleVerdict)
        assert (rows[0].confidence, rows[0].provider, rows[0].reasoning) == (0.7, "p", "r")


def test_only_approve_and_veto_are_final_decisions(
    repo: OracleDecisionRepository, saved_signal: StrategySignal
) -> None:
    sid = saved_signal.signal_id or ""
    assert repo.final_decision_for(sid) is None

    repo.save(_decision(saved_signal, HOLD))
    repo.save(_decision(saved_signal, HOLD, failed=True, reason="timeout"))
    assert repo.final_decision_for(sid) is None  # not now, and no answer, are not decisions

    repo.save(_decision(saved_signal, VETO, reason="no"))
    final = repo.final_decision_for(sid)
    assert final is not None and final.verdict is VETO and final.reason == "no"
    assert final.decision_timestamp.tzinfo is not None


def test_a_final_answer_is_served_from_the_record_even_after_a_restart(
    repo: OracleDecisionRepository, saved_signal: StrategySignal
) -> None:
    first = ScriptedOracle(VETO)
    RecordingOracle(first, repo).evaluate(OracleRequest(signal=saved_signal))

    # A new process: a new wrapper and a new provider that would answer differently.
    second = ScriptedOracle(APPROVE)
    again = RecordingOracle(second, repo).evaluate(OracleRequest(signal=saved_signal))

    assert again.verdict is VETO  # it does not change its mind
    assert second.requests == []  # and was not even asked


def test_a_hold_is_recorded_every_time_and_never_served(
    repo: OracleDecisionRepository, saved_signal: StrategySignal, session_factory: Any
) -> None:
    provider = ScriptedOracle(HOLD, HOLD, APPROVE)
    oracle = RecordingOracle(provider, repo)
    request = OracleRequest(signal=saved_signal)

    verdicts = [oracle.evaluate(request).verdict for _ in range(4)]

    assert verdicts == [HOLD, HOLD, APPROVE, APPROVE]
    assert len(provider.requests) == 3  # the fourth was served from the record
    with session_factory() as fresh:
        assert fresh.query(OracleDecisionModel).count() == 3  # cached answers are not re-recorded


def test_an_error_is_recorded_as_a_failed_hold_and_asked_about_again(
    repo: OracleDecisionRepository, saved_signal: StrategySignal, session_factory: Any
) -> None:
    """H1. A timeout used to be cached as a veto, which ended the signal for good."""
    provider = ScriptedOracle(ConnectionError("lemonade is down"), APPROVE)
    oracle = RecordingOracle(provider, repo)
    request = OracleRequest(signal=saved_signal)

    down = oracle.evaluate(request)
    up = oracle.evaluate(request)

    assert (down.verdict, down.failed) == (HOLD, True)
    assert (up.verdict, up.failed) == (APPROVE, False)
    with session_factory() as fresh:
        rows = fresh.query(OracleDecisionModel).order_by(OracleDecisionModel.id).all()
        assert [(r.verdict, r.failed) for r in rows] == [(HOLD, True), (APPROVE, False)]


def test_losing_the_record_does_not_lose_the_decision(saved_signal: StrategySignal) -> None:
    class BrokenRepository:
        def save(self, decision: OracleDecision) -> None:
            raise OSError("disk full")

        def final_decision_for(self, signal_id: str) -> None:
            raise OSError("disk full")

    oracle = RecordingOracle(ScriptedOracle(APPROVE), BrokenRepository())  # type: ignore[arg-type]

    assert oracle.evaluate(OracleRequest(signal=saved_signal)).verdict is APPROVE


def _count(verdict: str, kind: str) -> float:
    return (
        REGISTRY.get_sample_value("alpaca_oracle_answers_total", {"verdict": verdict, "kind": kind})
        or 0
    )


def test_metrics_tell_an_unreachable_oracle_from_one_that_says_no(
    repo: OracleDecisionRepository, saved_signal: StrategySignal
) -> None:
    before = {k: _count(*k) for k in (("VETO", "answer"), ("VETO", "cached"), ("HOLD", "failed"))}
    errors = REGISTRY.get_sample_value("alpaca_oracle_errors_total") or 0

    down = RecordingOracle(ScriptedOracle(OSError("x")), repo)
    down.evaluate(OracleRequest(signal=_signal(minute=0)))  # no answer
    says_no = RecordingOracle(ScriptedOracle(VETO), repo)
    says_no.evaluate(OracleRequest(signal=saved_signal))
    says_no.evaluate(OracleRequest(signal=saved_signal))  # from the record

    assert _count("HOLD", "failed") == before[("HOLD", "failed")] + 1
    assert _count("VETO", "answer") == before[("VETO", "answer")] + 1
    assert _count("VETO", "cached") == before[("VETO", "cached")] + 1
    assert (REGISTRY.get_sample_value("alpaca_oracle_errors_total") or 0) == errors + 1


def test_the_factory_records_only_when_given_a_repository(
    settings: Any, make_settings: Any, repo: OracleDecisionRepository
) -> None:
    enabled = make_settings(oracle_enabled=True)

    assert isinstance(create_oracle(enabled), NullOracle)
    recording = create_oracle(enabled, repo)
    assert isinstance(recording, RecordingOracle) and isinstance(recording.provider, NullOracle)


def test_a_veto_survives_a_restart_of_the_whole_cycle(run_cycle: Any, session_factory: Any) -> None:
    """The signal is vetoed in one run and the system restarts inside the signal's validity:
    the second run does not put the question to the model again."""
    repo = OracleDecisionRepository(session_factory)
    run_cycle(oracle=RecordingOracle(ScriptedOracle(VETO), repo))

    second_model = ScriptedOracle(APPROVE)
    second = run_cycle(oracle=RecordingOracle(second_model, repo))

    assert second_model.requests == []
    assert second.report.submitted == 0 and second.broker.submitted == []


# ------------------------------------------------------------ export for the backtest


def _populate(repo: OracleDecisionRepository, session_factory: Any) -> dict[str, StrategySignal]:
    signals = {
        name: _signal(name, minute) for name, minute in (("A", 0), ("B", 15), ("C", 30), ("D", 45))
    }
    for signal in signals.values():
        SignalRepository(session_factory).save_signal(signal)
    repo.save(_decision(signals["A"], HOLD))
    repo.save(_decision(signals["A"], APPROVE, reason="clean setup", provider="p"))
    repo.save(_decision(signals["B"], VETO, reason="weak volume"))
    repo.save(_decision(signals["C"], HOLD, reason="wait"))  # never decided
    repo.save(_decision(signals["D"], HOLD, failed=True, reason="timeout"))
    return signals


def test_the_export_has_the_final_decision_or_a_no_trade_for_the_undecided(
    repo: OracleDecisionRepository, session_factory: Any
) -> None:
    signals = _populate(repo, session_factory)

    export = repo.replay_export()

    sid = {name: s.signal_id for name, s in signals.items()}
    assert export[sid["A"]]["approved"] is True and export[sid["A"]]["final"] is True
    assert export[sid["A"]]["reason"] == "clean setup" and export[sid["A"]]["provider"] == "p"
    assert (export[sid["B"]]["approved"], export[sid["B"]]["verdict"]) == (False, "VETO")
    # Held until the signal lapsed, or no answer: no trade happened, and replay must say so.
    for name in ("C", "D"):
        assert (export[sid[name]]["approved"], export[sid[name]]["final"]) == (False, False)
        assert export[sid[name]]["verdict"] == "HOLD"


def test_the_backtest_replays_what_was_exported(
    repo: OracleDecisionRepository, session_factory: Any, make_settings: Any, tmp_path: Path
) -> None:
    import pandas as pd

    signals = _populate(repo, session_factory)
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(repo.replay_export()), encoding="utf-8")

    engine = BacktestEngine(
        make_settings(oracle_enabled=True),
        BacktestConfig(replay_oracle_path=str(path)),
        {"AAPL": pd.DataFrame()},
        {"AAPL": pd.DataFrame()},
    )

    assert [engine._apply_oracle(s) for s in signals.values()] == [True, False, False, False]
    unknown = _signal("MSFT", 0)
    assert engine._apply_oracle(unknown) is False  # not in the replay: the default veto


def test_the_command_writes_the_export(
    repo: OracleDecisionRepository,
    session_factory: Any,
    make_settings: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _populate(repo, session_factory)
    out = tmp_path / "export.json"

    with patch("trading_system.cli.Settings", return_value=make_settings()):
        assert main(["export-oracle-decisions", "--out", str(out)]) == 0

    exported = json.loads(out.read_text(encoding="utf-8"))
    assert len(exported) == 4
    printed = capsys.readouterr().out
    assert "Wrote 4 signals" in printed and "2 decided, 2 undecided" in printed


def test_a_failed_row_is_not_a_decision_even_if_it_somehow_says_veto(
    repo: OracleDecisionRepository, saved_signal: StrategySignal, session_factory: Any
) -> None:
    """The type forbids a failed VETO, but rows can come from older code or by hand, and a veto
    that is really an error is exactly the H1 bug. The query does not trust the verdict alone."""
    with session_factory.begin() as session:
        session.add(
            OracleDecisionModel(
                signal_id=saved_signal.signal_id,
                verdict=VETO,
                reasoning="timeout, stored as a veto by old code",
                provider="",
                failed=True,
                decision_timestamp=datetime.now(timezone.utc),
            )
        )

    assert repo.final_decision_for(saved_signal.signal_id or "") is None
