"""The signal context in the real loops (plan section 16.1): the live cycle and the observer journal it
AFTER their real work, and nothing it does can change or stop an order or the Oracle's questions."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from tests.integration.test_oracle_observer import (
    NOW,
    StubCandidate,
    _events,
    _run,
    _stub_signal,
)
from trading_system.domain.enums import OracleVerdict
from trading_system.runtime.orchestrator import Orchestrator
from tests.fixtures.oracles import ScriptedOracle
from trading_system.research.context import SIGNAL_CONTEXT, ContextRecorder
from trading_system.storage.models import RiskDecisionModel
from trading_system.storage.repositories import AuditRepository
from trading_system.strategy.config import RUNTIME_STRATEGY

RunCycle = Callable[..., SimpleNamespace]


def _spy() -> pd.DataFrame:
    days = pd.bdate_range(
        end=pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1),
        periods=300,
    )
    closes = 100 + pd.Series(range(300), dtype=float).to_numpy() * 0.1
    index = pd.DatetimeIndex([d + pd.Timedelta(hours=5) for d in days]).tz_localize("UTC")
    return pd.DataFrame(
        {"close": closes, "open": closes, "high": closes, "low": closes}, index=index
    )


def _recorder(session_factory: Any, source: str, groups: dict[str, str] | None = None) -> Any:
    return ContextRecorder(
        AuditRepository(session_factory),
        source,
        config=RUNTIME_STRATEGY,
        groups=groups or {"AAPL": "normal"},
        spy_loader=_spy,
        calendar=None,
    )


class Exploding:
    def record(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("a bug in the recorder")


# --------------------------------------------------------------------------- the live loop


def test_the_live_loop_journals_the_context_of_its_signal(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle(tweak=lambda cycle: setattr(cycle, "context", _recorder(session_factory, "trader")))

    events = _events(session_factory, SIGNAL_CONTEXT)
    (details,) = events
    assert details["symbol"] == "AAPL" and details["source"] == "trader"
    assert details["group"] == "normal" and details["atr_pct"] is not None
    assert details["spy_above_200d"] is True and details["earnings_known"] is False


def test_the_context_is_journalled_after_the_order_was_decided(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    seen: list[int] = []

    class Spy:
        def record(self, signals: Any, bars: Any, now: Any) -> int:
            with session_factory() as session:
                seen.append(session.query(RiskDecisionModel).count())
            return 0

    run_cycle(tweak=lambda cycle: setattr(cycle, "context", Spy()))

    assert seen == [1]  # the signal had its risk decision before anything of the context ran


def test_a_recorder_that_breaks_changes_nothing_in_the_orders(run_cycle: RunCycle) -> None:
    broken = run_cycle(tweak=lambda cycle: setattr(cycle, "context", Exploding()))
    plain = run_cycle()

    (a,), (b,) = broken.decisions, plain.decisions
    assert a.approved and b.approved and a.suggested_quantity == b.suggested_quantity


# --------------------------------------------------------------------------- the observer


def test_the_observer_journals_the_context_after_asking_the_oracle(
    make_cycle: Any, session_factory: Any
) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle)
    setup.cycle.context = _recorder(session_factory, "observer", {"MSFT": "risky"})

    report = _run(setup)

    assert report.errors == 0 and len(oracle.requests) == 1
    (details,) = _events(session_factory, SIGNAL_CONTEXT)
    assert details["symbol"] == "MSFT" and details["source"] == "observer"
    assert details["group"] == "risky"


def test_the_candidates_signals_get_a_context_with_their_own_source(
    make_cycle: Any, session_factory: Any
) -> None:
    name = "dip_in_uptrend"
    setup = make_cycle(
        oracle=ScriptedOracle(OracleVerdict.APPROVE),
        candidates=(StubCandidate(name, [_stub_signal(name)]),),
    )
    setup.cycle.context = _recorder(session_factory, "observer")

    setup.cycle.run_entries(
        NOW, __import__("trading_system.runtime.schedule", fromlist=["x"]).expected_bar_start(NOW)
    )

    sources = sorted(d["source"] for d in _events(session_factory, SIGNAL_CONTEXT))
    assert sources == ["candidate:dip_in_uptrend"] or sources == [
        "candidate:dip_in_uptrend",
        "observer",
    ]
    assert "candidate:dip_in_uptrend" in sources


def test_a_broken_recorder_leaves_the_observers_oracle_work_and_candidates_alone(
    make_cycle: Any, session_factory: Any
) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle, candidates=(StubCandidate("a", [_stub_signal("a")]),))
    setup.cycle.context = Exploding()

    report = _run(setup)

    assert len(oracle.requests) == 1
    assert report.errors == 0
    assert {d["strategy"] for d in _events(session_factory, "candidate_signal")} >= {"trend_cross"}


# --------------------------------------------------------------------------- the wiring


def test_it_is_off_by_default_and_builds_nothing_until_a_signal(
    make_settings: Any, tmp_path: Path
) -> None:
    off = Orchestrator(make_settings(), state_repo=object())
    assert off._context_recorder("trader") is None

    heartbeat = tmp_path / "var" / "heartbeat.json"
    on = Orchestrator(
        make_settings(
            signal_context_enabled=True,
            heartbeat_path=str(heartbeat),
            research_database_url=f"sqlite:///{(tmp_path / 'var' / 'research.db').as_posix()}",
        ),
        state_repo=object(),
    )
    recorder = on._context_recorder("paper3")

    assert isinstance(recorder, ContextRecorder) and recorder.source == "paper3"
    assert recorder.groups["AXTI"] == "risky" and recorder.groups["NKE"] == "normal"
    assert recorder.calendar is not None
    assert recorder.calendar.cache_path == tmp_path / "var" / "earnings_calendar.json"
    assert not (tmp_path / "var").exists()  # no file or folder is made before the first signal
