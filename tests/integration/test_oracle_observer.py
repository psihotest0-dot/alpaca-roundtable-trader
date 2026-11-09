"""The Oracle observer (runtime/observe.py): the live strategy over more stocks, each signal put to
the Oracle, and no order ever. Real cycle code, real SQLite, a fake Alpaca; the observation store is
filled with bars that end on a cross-up, so the strategy really signals."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.bars import make_bars
from tests.fixtures.fakes import FakeBroker, FakeStockClient
from tests.fixtures.oracles import ScriptedOracle
from trading_system.config import Settings
from trading_system.data.alpaca_data import AlpacaData
from trading_system.data.universe import (
    MARKET_SYMBOL,
    observation_universe,
    observed_symbols,
    research_universe,
)
from trading_system.domain.enums import OracleVerdict, SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.research.candidates import CANDIDATE_EVENT, CYCLE_EVENT
from trading_system.strategy.base import BaseStrategy
from trading_system.runtime.observe import OBSERVE_EVENT, ObservationCycle
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.runtime.runner import Runner
from trading_system.runtime.schedule import bar_boundary, expected_bar_start
from trading_system.storage.models import (
    BrokerOrderModel,
    OracleDecisionModel,
    RiskDecisionModel,
    StrategySignalModel,
)
from trading_system.storage.repositories import (
    AuditRepository,
    OracleDecisionRepository,
    SignalRepository,
    StateRepository,
)
from trading_system.storage.research import open_observe_store
from trading_system.oracle.recording import RecordingOracle
from trading_system.oracle.shadow import ShadowOracle
from trading_system.oracle.backend import BackendOracle

NOW = datetime.now(timezone.utc)


@pytest.fixture
def observe_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'observe.db').as_posix()}"


def _fill(store: Any, symbols: list[str], now: datetime) -> None:
    bars_15m, bars_1h = make_bars(now)
    for symbol in symbols:
        store.replace(symbol, "15m", bars_15m, adjustment="all", fetched_at=now)
        store.replace(symbol, "1h", bars_1h, adjustment="all", fetched_at=now)


class Setup:
    def __init__(self, cycle: ObservationCycle, broker: FakeBroker, updates: list[Any]) -> None:
        self.cycle, self.broker, self.updates = cycle, broker, updates


@pytest.fixture
def make_cycle(make_settings: Any, observe_url: str, db_url: str, session_factory: Any) -> Any:
    def _make(
        *,
        oracle: Any = None,
        observed: tuple[str, ...] = ("MSFT",),
        stored: tuple[str, ...] = ("MSFT", "AAPL"),
        update_error: Exception | None = None,
        live_age_seconds: float = 0.0,
        settings: Settings | None = None,
        fill: bool = True,
        candidates: tuple[Any, ...] = (),
    ) -> Setup:
        settings = settings or make_settings(watchlist="AAPL")
        store = open_observe_store(observe_url, db_url)
        if fill:
            _fill(store, list(stored), NOW)
        bars_15m, bars_1h = make_bars(NOW)
        data = AlpacaData(settings)
        data.client = FakeStockClient(  # type: ignore[assignment]
            {s: bars_15m for s in stored},
            {s: bars_1h for s in stored},
            live_age_seconds=live_age_seconds,
        )
        updates: list[Any] = []

        def updater(*args: Any, **kwargs: Any) -> None:
            updates.append((args, kwargs))
            if update_error is not None:
                raise update_error

        broker = FakeBroker()
        cycle = ObservationCycle(
            settings=settings,
            broker=broker,
            data=data,
            store=store,
            universe=list(stored),
            observed=list(observed),
            signal_repo=SignalRepository(session_factory),
            state_repo=StateRepository(session_factory),
            audit_repo=AuditRepository(session_factory),
            strategy_config=__import__(
                "trading_system.strategy.config", fromlist=["RUNTIME_STRATEGY"]
            ).RUNTIME_STRATEGY,
            oracle=oracle,
            updater=updater,
            candidates=candidates,
        )
        return Setup(cycle, broker, updates)

    return _make


def _count(session_factory: Any, model: Any) -> int:
    with session_factory() as session:
        return int(session.query(model).count())


def _run(setup: Setup) -> Any:
    return setup.cycle.run_entries(NOW, expected_bar_start(NOW))


# ------------------------------------------------------------------------------ what it does


def test_a_signal_on_an_observed_symbol_is_put_to_the_oracle_and_recorded(
    make_cycle: Any, session_factory: Any
) -> None:
    oracle = ScriptedOracle(OracleVerdict.VETO)
    setup = make_cycle(oracle=oracle)

    report = _run(setup)

    assert report.signals == 1 and report.errors == 0 and not report.retry_later
    (request,) = oracle.requests
    assert request.signal.symbol == "MSFT" and "close" in request.technicals
    assert request.live_price is not None and request.context["max_open_positions"] == 5
    assert _count(session_factory, StrategySignalModel) == 1  # the answer's record points at it


def test_it_places_no_order_and_makes_no_risk_decision(
    make_cycle: Any, session_factory: Any
) -> None:
    setup = make_cycle(oracle=ScriptedOracle(OracleVerdict.APPROVE))

    _run(setup)

    assert setup.broker.submitted == [] and setup.broker.sent_requests == []
    assert _count(session_factory, BrokerOrderModel) == 0
    assert _count(session_factory, RiskDecisionModel) == 0


def test_it_cannot_place_an_order_by_construction(make_cycle: Any) -> None:
    setup = make_cycle(oracle=ScriptedOracle())
    cycle = setup.cycle

    assert cycle.risk_engine is None and cycle.execution is None and cycle.order_repo is None
    assert cycle.run_housekeeping(NOW) is None and cycle.revisit_held(NOW) == 0
    with pytest.raises(AttributeError):  # the ordinary path to an order has nothing to stand on
        cycle._enter(None, None, {}, {}, NOW)  # type: ignore[arg-type]


def test_the_symbols_of_the_watchlist_are_left_to_the_live_loop(
    make_cycle: Any,
) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE, OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle, observed=("MSFT",), stored=("MSFT", "AAPL"))

    _run(setup)

    assert [r.signal.symbol for r in oracle.requests] == ["MSFT"]  # AAPL has bars, and a signal


def test_the_bars_are_brought_up_to_date_first_for_the_whole_universe(make_cycle: Any) -> None:
    setup = make_cycle(oracle=ScriptedOracle(OracleVerdict.APPROVE))

    _run(setup)

    ((args, kwargs),) = setup.updates
    assert args[2] == ["MSFT", "AAPL"] and args[3] == ["15m", "1h"]  # the universe, both sizes
    assert kwargs["now"] == NOW and (NOW - kwargs["start"]).days == 180


def test_it_journals_as_observe_cycle_and_not_as_the_live_loop(
    make_cycle: Any, session_factory: Any
) -> None:
    setup = make_cycle(oracle=ScriptedOracle(OracleVerdict.VETO))

    _run(setup)

    audit = AuditRepository(session_factory)
    start, end = (
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(2100, 1, 1, tzinfo=timezone.utc),
    )
    ((_, details),) = audit.events(OBSERVE_EVENT, start, end)
    assert details["signals"] == 1 and details["asked"] == 1 and details["no_price"] == 0
    assert audit.events("entries_cycle", start, end) == []  # the daily report stays the live loop's


# ---------------------------------------------------------------------- when it cannot go on


def test_a_failed_update_is_tried_again_and_nobody_is_asked(make_cycle: Any) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle, update_error=ConnectionError("alpaca is down"))

    report = _run(setup)

    assert report.retry_later and report.skipped == "bars could not be updated"
    assert oracle.requests == []


def test_a_bar_that_has_not_arrived_is_tried_again(make_cycle: Any) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle)

    later = expected_bar_start(NOW).replace(minute=0) + __import__("datetime").timedelta(hours=2)
    report = setup.cycle.run_entries(NOW, later)  # the store ends before the bar asked for

    assert report.retry_later and report.skipped == "bar not available yet"
    assert oracle.requests == []


def test_an_empty_store_is_tried_again(make_cycle: Any) -> None:
    setup = make_cycle(oracle=ScriptedOracle(), fill=False)

    report = _run(setup)

    assert report.retry_later and report.skipped == "bar not available yet"


def test_a_stale_price_means_the_signal_is_not_asked_about(
    make_cycle: Any, session_factory: Any
) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle, live_age_seconds=3600)  # the last trade is an hour old

    report = _run(setup)

    assert report.signals == 1 and report.errors == 0 and oracle.requests == []
    audit = AuditRepository(session_factory)
    ((_, details),) = audit.events(
        OBSERVE_EVENT,
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(2100, 1, 1, tzinfo=timezone.utc),
    )
    assert details["no_price"] == 1 and details["asked"] == 0


def test_an_oracle_that_raises_does_not_stop_the_pass(make_cycle: Any) -> None:
    setup = make_cycle(oracle=ScriptedOracle(ConnectionError("lemonade is down")))

    report = _run(setup)

    assert report.signals == 1 and report.errors == 0  # _ask_oracle turns it into a failed HOLD


def test_one_broken_signal_is_counted_and_the_others_go_on(make_cycle: Any) -> None:
    setup = make_cycle(
        oracle=ScriptedOracle(OracleVerdict.APPROVE, OracleVerdict.APPROVE),
        observed=("MSFT", "JPM"),
        stored=("MSFT", "JPM", "AAPL"),
    )
    original = setup.cycle._snapshot
    calls: list[int] = []

    def flaky(now: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("alpaca hiccup")
        return original(now)

    setup.cycle._snapshot = flaky  # type: ignore[method-assign]

    report = _run(setup)

    assert report.signals == 2 and report.errors == 1


def test_the_runner_drives_it_like_the_live_loop(make_cycle: Any) -> None:
    oracle = ScriptedOracle(OracleVerdict.VETO)
    setup = make_cycle(oracle=oracle)
    waits: list[float] = []
    # The Runner looks for entries only 30 s after a quarter hour; NOW is the clock at import, which
    # can fall in those 30 s (the test then failed about one run in thirty). Same quarter, safely later.
    later = bar_boundary(NOW) + timedelta(seconds=45)
    runner = Runner(setup.cycle, setup.broker, now=lambda: later, wait=waits.append)

    runner.tick()

    assert len(oracle.requests) == 1 and waits  # asked once, then it waited for the next bar


# ------------------------------------------------------------- the universe and the wiring


def test_the_observed_symbols_are_the_universe_less_the_watchlist_and_the_market() -> None:
    watchlist = ["AAPL", "MSFT", "OPEN"]  # OPEN is not in the research universe
    universe = observation_universe(watchlist)
    observed = observed_symbols(watchlist)

    assert universe == sorted(set(research_universe()) | set(watchlist))
    assert MARKET_SYMBOL in universe and "OPEN" in universe
    assert MARKET_SYMBOL not in observed and not set(watchlist) & set(observed)
    assert set(observed) | set(watchlist) | {MARKET_SYMBOL} == set(universe)


def test_with_the_real_watchlist_it_watches_over_a_hundred_more_stocks() -> None:
    watchlist = Settings(
        _env_file=None, alpaca_api_key="k", alpaca_secret_key="s"
    ).watchlist_symbols

    assert len(observation_universe(watchlist)) >= 150
    assert len(observed_symbols(watchlist)) >= 120


def test_the_orchestrator_builds_the_observer_without_a_way_to_order(
    make_settings: Any, observe_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(
        watchlist="AAPL,MSFT",
        observe_database_url=observe_url,
        oracle_enabled=True,
        oracle_provider="backend",
        oracle_backend_token="t" * 48,
    )
    monkeypatch.setattr("trading_system.broker.alpaca_broker.AlpacaBroker", lambda _s: object())

    cycle = Orchestrator(settings).build_observation_cycle()

    assert isinstance(cycle, ObservationCycle)
    assert cycle.risk_engine is None and cycle.execution is None and cycle.order_repo is None
    assert "MSFT" not in cycle.observed and "AAPL" not in cycle.observed
    assert "MSFT" in cycle.universe and MARKET_SYMBOL in cycle.universe
    # the same stack as the live loop: the record keeps the model's own word
    assert isinstance(cycle.oracle, ShadowOracle)
    assert isinstance(cycle.oracle.provider, RecordingOracle)
    assert isinstance(cycle.oracle.provider.provider, BackendOracle)


def test_the_answers_it_gets_are_recorded_with_the_models_own_word(
    make_settings: Any, make_cycle: Any, session_factory: Any
) -> None:
    from tests.fixtures.oracle_backend import GOOD_TOKEN, FakeOracleBackend, answer
    from trading_system.oracle import create_oracle

    with FakeOracleBackend(answer("VETO", "Новината е лоша.", request_id="req-obs")) as backend:
        settings = make_settings(
            watchlist="AAPL",
            oracle_enabled=True,
            oracle_provider="backend",
            oracle_mode="shadow",
            oracle_backend_url=backend.url,
            oracle_backend_token=GOOD_TOKEN,
        )
        oracle = create_oracle(settings, OracleDecisionRepository(session_factory))
        setup = make_cycle(oracle=oracle, settings=settings)

        _run(setup)

    with session_factory() as session:
        (row,) = session.query(OracleDecisionModel).all()
    assert row.verdict is OracleVerdict.VETO and row.mode == "shadow"
    assert (row.request_id, row.reasoning) == ("req-obs", "Новината е лоша.")
    assert backend.received[0].json["signal"]["symbol"] == "MSFT"
    assert Decimal(backend.received[0].json["live_price"]) > 0


def test_the_observer_refuses_to_start_when_there_is_nobody_to_ask(
    make_settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from unittest.mock import patch

    from trading_system.cli import main

    for overrides in ({}, {"oracle_enabled": True}):  # off; on but with the null provider
        settings = make_settings(**overrides)
        with pytest.raises(RuntimeError, match="nobody to ask"):
            Orchestrator(settings).run_observation(handle_signals=False)
        with patch("trading_system.cli.Settings", return_value=settings):
            assert main(["observe"]) == 1
        assert "nobody to ask" in capsys.readouterr().out


# ------------------------------------------------- the strategy candidates, forward in time
# (docs/strategy-research-plan.md, section 12.1; judged by research/candidates.py)


class StubCandidate(BaseStrategy):
    """A candidate that says what it is told to, or breaks."""

    def __init__(self, name: str, signals: list[StrategySignal], error: Exception | None = None):
        self.name, self._signals, self._error = name, signals, error

    @property
    def version(self) -> str:
        return f"{self.name}-stub"

    def generate_signals(self, data_15m: Any, data_1h: Any, eval_time: Any) -> list[StrategySignal]:
        if self._error is not None:
            raise self._error
        return list(self._signals)


def _stub_signal(strategy: str, symbol: str = "MSFT") -> StrategySignal:
    return StrategySignal(
        version=f"{strategy}-stub",
        symbol=symbol,
        timeframe="15m",
        closed_bar_timestamp=expected_bar_start(NOW),
        direction=SignalDirection.LONG,
    )


def _events(session_factory: Any, event: str) -> list[dict[str, Any]]:
    return [
        d
        for _, d in AuditRepository(session_factory).events(
            event,
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(2100, 1, 1, tzinfo=timezone.utc),
        )
    ]


def test_the_candidates_and_the_control_are_journalled_and_only_the_control_is_asked(
    make_cycle: Any, session_factory: Any
) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    name = "trend_cross_session"
    setup = make_cycle(oracle=oracle, candidates=(StubCandidate(name, [_stub_signal(name)]),))

    _run(setup)

    assert (
        len(oracle.requests) == 1
    )  # the live strategy's signal; the candidate is not put to the model
    journal = _events(session_factory, CANDIDATE_EVENT)
    assert sorted((d["strategy"], d["symbol"]) for d in journal) == [
        ("trend_cross", "MSFT"),  # the control: the live strategy over the same symbols and bars
        (name, "MSFT"),
    ]
    assert {d["strategy"]: d["version"] for d in journal}[name] == f"{name}-stub"
    ((_, cycle),) = [(0, d) for d in _events(session_factory, CYCLE_EVENT)]
    assert cycle["errors"] == 0 and cycle["trend_cross"] == 1 and cycle[name] == 1
    # the candidate's signal is not in the signals table the model's answers point at
    assert _count(session_factory, StrategySignalModel) == 1
    assert setup.broker.submitted == [] and _events(session_factory, "entries_cycle") == []


def test_a_candidate_that_breaks_stops_neither_the_others_nor_the_oracle(
    make_cycle: Any, session_factory: Any
) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    broken = StubCandidate("trend_cross_session", [], error=RuntimeError("a bug"))
    fine = StubCandidate("trend_cross_session_late", [_stub_signal("trend_cross_session_late")])
    setup = make_cycle(oracle=oracle, candidates=(broken, fine))

    report = _run(setup)

    assert report.errors == 0 and len(oracle.requests) == 1
    assert {d["strategy"] for d in _events(session_factory, CANDIDATE_EVENT)} == {
        "trend_cross",
        "trend_cross_session_late",
    }
    ((cycle),) = _events(session_factory, CYCLE_EVENT)
    assert cycle["errors"] == 1 and cycle["trend_cross_session"] == -1


def test_a_broken_journal_leaves_the_oracles_work_alone(make_cycle: Any) -> None:
    oracle = ScriptedOracle(OracleVerdict.APPROVE)
    setup = make_cycle(oracle=oracle, candidates=(StubCandidate("trend_cross_session", []),))
    setup.cycle.audit_repo.record = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[union-attr,method-assign]
        RuntimeError("the database is locked")
    )

    report = _run(setup)

    assert report.errors == 0 and len(oracle.requests) == 1


def test_without_candidates_nothing_is_journalled_for_them(
    make_cycle: Any, session_factory: Any
) -> None:
    setup = make_cycle(oracle=ScriptedOracle(OracleVerdict.APPROVE))

    _run(setup)

    assert _events(session_factory, CANDIDATE_EVENT) == []
    assert _events(session_factory, CYCLE_EVENT) == []


def test_a_bar_that_has_not_arrived_runs_no_candidate(
    make_cycle: Any, session_factory: Any
) -> None:
    setup = make_cycle(
        oracle=ScriptedOracle(OracleVerdict.APPROVE),
        candidates=(StubCandidate("trend_cross_session", [_stub_signal("trend_cross_session")]),),
    )
    later = expected_bar_start(NOW).replace(minute=0) + timedelta(hours=2)

    report = setup.cycle.run_entries(NOW, later)

    assert report.retry_later
    assert _events(session_factory, CANDIDATE_EVENT) == []
    assert _events(session_factory, CYCLE_EVENT) == []  # a pass that did not run is not a day run


def test_the_real_candidates_run_on_the_observers_bars_without_an_error(
    make_cycle: Any, session_factory: Any
) -> None:
    from trading_system.research.candidates import CANDIDATES
    from trading_system.strategy.registry import get_strategy

    setup = make_cycle(
        oracle=ScriptedOracle(OracleVerdict.APPROVE),
        candidates=tuple(get_strategy(name) for name in CANDIDATES),
    )

    _run(setup)

    ((cycle),) = _events(session_factory, CYCLE_EVENT)
    assert cycle["errors"] == 0 and all(cycle[name] >= 0 for name in CANDIDATES)


def test_the_orchestrator_gives_the_observer_every_recorded_candidate(
    make_settings: Any, observe_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_system.research.candidates import RECORDED

    settings = make_settings(
        watchlist="AAPL,MSFT",
        observe_database_url=observe_url,
        oracle_enabled=True,
        oracle_provider="backend",
        oracle_backend_token="t" * 48,
    )
    monkeypatch.setattr("trading_system.broker.alpaca_broker.AlpacaBroker", lambda _s: object())

    cycle = Orchestrator(settings).build_observation_cycle()

    assert [c.name for c in cycle.candidates] == list(RECORDED)
    # The pair of section 12.1 and the one added by section 14, in this order.
    assert list(RECORDED) == ["trend_cross_session", "trend_cross_session_late", "dip_in_uptrend"]


def test_the_candidates_are_counted_for_grafana(make_cycle: Any) -> None:
    from prometheus_client import REGISTRY

    def count(strategy: str) -> float:
        return (
            REGISTRY.get_sample_value("alpaca_candidate_signals_total", {"strategy": strategy})
            or 0.0
        )

    name = "trend_cross_session"
    candidate_before, control_before = count(name), count("trend_cross")
    setup = make_cycle(
        oracle=ScriptedOracle(OracleVerdict.APPROVE),
        candidates=(StubCandidate(name, [_stub_signal(name)]),),
    )

    _run(setup)

    assert count(name) == candidate_before + 1 and count("trend_cross") == control_before + 1
