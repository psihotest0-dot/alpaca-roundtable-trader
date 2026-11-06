"""The order of risk and the Oracle in one signal (task 2.2, T5; action plan 0.4 item 17).

With an Oracle: risk takes a cheap first look, then the model is asked, then risk decides on a
fresh snapshot of the account. So the model is never asked about a trade that risk would refuse,
what is on record in `risk_decisions` is one row per signal (the refusal of the first look, or the
final evaluation), and a change in the account while the model thinks still counts.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


from tests.fixtures.oracle_backend import GOOD_TOKEN, FakeOracleBackend, answer
from tests.fixtures.oracles import ScriptedOracle
from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision
from trading_system.oracle import OracleRequest, create_oracle
from trading_system.storage.models import BrokerOrderModel, RiskDecisionModel
from trading_system.storage.repositories import OracleDecisionRepository

RunCycle = Callable[..., SimpleNamespace]
APPROVE = OracleVerdict.APPROVE


def _risk_rows(session_factory: Any) -> list[RiskDecisionModel]:
    with session_factory() as session:
        return list(session.query(RiskDecisionModel).order_by(RiskDecisionModel.id).all())


def _orders(session_factory: Any) -> int:
    with session_factory() as session:
        return int(session.query(BrokerOrderModel).count())


def test_without_an_oracle_risk_is_evaluated_once_as_before(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    cycle = run_cycle(oracle=None)

    assert len(cycle.decisions) == 1 and cycle.report.submitted == 1
    assert len(_risk_rows(session_factory)) == 1


def test_a_trade_risk_refuses_is_never_put_to_the_model(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(APPROVE)

    cycle = run_cycle(oracle=oracle, equity="97000", last_equity="100000")  # -3 %: the loss guard

    assert oracle.requests == []  # no call spent on a trade that cannot be made
    assert cycle.report.submitted == 0 and cycle.report.held == 0
    assert _orders(session_factory) == 0
    (row,) = _risk_rows(session_factory)  # the refusal is recorded, once
    assert not row.approved and "Daily loss guard breached" in str(row.reasoning)
    assert len(cycle.decisions) == 1  # and nobody evaluated it a second time


def test_the_backend_is_not_called_for_a_trade_risk_refuses_not_even_in_shadow_mode(
    run_cycle: RunCycle, make_settings: Callable[..., Any], session_factory: Any
) -> None:
    with FakeOracleBackend(answer("APPROVE")) as backend:
        settings = make_settings(
            oracle_enabled=True,
            oracle_provider="backend",
            oracle_mode="shadow",
            oracle_backend_url=backend.url,
            oracle_backend_token=GOOD_TOKEN,
        )
        oracle = create_oracle(settings, OracleDecisionRepository(session_factory))
        cycle = run_cycle(oracle=oracle, equity="97000", last_equity="100000")

    assert backend.received == []
    assert cycle.report.submitted == 0 and len(_risk_rows(session_factory)) == 1


def test_a_trade_risk_allows_is_evaluated_twice_but_recorded_once(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(APPROVE)

    cycle = run_cycle(oracle=oracle)

    assert len(oracle.requests) == 1 and cycle.report.submitted == 1
    assert len(cycle.decisions) == 2  # the first look, and the final evaluation
    (row,) = _risk_rows(session_factory)  # only the final one is on record
    assert row.approved


def test_the_model_is_told_the_account_as_the_first_look_saw_it(run_cycle: RunCycle) -> None:
    snapshots: list[int] = []

    def count_snapshots(cycle: Any) -> None:
        original = cycle._snapshot

        def counting(now: Any) -> Any:
            snapshots.append(1)
            return original(now)

        cycle._snapshot = counting

    oracle = ScriptedOracle(APPROVE)
    run_cycle(oracle=oracle, tweak=count_snapshots)
    with_oracle = len(snapshots)
    snapshots.clear()
    run_cycle(oracle=None, tweak=count_snapshots)

    assert with_oracle == 2  # the first look (the model's context comes from it) and the final
    assert len(snapshots) == 1  # without an Oracle nothing else is added


def test_the_final_evaluation_sees_the_account_as_it_is_after_the_model_has_answered(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    """The model is slow. If the account moved while it thought (here: a 10 % drop, which breaks
    the daily loss guard), the order is judged on the account of now, not of before."""

    class SlowAndTheMarketFell:
        name = "slow"

        def __init__(self, broker: Any) -> None:
            self.broker = broker
            self.asked = 0

        def evaluate(self, request: OracleRequest) -> OracleDecision:
            self.asked += 1
            self.broker.equity = "90000"
            return OracleDecision(
                signal_id=request.signal.signal_id or "", verdict=APPROVE, provider=self.name
            )

    holder: dict[str, SlowAndTheMarketFell] = {}

    def install(cycle: Any) -> None:
        holder["oracle"] = SlowAndTheMarketFell(cycle.broker)
        cycle.oracle = holder["oracle"]

    cycle = run_cycle(oracle=None, last_equity="100000", tweak=install)

    assert holder["oracle"].asked == 1
    assert cycle.report.submitted == 0 and _orders(session_factory) == 0
    (row,) = _risk_rows(session_factory)  # the first look approved (not recorded); this refused
    assert not row.approved and "Daily loss guard breached" in str(row.reasoning)
    assert [d.approved for d in cycle.decisions] == [True, False]


def test_a_veto_after_an_approving_first_look_leaves_no_risk_record(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    cycle = run_cycle(oracle=ScriptedOracle(OracleVerdict.VETO))

    assert cycle.report.submitted == 0 and _risk_rows(session_factory) == []


def test_a_signal_held_by_the_oracle_gets_a_first_look_each_time_it_is_asked_again(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    from datetime import datetime, timezone

    oracle = ScriptedOracle(OracleVerdict.HOLD, APPROVE)
    cycle = run_cycle(oracle=oracle)
    assert cycle.report.held == 1 and len(cycle.decisions) == 1

    ordered = cycle.cycle.revisit_held(datetime.now(timezone.utc))

    assert ordered == 1 and len(oracle.requests) == 2
    assert len(cycle.decisions) == 3  # look, then (re-asked) look and final
    assert len(_risk_rows(session_factory)) == 1  # one row for the signal all the same
