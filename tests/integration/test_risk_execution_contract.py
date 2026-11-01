"""C3: an entry the risk engine approves must be something execution can send."""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.fakes import FakeBroker
from trading_system.domain.enums import OrderType
from trading_system.domain.models import TradingState
from trading_system.execution.engine import ExecutionEngine
from trading_system.risk.engine import RiskEngine


def _approved(settings: Any, signal: Any, account: Any) -> Any:
    state = TradingState(daily_start_equity=Decimal("100000"))
    return RiskEngine(settings).evaluate_signal(
        signal=signal,
        account=account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150.00"),
        state=state,
    )


def test_risk_engine_approves_a_healthy_signal(settings: Any, signal: Any, account: Any) -> None:
    """Precondition for the test below, and a record of what risk emits today."""
    decision, intent = _approved(settings, signal, account)

    assert decision.approved, decision.reasoning
    assert intent is not None
    assert intent.type == OrderType.MARKET


@pytest.mark.xfail(
    strict=True,
    reason="C3: risk emits a MARKET intent without limit/stop prices, execution only accepts "
    "LIMIT intents with both ('Only limit entries are supported'), so nothing is ever sent.",
)
def test_execution_engine_accepts_what_the_risk_engine_approves(
    settings: Any, signal: Any, account: Any
) -> None:
    _, intent = _approved(settings, signal, account)
    assert intent is not None

    state_repo = SimpleNamespace(get_state=lambda: TradingState(), save_state=lambda _state: None)
    order_repo = SimpleNamespace(save_intent=lambda _i: None, save_broker_order=lambda _o: None)
    engine = ExecutionEngine(
        FakeBroker(),
        order_repo,
        state_repo,
        settings,
        is_dry_run=True,
    )

    assert engine.submit_order(intent) is not None
