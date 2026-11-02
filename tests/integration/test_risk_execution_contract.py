"""C3: an entry the risk engine approves must be something execution can send."""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from tests.fixtures.fakes import FakeBroker
from trading_system.domain.enums import OrderType
from trading_system.domain.models import TradingState
from trading_system.execution.engine import ExecutionEngine
from trading_system.risk.daily_equity import day_marker, trading_day
from trading_system.risk.engine import RiskEngine


def _approved(settings: Any, signal: Any, account: Any) -> Any:
    state = TradingState(
        daily_start_equity=Decimal("100000"),
        daily_equity_date=day_marker(trading_day(datetime.now(timezone.utc))),
    )
    return RiskEngine(settings).evaluate_signal(
        signal=signal,
        account=account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150.00"),
        state=state,
    )


def test_risk_engine_approves_a_healthy_signal_with_a_complete_bracket(
    settings: Any, signal: Any, account: Any
) -> None:
    """Precondition for the test below: what risk emits is a limit entry with all three
    prices (150.00 plus the 0.1 % buffer, 4 % stop, 5 % target, rounded to the tick)."""
    decision, intent = _approved(settings, signal, account)

    assert decision.approved, decision.reasoning
    assert intent is not None
    assert intent.type == OrderType.LIMIT
    assert intent.limit_price == Decimal("150.15")
    assert intent.stop_price == Decimal("144.14")
    assert intent.take_profit_price == Decimal("157.65")
    assert decision.suggested_limit_price == intent.limit_price


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


def test_a_live_submission_sends_exactly_what_the_risk_engine_planned(
    settings: Any, signal: Any, account: Any
) -> None:
    """Not dry-run: the request that reaches the broker carries risk's prices unchanged and
    the quantity it sized, so the planned risk is the risk that is placed."""
    _, intent = _approved(settings, signal, account)
    assert intent is not None
    state_repo = SimpleNamespace(get_state=lambda: TradingState(), save_state=lambda _state: None)
    order_repo = SimpleNamespace(save_intent=lambda _i: None, save_broker_order=lambda _o: None)
    broker = FakeBroker()

    assert ExecutionEngine(broker, order_repo, state_repo, settings).submit_order(intent)

    assert len(broker.submitted) == 1
    sent = broker.sent_requests[0]
    assert Decimal(str(sent.limit_price)) == intent.limit_price
    assert Decimal(str(sent.stop_loss.stop_price)) == intent.stop_price
    assert Decimal(str(sent.take_profit.limit_price)) == intent.take_profit_price
    assert Decimal(str(sent.qty)) == intent.quantity
