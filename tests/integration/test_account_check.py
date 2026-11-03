"""`reconcile` and `resume-entries` look at the account again before they say it is fine
(Phase 6.5). A halt exists because something was wrong; lifting it without looking restarts
the thing that was wrong."""

import datetime as dt
from typing import Any
from unittest.mock import patch

import pytest
import requests
from alpaca.common.exceptions import APIError

from tests.fixtures.fakes import FakeBroker
from tests.unit.test_execution_engine import _position, _stop
from trading_system.cli import main
from trading_system.domain.enums import OrderSide, OrderStatus, OrderType
from trading_system.domain.models import BrokerOrder, TradingState
from trading_system.runtime.account_check import check_account, format_account_check
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.storage.repositories import OrderRepository, StateRepository

NOW = dt.datetime.now(dt.timezone.utc)


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> FakeBroker:
    fake = FakeBroker()
    monkeypatch.setattr("trading_system.broker.alpaca_broker.AlpacaBroker", lambda _settings: fake)
    return fake


def _halt(
    session_factory: Any, reason: str = "Unprotected open position(s): MSFT"
) -> StateRepository:
    repo = StateRepository(session_factory)
    repo.save_state(TradingState(is_trading_halted=True, halt_reason=reason))
    return repo


def _not_found() -> APIError:
    response = requests.Response()
    response.status_code = 404
    return APIError(
        '{"code": 40410000, "message": "order not found"}', requests.HTTPError(response=response)
    )


def _sent_order(order_id: str = "ord-1") -> BrokerOrder:
    old = NOW - dt.timedelta(minutes=10)
    return BrokerOrder(
        client_order_id=order_id,
        broker_id="broker-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        status=OrderStatus.SUBMITTED,
        quantity=10,
        filled_quantity=0,
        created_at=old,
        updated_at=old,
    )


def _check(settings: Any) -> Any:
    return Orchestrator(settings).reconcile_now()


# --- the check ------------------------------------------------------------------------------


def test_a_quiet_account_agrees_with_our_records(settings: Any, broker: FakeBroker) -> None:
    check = _check(settings)

    assert check.ok and check.problems == [] and check.positions == 0
    assert "No problems" in format_account_check(check)


def test_a_position_without_a_stop_is_a_problem_and_is_named(
    settings: Any, broker: FakeBroker
) -> None:
    broker.positions = [_position("MSFT", 10)]

    check = _check(settings)

    assert not check.ok and check.positions == 1
    assert check.problems == ["MSFT: 10 shares, only 0 covered by an active stop"]
    assert "PROBLEMS:" in format_account_check(check)


def test_a_position_with_its_stop_is_fine(settings: Any, broker: FakeBroker) -> None:
    broker.positions = [_position("MSFT", 10)]
    broker.open_orders = [_stop("MSFT", 10)]

    check = _check(settings)

    assert check.ok and check.positions == 1 and check.open_orders == 1


def test_a_broker_that_cannot_be_read_is_a_problem_not_a_pass(
    settings: Any, broker: FakeBroker
) -> None:
    broker.positions_error = ConnectionError("down")

    check = _check(settings)

    assert not check.ok
    assert check.problems == ["the broker could not be read: ConnectionError: down"]


def test_an_order_we_sent_and_the_broker_does_not_know_is_a_problem(
    settings: Any, broker: FakeBroker, session_factory: Any
) -> None:
    OrderRepository(session_factory).save_broker_order(_sent_order())

    def unknown(_client_order_id: str) -> Any:
        raise _not_found()

    broker.get_order_by_client_id = unknown  # type: ignore[method-assign]

    check = _check(settings)

    assert not check.ok
    assert check.unresolved == [("ord-1", "AAPL")]
    assert "ord-1 (AAPL) was sent, but the broker does not know it" in check.problems[0]


def test_an_order_that_could_not_be_looked_up_is_a_problem(
    settings: Any, broker: FakeBroker, session_factory: Any
) -> None:
    OrderRepository(session_factory).save_broker_order(_sent_order())
    # the fake raises LookupError, which is not a 404: a failed lookup, not "unknown"

    check = _check(settings)

    assert not check.ok and "1 order(s) could not be looked up" in check.problems


def test_the_check_trades_nothing_halts_nothing_and_closes_nothing(
    settings: Any, broker: FakeBroker, session_factory: Any
) -> None:
    broker.positions = [_position("MSFT", 10)]

    _check(settings)

    assert broker.submitted == [] and broker.cancelled == []
    assert not StateRepository(session_factory).get_state().is_trading_halted


def test_check_account_also_works_on_its_own(
    settings: Any, broker: FakeBroker, session_factory: Any
) -> None:
    cycle = Orchestrator(settings).build_cycle(False)
    broker.positions = [_position("A", 1), _position("B", 2)]

    check = check_account(cycle.broker, cycle.execution, cycle.order_repo, NOW)

    assert [gap.symbol for gap in check.unprotected] == ["A", "B"]


# --- reconcile --------------------------------------------------------------------------------


def test_reconcile_exits_0_when_the_account_agrees(
    settings: Any, broker: FakeBroker, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["reconcile"]) == 0

    assert "No problems" in capsys.readouterr().out


def test_reconcile_exits_1_and_lists_the_problems(
    settings: Any, broker: FakeBroker, capsys: pytest.CaptureFixture[str]
) -> None:
    broker.positions = [_position("MSFT", 10)]

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["reconcile"]) == 1

    assert "MSFT: 10 shares" in capsys.readouterr().out


# --- resume-entries ---------------------------------------------------------------------------


def test_a_halt_is_lifted_when_the_account_is_fine(
    settings: Any, broker: FakeBroker, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _halt(session_factory)

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["resume-entries"]) == 0

    assert not repo.get_state().is_trading_halted
    assert "Entries resumed." in capsys.readouterr().out


def test_a_halt_stays_when_a_position_has_no_stop(
    settings: Any, broker: FakeBroker, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _halt(session_factory)
    broker.positions = [_position("MSFT", 10)]

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["resume-entries"]) == 1

    state = repo.get_state()
    assert state.is_trading_halted and "Unprotected" in (state.halt_reason or "")
    out = capsys.readouterr().out
    assert "Entries stay halted." in out and "MSFT: 10 shares" in out and "--force" in out


def test_force_lifts_the_halt_over_the_problems_and_says_so(
    settings: Any, broker: FakeBroker, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _halt(session_factory)
    broker.positions = [_position("MSFT", 10)]

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["resume-entries", "--force"]) == 0

    assert not repo.get_state().is_trading_halted
    assert "Forced: the problems above stand." in capsys.readouterr().out


def test_nothing_is_checked_when_entries_are_not_halted(
    settings: Any, broker: FakeBroker, capsys: pytest.CaptureFixture[str]
) -> None:
    broker.positions_error = RuntimeError("the broker must not be asked")

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["resume-entries"]) == 0

    assert "not halted; nothing to resume" in capsys.readouterr().out


def test_the_halt_stays_when_the_broker_cannot_be_read(
    settings: Any, broker: FakeBroker, session_factory: Any
) -> None:
    repo = _halt(session_factory)
    broker.positions_error = ConnectionError("down")

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["resume-entries"]) == 1

    assert repo.get_state().is_trading_halted
