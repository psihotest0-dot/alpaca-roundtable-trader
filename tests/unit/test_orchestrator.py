"""Kill-switch bookkeeping of the Orchestrator, against an in-memory state repository.

The loop itself is covered by tests/integration/test_runtime_cycle.py.
"""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trading_system.config import Settings
from trading_system.domain.models import TradingState
from trading_system.runtime.orchestrator import Orchestrator


class InMemoryStateRepo:
    def __init__(self, state: TradingState | None = None) -> None:
        self.state = state

    def get_state(self) -> TradingState | None:
        return self.state

    def save_state(self, state: TradingState) -> None:
        self.state = state

    def save_daily_start_equity(self, equity: Decimal, equity_date: datetime) -> None:
        self.state = (self.state or TradingState()).model_copy(
            update={"daily_start_equity": equity, "daily_equity_date": equity_date}
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")


def test_pause_entries_halts_with_a_reason(settings: Settings) -> None:
    repo = InMemoryStateRepo()

    Orchestrator(settings, state_repo=repo).pause_entries()

    assert repo.state is not None
    assert repo.state.is_trading_halted is True
    assert repo.state.halt_reason == "Manual CLI pause"


def test_pause_entries_keeps_the_rest_of_the_state(settings: Settings) -> None:
    repo = InMemoryStateRepo(TradingState(daily_start_equity=None, halt_reason="old"))

    Orchestrator(settings, state_repo=repo).pause_entries()

    assert repo.state is not None
    assert repo.state.is_trading_halted is True
    assert repo.state.halt_reason == "Manual CLI pause"


def test_resume_entries_clears_the_halt(settings: Settings) -> None:
    repo = InMemoryStateRepo(TradingState(is_trading_halted=True, halt_reason="Manual CLI pause"))

    Orchestrator(settings, state_repo=repo).resume_entries()

    assert repo.state is not None
    assert repo.state.is_trading_halted is False
    assert repo.state.halt_reason is None


def test_status_reports_an_uninitialised_system(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    Orchestrator(settings, state_repo=InMemoryStateRepo()).status()

    assert "uninitialized" in capsys.readouterr().out


def test_status_reports_the_halt_reason(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = InMemoryStateRepo(TradingState(is_trading_halted=True, halt_reason="Manual CLI pause"))

    Orchestrator(settings, state_repo=repo).status()

    out = capsys.readouterr().out
    assert "Trading Halted: True" in out
    assert "Manual CLI pause" in out


class _Broker:
    def __init__(self, equity: str | None = "100000", last_equity: str | None = None):
        self.equity, self.last_equity = equity, last_equity

    def get_account(self) -> SimpleNamespace:
        if self.equity is None:
            raise ConnectionError("broker unreachable")
        return SimpleNamespace(equity=self.equity, last_equity=self.last_equity)


def test_the_daily_equity_is_recorded_on_the_first_cycle_of_the_day(settings: Settings) -> None:
    repo = InMemoryStateRepo(TradingState())

    state = Orchestrator(settings, state_repo=repo)._with_daily_equity(
        _Broker("100000"), repo.state
    )

    assert state.daily_start_equity == Decimal("100000")
    assert repo.state is not None and repo.state.daily_start_equity == Decimal("100000")


def test_a_second_call_the_same_day_changes_nothing(settings: Settings) -> None:
    repo = InMemoryStateRepo(TradingState())
    orchestrator = Orchestrator(settings, state_repo=repo)
    first = orchestrator._with_daily_equity(_Broker("100000"), repo.state)

    second = orchestrator._with_daily_equity(_Broker("90000"), first)

    assert second is first
    assert second.daily_start_equity == Decimal("100000")


def test_a_broker_failure_is_not_fatal_and_leaves_the_state_alone(settings: Settings) -> None:
    """Fail closed: with no baseline from today, risk rejects every entry."""
    repo = InMemoryStateRepo(TradingState(is_trading_halted=True, halt_reason="manual"))
    before = repo.state

    state = Orchestrator(settings, state_repo=repo)._with_daily_equity(_Broker(None), before)

    assert state is before
    assert repo.state is before
