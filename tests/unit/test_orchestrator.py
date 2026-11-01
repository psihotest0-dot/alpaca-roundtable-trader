"""Kill-switch bookkeeping of the Orchestrator, against an in-memory state repository.

The loop itself is covered by tests/integration/test_runtime_cycle.py.
"""

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
