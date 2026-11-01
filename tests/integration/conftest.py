"""Shared fixtures for the integration tests.

These tests use the real risk engine, execution engine, repositories and SQLite.
Only the network (Alpaca) is faked. Tests tagged xfail(strict=True) document a
known defect from docs/analysis-and-update-plan.md: they fail today, and the
strict flag makes them fail loudly the moment the defect is fixed, which is the
cue to delete the marker.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.bars import make_bars
from tests.fixtures.fakes import FakeBroker, FakeStockClient
from trading_system.config import Settings
from trading_system.data.alpaca_data import AlpacaData
from trading_system.domain.enums import SignalDirection
from trading_system.domain.models import AccountSnapshot, StrategySignal
from trading_system.risk.engine import RiskEngine
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.models import Base


@pytest.fixture
def db_url(tmp_path: Any) -> str:
    """A fresh SQLite file with the schema. The short busy timeout keeps tests that
    hit a lock fast (SQLite's default is 5 seconds)."""
    url = f"sqlite:///{(tmp_path / 'trading.db').as_posix()}?timeout=0.2"
    Base.metadata.create_all(get_engine(url))
    return url


@pytest.fixture
def make_settings(db_url: str) -> Callable[..., Settings]:
    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "alpaca_api_key": "test-key",
            "alpaca_secret_key": "test-secret",
            "database_url": db_url,
            "watchlist": "AAPL",
        }
        values.update(overrides)
        # _env_file=None: never read a developer's real .env during tests.
        return Settings(_env_file=None, **values)

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def session_factory(db_url: str) -> Any:
    return get_session_factory(get_engine(db_url))


@pytest.fixture
def signal() -> StrategySignal:
    now = datetime.now(timezone.utc)
    return StrategySignal(
        version="1.0.0",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=now - timedelta(minutes=15),
        direction=SignalDirection.LONG,
        expires_at=now + timedelta(minutes=15),
    )


@pytest.fixture
def account() -> AccountSnapshot:
    return AccountSnapshot(
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("400000"),
        day_trade_count=0,
        updated_at=datetime.now(timezone.utc),
    )


class StopLoop(Exception):
    """Raised from time.sleep to end Orchestrator.run after one iteration."""


@pytest.fixture
def run_cycle(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> Callable[..., SimpleNamespace]:
    """Run exactly one Orchestrator.run iteration against a fake Alpaca and a real SQLite
    database, then return what happened.

    The data layer is the real AlpacaData; only its network client is replaced, by one
    that answers in alpaca-py's real shape. Broker and data requests are both visible to
    the test (`broker.submitted`, `client.requests`).
    """

    # Taken once: running several cycles in one test must not wrap the spy in another spy.
    original = RiskEngine.evaluate_signal

    def _run(
        *, is_paper: bool = False, equity: str = "100000", last_equity: str | None = None
    ) -> SimpleNamespace:
        bars_15m, bars_1h = make_bars()
        broker = FakeBroker(equity=equity, last_equity=last_equity)
        client = FakeStockClient({"AAPL": bars_15m}, {"AAPL": bars_1h})
        data = AlpacaData(settings)
        data.client = client
        decisions: list[Any] = []

        def spy(self: RiskEngine, *args: Any, **kwargs: Any) -> Any:
            result = original(self, *args, **kwargs)
            decisions.append(result[0])
            return result

        def stop(_seconds: float) -> None:
            raise StopLoop

        monkeypatch.setattr(RiskEngine, "evaluate_signal", spy)
        monkeypatch.setattr("trading_system.broker.alpaca_broker.AlpacaBroker", lambda _s: broker)
        monkeypatch.setattr("trading_system.data.alpaca_data.AlpacaData", lambda _s: data)
        monkeypatch.setattr("time.sleep", stop)
        orchestrator = Orchestrator(settings)
        try:
            orchestrator.run(is_paper)
        except StopLoop:
            pass
        return SimpleNamespace(
            orchestrator=orchestrator, broker=broker, client=client, decisions=decisions
        )

    return _run
