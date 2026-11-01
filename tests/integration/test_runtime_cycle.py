"""One real Orchestrator.run iteration against a fake Alpaca: data requests, C2 and the end result.

The harness (see conftest.run_cycle) replaces only the network. The data layer, the
strategy, risk, execution, the repositories and SQLite are the real code, and the loop is
stopped at its first sleep.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from trading_system.storage.models import BrokerOrderModel

RunCycle = Callable[..., SimpleNamespace]


def test_cycle_requests_bars_with_valid_timeframes(run_cycle: RunCycle) -> None:
    """C1b. The orchestrator built the 15-minute timeframe as `TimeFrame.Minute * 15`,
    which raises TypeError in alpaca-py. The per-symbol try/except swallowed it, so market
    data was never fetched. Now: one request per timeframe, covering every symbol."""
    cycle = run_cycle()

    assert [str(r.timeframe) for r in cycle.client.requests] == ["15Min", "1Hour"]
    assert all(r.symbol_or_symbols == ["AAPL"] for r in cycle.client.requests)


def test_cycle_turns_alpaca_bars_into_one_signal_for_risk(run_cycle: RunCycle) -> None:
    """The whole data path in one go: Alpaca-shaped bars in, exactly one LONG signal
    evaluated by the risk engine."""
    cycle = run_cycle()

    assert len(cycle.decisions) == 1


@pytest.mark.xfail(
    strict=True,
    reason="C2: daily_start_equity is never recorded, so the risk engine rejects every signal "
    "with 'Missing daily start equity'.",
)
def test_cycle_risk_engine_approves_a_healthy_signal(run_cycle: RunCycle) -> None:
    cycle = run_cycle()

    (decision,) = cycle.decisions
    assert decision.approved, decision.reasoning


@pytest.mark.xfail(
    strict=True,
    reason="C2 + C3: the end-to-end result. A dry-run cycle on a healthy signal should leave a "
    "simulated order that another process can read. Today risk rejects the signal (C2) and, "
    "once it does not, execution rejects the MARKET intent (C3).",
)
def test_dry_run_cycle_persists_a_simulated_order(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle()

    with session_factory() as fresh:
        assert fresh.query(BrokerOrderModel).count() == 1
