"""One real Orchestrator.run iteration against fakes: C1b, C2 and the end result.

The harness (see conftest.run_cycle) swaps Alpaca for fakes and stops the loop at
its first sleep. Everything else is the real runtime, risk, execution and SQLite.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from trading_system.storage.models import BrokerOrderModel

RunCycle = Callable[..., SimpleNamespace]


@pytest.mark.xfail(
    strict=True,
    reason="C1b: the orchestrator builds its 15-minute timeframe as `TimeFrame.Minute * 15`, "
    "which raises TypeError in alpaca-py 0.36 (TimeFrame has no __mul__). The per-symbol "
    "try/except swallows it, so market data has never been fetched.",
)
def test_cycle_requests_bars_with_valid_timeframes(run_cycle: RunCycle) -> None:
    cycle = run_cycle(timeframe_workaround=False)

    assert ("AAPL", "15Min") in cycle.data.calls


def test_control_cycle_reaches_risk_evaluation_with_one_signal(run_cycle: RunCycle) -> None:
    """Guard for the harness, passes today: with the timeframe defect worked around,
    the engineered bars become exactly one LONG signal that reaches the risk engine."""
    cycle = run_cycle()

    assert ("AAPL", "15Min") in cycle.data.calls
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
