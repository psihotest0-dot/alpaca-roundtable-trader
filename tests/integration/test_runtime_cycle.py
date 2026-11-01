"""One real Orchestrator.run iteration against a fake Alpaca: data requests, C2 and the end result.

The harness (see conftest.run_cycle) replaces only the network. The data layer, the
strategy, risk, execution, the repositories and SQLite are the real code, and the loop is
stopped at its first sleep.
"""

from collections.abc import Callable
from types import SimpleNamespace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from trading_system.domain.models import TradingState
from trading_system.risk.daily_equity import trading_day
from trading_system.storage.models import BrokerOrderModel
from trading_system.storage.repositories import StateRepository

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


def test_cycle_risk_engine_approves_a_healthy_signal(run_cycle: RunCycle) -> None:
    """C2. daily_start_equity was never recorded, so risk rejected every signal with
    'Missing daily start equity' and the system could never open a position."""
    cycle = run_cycle()

    (decision,) = cycle.decisions
    assert decision.approved, decision.reasoning


def test_cycle_records_the_daily_start_equity_for_other_processes(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle(equity="100000")

    state = StateRepository(session_factory).get_state()
    assert state.daily_start_equity == Decimal("100000")
    assert state.daily_equity_date is not None
    assert state.daily_equity_date.date() == trading_day(datetime.now(timezone.utc))


def test_cycle_measures_the_day_from_the_previous_close_when_the_broker_reports_it(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle(equity="99000", last_equity="100000")

    assert StateRepository(session_factory).get_state().daily_start_equity == Decimal("100000")


def test_a_loss_beyond_the_daily_limit_stops_new_entries(run_cycle: RunCycle) -> None:
    """3 % below the previous close, with a 2 % limit."""
    cycle = run_cycle(equity="97000", last_equity="100000")

    (decision,) = cycle.decisions
    assert decision.approved is False
    assert "Daily loss guard breached" in decision.reasoning


def test_a_restart_during_the_day_keeps_the_morning_baseline(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    """Without a previous-close figure the first cycle's equity is the baseline. A restart
    after a 5 % loss must keep it, or the loss would be forgotten and trading would go on."""
    run_cycle(equity="100000")

    restarted = run_cycle(equity="95000")

    assert StateRepository(session_factory).get_state().daily_start_equity == Decimal("100000")
    (decision,) = restarted.decisions
    assert decision.approved is False
    assert "Daily loss guard breached" in decision.reasoning


def test_the_baseline_is_recorded_even_when_trading_is_halted(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    """A day that starts halted still needs its baseline for when trading resumes."""
    state_repo = StateRepository(session_factory)
    state_repo.save_state(TradingState(is_trading_halted=True, halt_reason="manual"))

    cycle = run_cycle(equity="100000")

    assert cycle.decisions == []  # halted: no signal was evaluated
    state = state_repo.get_state()
    assert state.is_trading_halted is True  # and the halt survived the update
    assert state.daily_start_equity == Decimal("100000")


@pytest.mark.xfail(
    strict=True,
    reason="C3: the end-to-end result. A dry-run cycle on a healthy signal should leave a "
    "simulated order that another process can read. Risk approves the signal now (C2 is "
    "fixed), but execution rejects its MARKET intent: 'Only limit entries are supported'.",
)
def test_dry_run_cycle_persists_a_simulated_order(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle()

    with session_factory() as fresh:
        assert fresh.query(BrokerOrderModel).count() == 1
