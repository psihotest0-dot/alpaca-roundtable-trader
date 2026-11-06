"""Looking at the backend Oracle from outside the loop: the ping for `doctor` and `health`, and
the probe that proves the whole road with one artificial signal (task 2.2, T7).

None of it is in the trading path. The loop never pings; these run when the owner or the Docker
healthcheck asks.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from trading_system.domain.enums import OracleVerdict, SignalDirection
from trading_system.domain.models import StrategySignal
from trading_system.oracle.backend import BackendOracle, PingResult
from trading_system.oracle.contract import OracleRequest

PROBE_VERSION = "oracle-probe"


def backend_is_on(settings: Any) -> bool:
    return bool(settings.oracle_enabled and settings.oracle_provider == "backend")


def ping_backend(settings: Any) -> Optional[PingResult]:
    """The backend's answer to a ping, or None when no backend Oracle is switched on."""
    if not backend_is_on(settings):
        return None
    try:
        return BackendOracle.from_settings(settings).ping()
    except Exception as e:  # `health` runs every minute and must never crash on this
        return PingResult(False, "internal", type(e).__name__)


def health_findings(
    settings: Any, ping: Optional[PingResult] = None
) -> tuple[list[str], list[str]]:
    """(problems, warnings) about the backend Oracle for the healthcheck.

    A backend that cannot be reached is a problem only in active mode, where it means no entries
    at all. In shadow mode the trading goes on exactly as before, so it is a warning: the
    container must not turn `unhealthy` because an experiment's far end is down."""
    if ping is None:
        ping = ping_backend(settings)
    if ping is None or ping.ok:
        return [], []
    text = f"Oracle backend {ping.kind}: {ping.detail}"
    if settings.oracle_mode == "active":
        return [text + " (ORACLE_MODE=active: no entries while it is down)"], []
    return [], [text + " (shadow mode: trading is not affected)"]


@dataclass
class ProbeResult:
    ok: bool
    lines: list[str]


def probe_signal(symbol: str, now: datetime) -> StrategySignal:
    """An artificial signal, marked by its version. Its id comes from the exact moment, so every
    probe is a new question and not an answer from the backend's cache."""
    return StrategySignal(
        version=PROBE_VERSION,
        symbol=symbol,
        timeframe="15m",
        closed_bar_timestamp=now,
        direction=SignalDirection.LONG,
        expires_at=now + timedelta(minutes=15),
    )


def run_probe(settings: Any, session_factory: Any, symbol: str) -> ProbeResult:
    """One artificial signal through the same stack as the loop (record and shadow wrappers
    included), then what the record says the MODEL answered. Writes one row to
    `strategy_signals` (the record's foreign key) and one to `oracle_decisions`, both marked
    `oracle-probe`; places no order and touches no risk state."""
    from trading_system.oracle.factory import create_oracle
    from trading_system.storage.models import OracleDecisionModel
    from trading_system.storage.repositories import OracleDecisionRepository, SignalRepository

    if not backend_is_on(settings):
        return ProbeResult(
            False,
            ["The backend Oracle is off: set ORACLE_ENABLED=true and ORACLE_PROVIDER=backend."],
        )
    ping = ping_backend(settings)
    if ping is not None and not ping.ok:
        return ProbeResult(False, [f"The ping failed ({ping.kind}): {ping.detail}"])

    now = datetime.now(timezone.utc)
    signal = probe_signal(symbol, now)
    SignalRepository(session_factory).save_signal(signal)
    oracle = create_oracle(settings, OracleDecisionRepository(session_factory))
    assert oracle is not None
    oracle.evaluate(
        OracleRequest(
            signal=signal,
            rationale=(
                "ORACLE PROBE: an artificial signal made only to test the connection. It is not "
                "a trading signal and there are no market facts with it."
            ),
            context={"probe": True},
        )
    )

    with session_factory() as session:
        row = (
            session.query(OracleDecisionModel)
            .filter(OracleDecisionModel.signal_id == signal.signal_id)
            .order_by(OracleDecisionModel.id.desc())
            .first()
        )
        if row is None:
            return ProbeResult(False, ["The answer was not recorded in oracle_decisions."])
        verdict = row.verdict.value if isinstance(row.verdict, OracleVerdict) else str(row.verdict)
        lines = [
            f"signal_id:   {signal.signal_id}",
            f"mode:        {row.mode or '-'}",
            f"verdict:     {verdict}" + (" (no answer)" if row.failed else ""),
            f"reason:      {row.reasoning}",
            f"model:       {row.model or '-'}",
            f"request_id:  {row.request_id or '-'}",
        ]
        if row.error_kind:
            lines.append(f"error_kind:  {row.error_kind}")
        failed = bool(row.failed)
    lines.append(
        "Find it on the backend: GET /oracle/decisions?symbol="
        f"{symbol} (signal_id above); here: oracle_decisions in trading.db."
    )
    return ProbeResult(not failed, lines)
