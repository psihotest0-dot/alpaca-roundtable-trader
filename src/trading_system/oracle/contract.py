"""What an Oracle is, whatever model or service stands behind it (Phase 2).

The Oracle is the second factor of a trade: the strategy produces a signal, the Oracle says
yes, no or not now, and the risk engine still has the last word. It is given facts and answers
with a verdict; it does not choose the order.
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from trading_system.domain.models import OracleDecision, StrategySignal


class OracleRequest(BaseModel):
    signal: StrategySignal
    technicals: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class OracleProvider(Protocol):
    """An implementation answers every request with an OracleDecision and does not raise:
    a timeout or an unreadable reply is a HOLD with `failed=True`."""

    name: str

    def evaluate(self, request: OracleRequest) -> OracleDecision: ...
