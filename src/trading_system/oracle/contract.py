"""What an Oracle is, whatever model or service stands behind it (Phase 2).

The Oracle is the second factor of a trade: the strategy produces a signal, the Oracle says
yes, no or not now, and the risk engine still has the last word. It is given facts and answers
with a verdict; it does not choose the order.
"""

from decimal import Decimal
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from trading_system.domain.models import OracleDecision, StrategySignal


class OracleRequest(BaseModel):
    """The facts the Oracle judges a signal by (see oracle/inputs.py, which builds it).

    Everything in it is a number or a fixed label computed here: no text that comes from the
    outside world, so there is nothing in it for a prompt injection to ride on. (News, in
    Phase 5, will be the first such text and gets its own handling.)"""

    signal: StrategySignal
    rationale: str = ""
    technicals: dict[str, Any] = Field(default_factory=dict)  # indicators of the last closed bar
    recent_bars: list[dict[str, Any]] = Field(default_factory=list)  # oldest first
    volume_ratio: Optional[float] = None  # last bar's volume against the average of the bars before
    live_price: Optional[Decimal] = None
    context: dict[str, Any] = Field(default_factory=dict)  # the account and the risk limits


class OracleProvider(Protocol):
    """An implementation answers every request with an OracleDecision and does not raise:
    a timeout or an unreadable reply is a HOLD with `failed=True`."""

    name: str

    def evaluate(self, request: OracleRequest) -> OracleDecision: ...
