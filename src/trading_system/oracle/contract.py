"""What an Oracle is, whatever model or service stands behind it (Phase 2).

The Oracle is the second factor of a trade: the strategy produces a signal, the Oracle says
yes, no or not now, and the risk engine still has the last word. It is given facts and answers
with a verdict; it does not choose the order.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from trading_system.domain.models import OracleDecision, StrategySignal


class OracleNews(BaseModel):
    """One article about the stock, as much of it as the Oracle is to see: plain, single-line text
    with no addresses or markup, a headline of at most 160 characters and a summary of at most 300."""

    article_id: int  # for the record of what the Oracle was shown; not for the model
    published_at: datetime
    age_hours: int = 0  # whole hours between the publication and the request: the model sees this
    source: str
    headline: str
    summary: str = ""
    other_symbols: int = 0  # how many other stocks the article names (0: about this one alone)


class OracleRequest(BaseModel):
    """The facts the Oracle judges a signal by (see oracle/inputs.py, which builds it).

    Everything in it is a number or a fixed label computed here, except `news`: the one piece of
    text that comes from the outside world, so the one place a prompt injection could ride on. It
    is cleaned and bounded before it gets here (oracle/news_input.py), it is data and never
    instructions, and it can do less harm than it looks: the Oracle only answers APPROVE, VETO or
    HOLD to a signal the strategy already made, so words in a headline can at worst turn a veto
    into an approval, never create a trade, change its size or skip the risk engine."""

    signal: StrategySignal
    rationale: str = ""
    technicals: dict[str, Any] = Field(default_factory=dict)  # indicators of the last closed bar
    recent_bars: list[dict[str, Any]] = Field(default_factory=list)  # oldest first
    volume_ratio: Optional[float] = None  # last bar's volume against the average of the bars before
    live_price: Optional[Decimal] = None
    context: dict[str, Any] = Field(default_factory=dict)  # the account and the risk limits
    news: list[OracleNews] = Field(default_factory=list)  # newest first; may well be empty


class OracleProvider(Protocol):
    """An implementation answers every request with an OracleDecision and does not raise:
    a timeout or an unreadable reply is a HOLD with `failed=True`."""

    name: str

    def evaluate(self, request: OracleRequest) -> OracleDecision: ...
