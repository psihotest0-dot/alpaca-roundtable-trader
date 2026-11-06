"""Shadow mode: the Oracle is asked and heard, and the loop is told yes (task 2.2).

The first weeks of a real model are for watching it, not for obeying it. In shadow mode the signal
is put to the wrapped provider exactly as in active mode (same request, same record), and then the
loop gets `APPROVE` whatever the model said, so nothing about the orders changes: a veto of the
model is a trade that was made anyway, and the record shows what would have been lost or saved.

The order of the wrappers is the whole point:

    ShadowOracle( RecordingOracle( BackendOracle ) )

The record is made INSIDE, so it holds what the model really said. Put the other way round, the
store would hold the APPROVE this class hands to the loop, and the experiment would measure
nothing. `create_oracle` builds them in this order; a test holds it.
"""

import logging

from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision
from trading_system.oracle.contract import OracleProvider, OracleRequest

logger = logging.getLogger(__name__)


class ShadowOracle:
    name = "shadow"

    def __init__(self, provider: OracleProvider) -> None:
        self.provider = provider

    def evaluate(self, request: OracleRequest) -> OracleDecision:
        signal_id = request.signal.signal_id or ""
        try:
            real = self.provider.evaluate(request)
        except Exception as e:  # a provider is meant not to raise; in shadow it must not matter
            real = OracleDecision(
                signal_id=signal_id,
                verdict=OracleVerdict.HOLD,
                reason=f"The Oracle raised: {type(e).__name__}",
                provider=getattr(self.provider, "name", ""),
                failed=True,
                mode="shadow",
                error_kind="internal",
            )
        said = real.verdict.value + (" (no answer)" if real.failed else "")
        logger.info(
            f"Shadow Oracle on {request.signal.symbol}: the model said {said} - {real.reason}; "
            "the signal goes on."
        )
        return OracleDecision(
            signal_id=signal_id,
            verdict=OracleVerdict.APPROVE,
            reason=f"Shadow mode: the model said {said}; the signal goes on regardless.",
            provider=self.name,
            mode="shadow",
        )
