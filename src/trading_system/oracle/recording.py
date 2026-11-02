"""Records what the Oracle says, and does not ask twice about a decision it has made (2.3).

Wraps any provider. A final answer (APPROVE or VETO) is stored and served from the store for
the same signal, also after a restart, so a model that could answer differently the second
time cannot change its mind about a signal. A HOLD is stored too, for the record, but it is
never served: "not now" has to be asked again. Neither is an answer that never came (H1: a
timeout used to be cached as a veto).
"""

import logging
import time

from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision
from trading_system.observability.metrics import (
    ORACLE_ANSWERS,
    ORACLE_ERRORS,
    ORACLE_LATENCY_SECONDS,
)
from trading_system.oracle.contract import OracleProvider, OracleRequest
from trading_system.storage.repositories import OracleDecisionRepository

logger = logging.getLogger(__name__)


class RecordingOracle:
    def __init__(self, provider: OracleProvider, repository: OracleDecisionRepository) -> None:
        self.provider = provider
        self.repository = repository
        self.name = provider.name

    def evaluate(self, request: OracleRequest) -> OracleDecision:
        signal_id = request.signal.signal_id or ""
        cached = self._final_decision(signal_id)
        if cached is not None:
            ORACLE_ANSWERS.labels(verdict=cached.verdict.value, kind="cached").inc()
            logger.info(f"Oracle: {cached.verdict.value} for {signal_id[:8]} is already decided.")
            return cached

        started = time.monotonic()
        try:
            decision = self.provider.evaluate(request)
        except Exception as e:
            decision = OracleDecision(
                signal_id=signal_id,
                verdict=OracleVerdict.HOLD,
                reason=f"The Oracle raised: {e}",
                provider=self.name,
                failed=True,
            )
        ORACLE_LATENCY_SECONDS.set(time.monotonic() - started)
        if decision.failed:
            ORACLE_ERRORS.inc()
        ORACLE_ANSWERS.labels(
            verdict=decision.verdict.value, kind="failed" if decision.failed else "answer"
        ).inc()

        try:
            self.repository.save(decision)
        except Exception as e:  # losing the record must not lose the decision
            logger.error(f"Could not record the Oracle decision for {signal_id[:8]}: {e}")
        return decision

    def _final_decision(self, signal_id: str) -> OracleDecision | None:
        try:
            return self.repository.final_decision_for(signal_id)
        except Exception as e:  # a cache miss is safe: it only means asking again
            logger.error(f"Could not read the Oracle record for {signal_id[:8]}: {e}")
            return None
