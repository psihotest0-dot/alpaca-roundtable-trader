from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision
from trading_system.oracle.contract import OracleRequest


class NullOracle:
    """The Oracle for when there is none: every signal passes. Selected with
    ORACLE_PROVIDER=null; with ORACLE_ENABLED=false the Oracle is not consulted at all."""

    name = "null"

    def evaluate(self, request: OracleRequest) -> OracleDecision:
        return OracleDecision(
            signal_id=request.signal.signal_id or "",
            verdict=OracleVerdict.APPROVE,
            reason="No Oracle is configured; the signal passes.",
            provider=self.name,
        )
