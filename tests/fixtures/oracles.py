"""An Oracle that answers from a script, for tests of everything around the model."""

from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision
from trading_system.oracle import OracleRequest


class ScriptedOracle:
    """Answers from a list, one per question; an Exception in the list is raised."""

    name = "scripted"

    def __init__(self, *answers: OracleVerdict | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[OracleRequest] = []

    def evaluate(self, request: OracleRequest) -> OracleDecision:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return OracleDecision(
            signal_id=request.signal.signal_id or "",
            verdict=answer,
            reason=f"scripted {answer.value}",
            provider=self.name,
        )
