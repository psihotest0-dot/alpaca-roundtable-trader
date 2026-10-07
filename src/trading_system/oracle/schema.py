from enum import Enum
from pydantic import BaseModel, Field

class LLMDecision(str, Enum):
    APPROVE = "APPROVE"
    VETO = "VETO"
    ABSTAIN = "ABSTAIN"

class AnythingLLMResponse(BaseModel):
    decision: LLMDecision
    reason: str = Field(..., max_length=500)
