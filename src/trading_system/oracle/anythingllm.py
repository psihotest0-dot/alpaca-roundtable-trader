import json
import logging
from typing import Any, Dict
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from trading_system.config import Settings
from trading_system.domain.models import OracleDecision, StrategySignal
from trading_system.oracle.schema import AnythingLLMResponse, LLMDecision

logger = logging.getLogger(__name__)

_decision_cache: Dict[str, OracleDecision] = {}

class AnythingLLMAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = self.settings.anythingllm_base_url.rstrip("/")
        self.workspace_slug = self.settings.anythingllm_workspace_slug

        api_key = ""
        if self.settings.anythingllm_api_key:
            api_key = self.settings.anythingllm_api_key.get_secret_value()

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.timeout = self.settings.oracle_timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with bounded retries"""
        url = f"{self.base_url}/api/v1/workspace/{self.workspace_slug}/chat"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=self.headers, json=payload)

            # For 429 and 5xx we want to retry, raise_for_status throws HTTPStatusError
            # But we must not retry on 400/401/403. We can raise a custom exception that tenacity won't catch.
            if response.status_code in (400, 401, 403):
                raise ValueError(f"HTTP {response.status_code}: {response.text}")

            response.raise_for_status()

            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected JSON response to be a dictionary")
            return result

    def _extract_and_parse_llm_response(self, chat_response: Dict[str, Any]) -> AnythingLLMResponse:
        """Extract the JSON response from AnythingLLM output and validate it."""
        try:
            # According to docs, textResponse contains the actual LLM output string
            text_response = chat_response.get("textResponse", "")

            if not text_response:
                raise ValueError("textResponse field is missing or empty")

            # Extract json if it's wrapped in markdown blocks
            text_response = text_response.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]

            data = json.loads(text_response.strip())
            return AnythingLLMResponse.model_validate(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}")
            raise ValueError(f"Invalid JSON from LLM: {str(e)}")
        except ValidationError as e:
            logger.error(f"Failed to validate LLM response schema: {e}")
            raise ValueError(f"Malformed output from LLM: {str(e)}")

    def evaluate_signal(self, signal: StrategySignal, technicals: Dict[str, Any], rationale: str) -> OracleDecision:
        """
        Evaluate a strategy signal with the Oracle.
        Returns a cached decision if one exists.
        Handles timeout and malformed outputs by producing an ABSTAIN/VETO decision.
        """
        signal_id = signal.signal_id or ""

        if not self.settings.oracle_enabled:
            return OracleDecision(
                signal_id=signal_id,
                approved=True,
                reasoning="Oracle is DISABLED",
                decision_timestamp=datetime.now(timezone.utc)
            )

        if signal_id in _decision_cache:
            logger.info(f"Returning cached Oracle decision for signal {signal_id}")
            return _decision_cache[signal_id]

        prompt = {
            "symbol": signal.symbol,
            "signal_timestamp": signal.closed_bar_timestamp.isoformat(),
            "strategy_version": signal.version,
            "direction": signal.direction.value,
            "technicals": technicals,
            "rationale": rationale
        }

        # Ensure we only ask for structured output
        message = (
            f"Analyze the following trading signal and context.\n"
            f"Respond ONLY with a valid JSON object in the exact format:\n"
            f"{{\"decision\": \"APPROVE\" | \"VETO\" | \"ABSTAIN\", \"reason\": \"short text\"}}\n\n"
            f"Context: {json.dumps(prompt)}"
        )

        payload = {
            "message": message,
            "mode": "chat",
            "sessionId": signal_id
        }

        try:
            logger.info(f"Requesting Oracle decision for {signal_id}")
            raw_response = self._make_request(payload)
            llm_response = self._extract_and_parse_llm_response(raw_response)

            approved = (llm_response.decision == LLMDecision.APPROVE)
            reason = llm_response.reason

            # ABSTAIN is treated as a rejection in the strict MVP model
            if llm_response.decision == LLMDecision.ABSTAIN:
                approved = False

        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.error(f"Oracle request failed/timed out: {e}")
            approved = False
            reason = "Oracle timeout or connection error"
        except httpx.HTTPStatusError as e:
            logger.error(f"Oracle HTTP error {e.response.status_code}: {e.response.text}")
            approved = False
            reason = f"Oracle HTTP error {e.response.status_code}"
        except ValueError as e:
            logger.error(f"Oracle data/request error: {e}")
            approved = False
            reason = f"Oracle data error: {str(e)}"
        except Exception as e:
            logger.error(f"Oracle evaluation failed: {e}")
            approved = False
            reason = f"Oracle evaluation failed: {str(e)}"

        decision = OracleDecision(
            signal_id=signal_id,
            approved=approved,
            reasoning=reason,
            decision_timestamp=datetime.now(timezone.utc)
        )

        _decision_cache[signal_id] = decision
        return decision
