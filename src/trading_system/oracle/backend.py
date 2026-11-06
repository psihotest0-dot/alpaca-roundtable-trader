"""The Oracle that asks the Go backend for a second opinion (task 2.2).

The bot never talks to the model. It sends the signal with the facts around it to the backend's
`POST /oracle/evaluate` (contract: the backend's docs/openapi.yaml and README, "The Oracle") and
reads one of three words back. The model, its slots, its prompt and its journal are the backend's.

The rules of this client, in the order they matter:

* **Nothing unclear is a yes.** A refused call, a timeout, a reply that is not JSON, a verdict
  that is not one of the three words, a missing field, a redirect, a reply that is too big: every
  one is a `HOLD` marked `failed`, with a kind (see below). Nothing here raises.
* **No retry inside.** The cycle asks again on its next tick; the backend finishes a call the bot
  stopped waiting for and keeps the answer (a HOLD for 3 minutes, a final verdict for 30), so the
  next question about the same signal is answered from its cache.
* **A failure costs the cycle once, not once per signal.** After a call that timed out or could
  not be made, further calls fail at once for FAILURE_COOLDOWN_SECONDS instead of each waiting out
  the timeout: five signals on one bar must not hold the loop (and its heartbeat) for a minute.
* **The reason carries a code, never a body.** What a failed call says about itself is the HTTP
  status and the backend's error code from a fixed list, or the name of the exception class. The
  token and the text of a reply that could not be understood never reach the log or the record.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import requests

from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision
from trading_system.oracle.contract import OracleRequest

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_BODY_BYTES = 64 * 1024  # the backend's limit for a request
MAX_REPLY_BYTES = 64 * 1024  # a reply is a few hundred bytes; more than this is not a reply
CONNECT_TIMEOUT_SECONDS = 3.0
PING_TIMEOUT_SECONDS = 5.0
FAILURE_COOLDOWN_SECONDS = 30.0

# Why there was no answer (OracleDecision.error_kind, and the `kind` of oracle_answers_total).
UNAVAILABLE = "unavailable"  # not reachable, or the model behind it did not answer (500, 502)
TIMEOUT = "timeout"  # no reply within ORACLE_TIMEOUT_SECONDS, or the backend's own 504
UNAUTHORIZED = "unauthorized"  # no usable token here, or the backend refused it (401)
BAD_REQUEST = "bad_request"  # the backend refused our request (400, 405, 413): a bug on our side
BAD_REPLY = "bad_reply"  # what came back is not in the contract
INTERNAL = "internal"  # an error in this client itself: a bug here, not in the backend

# The error codes of the contract. Anything else in a reply is not echoed anywhere.
KNOWN_ERROR_CODES = frozenset(
    {
        "unauthorized",
        "invalid_request",
        "unreadable_body",
        "body_too_large",
        "method_not_allowed",
        "model_unavailable",
        "model_timeout",
        "unusable_reply",
        "internal_error",
    }
)
_VERDICTS = {v.value: v for v in OracleVerdict}
_REQUIRED_REPLY_FIELDS = ("verdict", "reason", "request_id", "model", "cached", "latency_ms")


@dataclass(frozen=True)
class PingResult:
    """Whether the backend serves its Oracle route and accepts the token (it does not call the
    model). `kind` is one of the kinds above when `ok` is False."""

    ok: bool
    kind: Optional[str] = None
    detail: str = ""


def _one_line(text: str, limit: int = 200) -> str:
    """The backend's reason as one printable line: it is model output, and it goes to the log."""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(cleaned.split())[:limit]


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


_PROMPT_VERSION = re.compile(r"[0-9a-f]{6,64}")


def _prompt_version(value: Any) -> Optional[str]:
    """The backend's label of the prompt behind a reply (12 lowercase hex, since 2026-11-06 22:11
    German time). It is an extra field, not one of the six of the contract, so a reply without it,
    or with something else in it, is still a good reply: the label is then absent (None)."""
    if value is None:
        return None
    if isinstance(value, str) and _PROMPT_VERSION.fullmatch(value):
        return value
    logger.warning("Oracle backend: the prompt_version of a reply is not a hex label; ignored")
    return None


def _token_is_usable(token: str) -> bool:
    return bool(token) and token.isascii() and token.isprintable() and " " not in token


class BackendOracle:
    name = "backend"

    def __init__(
        self,
        url: str,
        token: str,
        *,
        mode: str = "shadow",
        timeout_seconds: float = 10.0,
        session: Optional[requests.Session] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url.rstrip("/")
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self._token = token
        self._clock = clock
        self._session = session if session is not None else requests.Session()
        # Proxy settings in the environment are not for this call: the token would go through them.
        self._session.trust_env = False
        self._skip_until = 0.0

    @classmethod
    def from_settings(cls, settings: Any) -> "BackendOracle":
        return cls(
            settings.oracle_backend_url,
            settings.oracle_backend_token.get_secret_value(),
            mode=settings.oracle_mode,
            timeout_seconds=float(settings.oracle_timeout_seconds),
        )

    # ------------------------------------------------------------------ the question

    def evaluate(self, request: OracleRequest) -> OracleDecision:
        signal_id = request.signal.signal_id or ""
        try:
            return self._evaluate(request, signal_id)
        except Exception as e:  # the protocol: an Oracle does not raise
            return self._failed(signal_id, INTERNAL, f"Oracle backend: {type(e).__name__}")

    def _evaluate(self, request: OracleRequest, signal_id: str) -> OracleDecision:
        if not _token_is_usable(self._token):
            return self._failed(
                signal_id, UNAUTHORIZED, "Oracle backend: no usable ORACLE_BACKEND_TOKEN is set"
            )
        waiting = self._skip_until - self._clock()
        if waiting > 0:
            return self._failed(
                signal_id,
                UNAVAILABLE,
                f"Oracle backend: skipped, the last call failed less than "
                f"{FAILURE_COOLDOWN_SECONDS:.0f} s ago",
            )
        try:
            body = self._body(request)
        except (TypeError, ValueError) as e:  # a NaN, something that is not JSON
            return self._failed(signal_id, BAD_REQUEST, f"Oracle backend: no request ({e})")
        if len(body) > MAX_BODY_BYTES:
            return self._failed(
                signal_id, BAD_REQUEST, f"Oracle backend: request is {len(body)} bytes, too big"
            )

        started = self._clock()
        try:
            status, content = self._call(
                "POST", "/oracle/evaluate", body=body, timeout=self.timeout_seconds
            )
        except requests.ConnectTimeout:  # also a Timeout, but nobody picked up: not "no reply"
            return self._transport_failure(
                signal_id, UNAVAILABLE, "Oracle backend: not reachable (ConnectTimeout)"
            )
        except requests.Timeout:
            return self._transport_failure(
                signal_id, TIMEOUT, f"Oracle backend: no reply in {self.timeout_seconds:g} s"
            )
        except requests.RequestException as e:
            return self._transport_failure(
                signal_id, UNAVAILABLE, f"Oracle backend: not reachable ({type(e).__name__})"
            )
        took = self._clock() - started

        if status != 200:
            return self._refused(signal_id, status, content)
        decision = self._answer(signal_id, content)
        if decision.failed:
            return decision
        logger.info(
            f"Oracle backend answered {decision.verdict.value} for {request.signal.symbol} "
            f"signal_id={signal_id} request_id={decision.request_id} model={decision.model} "
            f"({took:.1f} s)"
        )
        return decision

    def _body(self, request: OracleRequest) -> bytes:
        payload = request.model_dump(mode="json")
        payload["signal"] = _without_none(payload["signal"])
        for key in ("volume_ratio", "live_price"):  # the contract has no null for these
            if payload.get(key) is None:
                payload.pop(key, None)
        payload["schema_version"] = SCHEMA_VERSION
        payload["mode"] = self.mode
        return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def _call(
        self, method: str, path: str, *, body: Optional[bytes] = None, timeout: float
    ) -> tuple[int, bytes]:
        """One HTTP call: the status and at most MAX_REPLY_BYTES of the reply. No redirect is
        followed (the token must not go anywhere the address did not name)."""
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        response = self._session.request(
            method,
            self.url + path,
            data=body,
            headers=headers,
            timeout=(min(CONNECT_TIMEOUT_SECONDS, timeout), timeout),
            allow_redirects=False,
            stream=True,
        )
        try:
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > MAX_REPLY_BYTES:
                    break
            return response.status_code, content[: MAX_REPLY_BYTES + 1]
        finally:
            response.close()

    # ------------------------------------------------------------------ the reply

    def _answer(self, signal_id: str, content: bytes) -> OracleDecision:
        """A 200 reply, read strictly: anything outside the contract is a failure."""
        if len(content) > MAX_REPLY_BYTES:
            return self._failed(signal_id, BAD_REPLY, "Oracle backend: the reply is too big")
        try:
            reply = json.loads(content.decode("utf-8"))
        except ValueError:
            return self._failed(signal_id, BAD_REPLY, "Oracle backend: the reply is not JSON")
        if not isinstance(reply, dict):
            return self._failed(signal_id, BAD_REPLY, "Oracle backend: the reply is not an object")
        missing = [name for name in _REQUIRED_REPLY_FIELDS if name not in reply]
        if missing:
            return self._failed(
                signal_id, BAD_REPLY, f"Oracle backend: the reply has no {', '.join(missing)}"
            )
        verdict = reply["verdict"]
        if not isinstance(verdict, str) or verdict not in _VERDICTS:
            return self._failed(signal_id, BAD_REPLY, "Oracle backend: the verdict is unknown")
        if (
            not isinstance(reply["reason"], str)
            or not isinstance(reply["request_id"], str)
            or not isinstance(reply["model"], str)
            or not isinstance(reply["cached"], bool)
            or isinstance(reply["latency_ms"], bool)
            or not isinstance(reply["latency_ms"], int)
        ):
            return self._failed(signal_id, BAD_REPLY, "Oracle backend: a field has the wrong type")
        return OracleDecision(
            signal_id=signal_id,
            verdict=_VERDICTS[verdict],
            reason=_one_line(reply["reason"]),
            provider=self.name,
            mode=self.mode,
            request_id=_one_line(reply["request_id"], 128) or None,
            model=_one_line(reply["model"], 128) or None,
            prompt_version=_prompt_version(reply.get("prompt_version")),
        )

    def _refused(self, signal_id: str, status: int, content: bytes) -> OracleDecision:
        """A reply that is not a 200: no verdict, a kind from the status, a code if the body
        names one the contract knows."""
        code = _error_code(content)
        kind = _kind_for_status(status, code)
        said = f"HTTP {status}" + (f", {code}" if code else "")
        hint = " (is ORACLE_TOKEN set in the backend?)" if status == 404 else ""
        decision = self._failed(signal_id, kind, f"Oracle backend: {kind} ({said}){hint}")
        if kind in (UNAVAILABLE, TIMEOUT):
            self._skip_until = self._clock() + FAILURE_COOLDOWN_SECONDS
        return decision

    # ------------------------------------------------------------------ failures

    def _transport_failure(self, signal_id: str, kind: str, reason: str) -> OracleDecision:
        self._skip_until = self._clock() + FAILURE_COOLDOWN_SECONDS
        return self._failed(signal_id, kind, reason)

    def _failed(self, signal_id: str, kind: str, reason: str) -> OracleDecision:
        logger.warning(f"{reason} [signal_id={signal_id}, kind={kind}]")
        return OracleDecision(
            signal_id=signal_id,
            verdict=OracleVerdict.HOLD,
            reason=reason,
            provider=self.name,
            failed=True,
            mode=self.mode,
            error_kind=kind,
        )

    # ------------------------------------------------------------------ is it there?

    def ping(self) -> PingResult:
        """`GET /oracle/ping`: is the route on and the token right? It does not call the model,
        so it is cheap enough for `doctor` and `health`. Never raises."""
        if not _token_is_usable(self._token):
            return PingResult(False, UNAUTHORIZED, "no usable ORACLE_BACKEND_TOKEN is set")
        timeout = min(PING_TIMEOUT_SECONDS, self.timeout_seconds)
        try:
            status, content = self._call("GET", "/oracle/ping", timeout=timeout)
        except requests.ConnectTimeout:
            return PingResult(False, UNAVAILABLE, "not reachable (ConnectTimeout)")
        except requests.Timeout:
            return PingResult(False, TIMEOUT, f"no reply in {timeout:g} s")
        except requests.RequestException as e:
            return PingResult(False, UNAVAILABLE, f"not reachable ({type(e).__name__})")
        except Exception as e:
            return PingResult(False, INTERNAL, type(e).__name__)
        if status == 200:
            try:
                ok = json.loads(content.decode("utf-8")).get("status") == "ok"
            except (ValueError, AttributeError):
                ok = False
            return PingResult(True) if ok else PingResult(False, BAD_REPLY, "HTTP 200, not ok")
        code = _error_code(content)
        said = f"HTTP {status}" + (f", {code}" if code else "")
        if status == 404:
            said += ": the backend does not serve the Oracle (is ORACLE_TOKEN set there?)"
        return PingResult(False, _kind_for_status(status, code), said)


def _kind_for_status(status: int, code: Optional[str]) -> str:
    """Why a call that did not return 200 gave no verdict."""
    if status == 401:
        return UNAUTHORIZED
    if status in (400, 405, 413):
        return BAD_REQUEST
    if status == 504:
        return TIMEOUT
    if status == 502:  # the model did not answer, or answered twice with something unusable
        return BAD_REPLY if code == "unusable_reply" else UNAVAILABLE
    if status == 404:  # the backend serves its Oracle routes only with a token of 32 characters
        return UNAVAILABLE
    if 500 <= status < 600:
        return UNAVAILABLE
    return BAD_REPLY


def _error_code(content: bytes) -> Optional[str]:
    """The `error` of a refusal if it is one the contract lists, else None (never free text)."""
    try:
        body = json.loads(content.decode("utf-8"))
    except ValueError:
        return None
    code = body.get("error") if isinstance(body, dict) else None
    return code if isinstance(code, str) and code in KNOWN_ERROR_CODES else None
