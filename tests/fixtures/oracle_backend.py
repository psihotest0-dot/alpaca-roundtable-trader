"""A fake of the Go backend's Oracle routes: a real HTTP server on the loopback.

The client is tested against a socket, not against a mock of `requests`, because what has to be
shown is what goes over the wire (the path, the bearer header, the JSON body) and how every way a
call can end is read: a status, a body that is not JSON, a reply that never comes, a port with
nobody behind it.
"""

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

GOOD_TOKEN = "t" * 48


@dataclass
class Reply:
    status: int = 200
    body: Any = None  # a dict/list becomes JSON, bytes/str go as they are
    delay: float = 0.0  # seconds before the reply is sent
    headers: dict[str, str] = field(default_factory=dict)


def answer(
    verdict: str = "APPROVE",
    reason: str = "Сигналът изглежда добър.",
    *,
    cached: bool = False,
    request_id: str = "req-1",
    model: str = "test-model",
    latency_ms: int = 1500,
    prompt_version: Any = None,
) -> Reply:
    body: dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "request_id": request_id,
        "model": model,
        "cached": cached,
        "latency_ms": latency_ms,
    }
    if prompt_version is not None:  # the extra field of the backend since 2026-11-06 22:11
        body["prompt_version"] = prompt_version
    return Reply(200, body)


def error(status: int, code: str, **more: Any) -> Reply:
    return Reply(status, {"error": code, **more})


@dataclass
class Received:
    method: str
    path: str
    authorization: Optional[str]
    content_type: Optional[str]
    raw: bytes

    @property
    def json(self) -> Any:
        return json.loads(self.raw.decode("utf-8"))


class FakeOracleBackend:
    """Answers `script` in order, then `fallback`. Records everything it receives."""

    def __init__(self, *script: Reply, fallback: Optional[Reply] = None) -> None:
        self.script = list(script)
        self.fallback = fallback or answer()
        self.received: list[Received] = []
        self.token = GOOD_TOKEN
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                outer.received.append(
                    Received(
                        self.command,
                        self.path,
                        self.headers.get("Authorization"),
                        self.headers.get("Content-Type"),
                        raw,
                    )
                )
                reply = outer._next()
                if self.headers.get("Authorization") != f"Bearer {outer.token}":
                    reply = error(401, "unauthorized")
                if reply.delay:
                    # Not time.sleep: some tests patch that out to skip retry back-offs.
                    threading.Event().wait(reply.delay)
                body = reply.body
                if isinstance(body, (dict, list)):
                    payload = json.dumps(body).encode("utf-8")
                elif isinstance(body, str):
                    payload = body.encode("utf-8")
                else:
                    payload = body or b""
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    for name, value in reply.headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:  # the client gave up waiting (a timeout test)
                    pass

            do_GET = do_POST = _serve

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        # serve_forever looks for a shutdown only every poll_interval (0.5 s by default): every test
        # would wait that long to stop its server.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    def _next(self) -> Reply:
        return self.script.pop(0) if self.script else self.fallback

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def __enter__(self) -> "FakeOracleBackend":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
