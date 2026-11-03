"""The loop's pulse, as a small file that other processes can read.

The runner writes it every tick. Docker's healthcheck (`trading_system health`) reads it from
outside the loop: a loop that is stuck, or that has not looked at a bar for too long, cannot
write "I am fine" about itself, so what it wrote last and when is what is judged."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class Heartbeat:
    def __init__(self, path: Path) -> None:
        self.path = path

    def beat(self, fields: dict[str, Any], *, now: datetime) -> None:
        """Replaces the file with `fields` and the time. Written whole under another name and
        renamed, so a reader never sees half a file."""
        body = json.dumps({**fields, "at": now.isoformat(), "pid": os.getpid()}, default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, self.path)


def read_heartbeat(path: Path) -> Optional[dict[str, Any]]:
    """What the loop last wrote, or None if there is no file or it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
