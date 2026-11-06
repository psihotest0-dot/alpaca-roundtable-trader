"""`validate-config`: is the configuration usable, judged for real (Phase 6.4).

It used to put dummy keys into the environment first, so it passed on any machine, with or
without keys, and `extra="ignore"` let a misspelled or stale key in `.env` go unnoticed. Now the
keys must really be there, nothing is written to the environment, a setting nobody reads is
named, and only names are looked at: a value never appears in the output."""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import ValidationError

from trading_system.config import Settings

# Variables that are read by `compose.yaml` (not by Settings) and are legitimate in `.env`.
READ_BY_COMPOSE = {"BACKUP_KEEP", "BACKUP_INTERVAL_SECONDS"}


@dataclass
class ConfigReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def known_names() -> set[str]:
    return {name.upper() for name in Settings.model_fields} | READ_BY_COMPOSE


def unknown_key_warnings(env_file: str = ".env") -> list[str]:
    """One warning per name in the file that no setting reads. Only names are read."""
    path = Path(env_file)
    if not path.is_file():
        return []
    known = known_names()
    warnings = []
    for name in dotenv_values(path):
        if name.upper() in known:
            continue
        close = difflib.get_close_matches(name.upper(), sorted(known), n=1, cutoff=0.7)
        hint = f" (did you mean {close[0]}?)" if close else ""
        warnings.append(f"{name} in {env_file} is not a setting and is ignored{hint}")
    return warnings


# The backend serves its Oracle routes only for a token of at least this many characters.
MIN_ORACLE_TOKEN_LENGTH = 32
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def oracle_warnings(settings: Settings) -> list[str]:
    """What is wrong with an Oracle that is switched on, said without its token: the length of
    the token is shown, never a character of it. Warnings, not errors: a shadow Oracle that
    cannot be reached costs nothing, and the bot keeps trading."""
    if not (settings.oracle_enabled and settings.oracle_provider == "backend"):
        return []
    warnings = []
    token = settings.oracle_backend_token.get_secret_value()
    if not token:
        warnings.append(
            "ORACLE_BACKEND_TOKEN is not set: every question to the backend would be refused "
            "(401), so every answer is a HOLD that failed"
        )
    elif len(token) < MIN_ORACLE_TOKEN_LENGTH:
        warnings.append(
            f"ORACLE_BACKEND_TOKEN is {len(token)} characters long; the backend serves its "
            f"Oracle routes only for a token of at least {MIN_ORACLE_TOKEN_LENGTH}"
        )
    parts = urlsplit(settings.oracle_backend_url)
    if parts.scheme == "http" and (parts.hostname or "").lower() not in LOCAL_HOSTS:
        warnings.append(
            "ORACLE_BACKEND_URL is plain http to a host that is not this machine: the token "
            "would cross the network unencrypted"
        )
    if settings.oracle_mode == "active":
        warnings.append(
            "ORACLE_MODE=active: a backend that cannot be reached means no entries at all"
        )
    return warnings


def check_config(env_file: str = ".env") -> ConfigReport:
    report = ConfigReport(warnings=unknown_key_warnings(env_file))
    try:
        settings = Settings(_env_file=env_file)
        report.warnings.extend(oracle_warnings(settings))
    except ValidationError as e:
        for problem in e.errors():
            name = ".".join(str(part) for part in problem["loc"]).upper() or "SETTINGS"
            if problem["type"] == "missing":
                report.errors.append(f"{name} is not set (in the environment or in {env_file})")
            else:
                # The message only: the offending value could be a key.
                report.errors.append(f"{name}: {problem['msg']}")
    return report
