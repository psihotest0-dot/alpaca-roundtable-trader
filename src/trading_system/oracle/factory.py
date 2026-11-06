from typing import Optional

from trading_system.config import Settings
from trading_system.oracle.backend import BackendOracle
from trading_system.oracle.contract import OracleProvider
from trading_system.oracle.null import NullOracle
from trading_system.oracle.recording import RecordingOracle
from trading_system.oracle.shadow import ShadowOracle
from trading_system.storage.repositories import OracleDecisionRepository


def create_oracle(
    settings: Settings, repository: Optional[OracleDecisionRepository] = None
) -> Optional[OracleProvider]:
    """The Oracle the settings ask for, or None when it is switched off (ORACLE_ENABLED).

    With a repository its answers are recorded and final ones are not asked twice. A real
    provider in shadow mode (ORACLE_MODE=shadow, the default) is wrapped last, outside the
    recording: the record keeps what the model said and the loop is told APPROVE (oracle/shadow.py).
    """
    if not settings.oracle_enabled:
        return None
    provider: OracleProvider
    if settings.oracle_provider == "null":
        provider = NullOracle()
    elif settings.oracle_provider == "backend":
        provider = BackendOracle.from_settings(settings)
    else:
        raise ValueError(f"Unknown ORACLE_PROVIDER {settings.oracle_provider!r}")
    recorded: OracleProvider = (
        RecordingOracle(provider, repository) if repository is not None else provider
    )
    if settings.oracle_provider != "null" and settings.oracle_mode == "shadow":
        return ShadowOracle(recorded)
    return recorded
