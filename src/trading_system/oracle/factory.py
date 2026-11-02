from typing import Optional

from trading_system.config import Settings
from trading_system.oracle.contract import OracleProvider
from trading_system.oracle.null import NullOracle


def create_oracle(settings: Settings) -> Optional[OracleProvider]:
    """The Oracle the settings ask for, or None when it is switched off (ORACLE_ENABLED)."""
    if not settings.oracle_enabled:
        return None
    if settings.oracle_provider == "null":
        return NullOracle()
    raise ValueError(f"Unknown ORACLE_PROVIDER {settings.oracle_provider!r}")
