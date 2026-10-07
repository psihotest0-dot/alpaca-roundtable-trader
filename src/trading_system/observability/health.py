

class HealthStatus:
    """Tracks the health and readiness of the trading system."""

    def __init__(self) -> None:
        self._is_running = True

        # State tracking
        self.market_data_stale = False
        self.reconciliation_failed = False
        self.oracle_failed = False
        self.daily_loss_breached = False
        self.emergency_halt = False

    def check_liveness(self) -> bool:
        """
        Liveness indicates if the process is functioning.
        If this returns False, the orchestrator/container should be restarted.
        """
        return self._is_running

    def check_readiness(self) -> bool:
        """
        Readiness indicates if it is safe to process NEW inputs (e.g., new entries).
        If this returns False, new signals should be rejected, but exits/reconciliation
        can and must continue.
        """
        if not self._is_running:
            return False

        if self.emergency_halt:
            return False

        if self.market_data_stale:
            return False

        if self.reconciliation_failed:
            return False

        if self.daily_loss_breached:
            return False

        # Note: Oracle failure explicitly does NOT affect readiness.
        # Defensive exits and reconciliation mechanisms operate entirely
        # independently of the Oracle.
        # If the Oracle fails, we might reject new entries that *require* it,
        # but the system as a whole remains ready for exits and reconciliation.

        return True

    def mark_market_data_stale(self, is_stale: bool) -> None:
        self.market_data_stale = is_stale

    def mark_reconciliation_failed(self, has_failed: bool) -> None:
        self.reconciliation_failed = has_failed

    def mark_oracle_failed(self, has_failed: bool) -> None:
        self.oracle_failed = has_failed

    def mark_daily_loss_breached(self, is_breached: bool) -> None:
        self.daily_loss_breached = is_breached

    def trigger_emergency_halt(self) -> None:
        self.emergency_halt = True

    def set_running(self, is_running: bool) -> None:
        self._is_running = is_running
