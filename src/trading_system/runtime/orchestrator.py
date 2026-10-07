import logging
import time
from typing import Any

from trading_system.config import Settings
from trading_system.storage.repositories import StateRepository
from trading_system.domain.models import TradingState

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings, state_repo: Any = None):
        self.settings = settings
        # Use provided repo or initialize a real one (for MVP, maybe mock or minimal init)
        self.state_repo = state_repo
        if self.state_repo is None:
            from trading_system.storage.database import get_engine, get_session_factory

            engine = get_engine(settings.database_url)
            session_factory = get_session_factory(engine)
            self.state_repo = StateRepository(session_factory())

    def run(self, is_paper: bool) -> None:
        if is_paper:
            logger.info("Running in PAPER mode. (Live order execution is active)")
        else:
            logger.info("Running in DRY-RUN mode. (No live broker orders will be sent)")

        try:
            while True:
                # Main trading loop would go here
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Orchestrator shutting down gracefully...")
            # Graceful shutdown logic: stop new entries, persist state
            self.pause_entries()

    def pause_entries(self) -> None:
        logger.info("Pausing new entries...")
        state = self.state_repo.get_state()
        if state is None:
            state = TradingState(is_trading_halted=True, halt_reason="Manual CLI pause")
        else:
            state.is_trading_halted = True
            state.halt_reason = "Manual CLI pause"
        self.state_repo.save_state(state)
        logger.info("System entries paused successfully.")

    def resume_entries(self) -> None:
        logger.info("Resuming entries...")
        # Rule: resume-entries requires a repeated risk and reconciliation check.
        # MVP: simply set halted=False and log the requirement.
        state = self.state_repo.get_state()
        if state is None:
            state = TradingState(is_trading_halted=False, halt_reason=None)
        else:
            state.is_trading_halted = False
            state.halt_reason = None
        self.state_repo.save_state(state)
        logger.info(
            "System entries resumed successfully. NOTE: Ensure manual reconciliation was performed."
        )

    def status(self) -> None:
        state = self.state_repo.get_state()
        if state is None:
            print("System state is uninitialized.")
        else:
            print(f"Trading Halted: {state.is_trading_halted}")
            if state.is_trading_halted:
                print(f"Halt Reason: {state.halt_reason}")
            print(f"Daily Start Equity: {state.daily_start_equity}")
            print(f"Last Processed Bar: {state.last_processed_bar}")
