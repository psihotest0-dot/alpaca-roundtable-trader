import asyncio
import logging
import signal
import threading
import time
from typing import Any, Callable, Optional

from trading_system.config import Settings
from trading_system.data.rss_pipeline import RSSIngestionPipeline
from trading_system.domain.models import TradingState
from trading_system.runtime.cycle import TradingCycle
from trading_system.runtime.runner import Runner
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.repositories import SessionFactory, StateRepository
from trading_system.strategy.config import RUNTIME_STRATEGY

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings, state_repo: Any = None):
        self.settings = settings
        self._session_factory: Optional[SessionFactory] = None
        # Use the provided repo (tests) or a real one on the configured database.
        self.state_repo = (
            state_repo if state_repo is not None else StateRepository(self.session_factory)
        )

    @property
    def session_factory(self) -> SessionFactory:
        """One engine per Orchestrator, created on first use so that an injected
        state repository never needs a database."""
        if self._session_factory is None:
            self._session_factory = get_session_factory(get_engine(self.settings.database_url))
        return self._session_factory

    def build_cycle(self, is_paper: bool) -> TradingCycle:
        """Wires the broker, the data, the engines and the repositories into one cycle."""
        from trading_system.broker.alpaca_broker import AlpacaBroker
        from trading_system.data.alpaca_data import AlpacaData
        from trading_system.execution.engine import ExecutionEngine
        from trading_system.oracle.anythingllm import AnythingLLMAdapter
        from trading_system.risk.engine import RiskEngine
        from trading_system.storage.repositories import OrderRepository, SignalRepository

        broker = AlpacaBroker(self.settings)
        order_repo = OrderRepository(self.session_factory)
        return TradingCycle(
            settings=self.settings,
            broker=broker,
            data=AlpacaData(self.settings),
            risk_engine=RiskEngine(self.settings),
            execution=ExecutionEngine(
                broker=broker,
                order_repo=order_repo,
                state_repo=self.state_repo,
                settings=self.settings,
                is_dry_run=not is_paper,
            ),
            signal_repo=SignalRepository(self.session_factory),
            order_repo=order_repo,
            state_repo=self.state_repo,
            strategy_config=RUNTIME_STRATEGY,
            oracle=AnythingLLMAdapter(self.settings) if self.settings.oracle_enabled else None,
        )

    def run(self, is_paper: bool, handle_signals: bool = True) -> None:
        """Runs until SIGINT or SIGTERM (Ctrl+C, `docker stop`), then pauses new entries.

        `handle_signals=False` leaves the signals alone (tests, or a host that has its own)."""
        if is_paper:
            logger.info("Running in PAPER mode. (Live order execution is active)")
        else:
            logger.info("Running in DRY-RUN mode. (No live broker orders will be sent)")

        cycle = self.build_cycle(is_paper)
        runner = Runner(cycle, cycle.broker)
        self._start_rss_worker()
        restore = self._stop_on_signals(runner) if handle_signals else None
        try:
            runner.run()
        except KeyboardInterrupt:
            pass
        finally:
            if restore:
                restore()

        logger.info("Orchestrator shutting down gracefully...")
        # Graceful shutdown: stop new entries and persist that. Protective stops of positions
        # that are already open stay at the broker.
        self.pause_entries()

    def _stop_on_signals(self, runner: Runner) -> Optional[Callable[[], None]]:
        """Makes SIGINT and SIGTERM end the runner's wait at once. Returns a function that puts
        the previous handlers back, or None when this is not the main thread (where Python
        allows no signal handlers)."""
        if threading.current_thread() is not threading.main_thread():
            return None

        def handler(signum: int, _frame: Any) -> None:
            logger.info(f"Received signal {signum}; stopping.")
            runner.stop()

        names = ("SIGINT", "SIGTERM", "SIGBREAK")  # SIGBREAK: Ctrl+Break on Windows
        previous = {
            number: signal.signal(number, handler)
            for number in (getattr(signal, name) for name in names if hasattr(signal, name))
        }

        def restore() -> None:
            for number, old in previous.items():
                signal.signal(number, old)

        return restore

    def _start_rss_worker(self) -> None:
        if not (self.settings.rss_enabled and self.settings.oracle_enabled):
            return
        logger.info("Starting background RSS ingestion thread...")

        def rss_worker() -> None:
            while True:
                try:
                    asyncio.run(RSSIngestionPipeline(self.settings).run())
                except Exception as e:
                    logger.error(f"RSS Ingestion error: {e}")
                time.sleep(self.settings.rss_update_interval_minutes * 60)

        threading.Thread(target=rss_worker, daemon=True).start()

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
            print(f"Daily Equity Date: {state.daily_equity_date}")
            print(f"Last Updated: {state.last_updated}")
