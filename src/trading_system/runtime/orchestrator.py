import logging
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from trading_system.config import Settings
from trading_system.domain.models import TradingState
from trading_system.runtime.cycle import TradingCycle
from trading_system.runtime.account_check import AccountCheck
from trading_system.runtime.heartbeat import Heartbeat
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
        from trading_system.oracle import create_oracle
        from trading_system.risk.engine import RiskEngine
        from trading_system.observability.alerts import LoggingAlertManager
        from trading_system.storage.repositories import (
            AuditRepository,
            OracleDecisionRepository,
            OrderRepository,
            RiskDecisionRepository,
            SignalRepository,
        )

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
            oracle=create_oracle(self.settings, OracleDecisionRepository(self.session_factory)),
            risk_repo=RiskDecisionRepository(self.session_factory),
            audit_repo=AuditRepository(self.session_factory),
            alerts=LoggingAlertManager(logger),
            news=self._open_news(),
        )

    def build_observation_cycle(self) -> Any:
        """The Oracle observer (runtime/observe.py): a cycle that can ask the Oracle about signals
        and can place no order. It reads the bars from the observation store."""
        from trading_system.broker.alpaca_broker import AlpacaBroker
        from trading_system.data.alpaca_data import AlpacaData
        from trading_system.data.universe import observation_universe, observed_symbols
        from trading_system.oracle import create_oracle
        from trading_system.research.candidates import RECORDED
        from trading_system.runtime.observe import ObservationCycle
        from trading_system.storage.repositories import (
            AuditRepository,
            OracleDecisionRepository,
            SignalRepository,
        )
        from trading_system.storage.research import open_observe_store
        from trading_system.strategy.registry import get_strategy

        watchlist = self.settings.watchlist_symbols
        return ObservationCycle(
            settings=self.settings,
            broker=AlpacaBroker(self.settings),
            data=AlpacaData(self.settings),
            store=open_observe_store(
                self.settings.observe_database_url, self.settings.database_url
            ),
            universe=observation_universe(watchlist),
            observed=observed_symbols(watchlist),
            signal_repo=SignalRepository(self.session_factory),
            state_repo=self.state_repo,
            audit_repo=AuditRepository(self.session_factory),
            strategy_config=RUNTIME_STRATEGY,
            oracle=create_oracle(self.settings, OracleDecisionRepository(self.session_factory)),
            news=self._open_news(),
            candidates=[get_strategy(name) for name in RECORDED],
        )

    def run_observation(self, handle_signals: bool = True) -> None:
        """Runs the observer until SIGINT or SIGTERM. Nothing is paused on the way out: it holds
        no orders and no positions."""
        if not (self.settings.oracle_enabled and self.settings.oracle_provider == "backend"):
            raise RuntimeError(
                "the observer has nobody to ask: set ORACLE_ENABLED=true and "
                "ORACLE_PROVIDER=backend"
            )
        logger.info(
            "Running the Oracle OBSERVER: signals are put to the Oracle, no order is placed."
        )
        cycle = self.build_observation_cycle()
        self._start_metrics()
        runner = Runner(
            cycle, cycle.broker, heartbeat=Heartbeat(Path(self.settings.heartbeat_path))
        )
        restore = self._stop_on_signals(runner) if handle_signals else None
        try:
            runner.run()
        except KeyboardInterrupt:
            pass
        finally:
            if restore:
                restore()
        logger.info("Observer shutting down.")

    def _start_metrics(self) -> None:
        """Serves /metrics when METRICS_ENABLED is on (docs: the backend's Prometheus scrapes it). A
        port that cannot be taken is an error in the log and nothing more: the metrics are an
        addition, they must not stop the trading."""
        if not self.settings.metrics_enabled:
            return
        try:
            from trading_system.observability.metrics import start_metrics_server

            start_metrics_server(self.settings.metrics_port, self.settings.metrics_host)
            logger.info(
                f"Metrics on {self.settings.metrics_host}:{self.settings.metrics_port}/metrics."
            )
        except Exception as e:
            logger.error(
                f"Could not start the metrics server on {self.settings.metrics_host}:"
                f"{self.settings.metrics_port}: {e}. The trading goes on without it."
            )

    def _open_news(self) -> Any:
        """The news cache for the Oracle, or None if the Oracle is off or the cache cannot be
        opened (then the Oracle is asked without news; the cache is filled by `fetch-news`)."""
        if not self.settings.oracle_enabled:
            return None
        try:
            from trading_system.storage.news import open_news_store

            return open_news_store(
                self.settings.news_database_url,
                self.settings.database_url,
                self.settings.research_database_url,
            )
        except Exception as e:
            logger.warning(f"The news cache could not be opened; the Oracle gets no news: {e}")
            return None

    def run(
        self,
        is_paper: bool,
        handle_signals: bool = True,
        pause_on_shutdown: Optional[bool] = None,
    ) -> None:
        """Runs until SIGINT or SIGTERM (Ctrl+C, `docker stop`).

        When it stops, new entries are paused only if `pause_on_shutdown` says so (default: the
        PAUSE_ON_SHUTDOWN setting, which is off). Otherwise the next start carries on by itself.

        `handle_signals=False` leaves the signals alone (tests, or a host that has its own)."""
        if is_paper:
            logger.info("Running in PAPER mode. (Live order execution is active)")
        else:
            logger.info("Running in DRY-RUN mode. (No live broker orders will be sent)")

        cycle = self.build_cycle(is_paper)
        self._start_metrics()
        runner = Runner(
            cycle, cycle.broker, heartbeat=Heartbeat(Path(self.settings.heartbeat_path))
        )
        restore = self._stop_on_signals(runner) if handle_signals else None
        try:
            runner.run()
        except KeyboardInterrupt:
            pass
        finally:
            if restore:
                restore()

        logger.info("Orchestrator shutting down gracefully...")
        if pause_on_shutdown if pause_on_shutdown is not None else self.settings.pause_on_shutdown:
            # Stop new entries and persist that: the next start waits for `resume-entries`.
            self.pause_entries()
        else:
            logger.info("Entries are not paused: the next start carries on by itself.")
        # Either way, the protective stops of open positions stay at the broker.

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

    def reconcile_now(self, is_paper: bool = False) -> "AccountCheck":
        """Compares the account with our records once (see runtime/account_check.py)."""
        from trading_system.runtime.account_check import check_account

        cycle = self.build_cycle(is_paper)
        return check_account(cycle.broker, cycle.execution, cycle.order_repo)

    def resume_entries_checked(
        self, *, force: bool = False
    ) -> tuple[bool, Optional["AccountCheck"]]:
        """Lifts a halt only after the account has been looked at again. Returns (resumed, the
        check). Not halted: nothing to do, no check. Problems found: the halt stays, unless
        `force` says the owner has seen them and goes on anyway."""
        state = self.state_repo.get_state()
        if state is None or not state.is_trading_halted:
            logger.info("Entries are not halted; nothing to resume.")
            return True, None
        check = self.reconcile_now()
        if check.ok or force:
            if not check.ok:
                logger.warning(f"Resuming despite: {'; '.join(check.problems)}")
            self.resume_entries()
            return True, check
        logger.error(f"Not resuming: {'; '.join(check.problems)}")
        return False, check

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
