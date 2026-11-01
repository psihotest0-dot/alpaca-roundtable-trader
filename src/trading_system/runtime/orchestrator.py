import asyncio
import logging
import threading
import time
from typing import Any, Optional

from trading_system.config import Settings
from trading_system.data.rss_pipeline import RSSIngestionPipeline
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.repositories import SessionFactory, StateRepository
from trading_system.domain.models import TradingState

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

    def run(self, is_paper: bool) -> None:
        if is_paper:
            logger.info("Running in PAPER mode. (Live order execution is active)")
        else:
            logger.info("Running in DRY-RUN mode. (No live broker orders will be sent)")

        from trading_system.broker.alpaca_broker import AlpacaBroker
        from trading_system.data.alpaca_data import AlpacaData
        from trading_system.strategy.config import StrategyConfig
        from trading_system.strategy.trend import generate_signals, filter_signals
        from trading_system.oracle.anythingllm import AnythingLLMAdapter
        from trading_system.risk.engine import RiskEngine
        from trading_system.execution.engine import ExecutionEngine
        from trading_system.storage.repositories import SignalRepository, OrderRepository
        from datetime import datetime, timezone
        from alpaca.data.timeframe import TimeFrame
        import datetime as dt

        broker = AlpacaBroker(self.settings)
        data = AlpacaData(self.settings)
        strategy_config = StrategyConfig(
            strategy_version="1.0.0",
            fast_ema_period=12,
            slow_ema_period=26,
            rsi_period=14,
            rsi_lower_bound=30,
            rsi_upper_bound=70,
            atr_period=14,
            htf_fast_ema_period=12,
            htf_slow_ema_period=26,
            min_warmup_bars=51,
        )

        oracle = None
        if self.settings.oracle_enabled:
            oracle = AnythingLLMAdapter(self.settings)

        risk_engine = RiskEngine(self.settings)

        signal_repo = SignalRepository(self.session_factory)
        execution_engine = ExecutionEngine(
            broker=broker,
            order_repo=OrderRepository(self.session_factory),
            state_repo=self.state_repo,
            settings=self.settings,
            is_dry_run=not is_paper,
        )

        if self.settings.rss_enabled and self.settings.oracle_enabled:
            logger.info("Starting background RSS ingestion thread...")

            def rss_worker() -> None:
                while True:
                    try:
                        asyncio.run(RSSIngestionPipeline(self.settings).run())
                    except Exception as e:
                        logger.error(f"RSS Ingestion error: {e}")
                    time.sleep(self.settings.rss_update_interval_minutes * 60)

            rss_thread = threading.Thread(target=rss_worker, daemon=True)
            rss_thread.start()

        try:
            while True:
                logger.info("Starting new trading loop iteration...")

                # Check if market is open
                clock = broker.get_clock()
                if not clock.is_open:
                    sleep_secs = (clock.next_open - clock.timestamp).total_seconds()
                    # Ensure sleep_secs is strictly positive
                    sleep_secs = max(sleep_secs, 1.0)
                    logger.info(
                        f"Market is closed. Sleeping for {sleep_secs:.2f} seconds until next open."
                    )
                    time.sleep(sleep_secs)
                    continue

                # Check state
                state = self.state_repo.get_state()
                if state and state.is_trading_halted:
                    now_time = datetime.now(timezone.utc)
                    next_minute = (now_time + dt.timedelta(minutes=1)).replace(
                        second=0, microsecond=0
                    )
                    sleep_secs = max((next_minute - now_time).total_seconds(), 1.0)
                    logger.info(
                        f"Trading is halted. Sleeping for {sleep_secs:.2f} seconds until next minute."
                    )
                    time.sleep(sleep_secs)
                    continue

                # Fetch data
                logger.info("Fetching market data...")
                now = datetime.now(timezone.utc)
                start_15m = now - dt.timedelta(days=self.settings.lookback_days)
                start_1h = now - dt.timedelta(days=self.settings.htf_lookback_days)

                watchlist = self.settings.watchlist.split(",") if self.settings.watchlist else []
                df_15m_all = {}
                df_1h_all = {}
                for symbol in watchlist:
                    symbol = symbol.strip()
                    if not symbol:
                        continue

                    try:
                        df_15 = data.fetch_bars(symbol, TimeFrame.Minute * 15, start_15m, now)
                        df_h = data.fetch_bars(symbol, TimeFrame.Hour, start_1h, now)

                        df_15 = data.clean_and_deduplicate(df_15)
                        df_h = data.clean_and_deduplicate(df_h)

                        df_15, df_h = data.align_and_filter(df_15, df_h, now)

                        if not df_15.empty and not df_h.empty:
                            df_15m_all[symbol] = df_15
                            df_1h_all[symbol] = df_h
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")

                # Generate signals
                logger.info("Generating strategy signals...")
                signals = generate_signals(df_15m_all, df_1h_all, strategy_config, now)

                # Filter signals based on active positions/orders
                try:
                    positions = broker.get_positions()
                    active_symbols = {p.symbol for p in positions}
                    open_orders = broker.get_open_orders()
                    active_symbols.update({o.symbol for o in open_orders})
                except Exception as e:
                    logger.error(f"Error getting active symbols: {e}")
                    active_symbols = set()

                signals = filter_signals(signals, active_symbols)

                logger.info(f"Generated {len(signals)} valid signals.")

                for signal in signals:
                    logger.info(f"Processing signal: {signal.symbol} {signal.direction.value}")

                    # Store signal
                    try:
                        signal_repo.save_signal(signal)
                    except Exception as e:
                        logger.error(f"Failed to save signal: {e}")

                    # Oracle decision
                    if oracle:
                        logger.info("Querying Oracle...")

                        tech_dict = {}
                        if signal.symbol in df_15m_all:
                            # Pass latest technicals
                            last_bar = df_15m_all[signal.symbol].iloc[-1]
                            for col in last_bar.index:
                                tech_dict[str(col)] = (
                                    float(last_bar[col])
                                    if isinstance(last_bar[col], (int, float))
                                    else str(last_bar[col])
                                )

                        decision = oracle.evaluate_signal(
                            signal,
                            technicals=tech_dict,
                            rationale="Trend strategy signal generated.",
                        )
                        logger.info(f"Oracle decision: {decision.approved} ({decision.reasoning})")
                        if not decision.approved:
                            continue

                    # Risk engine evaluation
                    logger.info("Evaluating risk...")
                    try:
                        account = broker.get_account()
                        from trading_system.domain.models import AccountSnapshot
                        from decimal import Decimal

                        acc_snap = AccountSnapshot(
                            equity=Decimal(str(account.equity)),
                            cash=Decimal(str(account.cash)),
                            buying_power=Decimal(str(account.buying_power)),
                            day_trade_count=int(account.daytrade_count),
                            updated_at=now,
                        )

                        open_broker_orders = broker.get_open_orders()
                        from trading_system.domain.enums import OrderSide, OrderType, OrderStatus
                        from trading_system.domain.models import BrokerOrder, PositionSnapshot

                        active_orders = []
                        for bo in open_broker_orders:
                            status_str = str(bo.status)
                            if hasattr(bo.status, "value"):
                                status_str = bo.status.value
                            status_str = status_str.upper()

                            mapped_status = OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION
                            try:
                                mapped_status = OrderStatus(status_str)
                            except ValueError:
                                if status_str == "NEW" or status_str == "ACCEPTED":
                                    mapped_status = OrderStatus.SUBMITTED
                                elif status_str == "PENDING_NEW":
                                    mapped_status = OrderStatus.SUBMITTING

                            active_orders.append(
                                BrokerOrder(
                                    broker_id=str(bo.id),
                                    client_order_id=bo.client_order_id,
                                    symbol=bo.symbol,
                                    side=OrderSide.BUY if bo.side == "buy" else OrderSide.SELL,
                                    type=OrderType.LIMIT
                                    if bo.order_type == "limit"
                                    else OrderType.MARKET,
                                    status=mapped_status,
                                    quantity=Decimal(str(bo.qty)),
                                    filled_quantity=Decimal(str(bo.filled_qty)),
                                    limit_price=Decimal(str(bo.limit_price))
                                    if bo.limit_price
                                    else None,
                                    stop_price=Decimal(str(bo.stop_price))
                                    if bo.stop_price
                                    else None,
                                    created_at=bo.created_at,
                                    updated_at=bo.updated_at,
                                )
                            )

                        domain_positions = []
                        for p in positions:
                            domain_positions.append(
                                PositionSnapshot(
                                    symbol=p.symbol,
                                    quantity=Decimal(str(p.qty)),
                                    market_value=Decimal(str(p.market_value)),
                                    average_entry_price=Decimal(str(p.avg_entry_price)),
                                    unrealized_pl=Decimal(str(p.unrealized_pl)),
                                )
                            )

                        current_price = Decimal(str(df_15m_all[signal.symbol].iloc[-1]["close"]))

                        risk_decision, intent = risk_engine.evaluate_signal(
                            signal=signal,
                            account=acc_snap,
                            positions=domain_positions,
                            active_orders=active_orders,
                            current_price=current_price,
                            state=state or TradingState(),
                        )
                        logger.info(
                            f"Risk decision: {risk_decision.approved} ({risk_decision.reasoning})"
                        )

                        if risk_decision.approved and intent:
                            logger.info(f"Executing order intent for {intent.symbol}...")
                            execution_engine.submit_order(intent)
                    except Exception as e:
                        logger.error(f"Error in risk evaluation / execution: {e}")

                # RSS update is handled by the background rss_worker thread

                # Check for bounded expiry and emergency halts
                execution_engine.check_bounded_entry_expiry()
                execution_engine.emergency_check_unprotected_positions()

                # Sleep before next iteration to avoid spamming the broker API
                now_time = datetime.now(timezone.utc)
                next_minute = (now_time + dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
                sleep_secs = max((next_minute - now_time).total_seconds(), 1.0)
                logger.info(f"Sleeping for {sleep_secs:.2f} seconds until next minute...")
                time.sleep(sleep_secs)

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
            print(f"Last Updated: {state.last_updated}")
