import logging
import threading
from typing import Optional
from decimal import Decimal
import datetime
import time

from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderClass, OrderSide as AlpacaOrderSide, TimeInForce

from trading_system.broker.alpaca_broker import AlpacaBroker
from trading_system.storage.repositories import OrderRepository, StateRepository
from trading_system.domain.models import OrderIntent, BrokerOrder
from trading_system.domain.enums import OrderSide, OrderType, OrderStatus
from trading_system.config import Settings

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        broker: AlpacaBroker,
        order_repo: OrderRepository,
        state_repo: StateRepository,
        settings: Settings,
        is_dry_run: bool = False,
    ):
        self.broker = broker
        self.order_repo = order_repo
        self.state_repo = state_repo
        self.settings = settings
        self.is_dry_run = is_dry_run
        self._lock = threading.Lock()

    def _generate_client_order_id(self, signal_id: str) -> str:
        # Deterministic client_order_id according to signal ID
        return f"exec_{signal_id}"

    def halt_trading(self, reason: str) -> None:
        logger.critical(f"HALTING TRADING: {reason}")
        state = self.state_repo.get_state()
        state.is_trading_halted = True
        state.halt_reason = reason
        self.state_repo.save_state(state)

    from typing import Any

    def _validate_price_increment(self, price: Decimal, asset: Any) -> bool:
        # Validate price increments based on US Equity rules if not specified by asset
        # $0.01 for >= $1.00, $0.0001 for < $1.00

        inc = asset.price_increment
        if inc is None:
            if price >= Decimal("1.00"):
                inc = 0.01
            else:
                inc = 0.0001

        # Avoid float precision issues
        inc_dec = Decimal(str(inc))

        # Check if price modulo increment is 0
        return (price % inc_dec) == Decimal("0")

    def submit_order(self, intent: OrderIntent) -> Optional[BrokerOrder]:
        with self._lock:
            # Check state
            state = self.state_repo.get_state()
            if state.is_trading_halted:
                logger.error("Cannot submit order: Trading is halted.")
                return None

            client_order_id = self._generate_client_order_id(intent.signal_id)

            # MVP constraints
            if not self.settings.is_paper:
                logger.error("Live execution is not supported.")
                return None

            if intent.quantity != int(intent.quantity):
                logger.error("Only whole shares are supported.")
                return None

            if intent.side != OrderSide.BUY:
                logger.error("Only long entries are supported.")
                return None

            if intent.type != OrderType.LIMIT or intent.limit_price is None:
                logger.error("Only limit entries are supported.")
                return None

            if intent.stop_price is None:
                logger.error("Bracket orders require a stop price.")
                return None

            # Fetch asset info
            try:
                asset = self.broker.get_asset(intent.symbol)
            except Exception as e:
                logger.error(f"Failed to fetch asset info: {e}")
                return None

            if not self._validate_price_increment(intent.limit_price, asset):
                logger.error(f"Invalid limit price increment for {intent.limit_price}")
                return None

            if not self._validate_price_increment(intent.stop_price, asset):
                logger.error(f"Invalid stop price increment for {intent.stop_price}")
                return None

            # Calculate take profit price
            take_profit_price = intent.limit_price * (
                Decimal("1") + Decimal(str(self.settings.take_profit_pct))
            )

            if not self._validate_price_increment(take_profit_price, asset):
                # Round to nearest valid increment
                inc = asset.price_increment
                if inc is None:
                    inc = 0.01 if take_profit_price >= Decimal("1.00") else 0.0001
                inc_dec = Decimal(str(inc))
                take_profit_price = (take_profit_price // inc_dec) * inc_dec

            if self.is_dry_run:
                logger.info(f"[DRY-RUN] Would submit order {client_order_id} for {intent.symbol}")
                broker_order = BrokerOrder(
                    broker_id=f"dry_run_{client_order_id}",
                    client_order_id=client_order_id,
                    symbol=str(intent.symbol),
                    side=intent.side,
                    type=intent.type,
                    status=OrderStatus.SUBMITTED,
                    quantity=intent.quantity,
                    filled_quantity=Decimal("0"),
                    limit_price=intent.limit_price,
                    stop_price=intent.stop_price,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                )
                try:
                    self.order_repo.save_broker_order(broker_order)
                except Exception as e:
                    logger.error(f"Failed to persist dry-run broker order {client_order_id}: {e}")
                    return None
                return broker_order

            # Check for existing
            try:
                # In real scenario we would check DB. For idempotency, rely on client_order_id lookup later
                # We save intent first
                self.order_repo.save_intent(intent)
            except Exception as e:
                logger.error(f"Failed to persist intent: {e}")
                # We can return None since nothing was sent to broker
                return None

            req = LimitOrderRequest(
                symbol=intent.symbol,
                qty=float(intent.quantity),
                side=AlpacaOrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=float(intent.limit_price),
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=float(take_profit_price)),
                stop_loss=StopLossRequest(stop_price=float(intent.stop_price)),
                client_order_id=client_order_id,
                extended_hours=False,
            )

            # Persist an initial broker order in SUBMITTING state to track the in-flight API call
            # This handles the "crash after submission before DB update" by leaving a SUBMITTING record
            inflight_order = BrokerOrder(
                broker_id=None,  # the broker has not answered yet
                client_order_id=client_order_id,
                symbol=intent.symbol,
                side=intent.side,
                type=intent.type,
                status=OrderStatus.SUBMITTING,
                quantity=intent.quantity,
                filled_quantity=Decimal("0"),
                limit_price=intent.limit_price,
                stop_price=intent.stop_price,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                updated_at=datetime.datetime.now(datetime.timezone.utc),
            )
            try:
                self.order_repo.save_broker_order(inflight_order)
            except Exception as e:
                logger.error(f"Failed to save inflight order: {e}")
                return None

            try:
                alpaca_order = self.broker.submit_order(req)

                broker_order = BrokerOrder(
                    broker_id=str(alpaca_order.id),
                    client_order_id=client_order_id,
                    symbol=alpaca_order.symbol,
                    side=intent.side,
                    type=intent.type,
                    status=OrderStatus.SUBMITTED,
                    quantity=intent.quantity,
                    filled_quantity=Decimal("0"),
                    limit_price=intent.limit_price,
                    stop_price=intent.stop_price,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                )

                self.order_repo.save_broker_order(broker_order)

                # Track child orders (legs) if they exist
                if hasattr(alpaca_order, "legs") and alpaca_order.legs:
                    for leg in alpaca_order.legs:
                        leg_order = BrokerOrder(
                            broker_id=str(leg.id),
                            client_order_id=leg.client_order_id,
                            symbol=leg.symbol,
                            side=OrderSide.BUY if leg.side == "buy" else OrderSide.SELL,
                            type=OrderType.LIMIT if leg.order_type == "limit" else OrderType.STOP,
                            status=OrderStatus.CREATED,  # Usually they start created/held
                            quantity=Decimal(str(leg.qty)),
                            filled_quantity=Decimal(str(leg.filled_qty)),
                            limit_price=Decimal(str(leg.limit_price)) if leg.limit_price else None,
                            stop_price=Decimal(str(leg.stop_price)) if leg.stop_price else None,
                            created_at=leg.created_at,
                            updated_at=leg.updated_at,
                        )
                        self.order_repo.save_broker_order(leg_order)

                return broker_order

            except Exception as e:
                logger.error(f"Order submission error: {e}")

                # Check if it was accepted but we crashed/timed out
                # Wait a bit then reconcile
                try:
                    time.sleep(1)
                    reconciled_order = self.reconcile_order(client_order_id)
                    if reconciled_order:
                        return reconciled_order
                    else:
                        logger.critical(
                            "Reconciliation failed after submission crash. Halting trading to prevent conflicting inputs."
                        )
                        self.halt_trading("Reconciliation failed after order submission crash.")
                except Exception as rec_e:
                    logger.critical(f"Failed to reconcile after exception: {rec_e}")
                    self.halt_trading("Reconciliation exception after order submission crash.")

                return None

    def reconcile_order(self, client_order_id: str) -> Optional[BrokerOrder]:
        # Never blind resubmit
        try:
            alpaca_order = self.broker.get_order_by_client_id(client_order_id)

            # Found it, save it

            # Map status
            status_str = str(alpaca_order.status)
            if hasattr(alpaca_order.status, "value"):
                status_str = alpaca_order.status.value
            status_str = status_str.upper()

            mapped_status = OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION
            try:
                mapped_status = OrderStatus(status_str)
            except ValueError:
                if status_str == "NEW" or status_str == "ACCEPTED":
                    mapped_status = OrderStatus.SUBMITTED
                elif status_str == "PENDING_NEW":
                    mapped_status = OrderStatus.SUBMITTING
                pass

            broker_order = BrokerOrder(
                broker_id=str(alpaca_order.id),
                client_order_id=client_order_id,
                symbol=alpaca_order.symbol,
                side=OrderSide.BUY if alpaca_order.side == "buy" else OrderSide.SELL,
                type=OrderType.LIMIT if alpaca_order.order_type == "limit" else OrderType.MARKET,
                status=mapped_status,
                quantity=Decimal(str(alpaca_order.qty)),
                filled_quantity=Decimal(str(alpaca_order.filled_qty)),
                limit_price=Decimal(str(alpaca_order.limit_price))
                if alpaca_order.limit_price
                else None,
                stop_price=Decimal(str(alpaca_order.stop_price))
                if alpaca_order.stop_price
                else None,
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at,
            )
            self.order_repo.save_broker_order(broker_order)

            # Track child orders (legs) if they exist
            if hasattr(alpaca_order, "legs") and alpaca_order.legs:
                for leg in alpaca_order.legs:
                    leg_status_str = str(leg.status)
                    if hasattr(leg.status, "value"):
                        leg_status_str = leg.status.value
                    leg_status_str = leg_status_str.upper()

                    leg_mapped_status = OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION
                    try:
                        leg_mapped_status = OrderStatus(leg_status_str)
                    except ValueError:
                        if leg_status_str == "NEW" or leg_status_str == "ACCEPTED":
                            leg_mapped_status = OrderStatus.SUBMITTED
                        elif leg_status_str == "PENDING_NEW":
                            leg_mapped_status = OrderStatus.SUBMITTING
                        elif leg_status_str == "HELD":
                            leg_mapped_status = OrderStatus.CREATED

                    leg_order = BrokerOrder(
                        broker_id=str(leg.id),
                        client_order_id=leg.client_order_id,
                        symbol=leg.symbol,
                        side=OrderSide.BUY if leg.side == "buy" else OrderSide.SELL,
                        type=OrderType.LIMIT if leg.order_type == "limit" else OrderType.STOP,
                        status=leg_mapped_status,
                        quantity=Decimal(str(leg.qty)),
                        filled_quantity=Decimal(str(leg.filled_qty)),
                        limit_price=Decimal(str(leg.limit_price)) if leg.limit_price else None,
                        stop_price=Decimal(str(leg.stop_price)) if leg.stop_price else None,
                        created_at=leg.created_at,
                        updated_at=leg.updated_at,
                    )
                    self.order_repo.save_broker_order(leg_order)

            return broker_order
        except Exception as e:
            logger.warning(f"Reconciliation lookup failed for {client_order_id}: {e}")
            return None

    def cancel_bracket_order(self, client_order_id: str) -> None:
        """
        Cancels a bracket order.
        Alpaca behavior: When a parent limit order in a bracket is partially filled,
        the child orders (take-profit and stop-loss) are created for the executed quantity.
        If the parent is subsequently cancelled, the children remain active for the filled amount.
        """
        with self._lock:
            if self.is_dry_run:
                logger.info(f"[DRY-RUN] Would cancel order {client_order_id}")
                return

            try:
                alpaca_order = self.broker.get_order_by_client_id(client_order_id)
                if alpaca_order.status in ["filled", "canceled", "rejected", "expired"]:
                    logger.info(
                        f"Order {client_order_id} cannot be canceled in status {alpaca_order.status}"
                    )
                    return

                # Check partial fill safety
                if alpaca_order.status == "partially_filled":
                    logger.info(
                        f"Order {client_order_id} is partially filled. Canceling parent will leave children active for filled quantity."
                    )

                self.broker.cancel_order_by_id(str(alpaca_order.id))
            except Exception as e:
                logger.error(f"Failed to cancel order {client_order_id}: {e}")

    def check_bounded_entry_expiry(self, max_age_minutes: int = 15) -> None:
        # Bounded entry expiry policy
        # Query broker for open buy orders and cancel if they exceed TTL
        try:
            open_orders = self.broker.get_open_orders()
            now = datetime.datetime.now(datetime.timezone.utc)
            for order in open_orders:
                if order.side == "buy" and order.status in ["new", "accepted", "partially_filled"]:
                    age = (now - order.created_at).total_seconds() / 60
                    if age > max_age_minutes:
                        logger.warning(
                            f"Order {order.id} for {order.symbol} expired ({age:.1f} min). Canceling."
                        )
                        self.cancel_bracket_order(order.client_order_id)
        except Exception as e:
            logger.error(f"Failed to check bounded entry expiry: {e}")

    def emergency_check_unprotected_positions(self) -> None:
        # Paper emergency procedure: Check for open positions without active protective child orders
        try:
            positions = self.broker.get_positions()
            open_orders = self.broker.get_open_orders()

            # Check for STOP orders which act as stop loss
            sell_orders = [
                o
                for o in open_orders
                if o.side == "sell"
                and o.order_type == "stop"
                and o.status in ["new", "accepted", "partially_filled", "pending_new"]
            ]

            for pos in positions:
                sym = pos.symbol
                qty = Decimal(str(pos.qty))

                # Check if we have enough sell stop orders covering this position
                covering_qty = sum(Decimal(str(o.qty)) for o in sell_orders if o.symbol == sym)

                if covering_qty < qty:
                    # Unprotected position detected!
                    logger.critical(
                        f"UNPROTECTED POSITION DETECTED for {sym}: Position qty {qty}, covered {covering_qty}"
                    )
                    self.halt_trading(f"Unprotected open position for {sym}")

                    if self.is_dry_run:
                        logger.critical(f"[DRY-RUN] Would submit emergency market close for {sym}")
                        continue

                    # Documented action: In paper trading, submit a market sell order to close it out immediately
                    try:
                        from alpaca.trading.requests import MarketOrderRequest
                        from alpaca.trading.enums import OrderSide, TimeInForce

                        req = MarketOrderRequest(
                            symbol=sym,
                            qty=float(qty - covering_qty),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            client_order_id=f"emergency_close_{sym}_{int(time.time())}",
                        )
                        self.broker.submit_order(req)
                        logger.critical(f"Submitted emergency market close for {sym}")
                    except Exception as e:
                        logger.critical(f"Failed to submit emergency close for {sym}: {e}")
        except Exception as e:
            logger.error(f"Failed during emergency check: {e}")
