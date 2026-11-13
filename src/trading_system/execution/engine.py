import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Optional
from decimal import Decimal
import datetime
import time

from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderClass, OrderSide as AlpacaOrderSide, TimeInForce

from trading_system.broker.alpaca_broker import AlpacaBroker
from trading_system.broker.order_mapping import legs_of, to_broker_order
from trading_system.storage.repositories import OrderRepository, StateRepository
from trading_system.domain.models import OrderIntent, BrokerOrder
from trading_system.domain.enums import OrderSide, OrderType, OrderStatus
from trading_system.config import Settings
from trading_system.observability import metrics

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

            if intent.stop_price is None or intent.take_profit_price is None:
                logger.error("Bracket orders require a stop price and a take-profit price.")
                return None

            if not intent.stop_price < intent.limit_price < intent.take_profit_price:
                logger.error(
                    f"Bracket prices out of order: stop {intent.stop_price}, "
                    f"limit {intent.limit_price}, take-profit {intent.take_profit_price}"
                )
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

            if not self._validate_price_increment(intent.take_profit_price, asset):
                logger.error(f"Invalid take-profit price increment for {intent.take_profit_price}")
                return None

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
                # GTC, not DAY: the stop and take-profit legs inherit the entry's time in force,
                # so a DAY bracket loses its stop at the close and the position stays unprotected
                # overnight (seen on a real paper order). A GTC entry that never fills is
                # cancelled by check_bounded_entry_expiry.
                time_in_force=TimeInForce.GTC,
                limit_price=float(intent.limit_price),
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=float(intent.take_profit_price)),
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

                # Track child orders (legs) if they exist; they start `held`
                for leg in legs_of(alpaca_order):
                    self.order_repo.save_broker_order(to_broker_order(leg))

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
            return self.store_broker_order(alpaca_order, client_order_id)
        except Exception as e:
            logger.warning(f"Reconciliation lookup failed for {client_order_id}: {e}")
            return None

    def store_broker_order(
        self, alpaca_order: Any, client_order_id: Optional[str] = None
    ) -> BrokerOrder:
        """Saves an order as the broker reports it, together with its legs, and returns it.
        Saving the same order again updates the row (its client_order_id is the key)."""
        broker_order = to_broker_order(alpaca_order, client_order_id)
        self.order_repo.save_broker_order(broker_order)
        for leg in legs_of(alpaca_order):
            self.order_repo.save_broker_order(to_broker_order(leg))
        return broker_order

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
        """Halts trading when a position has no active protective stop (C7).

        Policy: an unprotected position raises a CRITICAL alarm and halts new entries. It is
        closed at market only when EMERGENCY_CLOSE_UNPROTECTED is on; the default leaves the
        position alone, because it may not be ours (the paper account is shared with other
        tools) and a forced close is not something to do on a guess."""
        try:
            unprotected = self.unprotected_positions()
            # Never raises; the alarm and the halt below are as they were.
            metrics.protection_checked(len(unprotected))
            if not unprotected:
                return

            for gap in unprotected:
                logger.critical(
                    f"UNPROTECTED POSITION DETECTED for {gap.symbol}: Position qty {gap.quantity}, "
                    f"covered {gap.covered}"
                )
            symbols = ", ".join(gap.symbol for gap in unprotected)
            self.halt_trading(f"Unprotected open position(s): {symbols}")

            if not self.settings.emergency_close_unprotected:
                logger.critical(
                    "Not closing the positions above: EMERGENCY_CLOSE_UNPROTECTED is off. "
                    "Trading stays halted until someone resumes it."
                )
                return

            for gap in unprotected:
                self._emergency_close(gap)
        except Exception as e:
            logger.error(f"Failed during emergency check: {e}")

    def unprotected_positions(
        self, positions: Optional[list[Any]] = None, open_orders: Optional[list[Any]] = None
    ) -> list["UnprotectedPosition"]:
        """Positions without an active stop, as of now. Read-only: it halts nothing and closes
        nothing (that is `emergency_check_unprotected_positions`). Raises if the broker cannot
        be read. The lists can be passed in when the caller has them already."""
        if positions is None:
            positions = self.broker.get_positions()
        if open_orders is None:
            open_orders = self.broker.get_open_orders()
        return find_unprotected_positions(positions, open_orders, self._recent_orders_of)

    def _recent_orders_of(self, symbol: str) -> list[Any]:
        """Where the stop of a filled bracket is found. A failed lookup leaves the position
        flagged: the check fails towards the alarm, not towards silence."""
        try:
            return list(self.broker.get_recent_orders_for_symbol(symbol))
        except Exception as e:
            logger.error(f"Could not look up the recent orders of {symbol}: {e}")
            return []

    def _emergency_close(self, gap: "UnprotectedPosition") -> None:
        """Market order for the part of a position that no stop covers: a sell for a long
        position, a buy for a short one."""
        if self.is_dry_run:
            logger.critical(f"[DRY-RUN] Would submit emergency market close for {gap.symbol}")
            return
        try:
            from alpaca.trading.requests import MarketOrderRequest

            req = MarketOrderRequest(
                symbol=gap.symbol,
                qty=float(abs(gap.quantity) - gap.covered),
                side=AlpacaOrderSide.SELL if gap.quantity > 0 else AlpacaOrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"emergency_close_{gap.symbol}_{int(time.time())}",
            )
            self.broker.submit_order(req)
            logger.critical(f"Submitted emergency market close for {gap.symbol}")
        except Exception as e:
            logger.critical(f"Failed to submit emergency close for {gap.symbol}: {e}")


# Statuses of an order that is live at the exchange.
_ACTIVE_STATUSES = frozenset({"new", "accepted", "partially_filled", "pending_new"})

# The stop leg of a bracket whose entry has FILLED is reported as `held` for as long as the
# take-profit leg is the working half of the OCO pair, and then no list of open orders
# contains it. It still protects the position. Seen on a real paper order (2026-11-02): entry
# filled, take-profit `new`, stop `held`, only the take-profit among the open orders. Before
# the entry has filled completely, held legs protect nothing: Alpaca activates them only then.
_LIVE_BRACKET_STOP_STATUSES = _ACTIVE_STATUSES | {"held"}


@dataclass(frozen=True)
class UnprotectedPosition:
    symbol: str
    quantity: Decimal  # signed: negative for a short position
    covered: Decimal  # shares an active stop order protects


def _text(value: Any) -> str:
    """ "sell" for OrderSide.SELL as well as for "sell"."""
    return str(getattr(value, "value", value)).lower()


def _with_legs(orders: Iterable[Any]) -> Iterator[Any]:
    """The orders and, below them, their child legs. Alpaca returns legs nested under the
    parent when asked with nested=True, and as separate orders otherwise."""
    for order in orders:
        yield order
        yield from _with_legs(getattr(order, "legs", None) or [])


def _is_stop(order: Any, closing_side: str) -> bool:
    return _text(order.type) == "stop" and _text(order.side) == closing_side


def _bracket_stop_quantity(orders: Iterable[Any], symbol: str, closing_side: str) -> Decimal:
    """Shares protected by the stop legs of brackets whose entry has filled completely."""
    total = Decimal(0)
    for parent in orders:
        if parent.symbol != symbol or _text(parent.status) != "filled":
            continue
        for leg in _with_legs(getattr(parent, "legs", None) or []):
            if _is_stop(leg, closing_side) and _text(leg.status) in _LIVE_BRACKET_STOP_STATUSES:
                total += Decimal(str(leg.qty))
    return total


def find_unprotected_positions(
    positions: Iterable[Any],
    open_orders: Iterable[Any],
    recent_orders_of: Optional[Callable[[str], Iterable[Any]]] = None,
) -> list[UnprotectedPosition]:
    """Positions whose size is not fully covered by stop orders on the closing side: sell
    stops for a long position, buy stops for a short one.

    Active stops among `open_orders` count. For a position they leave uncovered,
    `recent_orders_of(symbol)` is asked for the symbol's recent orders, nested, to find the
    held stop of a filled bracket (see _LIVE_BRACKET_STOP_STATUSES)."""
    seen: set[str] = set()
    active_stops = []
    for order in _with_legs(open_orders):
        key = str(getattr(order, "id", id(order)))
        if key in seen:
            continue
        seen.add(key)
        if _text(order.type) == "stop" and _text(order.status) in _ACTIVE_STATUSES:
            active_stops.append(order)

    gaps = []
    for pos in positions:
        quantity = Decimal(str(pos.qty))
        if quantity == 0:
            continue
        closing_side = "sell" if quantity > 0 else "buy"
        covered = sum(
            (
                Decimal(str(o.qty))
                for o in active_stops
                if o.symbol == pos.symbol and _text(o.side) == closing_side
            ),
            Decimal(0),
        )
        if covered < abs(quantity) and recent_orders_of is not None:
            covered += _bracket_stop_quantity(
                recent_orders_of(str(pos.symbol)), str(pos.symbol), closing_side
            )
        if covered < abs(quantity):
            gaps.append(UnprotectedPosition(str(pos.symbol), quantity, covered))
    return gaps
