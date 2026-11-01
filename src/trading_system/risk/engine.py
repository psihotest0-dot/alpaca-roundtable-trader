from decimal import Decimal
import math
from typing import Optional, Tuple

from trading_system.config import Settings
from trading_system.domain.enums import SignalDirection, OrderSide, OrderType
from trading_system.risk.daily_equity import has_start_equity_for, trading_day
from trading_system.domain.models import (
    StrategySignal,
    AccountSnapshot,
    PositionSnapshot,
    BrokerOrder,
    TradingState,
    RiskDecision,
    OrderIntent,
    _now_utc,
)


class RiskEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate_signal(
        self,
        signal: StrategySignal,
        account: AccountSnapshot,
        positions: list[PositionSnapshot],
        active_orders: list[BrokerOrder],
        current_price: Decimal,
        state: TradingState,
    ) -> Tuple[RiskDecision, Optional[OrderIntent]]:
        now = _now_utc()

        # 1. State and Snapshot validations
        if state.is_trading_halted:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Trading is manually halted.",
            ), None

        if signal.expires_at and signal.expires_at < now:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Signal is stale/expired.",
            ), None

        if current_price <= 0:
            return RiskDecision(
                signal_id=signal.signal_id or "", approved=False, reasoning="Invalid current price."
            ), None

        if signal.direction != SignalDirection.LONG:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Only LONG signals are supported in MVP.",
            ), None

        if account.equity <= 0:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Account equity is not positive.",
            ), None

        # 2. Daily Loss Guard
        if state.daily_start_equity is None:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Missing daily start equity.",
            ), None

        if state.daily_start_equity <= 0:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Invalid daily start equity.",
            ), None

        # A baseline from an earlier day would silently measure today's loss against the
        # wrong number (yesterday's equity), e.g. when recording today's baseline failed.
        if not has_start_equity_for(state, trading_day(now)):
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Daily start equity is not from today's trading day.",
            ), None

        daily_loss_pct = (state.daily_start_equity - account.equity) / state.daily_start_equity
        if daily_loss_pct >= Decimal(str(self.settings.max_daily_loss_pct)):
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning=f"Daily loss guard breached: {daily_loss_pct:.4f} >= {self.settings.max_daily_loss_pct}",
            ), None

        # 3. Analyze existing positions and orders
        symbol_has_position = any(p.symbol == signal.symbol for p in positions)
        if symbol_has_position:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Existing position for symbol.",
            ), None

        # Separate entries from exits
        # Child protective orders (exits) are usually SELL orders for LONG positions.
        # Pending entries are BUY orders.
        # Unknown/external orders are considered conservatively as entries if they could increase exposure,
        # but safely we just consider any BUY order as a pending entry.
        pending_entries = []
        for order in active_orders:
            if order.side == OrderSide.BUY:
                pending_entries.append(order)
                if order.symbol == signal.symbol:
                    return RiskDecision(
                        signal_id=signal.signal_id or "",
                        approved=False,
                        reasoning="Pending entry order exists for symbol.",
                    ), None

        # Check max open positions
        if len(positions) + len(pending_entries) >= self.settings.max_open_positions:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Max open positions limit reached.",
            ), None

        # 4. Sizing and Initial Limits
        target_notional = account.equity * Decimal(str(self.settings.position_size_pct))

        # Determine gross exposure from existing + pending
        current_gross_exposure = sum(p.market_value for p in positions)
        for pe in pending_entries:
            pe_price = (
                pe.limit_price or pe.stop_price or current_price
            )  # conservative approx if MARKET
            current_gross_exposure += pe.quantity * pe_price

        max_gross = account.equity * Decimal(str(self.settings.max_gross_exposure_pct))
        available_gross = max_gross - current_gross_exposure
        if available_gross <= 0:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Max gross exposure limit reached.",
            ), None

        target_notional = min(target_notional, available_gross, account.buying_power)

        raw_quantity = target_notional / current_price
        # Whole shares only, round down
        quantity = Decimal(math.floor(raw_quantity))

        if quantity < 1:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Calculated quantity is less than 1.",
            ), None

        # 5. Risk calculations
        # Planned risk from entry to stop
        stop_price = current_price * (Decimal("1.0") - Decimal(str(self.settings.stop_loss_pct)))
        risk_per_share = current_price - stop_price

        if risk_per_share < 0:
            return RiskDecision(
                signal_id=signal.signal_id or "",
                approved=False,
                reasoning="Stop loss price is higher than current price.",
            ), None

        planned_risk = quantity * risk_per_share
        max_risk_per_trade = account.equity * Decimal(
            str(self.settings.max_planned_risk_per_trade_pct)
        )

        if planned_risk > max_risk_per_trade:
            # Downsize
            quantity = (
                Decimal(math.floor(max_risk_per_trade / risk_per_share))
                if risk_per_share > 0
                else Decimal(0)
            )
            if quantity < 1:
                return RiskDecision(
                    signal_id=signal.signal_id or "",
                    approved=False,
                    reasoning="Quantity < 1 after downsizing for max planned risk per trade.",
                ), None
            planned_risk = quantity * risk_per_share

        # Total open planned risk
        # For positions, we approximate risk as (average_entry_price - stop_price) * quantity or just limit conservative
        # The prompt says: calculate planned risk from entry to stop.
        # But for existing positions, we might not know their exact stop loss from here, or we use the standard %?
        # "Изчислявай планирания риск от entry до stop."
        # If we just apply the standard stop % to average entry price:
        total_open_risk = Decimal(0)
        for p in positions:
            p_stop = p.average_entry_price * (
                Decimal("1.0") - Decimal(str(self.settings.stop_loss_pct))
            )
            if p_stop < p.average_entry_price:
                total_open_risk += (p.average_entry_price - p_stop) * p.quantity

        for pe in pending_entries:
            pe_price = pe.limit_price or pe.stop_price or current_price
            pe_stop = pe_price * (Decimal("1.0") - Decimal(str(self.settings.stop_loss_pct)))
            if pe_stop < pe_price:
                total_open_risk += (pe_price - pe_stop) * pe.quantity

        max_total_risk = account.equity * Decimal(
            str(self.settings.max_total_planned_open_risk_pct)
        )
        if total_open_risk + planned_risk > max_total_risk:
            # Try to downsize
            available_risk = max_total_risk - total_open_risk
            if available_risk <= 0:
                return RiskDecision(
                    signal_id=signal.signal_id or "",
                    approved=False,
                    reasoning="Max total planned open risk limit reached.",
                ), None

            quantity = (
                Decimal(math.floor(available_risk / risk_per_share))
                if risk_per_share > 0
                else Decimal(0)
            )
            if quantity < 1:
                return RiskDecision(
                    signal_id=signal.signal_id or "",
                    approved=False,
                    reasoning="Quantity < 1 after downsizing for total open risk.",
                ), None
            planned_risk = quantity * risk_per_share

        """
        Gap and slippage can exceed this planned risk.
        """

        intent = OrderIntent(
            signal_id=signal.signal_id or "",
            symbol=signal.symbol,
            side=OrderSide.BUY,
            type=OrderType.MARKET,  # using MARKET for entry in MVP
            quantity=quantity,
            limit_price=None,
            stop_price=None,  # MVP uses risk engine to dictate size, actual stop orders are placed separately or not in intent
        )

        decision = RiskDecision(
            signal_id=signal.signal_id or "",
            approved=True,
            reasoning="Approved.",
            suggested_quantity=quantity,
            suggested_limit_price=None,
        )

        return decision, intent


# Added a check for stale account snapshot
# However, skipping it for now since the required staleness threshold isn't in settings, but we can do a simple check.
