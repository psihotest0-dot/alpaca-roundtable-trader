from decimal import Decimal
from datetime import datetime, timezone, timedelta
import pytest

from trading_system.config import Settings
from trading_system.domain.enums import SignalDirection, OrderSide, OrderType, OrderStatus
from trading_system.domain.models import (
    StrategySignal,
    AccountSnapshot,
    PositionSnapshot,
    BrokerOrder,
    TradingState
)
from trading_system.risk.engine import RiskEngine

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

@pytest.fixture
def settings():
    return Settings(
        alpaca_api_key="test",
        alpaca_secret_key="test",
        is_paper=True,
        position_size_pct=0.04,
        max_open_positions=5,
        max_gross_exposure_pct=0.20,
        max_planned_risk_per_trade_pct=0.002,
        max_total_planned_open_risk_pct=0.01,
        max_daily_loss_pct=0.02,
        stop_loss_pct=0.04
    )

@pytest.fixture
def risk_engine(settings):
    return RiskEngine(settings)

@pytest.fixture
def base_signal():
    return StrategySignal(
        version="1.0",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=_now_utc() - timedelta(minutes=15),
        direction=SignalDirection.LONG,
        confidence=0.9,
        expires_at=_now_utc() + timedelta(minutes=5)
    )

@pytest.fixture
def base_account():
    return AccountSnapshot(
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("200000"),
        day_trade_count=0,
        updated_at=_now_utc()
    )

@pytest.fixture
def base_state():
    return TradingState(
        is_trading_halted=False,
        daily_start_equity=Decimal("100000")
    )

def test_valid_long_signal(risk_engine, base_signal, base_account, base_state):
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is True
    assert intent is not None
    assert intent.quantity == Decimal("26") # 100000 * 0.04 / 150 = 26.66 -> 26

def test_short_signal_rejected(risk_engine, base_signal, base_account, base_state):
    base_signal.direction = SignalDirection.SHORT
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert intent is None
    assert "LONG signals" in decision.reasoning

def test_trading_halted_rejected(risk_engine, base_signal, base_account, base_state):
    base_state.is_trading_halted = True
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "manually halted" in decision.reasoning

def test_daily_loss_breach(risk_engine, base_signal, base_account, base_state):
    base_account.equity = Decimal("97000") # 3% loss > 2% max
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "Daily loss guard breached" in decision.reasoning

def test_missing_daily_equity_rejected(risk_engine, base_signal, base_account, base_state):
    base_state.daily_start_equity = None
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "Missing daily start equity" in decision.reasoning

def test_existing_position_rejected(risk_engine, base_signal, base_account, base_state):
    positions = [
        PositionSnapshot(
            symbol="AAPL",
            quantity=Decimal("10"),
            market_value=Decimal("1500"),
            average_entry_price=Decimal("150"),
            unrealized_pl=Decimal("0")
        )
    ]
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=positions,
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "Existing position" in decision.reasoning

def test_pending_entry_rejected(risk_engine, base_signal, base_account, base_state):
    active_orders = [
        BrokerOrder(
            broker_id="1",
            client_order_id="c1",
            symbol="AAPL",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
            quantity=Decimal("10"),
            filled_quantity=Decimal("0"),
            created_at=_now_utc(),
            updated_at=_now_utc()
        )
    ]
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=active_orders,
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "Pending entry" in decision.reasoning

def test_max_open_positions(risk_engine, base_signal, base_account, base_state):
    positions = [
        PositionSnapshot(symbol=f"SYM{i}", quantity=Decimal("10"), market_value=Decimal("1000"), average_entry_price=Decimal("100"), unrealized_pl=Decimal("0"))
        for i in range(5)
    ]
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=positions,
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "Max open positions" in decision.reasoning

def test_insufficient_buying_power(risk_engine, base_signal, base_account, base_state):
    base_account.buying_power = Decimal("100")
    # 100 / 150 = 0.66 -> quantity 0
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "Calculated quantity is less than 1" in decision.reasoning

def test_max_planned_risk_per_trade_pct(risk_engine, base_signal, base_account, base_state):
    # equity 100k
    # max_planned_risk_per_trade = 100k * 0.002 = 200
    # price = 100, stop_loss_pct = 0.04 -> stop = 96, risk = 4 per share
    # qty = 100000 * 0.04 / 100 = 40
    # planned_risk = 40 * 4 = 160 < 200 (ok)

    # but if we set max risk to 0.001 -> 100
    risk_engine.settings.max_planned_risk_per_trade_pct = 0.001
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("100"),
        state=base_state
    )
    assert decision.approved is True
    # 100 / 4 = 25
    assert intent.quantity == Decimal("25")

def test_max_gross_exposure_pct(risk_engine, base_signal, base_account, base_state):
    # equity 100k, max gross = 20k
    positions = [
        PositionSnapshot(symbol="MSFT", quantity=Decimal("190"), market_value=Decimal("19000"), average_entry_price=Decimal("100"), unrealized_pl=Decimal("0"))
    ]
    # available gross = 1000
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=positions,
        active_orders=[],
        current_price=Decimal("100"),
        state=base_state
    )
    assert decision.approved is True
    assert intent.quantity == Decimal("10") # 1000 / 100 = 10

def test_max_total_planned_open_risk_pct(risk_engine, base_signal, base_account, base_state):
    # equity 100k, max total open risk = 100k * 0.01 = 1000
    # position risk: 100 shares * $4 risk = 400
    positions = [
        PositionSnapshot(symbol="MSFT", quantity=Decimal("100"), market_value=Decimal("10000"), average_entry_price=Decimal("100"), unrealized_pl=Decimal("0"))
    ]
    # new signal: AAPL, price 100, target qty 40, risk = 4 * 40 = 160
    # total risk = 560 < 1000
    risk_engine.settings.max_total_planned_open_risk_pct = 0.005 # 500

    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=positions,
        active_orders=[],
        current_price=Decimal("100"),
        state=base_state
    )
    assert decision.approved is True
    # available risk = 500 - 400 = 100
    # risk per share = 4
    # allowed qty = 100 / 4 = 25
    assert intent.quantity == Decimal("25")

def test_quantity_less_than_one(risk_engine, base_signal, base_account, base_state):
    # price 5000 -> qty = 4000 / 5000 = 0.8 -> 0
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("5000"),
        state=base_state
    )
    assert decision.approved is False
    assert "Calculated quantity is less than 1" in decision.reasoning

def test_unknown_orders_conservatism(risk_engine, base_signal, base_account, base_state):
    # pending BUY order increases gross exposure
    active_orders = [
        BrokerOrder(
            broker_id="2",
            client_order_id="c2",
            symbol="NVDA",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
            quantity=Decimal("100"),
            filled_quantity=Decimal("0"),
            created_at=_now_utc(),
            updated_at=_now_utc(),
            limit_price=None,
            stop_price=None
        )
    ]
    # pending NVDA increases gross exposure by 100 * 150 = 15000 (assumes current price)
    # available gross = 20000 - 15000 = 5000
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=active_orders,
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is True
    assert intent.quantity == Decimal("26") # target is 4000, 4000 < 5000 available gross, 4000 / 150 = 26.66 -> 26

    # Let's increase the pending order size
    active_orders[0].quantity = Decimal("130") # 130 * 150 = 19500
    # available gross = 500
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=active_orders,
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is True
    # allowed qty = 500 / 150 = 3
    assert intent.quantity == Decimal("3")

def test_stale_signal_rejected(risk_engine, base_signal, base_account, base_state):
    base_signal.expires_at = _now_utc() - timedelta(minutes=5)
    decision, intent = risk_engine.evaluate_signal(
        signal=base_signal,
        account=base_account,
        positions=[],
        active_orders=[],
        current_price=Decimal("150"),
        state=base_state
    )
    assert decision.approved is False
    assert "stale/expired" in decision.reasoning
