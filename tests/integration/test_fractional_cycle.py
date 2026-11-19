"""The fractional paper trial's loop (docs/fractional-paper-trial.md): the real cycle code and a real
SQLite journal, the real alpaca-py models for orders, positions, clock and account, and a fake Alpaca
that refuses what the real one refuses (fractional bracket and GTC stops are not in this fake: the loop
never sends them, and a test shows it)."""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AccountStatus, OrderStatus
from alpaca.trading.models import Clock, Order, Position, TradeAccount

from tests.fixtures.bars import make_bars
from tests.fixtures.fakes import FakeStockClient
from trading_system.data.alpaca_data import AlpacaData
from trading_system.fractional.cycle import FractionalCycle
from trading_system.fractional.plan import stop_price_for
from trading_system.runtime.schedule import expected_bar_start
from trading_system.storage.repositories import AuditRepository
from trading_system.strategy.config import RUNTIME_STRATEGY

NOW = datetime.now(timezone.utc)
REFUSED = APIError(
    '{"code":42210000,"message":"refused"}',
    http_error=SimpleNamespace(response=SimpleNamespace(status_code=422)),
)
OPEN_STATUSES = {"accepted", "new", "pending_new"}


class SecondAccount:
    """A fake of the second paper account that keeps its orders and positions, and refuses a second
    sell on a quantity that is already held for the first (as Alpaca does)."""

    def __init__(self, equity: str = "1000000") -> None:
        self.equity = equity
        self.market_open = True
        self.fill_entries = False
        self.stop_error: Exception | None = None
        self.not_fractionable: set[str] = set()
        self.positions: dict[str, Position] = {}
        self.orders: dict[uuid.UUID, Order] = {}
        self.requests: list[Any] = []
        self.cancelled: list[str] = []

    # -- what the loop reads
    def get_clock(self) -> Clock:
        return Clock(
            timestamp=NOW,
            is_open=self.market_open,
            next_open=NOW + timedelta(hours=1),
            next_close=NOW + timedelta(hours=7),
        )

    def get_account(self) -> TradeAccount:
        return TradeAccount(
            id=uuid.uuid4(),
            account_number="PA0000000001",
            status=AccountStatus.ACTIVE,
            currency="USD",
            equity=self.equity,
            cash=self.equity,
            buying_power=str(float(self.equity) * 2),
        )

    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    def get_asset(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(
            symbol=symbol, tradable=True, fractionable=symbol not in self.not_fractionable
        )

    def get_open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if _status(o) in OPEN_STATUSES]

    def get_order_by_id(self, order_id: str) -> Order:
        return self.orders[uuid.UUID(order_id)]

    # -- what the loop does
    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled.append(order_id)
        order = self.orders[uuid.UUID(order_id)]
        self.orders[order.id] = order.model_copy(update={"status": OrderStatus.CANCELED})

    def submit_order(self, request: Any) -> Order:
        self.requests.append(request)
        kind = type(request).__name__
        side = str(getattr(request.side, "value", request.side))
        if kind == "StopOrderRequest":
            if self.stop_error is not None:
                raise self.stop_error
            self._check_available(request.symbol, Decimal(str(request.qty)))
            return self._store(request, "stop", "new")
        if kind == "MarketOrderRequest" and side == "sell":
            self._check_available(request.symbol, Decimal(str(request.qty)))
            order = self._store(request, "market", "filled", filled=True)
            self.positions.pop(request.symbol, None)
            return order
        assert kind == "LimitOrderRequest" and side == "buy", f"the loop sent a {kind} {side}"
        assert str(request.time_in_force.value) == "day"  # fractional orders are DAY only
        assert getattr(request, "order_class", None) in (None, "simple")  # no bracket
        order = self._store(
            request,
            "limit",
            "filled" if self.fill_entries else "accepted",
            filled=self.fill_entries,
        )
        if self.fill_entries:
            self.add_position(request.symbol, request.qty, request.limit_price)
        return order

    # -- helpers for the tests
    def add_position(self, symbol: str, qty: Any, avg: Any, price: Any = None) -> None:
        price = avg if price is None else price
        self.positions[symbol] = Position(
            asset_id=uuid.uuid4(),
            symbol=symbol,
            exchange="NASDAQ",
            asset_class="us_equity",
            avg_entry_price=str(avg),
            qty=str(qty),
            side="long",
            market_value=str(Decimal(str(qty)) * Decimal(str(price))),
            cost_basis=str(Decimal(str(qty)) * Decimal(str(avg))),
            unrealized_pl="0",
            unrealized_plpc="0",
            unrealized_intraday_pl="0",
            unrealized_intraday_plpc="0",
            current_price=str(price),
            lastday_price=str(avg),
            change_today="0",
            qty_available=str(qty),
        )

    def set_price(self, symbol: str, price: Any) -> None:
        old = self.positions[symbol]
        self.positions[symbol] = old.model_copy(update={"current_price": str(price)})

    def add_stop(self, symbol: str, qty: Any, stop: Any) -> Order:
        request = SimpleNamespace(
            symbol=symbol,
            qty=qty,
            side="sell",
            stop_price=stop,
            limit_price=None,
            client_order_id=None,
        )
        return self._store(request, "stop", "new")

    def add_entry(self, symbol: str, age: timedelta) -> Order:
        request = SimpleNamespace(
            symbol=symbol,
            qty=0.1,
            side="buy",
            stop_price=None,
            limit_price=50.0,
            client_order_id=None,
        )
        return self._store(request, "limit", "accepted", created=NOW - age)

    def expire_day_orders(self) -> None:
        for order in self.get_open_orders():
            self.orders[order.id] = order.model_copy(update={"status": OrderStatus.EXPIRED})

    def stops(self, symbol: str) -> list[Order]:
        return [
            o for o in self.get_open_orders() if o.symbol == symbol and str(o.type.value) == "stop"
        ]

    def _check_available(self, symbol: str, qty: Decimal) -> None:
        position = self.positions.get(symbol)
        held = sum(
            (
                Decimal(str(o.qty))
                for o in self.get_open_orders()
                if o.symbol == symbol and str(o.side.value) == "sell"
            ),
            Decimal(0),
        )
        have = Decimal(str(position.qty)) if position else Decimal(0)
        if qty > have - held + Decimal("0.000000001"):
            raise APIError(
                '{"code":40310000,"message":"insufficient qty available for order"}',
                http_error=SimpleNamespace(response=SimpleNamespace(status_code=403)),
            )

    def _store(
        self,
        request: Any,
        kind: str,
        status: str,
        *,
        filled: bool = False,
        created: datetime | None = None,
    ) -> Order:
        qty = str(request.qty)
        limit = getattr(request, "limit_price", None)
        stop = getattr(request, "stop_price", None)
        client_id = getattr(request, "client_order_id", None)
        order = Order(
            id=uuid.uuid4(),
            client_order_id=client_id or str(uuid.uuid4()),
            created_at=created or NOW,
            updated_at=NOW,
            submitted_at=created or NOW,
            asset_id=uuid.uuid4(),
            symbol=request.symbol,
            asset_class="us_equity",
            order_class="simple",
            type=kind,
            side=str(getattr(request.side, "value", request.side)),
            time_in_force="day",
            status=status,
            extended_hours=False,
            qty=qty,
            limit_price=None if limit is None else str(limit),
            stop_price=None if stop is None else str(stop),
            filled_qty=qty if filled else "0",
            filled_avg_price=(str(limit) if filled and limit is not None else None),
            filled_at=NOW if filled else None,
        )
        self.orders[order.id] = order
        return order


def _status(order: Order) -> str:
    return str(order.status.value)


class RecordingAlerts:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_alert(self, title: str, message: str) -> None:
        self.sent.append((title, message))


class Setup:
    def __init__(
        self, cycle: FractionalCycle, broker: SecondAccount, alerts: RecordingAlerts, halt: Path
    ) -> None:
        self.cycle, self.broker, self.alerts, self.halt = cycle, broker, alerts, halt


@pytest.fixture
def make_cycle(make_settings: Any, session_factory: Any, tmp_path: Path) -> Any:
    def _make(
        *, watchlist: str = "MSFT", broker: SecondAccount | None = None, **overrides: Any
    ) -> Setup:
        halt = tmp_path / "fractional_halt"
        overrides.setdefault("fractional_symbols", "watchlist")  # the wide set has its own tests
        settings = make_settings(watchlist=watchlist, fractional_halt_path=str(halt), **overrides)
        bars_15m, bars_1h = make_bars(NOW)
        symbols = settings.watchlist_symbols
        data = AlpacaData(settings)
        data.client = FakeStockClient(  # type: ignore[assignment]
            {s: bars_15m for s in symbols}, {s: bars_1h for s in symbols}
        )
        broker = broker or SecondAccount()
        alerts = RecordingAlerts()
        cycle = FractionalCycle(
            settings=settings,
            broker=broker,
            data=data,
            audit_repo=AuditRepository(session_factory),
            strategy_config=RUNTIME_STRATEGY,
            alerts=alerts,
            halt_path=halt,
            sleep=lambda _seconds: None,
            clock=lambda: NOW,
        )
        return Setup(cycle, broker, alerts, halt)

    return _make


def _events(session_factory: Any, event: str) -> list[dict[str, Any]]:
    return [
        d
        for _, d in AuditRepository(session_factory).events(
            event,
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(2100, 1, 1, tzinfo=timezone.utc),
        )
    ]


def _entries(setup: Setup, now: datetime = NOW) -> Any:
    return setup.cycle.run_entries(now, expected_bar_start(NOW))


# ------------------------------------------------------------------------------------- entries


def test_a_signal_becomes_one_fractional_day_limit_order_sized_on_the_virtual_capital(
    make_cycle: Any, session_factory: Any
) -> None:
    setup = make_cycle()

    report = _entries(setup)

    assert report.signals == 1 and report.submitted == 1 and report.errors == 0
    (request,) = setup.broker.requests
    cost = Decimal(str(request.qty)) * Decimal(str(request.limit_price))
    # 4 % of 500 dollars (20), not 4 % of the million the account holds
    assert Decimal("15") < cost <= Decimal("20.01")
    assert Decimal(str(request.qty)) % 1 != 0  # a fraction of a share
    (baseline,) = _events(session_factory, "fractional_baseline")
    assert baseline["equity"] == "1000000" and baseline["budget"] == "500.0"
    (entry,) = _events(session_factory, "fractional_entry")
    assert entry["symbol"] == "MSFT" and entry["virtual_equity"] == "500.0"


def test_the_baseline_is_written_once_and_read_back(make_cycle: Any, session_factory: Any) -> None:
    setup = make_cycle()
    _entries(setup)
    setup.broker.equity = "1000100"  # the account made 100 dollars since

    again = make_cycle(broker=setup.broker)  # a restart: a new cycle on the same journal
    assert again.cycle.baseline_equity(NOW) == Decimal("1000000")

    assert len(_events(session_factory, "fractional_baseline")) == 1


def test_the_virtual_capital_moves_with_what_the_account_makes(make_cycle: Any) -> None:
    setup = make_cycle()
    setup.cycle.baseline_equity(NOW)
    setup.broker.equity = "1000100"

    account = setup.broker.get_account()

    assert setup.cycle._virtual_equity(account, NOW) == Decimal("600")


def test_a_filled_entry_gets_its_stop_at_once_from_the_real_entry_price(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.fill_entries = True
    setup = make_cycle(broker=broker)

    _entries(setup)

    (position,) = broker.get_positions()
    (stop,) = broker.stops("MSFT")
    assert Decimal(str(stop.qty)) == Decimal(str(position.qty))
    assert Decimal(str(stop.stop_price)) == stop_price_for(
        Decimal(str(position.avg_entry_price)), Decimal("0.04")
    )
    assert str(stop.time_in_force.value) == "day"
    (fill,) = _events(session_factory, "fractional_fill")
    assert fill["filled_qty"] == str(position.qty)
    (placed,) = _events(session_factory, "fractional_stop_placed")
    assert placed["symbol"] == "MSFT" and placed["gap_seconds"] is not None


def test_an_entry_that_has_not_filled_gets_no_stop_yet(make_cycle: Any) -> None:
    setup = make_cycle()

    _entries(setup)

    assert setup.broker.stops("MSFT") == [] and len(setup.broker.requests) == 1


def test_the_loop_never_sends_a_bracket_or_a_gtc_order(make_cycle: Any) -> None:
    broker = SecondAccount()
    broker.fill_entries = True
    setup = make_cycle(broker=broker)

    _entries(setup)

    for request in broker.requests:
        assert str(request.time_in_force.value) == "day"
        assert getattr(request, "order_class", None) in (None, "simple")
        assert getattr(request, "take_profit", None) is None
        assert getattr(request, "stop_loss", None) is None


def test_an_expired_signal_is_not_traded(make_cycle: Any, session_factory: Any) -> None:
    setup = make_cycle()

    report = _entries(setup, now=NOW + timedelta(hours=1))

    assert report.signals == 1 and report.submitted == 0 and setup.broker.requests == []
    (refused,) = _events(session_factory, "fractional_refused")
    assert refused["reason"] == "the signal has expired"


def test_a_symbol_that_is_already_held_is_not_entered_again(make_cycle: Any) -> None:
    broker = SecondAccount()
    broker.add_position("MSFT", "0.2", "100")
    broker.add_stop("MSFT", "0.2", "96")
    setup = make_cycle(broker=broker)

    report = _entries(setup)

    assert report.signals == 0 and broker.requests == []


def test_the_limit_of_open_positions_refuses_the_entry(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    for symbol in ("KO", "PEP", "PG", "WMT", "COST"):
        broker.add_position(symbol, "0.1", "60")
        broker.add_stop(symbol, "0.1", "57.6")
    setup = make_cycle(broker=broker)

    report = _entries(setup)

    assert report.submitted == 0 and broker.requests == []
    (refused,) = _events(session_factory, "fractional_refused")
    assert "Max open positions" in refused["reason"]


def test_an_order_above_the_hard_ceiling_is_refused_and_alarmed(
    make_cycle: Any, session_factory: Any
) -> None:
    setup = make_cycle(fractional_max_order_usd=5.0)  # the sizing wants about 20 dollars

    report = _entries(setup)

    assert report.errors == 1 and report.submitted == 0 and setup.broker.requests == []
    assert [title for title, _ in setup.alerts.sent] == ["Fractional order above the hard ceiling"]


def test_halted_entries_make_no_order(make_cycle: Any) -> None:
    setup = make_cycle()
    setup.halt.write_text("halted by a test\n")

    report = _entries(setup)

    assert report.skipped == "halted" and setup.broker.requests == []


# ---------------------------------------------------------------------------------- protection


def test_a_position_without_a_stop_is_given_one_and_the_gap_is_recorded(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00", price="60.10")
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    (stop,) = broker.stops("KO")
    assert Decimal(str(stop.qty)) == Decimal("0.3") and Decimal(str(stop.stop_price)) == Decimal(
        "57.60"
    )
    (episode,) = _events(session_factory, "fractional_unprotected")
    assert episode["symbol"] == "KO"
    (placed,) = _events(session_factory, "fractional_stop_placed")
    assert placed["market_open"] is True and placed["gap_seconds"] == 0

    setup.cycle.run_housekeeping(NOW + timedelta(minutes=1))  # nothing to do now
    assert len(broker.requests) == 1


def test_the_stop_that_expired_with_the_day_is_placed_again(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00")
    setup = make_cycle(broker=broker)
    setup.cycle.run_housekeeping(NOW)
    broker.expire_day_orders()

    setup.cycle.run_housekeeping(NOW + timedelta(days=1))

    assert len(broker.stops("KO")) == 1 and len(broker.requests) == 2
    assert len(_events(session_factory, "fractional_stop_placed")) == 2


def test_only_the_missing_part_of_a_quantity_is_covered(make_cycle: Any) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00")
    broker.add_stop("KO", "0.1", "57.60")
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    (request,) = broker.requests
    assert Decimal(str(request.qty)) == Decimal("0.2")


def test_a_stop_is_queued_after_the_close_without_starting_the_unprotected_clock(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.market_open = False
    broker.add_position("KO", "0.3", "60.00")
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    assert len(broker.stops("KO")) == 1
    assert _events(session_factory, "fractional_unprotected") == []
    (placed,) = _events(session_factory, "fractional_stop_placed")
    assert placed["market_open"] is False and placed["gap_seconds"] is None


def test_a_refused_stop_is_journalled_alarmed_once_and_tried_again_later(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.stop_error = REFUSED
    broker.add_position("KO", "0.3", "60.00")
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)
    setup.cycle.run_housekeeping(NOW + timedelta(minutes=1))  # too soon for another try
    assert len(broker.requests) == 1
    setup.cycle.run_housekeeping(NOW + timedelta(minutes=5, seconds=1))

    assert len(broker.requests) == 2
    refusals = _events(session_factory, "fractional_stop_refused")
    assert len(refusals) == 2 and "refused" in refusals[0]["message"]
    assert [t for t, _ in setup.alerts.sent].count("Fractional stop could not be placed") == 1


def test_a_position_unprotected_for_five_minutes_with_the_market_open_stops_the_entries(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.stop_error = REFUSED
    broker.add_position("KO", "0.3", "60.00")
    setup = make_cycle(broker=broker)
    setup.cycle.run_housekeeping(NOW)
    assert not setup.cycle.halted()

    setup.cycle.run_housekeeping(NOW + timedelta(minutes=6))

    assert setup.cycle.halted() and "KO" in setup.halt.read_text()
    assert "Fractional entries stopped" in [t for t, _ in setup.alerts.sent]
    assert _entries(setup, now=NOW + timedelta(minutes=6)).skipped == "halted"


def test_a_closed_market_never_halts_the_entries(make_cycle: Any) -> None:
    broker = SecondAccount()
    broker.market_open = False
    broker.stop_error = REFUSED
    broker.add_position("KO", "0.3", "60.00")
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)
    setup.cycle.run_housekeeping(NOW + timedelta(hours=3))

    assert (
        not setup.cycle.halted()
    )  # stops do not trigger overnight: the clock runs with the market


def test_at_the_target_the_stop_is_cancelled_and_the_whole_position_sold_at_market(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00")
    stop = broker.add_stop("KO", "0.3", "57.60")
    broker.set_price("KO", "63.00")  # the target: 60 x 1.05
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    assert broker.cancelled == [str(stop.id)]
    (sale,) = [r for r in broker.requests if type(r).__name__ == "MarketOrderRequest"]
    assert Decimal(str(sale.qty)) == Decimal("0.3") and str(sale.side.value) == "sell"
    assert broker.get_positions() == []
    (sold,) = _events(session_factory, "fractional_target_sold")
    assert sold["symbol"] == "KO" and sold["price"] == "63.00"


def test_the_target_is_not_sold_while_the_market_is_closed(make_cycle: Any) -> None:
    broker = SecondAccount()
    broker.market_open = False
    broker.add_position("KO", "0.3", "60.00")
    broker.add_stop("KO", "0.3", "57.60")
    broker.set_price("KO", "64.00")
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    assert broker.requests == [] and len(broker.get_positions()) == 1


def test_a_price_already_through_the_stop_is_sold_instead_of_placing_a_stop(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00", price="57.00")  # below the stop of 57.60
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    assert [type(r).__name__ for r in broker.requests] == ["MarketOrderRequest"]
    assert broker.get_positions() == []
    assert len(_events(session_factory, "fractional_stop_sold")) == 1


def test_a_sale_that_waits_for_the_stop_to_let_go_is_tried_again(make_cycle: Any) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00")
    broker.add_stop("KO", "0.3", "57.60")
    broker.set_price("KO", "63.50")
    broker.cancel_order_by_id = lambda _id: None  # type: ignore[method-assign]  # Alpaca is slow to cancel
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    assert (
        broker.requests == [] and len(broker.get_positions()) == 1
    )  # nothing sold, nothing broken


def test_a_stale_entry_is_cancelled_after_fifteen_minutes_and_not_before(make_cycle: Any) -> None:
    broker = SecondAccount()
    old = broker.add_entry("PEP", timedelta(minutes=16))
    fresh = broker.add_entry("PG", timedelta(minutes=14))
    setup = make_cycle(broker=broker)

    setup.cycle.run_housekeeping(NOW)

    assert broker.cancelled == [str(old.id)] and str(fresh.id) not in broker.cancelled


# -------------------------------------------------------------------------------------- resume


def test_a_halt_is_lifted_only_when_every_position_has_a_standing_stop(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.add_position("KO", "0.3", "60.00")
    setup = make_cycle(broker=broker)
    setup.cycle.halt("a test", NOW)

    assert setup.cycle.resume(NOW) == ["KO"] and setup.cycle.halted()

    broker.add_stop("KO", "0.3", "57.60")
    assert setup.cycle.resume(NOW) == [] and not setup.cycle.halted()
    events = AuditRepository(session_factory).events(
        "fractional_resumed", NOW - timedelta(days=1), NOW + timedelta(days=1)
    )
    assert len(events) == 1


# ------------------------------------------------------------- the symbols (plan, section 5)


def test_a_stock_that_cannot_be_bought_in_fractions_is_refused(
    make_cycle: Any, session_factory: Any
) -> None:
    broker = SecondAccount()
    broker.not_fractionable = {"MSFT"}
    setup = make_cycle(broker=broker)

    report = _entries(setup)

    assert report.signals == 1 and report.submitted == 0 and broker.requests == []
    (refused,) = _events(session_factory, "fractional_refused")
    assert "not fractionable" in refused["reason"]


def test_the_wide_setting_trades_the_observation_universe_without_the_market_etf_or_the_extras(
    make_cycle: Any,
) -> None:
    from trading_system.data.oracle_extra import ORACLE_EXTRA

    symbols = make_cycle(watchlist="AAPL", fractional_symbols="observation").cycle.watchlist

    assert "AAPL" in symbols and "KO" in symbols  # the watchlist and the observed stocks
    assert "SPY" not in symbols  # the market ETF is a yardstick, not a trade
    assert not set(symbols) & set(ORACLE_EXTRA)  # the Oracle's extra stocks stay out
    assert len(symbols) >= 140 and len(set(symbols)) == len(symbols)


def test_the_narrow_setting_trades_only_the_watchlist(make_cycle: Any) -> None:
    setup = make_cycle(watchlist="AAPL,MSFT", fractional_symbols="watchlist")

    assert setup.cycle.watchlist == ["AAPL", "MSFT"]


def test_a_pass_with_the_wide_setting_asks_for_the_bars_of_every_symbol(make_cycle: Any) -> None:
    setup = make_cycle(watchlist="MSFT", fractional_symbols="observation")

    report = _entries(setup)

    asked = {
        s
        for request in setup.cycle.data.client.requests  # type: ignore[attr-defined]
        for s in (
            [request.symbol_or_symbols]
            if isinstance(request.symbol_or_symbols, str)
            else request.symbol_or_symbols
        )
    }
    assert len(asked) >= 140 and "KO" in asked and "SPY" not in asked
    assert (
        report.signals == 1 and report.submitted == 1
    )  # MSFT is the only one the fake has bars for


# ------------------------------------------------------------------------------------ the walls


def test_the_fractional_package_does_not_reach_into_the_live_path() -> None:
    """The trial must not be able to touch the live loop's state, orders, risk or execution."""
    forbidden = (
        "ExecutionEngine",
        "RiskEngine",
        "StateRepository",
        "TradingState",
        "OrderRepository",
        "TradingCycle",
        "execution.engine",
        "risk.engine",
    )
    root = Path(__file__).resolve().parents[2] / "src" / "trading_system" / "fractional"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path.name} mentions {name}"
