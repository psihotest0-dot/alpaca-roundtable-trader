"""Sizing and prices of the fractional paper trial (docs/fractional-paper-trial.md, section 2)."""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from trading_system.fractional.plan import (
    HardCapExceeded,
    Refused,
    live_stops_of,
    size_entry,
    stop_cover,
    stop_price_for,
    target_price_for,
    virtual_equity,
)

D = Decimal


def _size(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "equity": D("500"),
        "reference_price": D("100"),
        "open_positions": 0,
        "open_gross": D("0"),
        "open_risk": D("0"),
        "max_positions": 5,
        "position_size_pct": D("0.04"),
        "max_gross_pct": D("0.20"),
        "max_trade_risk_pct": D("0.002"),
        "max_total_risk_pct": D("0.05"),
        "limit_buffer_pct": D("0.001"),
        "stop_loss_pct": D("0.04"),
        "take_profit_pct": D("0.05"),
        "max_order_notional": D("50"),
    }
    values.update(overrides)
    return size_entry(**values)


def test_the_virtual_capital_is_the_budget_moved_by_what_the_account_made() -> None:
    assert virtual_equity(D("500"), D("1000000"), D("1000000")) == D("500")
    assert virtual_equity(D("500"), D("1000012.5"), D("1000000")) == D("512.5")
    assert virtual_equity(D("500"), D("999990"), D("1000000")) == D("490")


def test_a_position_is_four_percent_of_the_virtual_capital_in_a_fraction_of_a_share() -> None:
    sized = _size(reference_price=D("349.55"))

    assert sized.quantity * sized.limit_price <= D("20")  # 4 % of 500
    assert sized.quantity * sized.limit_price > D("19.9")
    assert sized.quantity == sized.quantity.quantize(D("0.000001"))  # a millionth of a share
    assert 0 < sized.quantity < 1  # the whole point: less than one share of a 350-dollar stock
    assert sized.stop_price < sized.limit_price < sized.take_profit_price


def test_the_risk_of_one_trade_is_capped_at_a_fifth_of_a_percent_of_the_capital() -> None:
    # 4 % of 500 is 20 dollars; a 4 % stop on it risks 0.80, under the 1 dollar cap: not cut
    assert _size().planned_risk <= D("1")
    # a wider stop would risk more than the cap, and the quantity is cut to the cap
    sized = _size(stop_loss_pct=D("0.10"))
    assert sized.planned_risk <= D("1.0001") and sized.notional < D("20")


def test_the_gross_exposure_and_the_open_risk_leave_room_for_the_new_entry() -> None:
    with pytest.raises(Refused, match="gross exposure"):
        _size(open_gross=D("100"))  # 20 % of 500 is already used
    with pytest.raises(Refused, match="total planned open risk"):
        _size(open_risk=D("25"))  # 5 % of 500 is already at risk
    cut = _size(open_gross=D("90"))  # only 10 dollars of room
    assert cut.notional <= D("10")


def test_the_number_of_positions_is_limited() -> None:
    with pytest.raises(Refused, match="Max open positions"):
        _size(open_positions=5)
    assert _size(open_positions=4).quantity > 0


def test_an_order_under_one_dollar_is_not_sent() -> None:
    with pytest.raises(Refused, match="under the 1 dollar minimum"):
        _size(equity=D("20"))  # 4 % of 20 is 0.80


def test_an_order_above_the_hard_ceiling_is_refused_not_cut() -> None:
    with pytest.raises(HardCapExceeded):
        _size(max_order_notional=D("10"))  # the sizing wants about 20
    assert issubclass(HardCapExceeded, Refused)


def test_a_capital_that_is_not_positive_trades_nothing() -> None:
    with pytest.raises(Refused, match="not positive"):
        _size(equity=D("0"))
    with pytest.raises(Refused, match="not positive"):
        _size(equity=D("-5"))


def test_a_price_that_cannot_make_a_plan_is_refused() -> None:
    with pytest.raises(Refused, match="Entry plan rejected"):
        _size(reference_price=D("0"))


def test_the_stop_is_four_percent_under_the_real_entry_rounded_down_to_the_cent() -> None:
    assert stop_price_for(D("349.492"), D("0.04")) == D("335.51")  # 335.51232 down
    assert stop_price_for(D("60"), D("0.04")) == D("57.60")
    with pytest.raises(Refused):
        stop_price_for(D("0.10"), D("0.04"))  # less than a cent apart


def test_the_target_is_five_percent_over_the_entry() -> None:
    assert target_price_for(D("60"), D("0.05")) == D("63.00")


def _order(
    order_id: str, symbol: str, side: str, kind: str, status: str, qty: str, legs: Any = None
) -> Any:
    return SimpleNamespace(
        id=order_id, symbol=symbol, side=side, type=kind, status=status, qty=qty, legs=legs or []
    )


def test_the_cover_counts_standing_stop_sells_only() -> None:
    orders = [
        _order("1", "KO", "sell", "stop", "new", "0.1"),
        _order("2", "KO", "sell", "stop", "accepted", "0.2"),
        _order("3", "KO", "sell", "limit", "new", "0.5"),  # a target is not a stop
        _order("4", "KO", "sell", "stop", "canceled", "0.7"),  # not standing
        _order("5", "KO", "buy", "stop", "new", "0.9"),  # not a sell
        _order("6", "PEP", "sell", "stop_limit", "pending_new", "0.3"),
    ]

    cover = stop_cover(orders)

    assert cover == {"KO": D("0.3"), "PEP": D("0.3")}


def test_the_stop_leg_of_a_bracket_is_found_and_counted_once() -> None:
    leg = _order("L", "KO", "sell", "stop", "held", "2")
    parent = _order("P", "KO", "buy", "limit", "filled", "2", legs=[leg])

    assert stop_cover([parent, leg]) == {"KO": D("2")}
    assert live_stops_of([parent], "KO") == [leg]
    assert live_stops_of([parent], "PEP") == []
