from decimal import Decimal

import pytest

from trading_system.risk.entry_plan import InvalidEntryPlan, build_entry_plan, price_tick

D = Decimal


def _plan(price: str, *, buffer: str = "0", stop: str = "0.04", target: str = "0.05"):
    return build_entry_plan(
        D(price), limit_buffer_pct=D(buffer), stop_loss_pct=D(stop), take_profit_pct=D(target)
    )


@pytest.mark.parametrize(
    "price, limit, stop, target",
    [
        # Exact: nothing to round.
        ("100", "100.00", "96.00", "105.00"),
        # 149.995 is rounded to the nearest cent (up), stop and target go down:
        # 150.00 * 0.96 = 144.00 and 150.00 * 1.05 = 157.50 are already on the tick.
        ("149.995", "150.00", "144.00", "157.50"),
        # 143.99 * 0.96 = 138.2304 -> 138.23; 143.99 * 1.05 = 151.1895 -> 151.18 (down, not 151.19)
        ("143.99", "143.99", "138.23", "151.18"),
        # Under $1 the tick is 0.0001: 0.37 * 0.96 = 0.3552, 0.37 * 1.05 = 0.3885.
        ("0.37", "0.3700", "0.3552", "0.3885"),
        # 0.3737 * 0.96 = 0.358752 -> 0.3587; 0.3737 * 1.05 = 0.392385 -> 0.3923
        ("0.3737", "0.3737", "0.3587", "0.3923"),
    ],
)
def test_prices_are_rounded_to_the_tick(price: str, limit: str, stop: str, target: str) -> None:
    plan = _plan(price)

    assert (plan.limit_price, plan.stop_price, plan.take_profit_price) == (
        D(limit),
        D(stop),
        D(target),
    )


def test_every_price_is_a_multiple_of_its_own_tick() -> None:
    for cents in range(105, 100_000, 7):  # $1.05 .. $1000, an irregular walk through the range
        plan = _plan(str(D(cents) / 100), buffer="0.001")
        for price in (plan.limit_price, plan.stop_price, plan.take_profit_price):
            assert price % price_tick(price) == 0, (cents, price)
        assert plan.stop_price < plan.limit_price < plan.take_profit_price


def test_the_limit_sits_a_buffer_above_the_reference_and_the_legs_are_measured_from_it() -> None:
    plan = _plan("150.00", buffer="0.001")

    assert plan.limit_price == D("150.15")
    assert plan.stop_price == D("144.14")  # 150.15 * 0.96 = 144.144, down
    assert plan.take_profit_price == D("157.65")  # 150.15 * 1.05 = 157.6575, down


def test_the_legs_straddle_the_dollar_boundary() -> None:
    """A stop under $1 uses the finer tick, a target over $1 the cent."""
    plan = _plan("1.02", buffer="0", stop="0.04", target="0.05")

    assert plan.stop_price == D("0.9792")  # 1.02 * 0.96
    assert plan.take_profit_price == D("1.07")  # 1.02 * 1.05 = 1.071, down to the cent


@pytest.mark.parametrize("price", ["0", "-5"])
def test_a_non_positive_price_is_refused(price: str) -> None:
    with pytest.raises(InvalidEntryPlan, match="not positive"):
        _plan(price)


@pytest.mark.parametrize(
    "stop, target, message",
    [
        ("0", "0.05", "Stop"),  # no stop at all
        ("0.04", "0", "Take-profit"),  # no target
        ("1", "0.05", "must be positive"),  # a 100 % stop is a stop at zero
    ],
)
def test_a_missing_stop_or_target_is_refused(stop: str, target: str, message: str) -> None:
    with pytest.raises(InvalidEntryPlan, match=message):
        _plan("100", stop=stop, target=target)


def test_legs_closer_than_a_cent_to_the_limit_are_refused() -> None:
    """A $0.20 stock with a 4 % stop has a stop 0.8 cents away: Alpaca would reject it."""
    with pytest.raises(InvalidEntryPlan, match="less than 0.01 below the limit"):
        _plan("0.20")


def test_a_target_that_rounds_onto_the_limit_is_refused() -> None:
    with pytest.raises(InvalidEntryPlan, match="Take-profit"):
        _plan("100", target="0.00001")


def test_a_price_too_small_to_round_to_a_tick_is_refused() -> None:
    with pytest.raises(InvalidEntryPlan, match="must be positive"):
        _plan("0.00004")
