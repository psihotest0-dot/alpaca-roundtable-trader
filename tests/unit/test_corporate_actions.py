"""Corporate actions: what the feed makes of Alpaca's answer.

The events come in the shape the real API sends them, through alpaca-py's own CorporateActionsSet
(a hand-made object with the right attributes would hide a changed field name). The facts these
tests hold down were found by probing the live API on 2026-11-06: one call returns at most 1000
events and cuts the rest silently, and a merger is listed under the buyer as well as the bought
company."""

import uuid
from datetime import date, datetime, timezone
from typing import Any

import pytest
from alpaca.data.models.corporate_actions import CorporateActionsSet
from alpaca.data.requests import CorporateActionsRequest
from alpaca.trading.enums import OrderClass, OrderStatus, TimeInForce
from alpaca.trading.enums import AssetClass, AssetExchange, PositionSide
from alpaca.trading.models import Order, Position

from trading_system.config import Settings
from trading_system.data import corporate_actions as ca
from trading_system.data.corporate_actions import (
    CorporateActionsFeed,
    CorporateActionsIncomplete,
    CorporateEvent,
    check_holdings,
    event_from,
    format_events,
    held_symbols,
    in_window,
)


def _split(symbol: str, day: str, new: float = 4.0, old: float = 1.0) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "cusip": "037833100",
        "old_cusip": "037833100",  # a reverse split carries these two instead of `cusip`
        "new_cusip": "037833100",
        "new_rate": new,
        "old_rate": old,
        "process_date": day,
        "ex_date": day,
    }


def _spin_off(source: str, new: str, day: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "source_symbol": source,
        "source_cusip": "1",
        "source_rate": 1.0,
        "new_symbol": new,
        "new_cusip": "2",
        "new_rate": 0.25,
        "process_date": day,
        "ex_date": day,
    }


def _cash_merger(acquirer: str, acquiree: str, day: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "acquirer_symbol": acquirer,
        "acquirer_cusip": "1",
        "acquiree_symbol": acquiree,
        "acquiree_cusip": "2",
        "rate": 115.0,
        "process_date": day,
        "effective_date": day,
    }


def _name_change(old: str, new: str, day: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "old_symbol": old,
        "old_cusip": "1",
        "new_symbol": new,
        "new_cusip": "1",
        "process_date": day,
    }


def _only(raw: dict[str, list[dict[str, Any]]]) -> tuple[str, Any]:
    ((kind, items),) = CorporateActionsSet(raw).data.items()
    return kind, items[0]


# ---- one event as we keep it -------------------------------------------------------------------


def test_a_split_is_described_in_plain_words_and_dated_by_its_ex_date():
    kind, item = _only({"forward_splits": [_split("NVDA", "2024-06-10", 10, 1)]})
    event = event_from(kind, item)

    assert event == CorporateEvent(
        kind="forward_splits",
        affected=("NVDA",),
        day=date(2024, 6, 10),
        detail="split: 10 new share(s) for 1 old",
    )


def test_a_reverse_split_reads_as_fewer_new_shares_for_more_old_ones():
    kind, item = _only({"reverse_splits": [_split("GE", "2021-08-02", 1, 8)]})

    assert event_from(kind, item).detail == "split: 1 new share(s) for 8 old"  # type: ignore[union-attr]


def test_a_spin_off_affects_the_parent_and_the_new_company():
    kind, item = _only({"spin_offs": [_spin_off("GE", "GEV", "2024-04-02")]})
    event = event_from(kind, item)

    assert event is not None
    assert event.affected == ("GE", "GEV")
    assert event.detail == "spin-off: GEV is handed out by GE"


def test_in_a_merger_only_the_bought_company_is_affected_not_the_buyer():
    kind, item = _only({"cash_mergers": [_cash_merger("CSCO", "ACIA", "2021-03-01")]})
    event = event_from(kind, item)

    assert event is not None
    assert event.affected == ("ACIA",)
    assert event.detail == "ACIA is bought by CSCO for cash"


def test_a_name_change_affects_the_old_and_the_new_symbol():
    kind, item = _only({"name_changes": [_name_change("UTX", "RTX", "2020-04-03")]})
    event = event_from(kind, item)

    assert event is not None
    assert event.affected == ("UTX", "RTX")
    assert event.detail == "renamed UTX -> RTX"


def test_an_event_type_this_table_does_not_know_counts_every_symbol_in_it_as_affected():
    class Unknown:
        symbol = "AAA"
        acquirer_symbol = "BBB"
        process_date = date(2026, 12, 1)

    event = event_from("something_new", Unknown())

    assert event is not None
    assert event.affected == ("AAA", "BBB")  # a false alarm costs a look; silence costs a position


def test_an_event_without_any_date_is_dropped():
    class Undated:
        symbol = "AAA"

    assert event_from("forward_splits", Undated()) is None


# ---- the feed ----------------------------------------------------------------------------------


class FakeActionsClient:
    """Answers like the API: the events inside the requested days, at most `cap` of them in total
    (the rest cut without a word), as alpaca-py's own CorporateActionsSet."""

    def __init__(self, raw: dict[str, list[dict[str, Any]]], cap: int = 1000) -> None:
        self.raw = raw
        self.cap = cap
        self.requests: list[CorporateActionsRequest] = []

    def get_corporate_actions(self, request: CorporateActionsRequest) -> CorporateActionsSet:
        self.requests.append(request)
        assert request.start is not None and request.end is not None
        asked = set(request.types or [])
        answer: dict[str, list[dict[str, Any]]] = {}
        taken = 0
        for kind, items in self.raw.items():
            if asked and kind[:-1] not in asked:  # the API's plural against the request's singular
                continue
            for item in items:
                day = item.get("ex_date") or item.get("effective_date") or item["process_date"]
                names = [v for k, v in item.items() if k.endswith("symbol")]
                if not (request.start <= date.fromisoformat(day) <= request.end):
                    continue
                if request.symbols and not set(names) & set(request.symbols):
                    continue
                if taken >= self.cap:
                    continue
                answer.setdefault(kind, []).append(item)
                taken += 1
        return CorporateActionsSet(answer)


def _feed(
    raw: dict[str, list[dict[str, Any]]], cap: int = 1000
) -> tuple[CorporateActionsFeed, Any]:
    client = FakeActionsClient(raw, cap)
    settings = Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")  # type: ignore[call-arg]
    return CorporateActionsFeed(settings, client=client), client


def test_dividends_are_not_asked_for_but_the_events_that_break_a_price_are():
    feed, client = _feed({"forward_splits": [_split("AAPL", "2020-08-31")]})

    feed.events(["AAPL"], date(2020, 1, 1), date(2020, 12, 31))

    asked = client.requests[0].types
    assert "cash_dividend" not in asked
    assert {"forward_split", "reverse_split", "spin_off", "name_change", "cash_merger"} <= set(
        asked
    )


def test_a_window_the_api_fills_to_its_cap_is_halved_until_nothing_is_cut(
    monkeypatch: pytest.MonkeyPatch,
):
    """The live API cut a ten-year answer at exactly 1000 and said nothing. Here the cap is 3."""
    monkeypatch.setattr(ca, "MAX_EVENTS_PER_CALL", 3)
    days = ["2020-02-01", "2020-04-01", "2020-06-01", "2020-08-01", "2020-10-01"]
    raw = {"forward_splits": [_split("AAPL", d) for d in days]}
    feed, client = _feed(raw, cap=3)

    found = feed.events(["AAPL"], date(2020, 1, 1), date(2020, 12, 31))

    assert [e.day.isoformat() for e in found] == days  # all five, though one call returns three
    assert len(client.requests) > 1


def test_a_single_day_with_more_events_than_one_call_returns_is_an_error_not_a_half_list(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ca, "MAX_EVENTS_PER_CALL", 2)
    raw = {"forward_splits": [_split("AAPL", "2020-08-31") for _ in range(3)]}
    feed, _ = _feed(raw, cap=2)

    with pytest.raises(CorporateActionsIncomplete):
        feed.events(["AAPL"], date(2020, 8, 31), date(2020, 8, 31))


def test_a_merger_the_stock_only_buys_is_not_an_event_for_it():
    """CSCO buys ACIA: the API lists the deal under CSCO too, but CSCO's holders keep their shares."""
    feed, _ = _feed({"cash_mergers": [_cash_merger("CSCO", "ACIA", "2021-03-01")]})

    assert feed.events(["CSCO"], date(2021, 1, 1), date(2021, 12, 31)) == []
    assert [e.affected for e in feed.events(["ACIA"], date(2021, 1, 1), date(2021, 12, 31))] == [
        ("ACIA",)
    ]


def test_events_come_oldest_first_and_only_for_the_symbols_asked():
    raw = {
        "forward_splits": [_split("NVDA", "2024-06-10"), _split("AAPL", "2020-08-31")],
        "spin_offs": [_spin_off("GE", "GEV", "2024-04-02")],
    }
    feed, _ = _feed(raw)

    found = feed.events(["NVDA", "GE"], date(2020, 1, 1), date(2024, 12, 31))

    assert [(e.day.isoformat(), e.affected) for e in found] == [
        ("2024-04-02", ("GE", "GEV")),
        ("2024-06-10", ("NVDA",)),
    ]


def test_a_long_symbol_list_is_sent_in_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ca, "SYMBOLS_PER_CALL", 2)
    feed, client = _feed({"forward_splits": [_split("EEE", "2020-08-31")]})

    found = feed.events(["aaa", "BBB", "CCC", "DDD", "EEE"], date(2020, 1, 1), date(2020, 12, 31))

    assert len(client.requests) == 3
    assert [len(r.symbols or []) for r in client.requests] == [2, 2, 1]
    assert [e.affected for e in found] == [("EEE",)]


def test_nothing_in_the_answer_is_an_empty_list():
    feed, _ = _feed({})

    assert feed.events(["AAPL"], date(2026, 11, 7), date(2026, 12, 7)) == []


# ---- the small helpers and the doctor's lines -------------------------------------------------


def _event(symbol: str, day: date) -> CorporateEvent:
    return CorporateEvent("forward_splits", (symbol,), day, "split: 2 new share(s) for 1 old")


def test_in_window_includes_both_ends():
    events = [
        _event("A", date(2026, 11, 6)),
        _event("B", date(2026, 11, 13)),
        _event("C", date(2026, 11, 14)),
    ]

    assert [e.affected[0] for e in in_window(events, date(2026, 11, 6), date(2026, 11, 13))] == [
        "A",
        "B",
    ]


def test_the_lines_name_the_day_the_symbols_and_what_happens():
    text = format_events([_event("SNDK", date(2026, 11, 20))])

    assert "2026-11-20" in text and "SNDK" in text and "split: 2 new share(s) for 1 old" in text


class _Broker:
    """The real alpaca-py Position and Order models, filled with the least they require."""

    def __init__(self, positions: list[str], orders: list[str]) -> None:
        self._positions = positions
        self._orders = orders

    def get_positions(self) -> list[Position]:
        return [
            Position(
                asset_id=uuid.uuid4(),
                symbol=s,
                exchange=AssetExchange.NASDAQ,
                asset_class=AssetClass.US_EQUITY,
                avg_entry_price="100",
                qty="10",
                side=PositionSide.LONG,
                cost_basis="1000",
            )
            for s in self._positions
        ]

    def get_open_orders(self) -> list[Order]:
        now = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)
        return [
            Order(
                id=uuid.uuid4(),
                client_order_id=f"c-{s}",
                created_at=now,
                updated_at=now,
                submitted_at=now,
                order_class=OrderClass.SIMPLE,
                time_in_force=TimeInForce.DAY,
                status=OrderStatus.NEW,
                extended_hours=False,
                symbol=s,
            )
            for s in self._orders
        ]


class _Feed:
    def __init__(self, events: list[CorporateEvent]) -> None:
        self.events_to_return = events
        self.asked: list[tuple[list[str], date, date]] = []

    def events(self, symbols: Any, start: date, end: date) -> list[CorporateEvent]:
        self.asked.append((list(symbols), start, end))
        return self.events_to_return


def test_what_we_hold_is_the_positions_and_the_symbols_of_open_orders():
    assert held_symbols(_Broker(["MSFT", "aapl"], ["NVDA", "AAPL"])) == ["AAPL", "MSFT", "NVDA"]


def test_nothing_held_says_so_and_asks_nothing():
    feed = _Feed([])

    lines = check_holdings(feed, _Broker([], []), date(2026, 11, 6))

    assert lines == ["Corporate actions: no open positions or orders to check."]
    assert feed.asked == []


def test_a_quiet_week_for_what_we_hold_is_one_plain_line():
    feed = _Feed([])

    lines = check_holdings(feed, _Broker(["AAPL"], []), date(2026, 11, 6), days=7)

    assert feed.asked == [(["AAPL"], date(2026, 11, 6), date(2026, 11, 13))]
    assert lines == ["Corporate actions: none announced for AAPL in the next 7 days."]


def test_an_event_on_a_holding_is_a_warning_with_the_event_under_it():
    feed = _Feed([_event("AAPL", date(2026, 11, 9))])

    lines = check_holdings(feed, _Broker(["AAPL"], []), date(2026, 11, 6))

    assert lines[0].startswith("WARNING")
    assert "2026-11-09" in lines[1] and "AAPL" in lines[1]
