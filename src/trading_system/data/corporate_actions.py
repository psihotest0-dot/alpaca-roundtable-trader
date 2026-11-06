"""Corporate actions: splits, spin-offs, mergers and name changes of the stocks we hold or watch.

Why this exists. The candles we store are adjusted for splits and dividends and for nothing else, so
a spin-off or a merger shows up as a price jump that never happened to a holder (RTX on 2020-04-03
looked like -71 %, see data/universe.py and docs/strategy-research-plan.md). For a position it is
worse: a bracket order's stop can fire on the ex-date of a spin-off at a "loss" that is only the
value of the shares handed out on the side. This module answers one question, read-only: what is
about to happen, or has happened, to these symbols?

Facts found by probing the live API on 2026-11-06 (they are in no documentation):

* One call returns at most 1000 events IN TOTAL and cuts the rest without a word. A ten-year query
  for our 145 stocks came back with exactly 1000 events, 978 of them cash dividends; the real
  total is 5,514, of which 5,385 are dividends. So dividends (already in the adjusted prices) are
  left out of the request, and a window that comes back full is halved until no window is full.
* The market-data API lists events that are announced but not yet due, about four weeks ahead
  (on 2026-11-06 it knew of a split on 2026-11-20 and a reverse split on 2026-11-09). Nothing
  further out is known, so "nothing found" means "nothing announced yet", never "nothing will
  happen".
* A merger is listed under the buyer as well as the bought company. Only the bought company's
  holders have their shares replaced, so only it counts as affected here.
* alpaca-py silently drops an event type it does not know. A new type from Alpaca would be missing
  from the result until the library learns it.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Optional, Protocol, Sequence

from alpaca.data.enums import CorporateActionsType
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import CorporateActionsRequest
from pydantic import SecretStr

from trading_system.common.retry import retry_read_request
from trading_system.config import Settings

logger = logging.getLogger(__name__)

# The API's cap on the events one call returns (see the module docstring).
MAX_EVENTS_PER_CALL = 1000
# A comma-separated symbol list goes into the URL; 145 symbols worked, stay well inside any limit.
SYMBOLS_PER_CALL = 100

# Every event type but the cash dividend: dividends are in the adjusted prices already and are
# 98 % of the events. Taken from the library's own list so that a type added later is included.
EVENT_TYPES = [t.value for t in CorporateActionsType if t is not CorporateActionsType.CASH_DIVIDEND]

# The symbol fields of an event whose holders are changed by it, by the API's name for the type.
_AFFECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "forward_splits": ("symbol",),
    "reverse_splits": ("symbol",),
    "unit_splits": ("old_symbol", "new_symbol", "alternate_symbol"),
    "stock_dividends": ("symbol",),
    "spin_offs": ("source_symbol", "new_symbol"),
    "cash_mergers": ("acquiree_symbol",),
    "stock_mergers": ("acquiree_symbol",),
    "stock_and_cash_mergers": ("acquiree_symbol",),
    "redemptions": ("symbol",),
    "name_changes": ("old_symbol", "new_symbol"),
    "worthless_removals": ("symbol",),
    "rights_distributions": ("source_symbol", "new_symbol"),
}
# Every field that can hold a symbol: used for a type this table does not know, where the safe
# reading is "everything named is affected" (a false alarm costs a look; silence costs a position).
_ANY_SYMBOL_FIELDS = (
    "symbol",
    "old_symbol",
    "new_symbol",
    "alternate_symbol",
    "source_symbol",
    "acquirer_symbol",
    "acquiree_symbol",
)


class CorporateActionsIncomplete(RuntimeError):
    """A single day alone holds more events than one call can return: the list cannot be made
    complete, and a half list must not pass for a whole one."""


@dataclass(frozen=True)
class CorporateEvent:
    kind: str  # the API's name for the type: "spin_offs", "forward_splits", ...
    affected: tuple[str, ...]  # the symbols whose holders are changed by the event
    day: date  # the ex-date, else the effective date, else the processing date
    detail: str  # one plain line: what happens


def _symbols_of(item: Any, fields: Iterable[str]) -> tuple[str, ...]:
    found: list[str] = []
    for name in fields:
        value = getattr(item, name, None)
        if isinstance(value, str) and value.strip() and value.strip().upper() not in found:
            found.append(value.strip().upper())
    return tuple(found)


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else "?"


def _describe(kind: str, item: Any) -> str:
    def text(name: str) -> str:
        return str(getattr(item, name, None) or "?")

    if kind in ("forward_splits", "reverse_splits", "unit_splits"):
        return (
            f"split: {_number(getattr(item, 'new_rate', None))} new share(s) for "
            f"{_number(getattr(item, 'old_rate', None))} old"
        )
    if kind == "spin_offs":
        return f"spin-off: {text('new_symbol')} is handed out by {text('source_symbol')}"
    if kind == "cash_mergers":
        return f"{text('acquiree_symbol')} is bought by {text('acquirer_symbol')} for cash"
    if kind == "stock_mergers":
        return f"{text('acquiree_symbol')} is bought by {text('acquirer_symbol')} for shares"
    if kind == "stock_and_cash_mergers":
        return (
            f"{text('acquiree_symbol')} is bought by {text('acquirer_symbol')} for shares and cash"
        )
    if kind == "name_changes":
        return f"renamed {text('old_symbol')} -> {text('new_symbol')}"
    return kind.replace("_", " ")


def event_from(kind: str, item: Any) -> Optional[CorporateEvent]:
    """The event as we keep it, or None for one that carries no date at all."""
    day = (
        getattr(item, "ex_date", None)
        or getattr(item, "effective_date", None)
        or getattr(item, "process_date", None)
    )
    if day is None:
        return None
    fields = _AFFECTED_FIELDS.get(kind, _ANY_SYMBOL_FIELDS)
    return CorporateEvent(
        kind=kind, affected=_symbols_of(item, fields), day=day, detail=_describe(kind, item)
    )


class _Client(Protocol):
    def get_corporate_actions(self, request_params: CorporateActionsRequest) -> Any: ...


class CorporateActionsFeed:
    """Reads corporate actions from Alpaca's market-data API (the same keys as the bars)."""

    def __init__(self, settings: Settings, client: Optional[_Client] = None) -> None:
        if client is None:
            api_key = (
                settings.alpaca_api_key.get_secret_value()
                if isinstance(settings.alpaca_api_key, SecretStr)
                else settings.alpaca_api_key
            )
            secret_key = (
                settings.alpaca_secret_key.get_secret_value()
                if isinstance(settings.alpaca_secret_key, SecretStr)
                else settings.alpaca_secret_key
            )
            client = CorporateActionsClient(api_key=api_key, secret_key=secret_key)
        self.client = client

    @retry_read_request()
    def _fetch(
        self, symbols: Sequence[str], start: date, end: date
    ) -> tuple[list[CorporateEvent], int]:
        """The events of one request and how many the API returned in it (the second number is
        what tells a full, cut-off answer from a complete one)."""
        result = self.client.get_corporate_actions(
            CorporateActionsRequest(symbols=list(symbols), start=start, end=end, types=EVENT_TYPES)
        )
        data = getattr(result, "data", None) or {}
        returned = sum(len(items) for items in data.values())
        events = [
            event
            for kind, items in data.items()
            for item in items
            if (event := event_from(kind, item)) is not None
        ]
        return events, returned

    def _window(self, symbols: Sequence[str], start: date, end: date) -> list[CorporateEvent]:
        events, returned = self._fetch(symbols, start, end)
        if returned < MAX_EVENTS_PER_CALL:
            return events
        if start >= end:
            raise CorporateActionsIncomplete(
                f"{returned} corporate actions on {start} alone for {len(symbols)} symbols: "
                "more than one call can return"
            )
        middle = start + (end - start) // 2
        return self._window(symbols, start, middle) + self._window(
            symbols, middle + timedelta(days=1), end
        )

    def events(self, symbols: Sequence[str], start: date, end: date) -> list[CorporateEvent]:
        """The events from `start` to `end` (both included) that change the holders of any of
        `symbols`, oldest first and without repeats."""
        wanted = sorted({s.strip().upper() for s in symbols if s.strip()})
        found: set[CorporateEvent] = set()
        for first in range(0, len(wanted), SYMBOLS_PER_CALL):
            chunk = wanted[first : first + SYMBOLS_PER_CALL]
            found.update(self._window(chunk, start, end))
        return sorted(
            (e for e in found if set(e.affected) & set(wanted)),
            key=lambda e: (e.day, e.kind, e.affected),
        )


def in_window(events: Iterable[CorporateEvent], first: date, last: date) -> list[CorporateEvent]:
    """The events with `first` <= day <= `last`."""
    return [e for e in events if first <= e.day <= last]


def format_events(events: Sequence[CorporateEvent]) -> str:
    """One line per event: the date, the symbols, what happens."""
    return "\n".join(f"  {e.day}  {', '.join(e.affected):<14} {e.detail}" for e in events)


class _Holdings(Protocol):
    def get_positions(self) -> Any: ...

    def get_open_orders(self) -> Any: ...


def held_symbols(broker: _Holdings) -> list[str]:
    """The symbols of the open positions and of the open orders, sorted: the ones an event
    would hit at once."""
    names = {str(p.symbol).upper() for p in broker.get_positions()}
    names |= {str(o.symbol).upper() for o in broker.get_open_orders()}
    return sorted(names)


class _Feed(Protocol):
    def events(self, symbols: Sequence[str], start: date, end: date) -> list[CorporateEvent]: ...


def check_holdings(feed: _Feed, broker: _Holdings, today: date, days: int = 7) -> list[str]:
    """The lines `doctor --network` prints: what, if anything, is due within `days` days for the
    positions and open orders. A warning to read, never a reason to stop on its own."""
    names = held_symbols(broker)
    if not names:
        return ["Corporate actions: no open positions or orders to check."]
    due = feed.events(names, today, today + timedelta(days=days))
    if not due:
        return [
            f"Corporate actions: none announced for {', '.join(names)} in the next {days} days."
        ]
    return [
        f"WARNING, corporate actions due within {days} days on what we hold:",
        format_events(due),
    ]
