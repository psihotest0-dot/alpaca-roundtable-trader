"""The Oracle observer: the live strategy over ~133 more stocks, asked about, never traded.

Why. The model is judged on its vetoes, and the 22 watchlist symbols give about 0.8 signals a day:
30 vetoes would take months (docs/oracle-provider-task.md, section 5). The same strategy over the
rest of the observation universe (the research universe less the watchlist) gives several times as
many signals. Here each one is put to the Oracle exactly as the live loop does it (same request, same
record in `oracle_decisions`, same news), and that is all: **no order, ever**.

**How it cannot trade.** `ObservationCycle` is a `TradingCycle` built WITHOUT a risk engine, an
execution engine or an order repository (they are `None`), so a code path that tried to place an
order would fail with an AttributeError instead of placing it. It overrides the three entry points
the `Runner` calls and uses only `_ask_oracle`, `_snapshot` and `_news_for` of its parent. Its journal
entry is `observe_cycle`, not `entries_cycle`, so the daily report keeps counting the live loop only.

**Where the bars come from.** Not from a download per cycle (155 symbols would take minutes) but from
the observation store `var/observe.db`, brought up to date at the start of every cycle with the same
`update_history` as `fetch-history`, then read back through the live loop's own `align_and_filter`:
the strategy sees what it would see live. The same store feeds the hours research.

A signal's outcome is not looked at here; the judgement is a separate, written procedure.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Sequence

import pandas as pd

from trading_system.data.history import TIMEFRAMES, update_history
from trading_system.data.live_price import UnusableLivePrice, check_live_price
from trading_system.domain.models import StrategySignal, TradingState
from trading_system.runtime.cycle import EntriesReport, TradingCycle
from trading_system.storage.repositories import MarketBarRepository
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import filter_signals, generate_signals

logger = logging.getLogger(__name__)

# How far back the observation store is kept (the hours research reads all of it).
HISTORY_DAYS = 180
OBSERVE_EVENT = "observe_cycle"


class ObservationCycle(TradingCycle):
    def __init__(
        self,
        *,
        settings: Any,
        broker: Any,
        data: Any,
        store: MarketBarRepository,
        universe: Sequence[str],
        observed: Sequence[str],
        signal_repo: Any,
        state_repo: Any,
        audit_repo: Any,
        strategy_config: StrategyConfig,
        oracle: Any,
        news: Any = None,
        updater: Callable[..., Any] = update_history,
    ) -> None:
        super().__init__(
            settings=settings,
            broker=broker,
            data=data,
            risk_engine=None,  # no order can be sized ...
            execution=None,  # ... or sent: these two are what an order would need
            signal_repo=signal_repo,
            order_repo=None,
            state_repo=state_repo,
            strategy_config=strategy_config,
            oracle=oracle,
            risk_repo=None,
            audit_repo=audit_repo,
            alerts=None,
            news=news,
        )
        self.store = store
        self.universe = list(universe)  # the bars kept up to date
        self.observed = list(observed)  # the symbols the Oracle is asked about
        self._updater = updater

    # -- what the Runner calls ----------------------------------------------------------

    def run_housekeeping(self, now: datetime) -> None:
        """Nothing to keep: there are no orders, positions or fills of this cycle's own."""

    def revisit_held(self, now: datetime) -> int:
        return 0

    def run_entries(self, now: datetime, expected_bar_start: datetime) -> EntriesReport:
        try:
            report, detail = self._observe(now, expected_bar_start)
        except Exception as e:
            self._record(now, expected_bar_start, EntriesReport(errors=1, skipped=f"crashed: {e}"))
            raise
        self._record(now, expected_bar_start, report, detail)
        return report

    # -- one pass over the newest bar -----------------------------------------------------

    def _observe(
        self, now: datetime, expected_bar_start: datetime
    ) -> tuple[EntriesReport, dict[str, int]]:
        try:
            self._updater(
                self.data,
                self.store,
                self.universe,
                list(TIMEFRAMES),
                start=now - timedelta(days=HISTORY_DAYS),
                now=now,
            )
        except Exception as e:
            logger.error(f"Could not bring the observation bars up to date: {e}")
            return EntriesReport(skipped="bars could not be updated", retry_later=True), {}

        bars_15m, bars_1h = self._load(now)
        newest = max((df.index[-1] for df in bars_15m.values()), default=None)
        if newest is None or pd.Timestamp(newest) < pd.Timestamp(expected_bar_start):
            logger.warning(
                f"The bar that started at {expected_bar_start:%H:%M} is not in the observation "
                f"store yet (newest: {newest})."
            )
            return EntriesReport(skipped="bar not available yet", retry_later=True), {}

        signals = filter_signals(
            generate_signals(bars_15m, bars_1h, self.strategy_config, now), set()
        )
        signals = [s for s in signals if s.symbol in self.observed]
        logger.info(f"Observing {len(self.observed)} symbols: {len(signals)} signal(s).")
        report = EntriesReport(signals=len(signals))
        detail: dict[str, int] = {"symbols": len(bars_15m), "asked": 0, "no_price": 0}
        state = self.state_repo.get_state() or TradingState()
        for signal in signals:
            try:
                asked = self._ask_about(signal, state, bars_15m, bars_1h, now)
            except Exception as e:  # one broken signal must not stop the others
                logger.error(f"Error observing {signal.symbol}: {e}")
                report.errors += 1
                continue
            detail["asked" if asked else "no_price"] += 1
        return report, detail

    def _load(self, now: datetime) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """The stored bars of the observed symbols, cut and aligned as the live loop does."""
        stored_15m = self.store.load(
            self.observed, "15m", start=now - timedelta(days=self.settings.lookback_days)
        )
        stored_1h = self.store.load(
            self.observed, "1h", start=now - timedelta(days=self.settings.htf_lookback_days)
        )
        out_15m: dict[str, pd.DataFrame] = {}
        out_1h: dict[str, pd.DataFrame] = {}
        for symbol in self.observed:
            if symbol not in stored_15m or symbol not in stored_1h:
                continue
            fifteen, hourly = self.data.align_and_filter(stored_15m[symbol], stored_1h[symbol], now)
            if not fifteen.empty and not hourly.empty:
                out_15m[symbol], out_1h[symbol] = fifteen, hourly
        return out_15m, out_1h

    def _ask_about(
        self,
        signal: StrategySignal,
        state: TradingState,
        bars_15m: dict[str, pd.DataFrame],
        bars_1h: dict[str, pd.DataFrame],
        now: datetime,
    ) -> bool:
        """Puts one signal to the Oracle. False if there was no usable price to put with it. The
        verdict is ignored: it is recorded by the Oracle's own record, which is the point."""
        try:
            self.signal_repo.save_signal(signal)  # the record of an answer points at it
        except Exception as e:
            logger.error(f"Could not save the observed signal: {e}")
        bar_close = Decimal(str(bars_15m[signal.symbol].iloc[-1]["close"]))
        try:
            live = self.data.fetch_latest_prices([signal.symbol]).get(signal.symbol)
        except Exception as e:
            logger.error(f"Could not fetch a live price for {signal.symbol}: {e}")
            live = None
        try:
            price = check_live_price(
                live,
                reference_close=bar_close,
                now=datetime.now(timezone.utc),
                max_age_seconds=self.settings.live_price_max_age_seconds,
                max_deviation_pct=Decimal(str(self.settings.live_price_max_deviation_pct)),
            )
        except UnusableLivePrice as e:
            logger.warning(f"Not asking about {signal.symbol}: {e}")
            return False
        account, positions, _ = self._snapshot(now)
        self._ask_oracle(signal, state, bars_15m, bars_1h, price, now, account, positions)
        return True

    # -- the journal ----------------------------------------------------------------------------

    def _record(
        self,
        now: datetime,
        bar_start: datetime,
        report: EntriesReport,
        detail: dict[str, int] | None = None,
    ) -> None:
        if self.audit_repo is None:
            return
        try:
            self.audit_repo.record(
                OBSERVE_EVENT,
                {
                    "bar_start": bar_start.isoformat(),
                    "skipped": report.skipped,
                    "retry_later": report.retry_later,
                    "signals": report.signals,
                    "errors": report.errors,
                    **(detail or {}),
                },
                when=now,
            )
        except Exception as e:  # the journal is better with a gap than with no observation
            logger.warning(f"Could not journal the observation: {e}")
