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

**The extra stocks (docs/oracle-provider-task.md, section 5.2).** Besides the observed symbols the
Oracle is asked about ~300 more liquid stocks (`data/oracle_extra.py`, picked by a written rule that
looks at no return), to reach the minimum of vetoes in weeks instead of months. They are kept in the
same store with a shorter history and are NOT part of the observation universe: the candidates'
forward test and the hours research see the registered symbols only.

**Expired signals are not asked about** (section 5.3): the live loop's risk refuses them before the
Oracle, and the observer, which has no risk, keeps that rule itself.

A signal's outcome is not looked at here; the judgement is a separate, written procedure.

**The strategy candidates (docs/strategy-research-plan.md, section 12.1).** When given `candidates`, a
pass also journals their signals over the same bars (and the live strategy's own, as the control)
through `research/candidates.py`: they are NOT put to the Oracle (that would change the sample the
model is judged on) and no order is possible. This step runs first and cannot stop the Oracle's work.
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
from trading_system.research.candidates import record_pass
from trading_system.storage.repositories import MarketBarRepository
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import filter_signals, generate_signals

logger = logging.getLogger(__name__)

# How far back the observation store is kept (the hours research reads all of it).
HISTORY_DAYS = 180
# The extra stocks (data/oracle_extra.py) feed only the Oracle's questions: the strategy needs
# `lookback_days` of 15-minute and `htf_lookback_days` of hourly bars (30 and 30), and a shadow trade
# needs the bars after its signal. 45 days cover both and make the first download about a quarter of the
# size; nothing else reads their bars.
EXTRA_HISTORY_DAYS = 45
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
        extra: Sequence[str] = (),
        signal_repo: Any,
        state_repo: Any,
        audit_repo: Any,
        strategy_config: StrategyConfig,
        oracle: Any,
        news: Any = None,
        updater: Callable[..., Any] = update_history,
        candidates: Sequence[BaseStrategy] = (),
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
        self.observed = list(
            observed
        )  # the symbols the Oracle is asked about, and the candidates run on
        # The extra stocks (docs/oracle-provider-task.md, section 5.2): asked about like the others,
        # but outside the observation universe, so the candidates' registered forward test (variants
        # 18-20) and the hours research never see them.
        known = set(self.universe) | set(self.observed)
        self.extra = [s for s in extra if s not in known]
        self._updater = updater
        self.candidates = list(candidates)  # run forward in time, never asked about, never traded

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
        self._update_extra(now)

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
        registered = set(self.observed)
        extra = set(self.extra)
        signals = [s for s in signals if s.symbol in registered or s.symbol in extra]
        report = EntriesReport(signals=len(signals))
        detail: dict[str, int] = {
            "symbols": len(bars_15m),
            "extra_symbols": len(extra),
            "extra_signals": sum(1 for s in signals if s.symbol in extra),
            "asked": 0,
            "extra_asked": 0,
            "no_price": 0,
        }
        # The candidates' forward test was registered over the observed symbols alone: the control and
        # the bars they get are cut to that set, whatever else the Oracle is asked about.
        detail.update(
            self._journal_candidates(
                [s for s in signals if s.symbol in registered],
                {k: v for k, v in bars_15m.items() if k in registered},
                {k: v for k, v in bars_1h.items() if k in registered},
                now,
                expected_bar_start,
            )
        )
        # The live loop's first look at risk refuses an expired signal before the Oracle is asked
        # (risk/engine.py); the observer has no risk, so it keeps that rule here. Without it the
        # first pass of a session asked, twice, about a cross on the last bar of the day before
        # (docs/oracle-provider-task.md, section 5.3). The candidates above saw every signal, as before.
        fresh = [s for s in signals if s.expires_at is None or s.expires_at >= now]
        detail["expired"] = len(signals) - len(fresh)
        logger.info(
            f"Observing {len(self.observed)} + {len(self.extra)} extra symbols: "
            f"{len(signals)} signal(s)"
            + (f", {detail['expired']} expired and not asked." if detail["expired"] else ".")
        )
        state = self.state_repo.get_state() or TradingState()
        for signal in fresh:
            try:
                asked = self._ask_about(signal, state, bars_15m, bars_1h, now)
            except Exception as e:  # one broken signal must not stop the others
                logger.error(f"Error observing {signal.symbol}: {e}")
                report.errors += 1
                continue
            detail["asked" if asked else "no_price"] += 1
            if asked and signal.symbol in extra:
                detail["extra_asked"] += 1
        return report, detail

    def _update_extra(self, now: datetime) -> None:
        """The extra stocks' bars, from their own, shorter window. Whatever goes wrong here is logged
        and leaves the pass on the registered symbols alone: a stock with stale bars gives only
        expired signals, and those are not asked about."""
        if not self.extra:
            return
        try:
            self._updater(
                self.data,
                self.store,
                self.extra,
                list(TIMEFRAMES),
                start=now - timedelta(days=EXTRA_HISTORY_DAYS),
                now=now,
            )
        except Exception as e:
            logger.error(f"Could not bring the extra stocks' bars up to date: {e}")

    def _journal_candidates(
        self,
        control: list[StrategySignal],
        bars_15m: dict[str, pd.DataFrame],
        bars_1h: dict[str, pd.DataFrame],
        now: datetime,
        bar_start: datetime,
    ) -> dict[str, int]:
        """The candidates' signals (and the control's) into the journal. Whatever goes wrong here is
        logged and leaves the Oracle's work untouched."""
        if not self.candidates or self.audit_repo is None:
            return {}
        try:
            return record_pass(
                self.audit_repo, self.candidates, control, bars_15m, bars_1h, now, bar_start
            )
        except Exception as e:
            logger.error(f"Could not journal the candidates' signals: {e}")
            return {"candidate_errors": 1}

    def _load(self, now: datetime) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """The stored bars of the observed symbols, cut and aligned as the live loop does."""
        asked = self.observed + self.extra
        stored_15m = self.store.load(
            asked, "15m", start=now - timedelta(days=self.settings.lookback_days)
        )
        stored_1h = self.store.load(
            asked, "1h", start=now - timedelta(days=self.settings.htf_lookback_days)
        )
        out_15m: dict[str, pd.DataFrame] = {}
        out_1h: dict[str, pd.DataFrame] = {}
        for symbol in asked:
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
