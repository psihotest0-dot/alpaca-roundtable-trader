import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional
from decimal import Decimal
import logging
from datetime import date, datetime, timedelta, timezone
import json

from trading_system.config import Settings
from trading_system.data.live_price import LivePrice, UnusableLivePrice, check_live_price
from trading_system.data.bars import normalize_bars
from trading_system.domain.enums import OrderSide
from trading_system.domain.models import (
    AccountSnapshot,
    PositionSnapshot,
    StrategySignal,
    TradingState,
)
from trading_system.risk.daily_equity import day_marker, trading_day
from trading_system.risk.engine import RiskEngine
from trading_system.risk.entry_plan import (
    AtrExit,
    InvalidEntryPlan,
    atr_exit_percentages,
    build_entry_plan,
)
from trading_system.runtime.runner import ENTRY_DELAY
from trading_system.runtime.schedule import BAR, is_regular_session
from trading_system.strategy.atr import atr_and_close
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.config import RUNTIME_STRATEGY
from trading_system.strategy.trend import filter_signals
from trading_system.strategy.trend_cross import TrendCrossStrategy
from trading_system.backtest.exits import resolve_exit
from trading_system.backtest.models import (
    BacktestConfig,
    BacktestTrade,
    BacktestPosition,
    BacktestResult,
    ShadowTrade,
)
from trading_system.backtest.shadow import simulate_shadow_trade

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class BacktestEngine:
    def __init__(
        self,
        settings: Settings,
        bt_config: BacktestConfig,
        df_15m: Dict[str, pd.DataFrame],
        df_1h: Dict[str, pd.DataFrame],
        prefilter: bool = True,
        strategy: Optional[BaseStrategy] = None,
        atr_exit: Optional[AtrExit] = None,
    ):
        """`strategy`: what produces the signals; by default the one the live loop runs.

        `atr_exit`: a CANDIDATE (research plan, section 13.1). The stop and the target of each entry
        are multiples of the ATR at the signal bar instead of the live percentages; nothing else
        changes (not the size either). The live loop never sets it.

        `prefilter`: ask the strategy only about symbols whose last closed bar could be a
        signal, according to the strategy's own hint (`candidate_bars`, read off indicators
        computed once for the whole history). It gives the same signals as asking about every
        symbol at every bar, because a strategy judges each symbol on its own, and it is many
        times faster. Turn it off to get the slow, exhaustive behaviour (the tests compare)."""
        self.settings = settings
        self.bt_config = bt_config

        self.df_15m = df_15m
        self.df_1h = df_1h

        # By default the strategy and parameters the live loop runs with: a backtest of other
        # ones says nothing about the system that trades (it is useful for comparing them).
        self.strategy: BaseStrategy = strategy or TrendCrossStrategy(RUNTIME_STRATEGY)

        # The live system's risk engine, not a copy of it. It is told the time of the bar
        # being replayed (see _clock), so a signal from last month is judged by the same rules.
        self._decision_time: datetime = _EPOCH
        self.risk_engine = RiskEngine(settings, clock=self._clock)
        self.atr_exit = atr_exit

        self.cash = bt_config.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []

        self.equity_curve: List[float] = []
        self.max_equity = float(self.cash)
        self.max_drawdown = 0.0
        self.peak_exposure = 0.0

        # The daily loss guard measures from the equity at the end of the previous New York day.
        self._last_equity: Decimal = bt_config.initial_capital
        self._day: Optional[date] = None
        self._day_start_equity: Decimal = bt_config.initial_capital

        self.signals_seen = 0
        self.shadow_skipped = 0
        self.not_traded: Counter[str] = Counter()
        self.daily_equity: Dict[str, float] = {}

        self.prefilter = prefilter
        self._bars_15m: Dict[str, pd.DataFrame] = {}
        self._bars_1h: Dict[str, pd.DataFrame] = {}
        self._could_signal: Dict[str, np.ndarray] = {}

        self.oracle_decisions: Dict[str, bool] = {}
        self.oracle_verdicts: Dict[str, str] = {}
        self.shadow_trades: List[ShadowTrade] = []
        # A shadow position is held like a real one: no second one on a symbol while it is open.
        self._shadow_busy_until: Dict[str, datetime] = {}
        if bt_config.replay_oracle_path:
            self._load_replay_oracle(bt_config.replay_oracle_path)

    def _clock(self) -> datetime:
        return self._decision_time

    def _load_replay_oracle(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                # Assume dict mapping signal_id -> approved bool
                for k, v in data.items():
                    if isinstance(v, dict) and "approved" in v:
                        self.oracle_decisions[k] = bool(v["approved"])
                        fallback = "APPROVE" if v["approved"] else "VETO"
                        self.oracle_verdicts[k] = str(v.get("verdict") or fallback)
                    else:
                        self.oracle_decisions[k] = bool(v)
                        self.oracle_verdicts[k] = "APPROVE" if v else "VETO"
            logger.info(f"Loaded {len(self.oracle_decisions)} oracle decisions from replay.")
        except Exception as e:
            logger.error(f"Failed to load replay oracle: {e}")

    def run(self) -> BacktestResult:
        self._prepare()

        # Align all unique timestamps from 15m data
        all_timestamps = set()
        for df in self.df_15m.values():
            if not df.empty:
                all_timestamps.update(df["timestamp"].tolist())

        sorted_timestamps = sorted(list(all_timestamps))

        for current_time in sorted_timestamps:
            # A bar starting at T closes at T + 15m. At T the bar before it has just closed,
            # and that is when the live loop looks at it (ENTRY_DELAY after the close).
            self._start_day_if_new(current_time)

            # 1. Decide on the signals of the bar that just closed. This comes before the
            #    stops of the new bar are looked at: the decision is taken at the start of
            #    the bar, with the positions and cash as they were then.
            #    The live loop looks for entries only while the market is open, so the
            #    backtest does too (stops below are still checked on every bar).
            if is_regular_session(current_time):
                self._look_for_entries(current_time)

            # 2. Stops and targets on the new bar, new positions included.
            self._process_bar_executions(current_time)

            # 3. Record the equity at the end of the bar.
            total_equity = self._calculate_equity(current_time)
            self._last_equity = total_equity
            self.equity_curve.append(float(total_equity))
            self.daily_equity[trading_day(current_time).isoformat()] = float(total_equity)
            if total_equity > self.max_equity:
                self.max_equity = float(total_equity)
            drawdown = (self.max_equity - float(total_equity)) / self.max_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

            # Track peak exposure
            invested = float(total_equity) - float(self.cash)
            exposure_pct = invested / float(total_equity) if total_equity > 0 else 0
            if exposure_pct > self.peak_exposure:
                self.peak_exposure = exposure_pct

        return self._generate_report()

    def _start_day_if_new(self, current_time: datetime) -> None:
        day = trading_day(current_time)
        if day != self._day:
            self._day = day
            self._day_start_equity = self._last_equity

    def _look_for_entries(self, current_time: datetime) -> None:
        self._decision_time = current_time + ENTRY_DELAY
        # The strategy works on closed bars only, judged by the evaluation time.
        data_15m, data_1h = self._strategy_inputs(current_time)
        signals = self.strategy.generate_signals(data_15m, data_1h, current_time)
        fresh = filter_signals(signals, set(self.positions.keys()))
        self.signals_seen += len(fresh)
        for sig in fresh:
            if not self._apply_oracle(sig):
                self.not_traded["Oracle: no approval"] += 1
                if (
                    sig.signal_id in self.oracle_decisions
                ):  # the Oracle was asked and did not approve
                    self._shadow(sig, current_time)
                continue
            self._enter(sig, current_time)

    def _prepare(self) -> None:
        """Brings every symbol's bars into one shape (sorted, one row per instant, UTC) and, if
        the prefilter is on, marks the bars where a signal is possible at all."""
        self._bars_15m = {s: normalize_bars(df) for s, df in self.df_15m.items()}
        self._bars_1h = {s: normalize_bars(df) for s, df in self.df_1h.items()}
        self._could_signal = {}
        if not self.prefilter:
            return
        for symbol, frame in self._bars_15m.items():
            if frame.empty:
                continue
            flags = self.strategy.candidate_bars(frame, self._bars_1h.get(symbol))
            if flags is not None:
                self._could_signal[symbol] = flags

    def _may_signal(self, symbol: str, current_time: datetime) -> bool:
        """Whether the last bar closed by `current_time` could be a signal bar for `symbol`."""
        flags = self._could_signal.get(symbol)
        if flags is None:
            return True  # the strategy gave no hint for this symbol: ask
        position = self._bars_15m[symbol].index.searchsorted(current_time - BAR, side="right") - 1
        return position >= 1 and bool(flags[position])

    def _strategy_inputs(
        self, current_time: datetime
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """What the strategy is shown at `current_time`: for each symbol the bars of the live
        loop's window (LOOKBACK_DAYS of 15-minute bars, HTF_LOOKBACK_DAYS of 1-hour bars) that
        started before now. Nothing from the future, and no more history than the live loop has."""
        span_15m = pd.Timedelta(days=self.settings.lookback_days)
        # The live loop's window, or more when the strategy says its hourly indicator needs it.
        span_1h = pd.Timedelta(
            days=max(self.settings.htf_lookback_days, self.strategy.htf_history_days or 0)
        )
        data_15m: Dict[str, pd.DataFrame] = {}
        data_1h: Dict[str, pd.DataFrame] = {}
        for symbol, frame in self._bars_15m.items():
            hourly = self._bars_1h.get(symbol)
            if frame.empty or hourly is None or hourly.empty:
                continue
            if self.prefilter and not self._may_signal(symbol, current_time):
                continue
            data_15m[symbol] = self._window(frame, current_time, span_15m)
            data_1h[symbol] = self._window(hourly, current_time, span_1h)
        return data_15m, data_1h

    @staticmethod
    def _window(frame: pd.DataFrame, current_time: datetime, span: pd.Timedelta) -> pd.DataFrame:
        first = frame.index.searchsorted(current_time - span, side="left")
        end = frame.index.searchsorted(current_time, side="left")
        return frame.iloc[first:end]

    def _apply_oracle(self, sig: StrategySignal) -> bool:
        # Only a replay file brings an Oracle into a backtest. ORACLE_ENABLED is the live loop's
        # switch: with it on and no file, every signal used to be refused ("not in the replay")
        # and the backtest showed no trades at all.
        if not self.bt_config.replay_oracle_path:
            return True  # no Oracle to replay: every signal goes on
        if sig.signal_id in self.oracle_decisions:
            return self.oracle_decisions[sig.signal_id]
        return False  # Abstain / default veto if not found in replay

    def _bars_before(self, symbol: str, moment: datetime) -> Optional[pd.DataFrame]:
        """The bars of `symbol` that had closed by `moment`, or None if there are none."""
        df = self.df_15m.get(symbol)
        if df is None or df.empty:
            return None
        closed = df[df["timestamp"] + BAR <= moment]
        return None if closed.empty else closed

    def _account_at(self, current_time: datetime) -> tuple[AccountSnapshot, list[PositionSnapshot]]:
        """The account as the risk engine sees it at the decision: positions marked at the last
        closed bar (what was known then, not the close of the bar being replayed)."""
        equity = self.cash
        snapshots: list[PositionSnapshot] = []
        for sym, pos in self.positions.items():
            closed = self._bars_before(sym, current_time)
            mark = (
                Decimal(str(closed.iloc[-1]["close"]))
                if closed is not None
                else pos.average_entry_price
            )
            value = pos.quantity * mark
            equity += value
            snapshots.append(
                PositionSnapshot(
                    symbol=sym,
                    quantity=pos.quantity,
                    market_value=value,
                    average_entry_price=pos.average_entry_price,
                    unrealized_pl=value - pos.quantity * pos.average_entry_price,
                )
            )
        account = AccountSnapshot(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash,  # no margin
            day_trade_count=0,  # the day trader rule is Phase 4
            updated_at=self._decision_time,
        )
        return account, snapshots

    def _entry_price(self, symbol: str, current_time: datetime) -> tuple[Optional[Decimal], str]:
        """The price an entry at `current_time` would be planned around: the first print of the
        bar that just started, through the same freshness and distance check as the live price.
        (price, "") or (None, why not)."""
        closed = self._bars_before(symbol, current_time)
        df = self.df_15m.get(symbol)
        starting = df[df["timestamp"] == current_time] if df is not None else None
        if closed is None or starting is None or starting.empty:
            # No print after the signal bar: the live system would have no fresh price either.
            return None, "No bar after the signal bar"

        reference_close = Decimal(str(closed.iloc[-1]["close"]))
        first_print = Decimal(str(starting.iloc[0]["open"]))
        try:
            price = check_live_price(
                LivePrice(first_print, current_time),
                reference_close=reference_close,
                now=self._decision_time,
                max_age_seconds=self.settings.live_price_max_age_seconds,
                max_deviation_pct=Decimal(str(self.settings.live_price_max_deviation_pct)),
            )
        except UnusableLivePrice:
            return None, "No usable price"
        return price, ""

    def _exit_settings(self, symbol: str, current_time: datetime) -> tuple[Optional[Settings], str]:
        """The settings an entry's plan is built from: the live ones, or, with an ATR exit, a copy
        whose stop and target percentages are the ATR ones of this signal (the ATR of the live window
        of bars up to the signal bar). (None, why) when there is no ATR."""
        if self.atr_exit is None:
            return self.settings, ""
        closed = self._bars_before(symbol, current_time)
        if closed is None:
            return None, "No ATR"
        since = current_time - timedelta(days=self.settings.lookback_days)
        pair = atr_and_close(closed[closed["timestamp"] > since])
        if pair is None:
            return None, "No ATR"
        try:
            stop, target = atr_exit_percentages(*pair, self.atr_exit)
        except InvalidEntryPlan:
            return None, "ATR exit is not valid"
        return (
            self.settings.model_copy(
                update={"stop_loss_pct": float(stop), "take_profit_pct": float(target)}
            ),
            "",
        )

    def _shadow(self, sig: StrategySignal, current_time: datetime) -> None:
        """Plays out a signal the Oracle did not approve, as if it had been traded (Phase 3.5)."""
        busy = self._shadow_busy_until.get(sig.symbol)
        if busy is not None and current_time < busy:
            return  # an earlier shadow position in this symbol is still open
        price, _ = self._entry_price(sig.symbol, current_time)
        bars = self._bars_15m.get(sig.symbol)
        exit_settings, _ = self._exit_settings(sig.symbol, current_time)
        if price is None or bars is None or bars.empty or exit_settings is None:
            self.shadow_skipped += 1
            return
        try:
            plan = build_entry_plan(
                price,
                limit_buffer_pct=Decimal(str(exit_settings.entry_limit_buffer_pct)),
                stop_loss_pct=Decimal(str(exit_settings.stop_loss_pct)),
                take_profit_pct=Decimal(str(exit_settings.take_profit_pct)),
            )
        except InvalidEntryPlan:
            self.shadow_skipped += 1
            return
        fill = min(
            plan.limit_price, price * (Decimal("1") + Decimal(str(self.bt_config.slippage_pct)))
        )
        shadow = simulate_shadow_trade(
            symbol=sig.symbol,
            signal_id=sig.signal_id or "",
            verdict=self.oracle_verdicts.get(sig.signal_id or "", "VETO"),
            bars=bars,
            entry_time=current_time,
            entry_price=fill,
            stop_loss=plan.stop_price,
            take_profit=plan.take_profit_price,
            slippage_pct=self.bt_config.slippage_pct,
            fee_pct=self.bt_config.fee_pct_per_trade,
        )
        self.shadow_trades.append(shadow)
        self._shadow_busy_until[sig.symbol] = (
            datetime.max.replace(tzinfo=timezone.utc)
            if shadow.exit_reason == "OPEN"
            else shadow.exit_time
        )

    def _enter(self, sig: StrategySignal, current_time: datetime) -> None:
        """One signal through the live price check, the real risk engine, and a fill."""
        price, why_not = self._entry_price(sig.symbol, current_time)
        if price is None:
            self.not_traded[why_not] += 1
            return

        exit_settings, why_not = self._exit_settings(sig.symbol, current_time)
        if exit_settings is None:
            self.not_traded[why_not] += 1
            return
        # The real risk engine; with an ATR exit, one built on the settings of this signal's exit.
        risk_engine = (
            self.risk_engine
            if exit_settings is self.settings
            else RiskEngine(exit_settings, clock=self._clock)
        )

        account, positions = self._account_at(current_time)
        state = TradingState(
            daily_start_equity=self._day_start_equity,
            daily_equity_date=day_marker(trading_day(current_time)),
        )
        decision, intent = risk_engine.evaluate_signal(
            signal=sig,
            account=account,
            positions=positions,
            active_orders=[],  # an entry fills at once here, so none is ever waiting
            current_price=price,
            state=state,
        )
        if not (decision.approved and intent):
            self.not_traded[f"Risk: {decision.reasoning.split(':')[0]}"] += 1
            return
        if (
            intent.limit_price is None
            or intent.stop_price is None
            or intent.take_profit_price is None
        ):
            self.not_traded["Incomplete entry plan"] += 1
            return

        # A marketable limit fills at once at about the price; the limit caps what it can cost.
        fill_price = min(
            intent.limit_price, price * (Decimal("1") + Decimal(str(self.bt_config.slippage_pct)))
        )
        notional = intent.quantity * fill_price
        fee = self.bt_config.fee_per_trade + notional * Decimal(
            str(self.bt_config.fee_pct_per_trade)
        )
        if self.cash < notional + fee:
            self.not_traded["Not enough cash"] += 1
            return

        self.cash -= notional + fee
        entry_slippage = intent.quantity * (fill_price - price)
        # The legs sit where the entry plan put them (measured from the limit), not where the
        # fill happened to land.
        self.positions[sig.symbol] = BacktestPosition(
            symbol=sig.symbol,
            quantity=intent.quantity,
            average_entry_price=fill_price,
            entry_time=current_time,
            entry_fee=fee,
            entry_slippage=entry_slippage,
            stop_loss=intent.stop_price,
            take_profit=intent.take_profit_price,
        )

    def _process_bar_executions(self, current_time: datetime) -> None:
        closed_positions = []
        for sym, pos in self.positions.items():
            df = self.df_15m.get(sym)
            if df is None:
                continue

            bar = df[df["timestamp"] == current_time]
            if bar.empty:
                continue
            bar = bar.iloc[0]

            hit = resolve_exit(
                Decimal(str(bar["open"])),
                Decimal(str(bar["high"])),
                Decimal(str(bar["low"])),
                pos.stop_loss,
                pos.take_profit,
            )

            if hit is not None:
                reason, exit_price = hit
                # Apply slippage (we sell slightly lower)
                price_seen = exit_price
                exit_price = exit_price * (Decimal("1") - Decimal(str(self.bt_config.slippage_pct)))
                exit_slippage = pos.quantity * (price_seen - exit_price)

                notional = pos.quantity * exit_price
                fee = self.bt_config.fee_per_trade + (
                    notional * Decimal(str(self.bt_config.fee_pct_per_trade))
                )

                self.cash += notional - fee

                entry_notional = pos.quantity * pos.average_entry_price
                pnl = (notional - fee) - entry_notional - pos.entry_fee
                cost = entry_notional + pos.entry_fee
                pnl_pct = float(pnl / cost) if cost > 0 else 0.0

                trade = BacktestTrade(
                    symbol=sym,
                    side=OrderSide.BUY,  # Always long MVP
                    quantity=pos.quantity,
                    entry_price=pos.average_entry_price,
                    entry_time=pos.entry_time,
                    exit_price=exit_price,
                    exit_time=current_time,
                    exit_reason=reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    fees_paid=fee + pos.entry_fee,
                    slippage_paid=pos.entry_slippage + exit_slippage,  # already in the prices
                )
                self.trades.append(trade)
                closed_positions.append(sym)

        for sym in closed_positions:
            self.positions.pop(sym, None)

    def _calculate_equity(self, current_time: datetime) -> Decimal:
        equity = self.cash
        for sym, pos in self.positions.items():
            df = self.df_15m.get(sym)
            if df is not None:
                # Get the close of the current bar if available, else last known close
                bar = df[df["timestamp"] <= current_time]
                if not bar.empty:
                    close_px = Decimal(str(bar.iloc[-1]["close"]))
                    equity += pos.quantity * close_px
                else:
                    equity += pos.quantity * pos.average_entry_price
        return equity

    def _generate_report(self) -> BacktestResult:
        # Close all open positions at last known price for final equity calculation
        final_equity = self.cash
        for sym, pos in self.positions.items():
            df = self.df_15m.get(sym)
            if df is not None and not df.empty:
                close_px = Decimal(str(df.iloc[-1]["close"]))
                final_equity += pos.quantity * close_px

        initial = self.bt_config.initial_capital
        total_ret = float((final_equity - initial) / initial)

        wins = [t for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.trades if t.pnl and t.pnl <= 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0.0

        gross_profit_val = sum((t.pnl for t in wins if t.pnl), Decimal("0.0"))
        gross_loss_val = abs(sum((t.pnl for t in losses if t.pnl), Decimal("0.0")))

        profit_factor = 0.0
        if gross_loss_val > Decimal("0.0"):
            profit_factor = float(gross_profit_val / gross_loss_val)
        elif gross_profit_val > Decimal("0.0"):
            profit_factor = float("inf")

        total_fees = Decimal("0.0")
        total_slippage = Decimal("0.0")
        for t in self.trades:
            total_fees += t.fees_paid
            total_slippage += t.slippage_paid

        return BacktestResult(
            initial_capital=initial,
            final_capital=final_equity,
            total_return_pct=total_ret,
            max_drawdown_pct=self.max_drawdown,
            trade_count=len(self.trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_fees=total_fees,
            total_slippage=total_slippage,
            max_exposure_pct=self.peak_exposure,
            trades=self.trades,
            signals=self.signals_seen,
            not_traded=dict(self.not_traded),
            daily_equity=dict(self.daily_equity),
            shadow_trades=list(self.shadow_trades),
        )
