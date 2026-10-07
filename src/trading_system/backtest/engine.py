import pandas as pd
from typing import Dict, List
from decimal import Decimal
import logging
from datetime import datetime
import json

from trading_system.config import Settings
from trading_system.domain.enums import OrderSide
from trading_system.domain.models import StrategySignal
from trading_system.strategy.config import StrategyConfig
from trading_system.strategy.trend import generate_signals, filter_signals
from trading_system.backtest.models import (
    BacktestConfig,
    BacktestTrade,
    BacktestPosition,
    BacktestResult,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        settings: Settings,
        bt_config: BacktestConfig,
        df_15m: Dict[str, pd.DataFrame],
        df_1h: Dict[str, pd.DataFrame],
    ):
        self.settings = settings
        self.bt_config = bt_config

        self.df_15m = df_15m
        self.df_1h = df_1h

        self.strategy_config = StrategyConfig(
            strategy_version="1.0",
            fast_ema_period=20,
            slow_ema_period=50,
            rsi_period=14,
            rsi_lower_bound=50.0,
            rsi_upper_bound=70.0,
            atr_period=14,
            htf_fast_ema_period=20,
            htf_slow_ema_period=50,
            min_warmup_bars=settings.htf_lookback_days * 6,
        )

        self.cash = bt_config.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.pending_orders: Dict[str, StrategySignal] = {}

        self.equity_curve: List[float] = []
        self.max_equity = float(self.cash)
        self.max_drawdown = 0.0
        self.peak_exposure = 0.0

        self.oracle_decisions: Dict[str, bool] = {}
        if bt_config.replay_oracle_path:
            self._load_replay_oracle(bt_config.replay_oracle_path)

    def _load_replay_oracle(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                # Assume dict mapping signal_id -> approved bool
                for k, v in data.items():
                    if isinstance(v, dict) and "approved" in v:
                        self.oracle_decisions[k] = bool(v["approved"])
                    else:
                        self.oracle_decisions[k] = bool(v)
            logger.info(f"Loaded {len(self.oracle_decisions)} oracle decisions from replay.")
        except Exception as e:
            logger.error(f"Failed to load replay oracle: {e}")

    def run(self) -> BacktestResult:
        # Align all unique timestamps from 15m data
        all_timestamps = set()
        for df in self.df_15m.values():
            if not df.empty:
                all_timestamps.update(df["timestamp"].tolist())

        sorted_timestamps = sorted(list(all_timestamps))

        for current_time in sorted_timestamps:
            # Reconstruct point-in-time view (No look-ahead)
            # A 15m bar starting at T closes at T + 15m.
            # We process at the START of the current_time bar. This means the PREVIOUS bar is now closed.
            # To evaluate signals generated on the PREVIOUS closed bar, we pass the current_time as the eval_time.
            eval_time = current_time

            # 1. Update existing positions (check stops/targets on the incoming bar)
            self._process_bar_executions(current_time)

            # 2. Check pending limit orders against the incoming bar
            self._process_pending_orders(current_time)

            # Record Equity Curve
            total_equity = self._calculate_equity(current_time)
            self.equity_curve.append(float(total_equity))
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

            # 3. Generate new signals based on data closed *before* eval_time
            # We provide the full dataframes but `generate_signals` filters internally using eval_time
            signals = generate_signals(self.df_15m, self.df_1h, self.strategy_config, eval_time)

            # Active symbols: positions + pending orders
            active_symbols = set(self.positions.keys()).union(set(self.pending_orders.keys()))
            filtered_signals = filter_signals(signals, active_symbols)

            for sig in filtered_signals:
                if self._apply_oracle(sig):
                    self._apply_risk_and_queue(sig, total_equity, current_time)

            # Expire bounded entry orders
            self._expire_orders(current_time)

        return self._generate_report()

    def _apply_oracle(self, sig: StrategySignal) -> bool:
        if not self.settings.oracle_enabled:
            return True
        if sig.signal_id in self.oracle_decisions:
            return self.oracle_decisions[sig.signal_id]
        return False  # Abstain / default veto if not found in replay

    def _apply_risk_and_queue(
        self, sig: StrategySignal, equity: Decimal, eval_time: datetime
    ) -> None:
        # Determine current price for risk logic using the closed bar
        df = self.df_15m.get(sig.symbol)
        if df is None or df.empty:
            return

        closed_bars = df[df["timestamp"] + pd.Timedelta(minutes=15) <= eval_time]
        if closed_bars.empty:
            return

        last_close = Decimal(str(closed_bars.iloc[-1]["close"]))

        # Calculate risk based on config percentages
        risk_per_trade = equity * Decimal(str(self.settings.max_planned_risk_per_trade_pct))
        stop_dist = last_close * Decimal(str(self.settings.stop_loss_pct))
        if stop_dist == 0:
            return

        shares = risk_per_trade / stop_dist
        if not self.bt_config.allow_fractional:
            shares = Decimal(int(shares))

        position_size = shares * last_close
        max_size = equity * Decimal(str(self.settings.position_size_pct))

        if position_size > max_size:
            shares = max_size / last_close
            if not self.bt_config.allow_fractional:
                shares = Decimal(int(shares))

        if shares <= 0:
            return

        required_cash = shares * last_close
        if self.cash >= required_cash:
            # We strictly enforce 1 active entry per symbol. filter_signals handles this, but double check.
            if sig.symbol not in self.positions and sig.symbol not in self.pending_orders:
                self.pending_orders[sig.symbol] = sig
                # In Backtest we just track it. The actual limit entry price is usually the close of the signal bar.
                sig._suggested_limit = last_close
                sig._suggested_shares = shares

    def _process_pending_orders(self, current_time: datetime) -> None:
        # Pending orders are assumed to be limit orders placed at the close of the signal bar.
        # If the low of the *current* bar is <= limit price, we get filled.
        filled = []
        for sym, sig in self.pending_orders.items():
            df = self.df_15m.get(sym)
            if df is None:
                continue

            bar = df[df["timestamp"] == current_time]
            if bar.empty:
                continue
            bar = bar.iloc[0]

            low = Decimal(str(bar["low"]))
            open_px = Decimal(str(bar["open"]))
            limit_px = getattr(sig, "_suggested_limit", Decimal("0"))
            shares = getattr(sig, "_suggested_shares", Decimal("0"))

            if limit_px > 0 and low <= limit_px:
                # Gap down below limit? We get filled at open if open < limit, else limit.
                fill_price = open_px if open_px < limit_px else limit_px

                # Apply slippage (we buy slightly higher)
                fill_price = fill_price * (Decimal("1") + Decimal(str(self.bt_config.slippage_pct)))

                # Calculate fees
                notional = shares * fill_price
                fee = self.bt_config.fee_per_trade + (
                    notional * Decimal(str(self.bt_config.fee_pct_per_trade))
                )

                total_cost = notional + fee
                if self.cash >= total_cost:
                    self.cash -= total_cost

                    stop_loss = fill_price * (
                        Decimal("1") - Decimal(str(self.settings.stop_loss_pct))
                    )
                    take_profit = fill_price * (
                        Decimal("1") + Decimal(str(self.settings.take_profit_pct))
                    )

                    self.positions[sym] = BacktestPosition(
                        symbol=sym,
                        quantity=shares,
                        average_entry_price=fill_price,
                        entry_time=current_time,  # Actually it's the timestamp of this bar
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )

                    filled.append(sym)
                else:
                    # Insufficient cash for fill, cancel order
                    filled.append(sym)

        for sym in filled:
            self.pending_orders.pop(sym, None)

    def _expire_orders(self, current_time: datetime) -> None:
        expired = []
        for sym, sig in self.pending_orders.items():
            if sig.expires_at and current_time >= sig.expires_at:
                expired.append(sym)
        for sym in expired:
            self.pending_orders.pop(sym, None)

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

            high = Decimal(str(bar["high"]))
            low = Decimal(str(bar["low"]))
            open_px = Decimal(str(bar["open"]))

            hit_stop = False
            hit_tp = False
            exit_price = Decimal("0")
            reason = ""

            # Gap aware stops
            if pos.stop_loss and low <= pos.stop_loss:
                hit_stop = True
                exit_price = pos.stop_loss
                if open_px < pos.stop_loss:
                    exit_price = open_px  # Gap down, filled worse
                reason = "STOP_LOSS"

            if pos.take_profit and high >= pos.take_profit:
                hit_tp = True
                tp_price = pos.take_profit
                if open_px > pos.take_profit:
                    tp_price = open_px  # Gap up, filled better

                # Conservative intra-bar collision:
                if hit_stop:
                    # Both hit. Assume worst case: stop hit first.
                    pass
                else:
                    exit_price = tp_price
                    reason = "TAKE_PROFIT"

            if hit_stop or hit_tp:
                # Apply slippage (we sell slightly lower)
                exit_price = exit_price * (Decimal("1") - Decimal(str(self.bt_config.slippage_pct)))

                notional = pos.quantity * exit_price
                fee = self.bt_config.fee_per_trade + (
                    notional * Decimal(str(self.bt_config.fee_pct_per_trade))
                )

                self.cash += notional - fee

                entry_notional = pos.quantity * pos.average_entry_price
                pnl = (notional - fee) - entry_notional
                pnl_pct = float(pnl / entry_notional) if entry_notional > 0 else 0.0

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
                    fees_paid=fee,  # Simplified, missing entry fee here but reflected in cash
                    slippage_paid=Decimal("0"),  # Already baked into price
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
        )
