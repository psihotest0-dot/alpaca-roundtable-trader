import argparse
import sys
from typing import Optional, Sequence

from pydantic import ValidationError

from trading_system.config import Settings


def inject_offline_dummy_keys() -> None:
    import os

    os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy_offline_key")
    os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy_offline_secret")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpaca Roundtable Trader - Automated paper trading system."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate-config command
    subparsers.add_parser(
        "validate-config", help="Validates the local configuration without making network requests"
    )

    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Runs system checks (local by default).")
    doctor_parser.add_argument(
        "--network",
        action="store_true",
        help="Perform read-only network checks against Alpaca/Oracle.",
    )

    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Replays the stored candles (see fetch-history) through the strategy."
    )
    backtest_parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="How many days of stored candles to replay. The first days are only warm-up.",
    )
    backtest_parser.add_argument(
        "--replay-oracle",
        type=str,
        help="Path to JSON file containing historical Oracle decisions.",
    )
    backtest_parser.add_argument(
        "--strategy",
        default="trend_cross",
        help="Which strategy to replay (default trend_cross, the one the live loop runs). "
        "See compare-strategies for the choices.",
    )
    backtest_parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Cut the replay into this many consecutive periods and report each (default 4).",
    )
    backtest_parser.add_argument(
        "--cost-sweep",
        action="store_true",
        help="Also replay at higher slippage (0 to 0.2 % a side) to see how fragile the result is.",
    )

    # compare-strategies command
    compare_parser = subparsers.add_parser(
        "compare-strategies",
        help="Replays several strategies on the same stored candles and costs, side by side.",
    )
    compare_parser.add_argument("--days", type=int, default=90, help="Days of candles to replay.")
    compare_parser.add_argument(
        "--strategies", type=str, help="Comma-separated names. Default: all of them."
    )

    # fetch-history command
    history_parser = subparsers.add_parser(
        "fetch-history", help="Downloads candles from Alpaca into the database."
    )
    history_parser.add_argument(
        "--days", type=int, default=180, help="How far back the stored history should reach."
    )
    history_parser.add_argument(
        "--symbols", type=str, help="Comma-separated symbols. Default: the WATCHLIST."
    )

    # run command
    run_parser = subparsers.add_parser("run", help="Runs the trading system.")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no live broker execution).",
    )
    run_parser.add_argument("--paper", action="store_true", help="Run in paper trading mode.")
    run_parser.add_argument(
        "--pause-on-exit",
        action="store_true",
        help="When stopped, pause new entries, so the next start waits for resume-entries "
        "(the PAUSE_ON_SHUTDOWN setting does the same for every run).",
    )

    # pause-entries command
    subparsers.add_parser("pause-entries", help="Halts the trading system, preventing new entries.")

    # resume-entries command
    resume_parser = subparsers.add_parser(
        "resume-entries",
        help="Lifts a halt after looking at the account again (positions protected, orders "
        "accounted for).",
    )
    resume_parser.add_argument(
        "--force",
        action="store_true",
        help="Resume even if the check finds problems (you have seen them and go on).",
    )

    # reconcile command
    subparsers.add_parser(
        "reconcile",
        help="Compares the account with our records (orders, positions, stops). Exit 0 if "
        "they agree, 1 if not.",
    )

    # export-oracle-decisions command
    export_parser = subparsers.add_parser(
        "export-oracle-decisions",
        help="Writes the recorded Oracle decisions as a file the backtest can replay.",
    )
    export_parser.add_argument("--out", required=True, help="Path of the JSON file to write.")

    # status command
    subparsers.add_parser("status", help="Prints the current status of the trading system.")

    # report command
    report_parser = subparsers.add_parser(
        "report", help="What the system did on one New York trading day (default: today)."
    )
    report_parser.add_argument("--date", help="The day, YYYY-MM-DD (New York).")

    # health command
    subparsers.add_parser(
        "health",
        help="Exit 0 if the trading loop is doing its job, 1 if not (for Docker's healthcheck).",
    )

    # backup-db command
    backup_parser = subparsers.add_parser(
        "backup-db",
        help="Writes a checked, compressed copy of the SQLite database and removes old copies.",
    )
    backup_parser.add_argument(
        "--out", default="var/backups", help="Folder for the archives (default: var/backups)."
    )
    backup_parser.add_argument(
        "--keep", type=int, default=28, help="How many of the newest archives to keep."
    )

    return parser


LOG_FILE = "var/alpaca_trading.log"


def log_file_for(command: Optional[str]) -> Optional[str]:
    """Only the long-running loop writes the log file. A short command (a healthcheck every
    minute, `status`, `report`) that also opened it would add a handler for a few seconds to a
    file that is being rotated under it; its output goes to the console only."""
    return LOG_FILE if command == "run" else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    import logging
    from trading_system.observability.logging import setup_json_logger

    parser = create_parser()
    args = parser.parse_args(argv)

    setup_json_logger("", level=logging.INFO, log_file=log_file_for(args.command))

    if args.command == "validate-config":
        from trading_system.config_check import check_config

        config_report = check_config()
        for warning in config_report.warnings:
            print(f"WARNING: {warning}")
        if config_report.valid:
            print("Configuration is valid.")
            return 0
        print("Configuration is NOT valid:")
        for error in config_report.errors:
            print(f"  - {error}")
        return 1

    elif args.command == "doctor":
        try:
            from trading_system.config_check import unknown_key_warnings

            settings = Settings()
            for warning in unknown_key_warnings():
                print(f"WARNING: {warning}")
            print("Local configuration is valid.")

            if args.network:
                print("Running network checks...")
                try:
                    from trading_system.broker.alpaca_broker import AlpacaBroker

                    broker = AlpacaBroker(settings)
                    clock = broker.get_clock()
                    print(
                        f"Broker connection successful. Market is currently {'open' if clock.is_open else 'closed'}."
                    )
                except ImportError:
                    print("Error: AlpacaBroker not yet implemented. Cannot run network checks.")
                    return 1
                except Exception as e:
                    print(f"Network check failed: {e}")
                    return 1
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1

    elif args.command == "backtest":
        print("Starting offline backtest...")
        try:
            # For backtest, we don't strictly need real API keys if they are just validated.
            # We bypass the strict check by providing dummy values just for the engine init.
            import os

            os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy")
            os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy")
            settings = Settings()
            from trading_system.backtest.models import BacktestConfig
            from trading_system.backtest.engine import BacktestEngine
            from trading_system.backtest.loader import NoStoredCandles, load_candles
            from trading_system.strategy.registry import get_strategy

            try:
                strategy = get_strategy(args.strategy)
                df_15m, df_1h = load_candles(settings, args.days)
            except (ValueError, NoStoredCandles) as e:
                print(e)
                return 1
            symbols = list(df_15m)
            first = min(df["timestamp"].iloc[0] for df in df_15m.values())
            last = max(df["timestamp"].iloc[-1] for df in df_15m.values())
            print(f"Replaying {len(symbols)} symbols, {first:%Y-%m-%d} to {last:%Y-%m-%d}.")

            config = BacktestConfig(replay_oracle_path=args.replay_oracle)
            engine = BacktestEngine(settings, config, df_15m, df_1h, strategy=strategy)
            print(f"Strategy: {strategy.name} ({strategy.version})")

            result = engine.run()

            print("\n=== Backtest Report ===")
            print(f"Initial Capital: ${result.initial_capital:,.2f}")
            print(f"Final Capital:   ${result.final_capital:,.2f}")
            print(f"Total Return:    {result.total_return_pct * 100:.2f}%")
            print(f"Max Drawdown:    {result.max_drawdown_pct * 100:.2f}%")
            print(f"Signals:         {result.signals}")
            print(f"Trade Count:     {result.trade_count}")
            print(f"Win Rate:        {result.win_rate * 100:.2f}%")
            print(f"Profit Factor:   {result.profit_factor:.2f}")
            print(f"Max Exposure:    {result.max_exposure_pct * 100:.2f}%")
            print(f"Fees Paid:       ${result.total_fees:,.2f}")
            print(f"Slippage Paid:   ${result.total_slippage:,.2f} (already inside the prices)")

            days = result.daily_returns()
            if days:
                best = max(days, key=lambda d: d[1])
                worst = min(days, key=lambda d: d[1])
                down = sum(1 for _, r in days if r < 0)
                print(f"Days:            {len(days)} ({down} down)")
                print(f"Best Day:        {best[1] * 100:+.2f}% ({best[0]})")
                print(f"Worst Day:       {worst[1] * 100:+.2f}% ({worst[0]})")

            if result.not_traded:
                print("\n=== Signals that did not become a trade ===")
                for reason, count in sorted(result.not_traded.items(), key=lambda kv: -kv[1]):
                    print(f"- {reason}: {count}")

            from trading_system.backtest.analysis import (
                cost_sweep,
                format_analysis,
                format_cost_sweep,
                format_oracle_value,
                oracle_value,
            )

            print("\n" + format_analysis(result, periods=args.windows))
            value = oracle_value(result)
            if value is not None:
                print("\n" + format_oracle_value(value))
            if args.cost_sweep:
                print("\nRunning the replay again at higher costs...")
                print(
                    "\n" + format_cost_sweep(cost_sweep(settings, config, df_15m, df_1h, strategy))
                )

            print("\n=== Assumptions & Limitations ===")
            for a in result.assumptions_and_limitations:
                print(f"- {a}")

            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except Exception as e:
            print(f"Backtest error: {e}")
            return 1

    elif args.command == "fetch-history":
        try:
            from datetime import datetime, timedelta, timezone

            from trading_system.data.alpaca_data import AlpacaData
            from trading_system.data.history import TIMEFRAMES, update_history
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import MarketBarRepository

            settings = Settings()
            symbols = (
                [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols
                else settings.watchlist_symbols
            )
            now = datetime.now(timezone.utc)
            repo = MarketBarRepository(get_session_factory(get_engine(settings.database_url)))
            print(f"Updating {len(symbols)} symbols, {args.days} days back...")
            report = update_history(
                AlpacaData(settings),
                repo,
                symbols,
                list(TIMEFRAMES),
                start=now - timedelta(days=args.days),
                now=now,
            )
            for name in TIMEFRAMES:
                written = sum(n for (_, tf), n in report.rows.items() if tf == name)
                again = [s for s, tf in report.downloaded_again if tf == name]
                line = f"{name}: {written:,} rows written"
                if again:
                    line += f"; prices out of date, downloaded again: {', '.join(again)}"
                print(line)
            if report.no_data:
                print(
                    "No data from Alpaca for: " + ", ".join(sorted({s for s, _ in report.no_data}))
                )
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except Exception as e:
            print(f"fetch-history error: {e}")
            return 1

    elif args.command == "run":
        is_paper = getattr(args, "paper", False)
        mode = "Paper" if is_paper else "Dry-Run"
        print(f"Starting Orchestrator in {mode} mode...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()

            # Use appropriate repo logic here
            orchestrator = Orchestrator(settings)

            # Simple simulation for CLI demo
            print("Orchestrator running. (Press Ctrl+C to stop)")
            # None leaves it to the PAUSE_ON_SHUTDOWN setting; the flag can only add a pause.
            orchestrator.run(is_paper, pause_on_shutdown=True if args.pause_on_exit else None)
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except KeyboardInterrupt:
            print("Shutting down Orchestrator gracefully...")
            return 0
        except Exception as e:
            print(f"Run error: {e}")
            return 1

    elif args.command == "pause-entries":
        print("Pausing entries...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            orchestrator.pause_entries()
            return 0
        except Exception as e:
            print(f"Pause error: {e}")
            return 1

    elif args.command == "reconcile":
        print("Comparing the account with our records...")
        try:
            from trading_system.runtime.account_check import format_account_check
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            check = Orchestrator(settings).reconcile_now()
            print(format_account_check(check))
            return 0 if check.ok else 1
        except Exception as e:
            print(f"Reconcile error: {e}")
            return 1

    elif args.command == "resume-entries":
        print("Resuming entries...")
        try:
            from trading_system.runtime.account_check import format_account_check
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            resumed, looked = orchestrator.resume_entries_checked(force=args.force)
            if looked is None:
                print("Entries are not halted; nothing to resume.")
                return 0
            print(format_account_check(looked))
            if resumed:
                print(
                    "Entries resumed."
                    + ("" if looked.ok else " (Forced: the problems above stand.)")
                )
                return 0
            print(
                "Entries stay halted. Put the problems right and run this again, or use "
                "--force if you have seen them and want to go on."
            )
            return 1
        except Exception as e:
            print(f"Resume error: {e}")
            return 1

    elif args.command == "export-oracle-decisions":
        try:
            import json

            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import OracleDecisionRepository

            settings = Settings()
            repository = OracleDecisionRepository(
                get_session_factory(get_engine(settings.database_url))
            )
            decisions = repository.replay_export()
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(decisions, f, indent=2, sort_keys=True)
            decided = sum(1 for d in decisions.values() if d["final"])
            print(
                f"Wrote {len(decisions)} signals to {args.out} "
                f"({decided} decided, {len(decisions) - decided} undecided: no trade)."
            )
            return 0
        except Exception as e:
            print(f"Export error: {e}")
            return 1

    elif args.command == "status":
        print("Fetching system status...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            orchestrator.status()
            return 0
        except Exception as e:
            print(f"Status error: {e}")
            return 1

    elif args.command == "compare-strategies":
        print("Comparing strategies on the stored candles...")
        try:
            import os

            os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy")
            os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy")
            settings = Settings()
            from trading_system.backtest.comparison import compare_strategies, format_comparison
            from trading_system.backtest.loader import NoStoredCandles, load_candles
            from trading_system.backtest.models import BacktestConfig
            from trading_system.strategy.registry import strategy_names

            names = (
                [n.strip() for n in args.strategies.split(",") if n.strip()]
                if args.strategies
                else strategy_names()
            )
            unknown = [n for n in names if n not in strategy_names()]
            if unknown:
                print(
                    f"unknown strategy {', '.join(unknown)}; "
                    f"the choices are: {', '.join(strategy_names())}"
                )
                return 1
            try:
                df_15m, df_1h = load_candles(settings, args.days)
                config = BacktestConfig()
                rows = compare_strategies(settings, config, df_15m, df_1h, names)
            except (ValueError, NoStoredCandles) as e:
                print(e)
                return 1
            first = min(df["timestamp"].iloc[0] for df in df_15m.values())
            last = max(df["timestamp"].iloc[-1] for df in df_15m.values())
            print(f"{len(df_15m)} symbols, {first:%Y-%m-%d} to {last:%Y-%m-%d}.\n")
            print(format_comparison(rows))
            print(
                "\nSame data, same risk engine and entry plan, same costs "
                f"(fee {config.fee_pct_per_trade * 100:.2f}%, "
                f"slippage {config.slippage_pct * 100:.2f}% a side)."
                "\nReturn, win rate and costs are for the whole replay; PF is the profit factor; "
                "t is the average trade over its standard error (below 2 in absolute value the "
                "average cannot be told from luck)."
                "\nWith more signals than room (the risk engine's position limit) the symbols "
                "first in the watchlist get the places: a signal count is not a trade count."
                "\nA few dozen trades in one market regime cannot tell a good strategy from a "
                "lucky one: read the differences with that in mind."
            )
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except Exception as e:
            print(f"compare-strategies error: {e}")
            return 1

    elif args.command == "report":
        try:
            from datetime import date, datetime, timezone

            from trading_system.observability.report import (
                build_daily_report,
                format_daily_report,
            )
            from trading_system.risk.daily_equity import trading_day
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import StateRepository

            inject_offline_dummy_keys()
            settings = Settings()
            day = (
                date.fromisoformat(args.date)
                if args.date
                else trading_day(datetime.now(timezone.utc))
            )
            factory = get_session_factory(get_engine(settings.database_url))
            print(
                format_daily_report(
                    build_daily_report(factory, day, StateRepository(factory).get_state())
                )
            )
            return 0
        except ValueError as e:
            print(f"Report error: {e} (the date is YYYY-MM-DD)")
            return 1
        except Exception as e:
            print(f"Report error: {e}")
            return 1

    elif args.command == "health":
        try:
            from datetime import datetime, timezone
            from pathlib import Path

            from trading_system.observability.health import evaluate_heartbeat
            from trading_system.runtime.heartbeat import read_heartbeat

            inject_offline_dummy_keys()
            settings = Settings()
            verdict = evaluate_heartbeat(
                read_heartbeat(Path(settings.heartbeat_path)), datetime.now(timezone.utc)
            )
        except Exception as e:
            print(f"UNHEALTHY: could not judge the heartbeat: {e}")
            return 1
        if verdict.healthy:
            print("healthy")
            return 0
        print("UNHEALTHY: " + "; ".join(verdict.reasons))
        return 1

    elif args.command == "backup-db":
        try:
            from pathlib import Path

            from trading_system.storage.backup import BackupError, backup_database

            # Only the database location is needed: no Alpaca keys, so the container that
            # takes the backups is not given any.
            inject_offline_dummy_keys()
            settings = Settings()
            try:
                archive = backup_database(settings.database_url, Path(args.out), keep=args.keep)
            except BackupError as e:
                print(f"Backup FAILED: {e}")
                return 1
            print(
                f"Backup written: {archive.path} ({archive.size_bytes / 1e6:.1f} MB); "
                f"{archive.kept} kept, {len(archive.removed)} old removed."
            )
            return 0
        except Exception as e:
            print(f"Backup FAILED: {e}")
            return 1

    # Normally we would start the app here
    print("Alpaca Roundtable Trader MVP Skeleton")
    if not args.command:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
