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
    subparsers.add_parser(
        "resume-entries", help="Resumes the trading system, allowing new entries."
    )

    # export-oracle-decisions command
    export_parser = subparsers.add_parser(
        "export-oracle-decisions",
        help="Writes the recorded Oracle decisions as a file the backtest can replay.",
    )
    export_parser.add_argument("--out", required=True, help="Path of the JSON file to write.")

    # status command
    subparsers.add_parser("status", help="Prints the current status of the trading system.")

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    import logging
    from trading_system.observability.logging import setup_json_logger

    setup_json_logger("", level=logging.INFO, log_file="var/alpaca_trading.log")

    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        try:
            inject_offline_dummy_keys()
            Settings()
            print("Configuration is valid.")
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1

    elif args.command == "doctor":
        try:
            if not args.network:
                inject_offline_dummy_keys()
            settings = Settings()
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
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import MarketBarRepository
            from datetime import datetime, timezone, timedelta

            symbols = [s.strip() for s in settings.watchlist.split(",") if s.strip()]
            since = datetime.now(timezone.utc) - timedelta(days=args.days)
            repo = MarketBarRepository(get_session_factory(get_engine(settings.database_url)))
            df_15m = repo.load(symbols, "15m", start=since)
            df_1h = repo.load(symbols, "1h", start=since)
            symbols = [s for s in symbols if s in df_15m and s in df_1h]
            if not symbols:
                print(
                    f"No stored candles for the watchlist in the last {args.days} days. "
                    "Run `fetch-history` first."
                )
                return 1
            df_15m = {s: df_15m[s] for s in symbols}
            df_1h = {s: df_1h[s] for s in symbols}
            first = min(df["timestamp"].iloc[0] for df in df_15m.values())
            last = max(df["timestamp"].iloc[-1] for df in df_15m.values())
            print(f"Replaying {len(symbols)} symbols, {first:%Y-%m-%d} to {last:%Y-%m-%d}.")

            config = BacktestConfig(replay_oracle_path=args.replay_oracle)
            engine = BacktestEngine(settings, config, df_15m, df_1h)

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
            wanted = args.symbols if args.symbols else settings.watchlist
            symbols = [s.strip().upper() for s in wanted.split(",") if s.strip()]
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

    elif args.command == "resume-entries":
        print("Resuming entries...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            orchestrator.resume_entries()
            return 0
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
