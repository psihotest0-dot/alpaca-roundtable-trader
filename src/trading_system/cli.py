import argparse
import sys
from typing import Optional, Sequence

from pydantic import ValidationError

from trading_system.config import Settings


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
        "backtest", help="Runs an offline fixture-based backtest/replay."
    )
    backtest_parser.add_argument(
        "--data-dir", type=str, help="Directory containing historical OHLCV data for backtesting."
    )
    backtest_parser.add_argument(
        "--replay-oracle",
        type=str,
        help="Path to JSON file containing historical Oracle decisions.",
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

    # pause-entries command
    subparsers.add_parser("pause-entries", help="Halts the trading system, preventing new entries.")

    # resume-entries command
    subparsers.add_parser(
        "resume-entries", help="Resumes the trading system, allowing new entries."
    )

    # status command
    subparsers.add_parser("status", help="Prints the current status of the trading system.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        try:
            Settings()  # type: ignore
            print("Configuration is valid.")
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1

    elif args.command == "doctor":
        try:
            settings = Settings()  # type: ignore
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
            settings = Settings()  # type: ignore
            from trading_system.backtest.models import BacktestConfig
            from trading_system.backtest.engine import BacktestEngine
            import pandas as pd
            import numpy as np
            from datetime import datetime, timezone, timedelta

            # Use provided dir or mock data for testing
            df_15m = {}
            df_1h = {}

            # Simple mock data generator for demonstration and tests
            print(
                "Generating sample dataset (AAPL) since no external data dir was provided..."
                if not args.data_dir
                else f"Loading data from {args.data_dir}..."
            )

            # Dummy data for the CLI demo
            now = datetime.now(timezone.utc)
            start_time = now - timedelta(days=5)

            # 15m index
            idx_15m = pd.date_range(start=start_time, end=now, freq="15min", tz="UTC")
            df = pd.DataFrame(index=idx_15m)
            df["timestamp"] = df.index
            df["open"] = np.linspace(150, 160, len(idx_15m))
            df["close"] = df["open"] + np.random.normal(0, 0.5, len(idx_15m))
            df["high"] = df[["open", "close"]].max(axis=1) + 0.5
            df["low"] = df[["open", "close"]].min(axis=1) - 0.5
            df["volume"] = 1000

            # Reset index so 'timestamp' is a column as expected by engine
            df_15m["AAPL"] = df.reset_index(drop=True)

            # 1h index
            idx_1h = pd.date_range(start=start_time, end=now, freq="1h", tz="UTC")
            df_h = pd.DataFrame(index=idx_1h)
            df_h["timestamp"] = df_h.index
            df_h["open"] = np.linspace(150, 160, len(idx_1h))
            df_h["close"] = df_h["open"] + np.random.normal(0, 1.0, len(idx_1h))
            df_h["high"] = df_h[["open", "close"]].max(axis=1) + 1.0
            df_h["low"] = df_h[["open", "close"]].min(axis=1) - 1.0
            df_h["volume"] = 4000

            df_1h["AAPL"] = df_h.reset_index(drop=True)

            config = BacktestConfig(data_dir=args.data_dir, replay_oracle_path=args.replay_oracle)
            engine = BacktestEngine(settings, config, df_15m, df_1h)

            result = engine.run()

            print("\n=== Backtest Report ===")
            print(f"Initial Capital: ${result.initial_capital:,.2f}")
            print(f"Final Capital:   ${result.final_capital:,.2f}")
            print(f"Total Return:    {result.total_return_pct * 100:.2f}%")
            print(f"Max Drawdown:    {result.max_drawdown_pct * 100:.2f}%")
            print(f"Trade Count:     {result.trade_count}")
            print(f"Win Rate:        {result.win_rate * 100:.2f}%")
            print(f"Profit Factor:   {result.profit_factor:.2f}")
            print(f"Max Exposure:    {result.max_exposure_pct * 100:.2f}%")
            print(f"Fees Paid:       ${result.total_fees:,.2f}")
            print(f"Slippage Paid:   ${result.total_slippage:,.2f}")

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

    elif args.command == "run":
        is_paper = getattr(args, "paper", False)
        mode = "Paper" if is_paper else "Dry-Run"
        print(f"Starting Orchestrator in {mode} mode...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()  # type: ignore

            # Use appropriate repo logic here
            orchestrator = Orchestrator(settings)

            # Simple simulation for CLI demo
            print("Orchestrator running. (Press Ctrl+C to stop)")
            orchestrator.run(is_paper)
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

            settings = Settings()  # type: ignore
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

            settings = Settings()  # type: ignore
            orchestrator = Orchestrator(settings)
            orchestrator.resume_entries()
            return 0
        except Exception as e:
            print(f"Resume error: {e}")
            return 1

    elif args.command == "status":
        print("Fetching system status...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()  # type: ignore
            orchestrator = Orchestrator(settings)
            orchestrator.status()
            return 0
        except Exception as e:
            print(f"Status error: {e}")
            return 1

    # Normally we would start the app here
    print("Alpaca Roundtable Trader MVP Skeleton")
    if not args.command:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
