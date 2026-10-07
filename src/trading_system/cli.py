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
        "validate-config",
        help="Validates the local configuration without making network requests"
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Runs system checks (local by default)."
    )
    doctor_parser.add_argument(
        "--network",
        action="store_true",
        help="Perform read-only network checks against Alpaca/Oracle."
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        try:
            Settings() # type: ignore
            print("Configuration is valid.")
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1

    elif args.command == "doctor":
        try:
            settings = Settings() # type: ignore
            print("Local configuration is valid.")

            if args.network:
                print("Running network checks...")
                try:
                    from trading_system.broker.alpaca_broker import AlpacaBroker
                    broker = AlpacaBroker(settings)
                    clock = broker.get_clock()
                    print(f"Broker connection successful. Market is currently {'open' if clock.is_open else 'closed'}.")
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

    # Normally we would start the app here
    print("Alpaca Roundtable Trader MVP Skeleton")
    return 0


if __name__ == "__main__":
    sys.exit(main())
