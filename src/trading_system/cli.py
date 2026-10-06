import argparse
import sys
from typing import Optional, Sequence

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpaca Roundtable Trader - Automated paper trading system."
    )
    # Add minimal commands or arguments here
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    _args = parser.parse_args(argv)

    # Normally we would start the app here
    print("Alpaca Roundtable Trader MVP Skeleton")
    return 0

if __name__ == "__main__":
    sys.exit(main())
