# Alpaca Roundtable Trader

Automated trading system for US stocks using Alpaca for market data and paper trading. This project utilizes a 15-minute technical strategy with a 1-hour confirmation and an optional AnythingLLM Oracle.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Install Poetry if you haven't already.
2. Run `poetry install` to install dependencies from the lock file.

## Usage

You can test the CLI locally:

```bash
poetry run python -m trading_system --help
```

You can validate your local configuration without making network requests or sending orders:

```bash
poetry run python -m trading_system validate-config
```

Or using Docker:

```bash
docker compose run --rm app python -m trading_system --help
```

## Checks and Tests

Run the following commands to ensure code quality and tests pass:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```
