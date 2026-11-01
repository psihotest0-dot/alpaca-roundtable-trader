# Alpaca Roundtable Trader

Automated trading system for US stocks using Alpaca for market data and paper trading. This project utilizes a 15-minute technical strategy with a 1-hour confirmation and an optional AnythingLLM Oracle.

## Installation (Clean Install)

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Ensure you have Python 3.11+ installed.
2. Install Poetry if you haven't already.
3. Clone this repository.
4. Run `poetry install` to install dependencies from the lock file in a fresh virtual environment.
5. Create `.env` file from the example: `cp .env.example .env` and configure your API keys. Do not use quotes around values.

### Database Setup
The system uses SQLite. Before starting, initialize the database via migrations:
```bash
poetry run alembic upgrade head
```

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

### Main CLI Commands

- `validate-config`: Validates the `.env` settings offline.
- `doctor`: Runs diagnostic checks (use `--network` for API validation).
- `run`: Starts the trading loop in `--dry-run` mode (default). Use `--paper` to enable active broker orders.
- `pause-entries`: Manual kill switch to halt new trades.
- `resume-entries`: Removes the trading halt (requires manual reconciliation check).
- `status`: Displays current system status (halt state, equity).
- `backtest`: Runs an offline fixture-based historical simulation.

For detailed command behaviors, refer to `docs/operations.md`.

## Checks and Tests

Run the following commands to ensure code quality and tests pass:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

### Guard rails against leaking secrets

Install the pre-commit hooks once per machine. They block env files, databases,
logs, keys and broker credentials from being committed (CI runs the same checks):

```bash
pip install pre-commit
pre-commit install
```

See `SECURITY.md` and `docs/action-plan.md` for the workflow and what to do if a
key is ever committed.
