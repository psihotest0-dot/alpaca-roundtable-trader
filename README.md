# Alpaca Roundtable Trader

Automated trading system for US stocks using Alpaca for market data and paper trading. This project utilizes a 15-minute technical strategy with a 1-hour confirmation and an optional Oracle: a yes / no / hold second opinion on each signal.

## Status

Under active development and **paper-only**. The trading loop runs against Alpaca's paper account: data, strategy, the optional Oracle, risk, bracket orders with a protection check, reconciliation and a daily loss guard, all covered by tests. A full session on live data that produced a signal has not been observed yet. The Oracle is an interface (`APPROVE` / `VETO` / `HOLD`); no real model is connected, and `ORACLE_PROVIDER=null` lets every signal through. Where things stand and what comes next: [`docs/action-plan.md`](docs/action-plan.md). Why: [`docs/analysis-and-update-plan.md`](docs/analysis-and-update-plan.md). Decisions: [`docs/decisions.md`](docs/decisions.md).

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

**The system's data lives in Docker.** The containers share one volume (`alpaca_trader_var`), which holds the only copy of the database and the log: `alpaca-trader` runs the trading loop in dry-run mode (real data, simulated orders), `alpaca-trader-data` keeps the candle store up to date, and `alpaca-trader-backup` writes a checked archive of the database to `var/backups` on the PC every 6 hours, outside Docker. All are visible in Docker Desktop. They need `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`. Run the CLI commands below in the container (`docker compose exec trader python -m trading_system status`), not on the host, which would create a second, empty database:

```bash
docker compose build                                 # after code changes
docker compose up -d                                 # start both; each brings the database to the latest migration first
docker compose logs -f trader                        # follow the trading loop
docker compose exec trader python -m trading_system backtest --days 30
```

### Main CLI Commands

- `validate-config`: Validates the `.env` settings offline.
- `doctor`: Runs diagnostic checks (use `--network` for API validation).
- `run`: Starts the trading loop in `--dry-run` mode (default). Use `--paper` to enable active broker orders.
- `pause-entries`: Manual kill switch to halt new trades.
- `resume-entries`: Removes the trading halt (requires manual reconciliation check).
- `status`: Displays current system status (halt state, equity).
- `backup-db`: Writes a checked, compressed copy of the database to a folder (`--out`, default `var/backups`) and keeps the newest `--keep` (default 28). The `backup` container runs it every 6 hours.
- `fetch-history`: Downloads candles from Alpaca into the database (`--days`, default 180). The first run takes about a minute; later runs only fetch what is missing.
- `compare-strategies`: Replays several strategies (`--strategies a,b`, default all) on the same stored candles, risk engine, entry plan and costs, and prints them side by side with the t-statistic of the average trade (`--days`, default 90). The choices: `trend_cross` (what the live loop runs), `ema_crossover` and `strict_ema_crossover` (the two strategies of the original Money_bot) and their `_5pct` variants (pullback distance widened to 5 %).
- `backtest`: Replays the stored candles through one strategy (`--strategy NAME`, default `trend_cross`; `--days`, default 60). Run `fetch-history` first. Entries go through the live system's `RiskEngine` and entry plan; the report lists the assumptions that remain and which signals did not become a trade, and why.

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

## License

[MIT](LICENSE). The software is provided as is, without warranty of any kind. It places orders on a brokerage account and is paper-only today; nothing here is financial advice, and anyone who runs it with real money does so at their own risk.
