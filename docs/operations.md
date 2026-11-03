# Operations

## Installation & Setup

1. **Dependencies**: Install using `poetry install`.
2. **Configuration**: Copy `.env.example` to `.env` and fill the variables.
3. **Database Migration**: Ensure the SQLite database schema is applied before startup:
   ```bash
   poetry run alembic upgrade head
   ```

## Upgrading an existing installation

After pulling new code, run `poetry install` and then `poetry run alembic upgrade head`. The revision `c3f1a9e47d52` (2026-11-01) rebuilds `broker_orders` so that orders are keyed by `client_order_id`; it keeps all rows and has a working `downgrade`. A database created before it (for example the old `var/trading.db`) is empty or migrated in place.

## Local Operations Commands

> **With the Docker setup the commands below are run in the container.** Read `poetry run python -m trading_system X` as `docker compose exec trader python -m trading_system X`. The data is in the Docker volume `alpaca_trader_var`; a command run on the host would create a second, empty `var/trading.db`.

- **Config Validation**: `poetry run python -m trading_system validate-config` (Verifies Pydantic settings).
- **Diagnostics**: `poetry run python -m trading_system doctor [--network]` (Checks Alpaca connection).
- **Candles**: `poetry run python -m trading_system fetch-history [--days 180] [--symbols AAPL,MSFT]` downloads or updates the stored candles (needs the Alpaca keys; run `alembic upgrade head` first). It is safe to run while the system is running, and as often as you like: it only fetches what is missing and checks that the stored prices are still current.
- **Backtest**: `poetry run python -m trading_system backtest [--days 60]` replays the stored candles. The first days of the window are warm-up for the indicators. It takes about 40 s for 30 days and 2.5 minutes for 90; it grows a little faster than the window (Phase 3.3 fixes that).
- **Dry Run**: `poetry run python -m trading_system run --dry-run` (Executes the loop without sending API orders).
- **Paper Run**: `poetry run python -m trading_system run --paper` (Live paper trading against Alpaca).

## Control & Incidents

### Pause / Resume (Kill Switch)
- **Pause**: `poetry run python -m trading_system pause-entries`
  This acts as a manual kill switch. It stops any *new* entries from being submitted. Open positions are not automatically liquidated.
- **Resume**: `poetry run python -m trading_system resume-entries`
  *WARNING*: Before resuming, you MUST perform a manual reconciliation to ensure local state matches the broker.
- **Status**: `poetry run python -m trading_system status`
  Shows whether entries are halted and the daily loss baseline (start-of-day equity and the trading day it belongs to).

### Incident Procedures
- **Reconciliation**: If the system crashes mid-submission or enters an unknown state, it halts entries. The `reconcile_order` function attempts to recover status on restart.
- **Unprotected Positions**: The check `emergency_check_unprotected_positions` runs on every tick of the runner, also while entries are halted or the market is closed. A position without a covering stop halts new entries and raises an alarm; it is not sold. Closing it at market happens only with `EMERGENCY_CLOSE_UNPROTECTED=true` (default `false`). Undo the halt with `resume-entries` after checking the account.

## Database Backup and Restore (SQLite)

- **Backup**: SQLite stores the entire state in a single file (default: `./var/trading.db`; in the Docker setup it is `/app/var/trading.db` in the volume). To back up safely while running (using WAL mode):
  ```bash
  sqlite3 ./var/trading.db ".backup './var/trading_backup.db'"
  ```
  In the Docker setup (the image has no `sqlite3` command, Python does the same):
  ```bash
  docker compose exec trader python -c "import sqlite3; s=sqlite3.connect('var/trading.db'); d=sqlite3.connect('var/backup.db'); s.backup(d); d.close()"
  docker cp alpaca-trader:/app/var/backup.db ./var/backups/trading-$(date +%F).db
  ```
- **Restore**:
  1. Stop all users/processes interacting with the database.
  2. Backup existing state (even if corrupted) before overwrite. Ensure WAL and SHM files are handled/removed.
  3. Copy the backup file over the active database file:
     ```bash
     cp ./var/trading_backup.db ./var/trading.db
     ```
  4. Run integrity and schema checks:
     ```bash
     sqlite3 ./var/trading.db "PRAGMA integrity_check;"
     poetry run alembic upgrade head
     ```
  5. Manually reconcile the broker state before allowing new entries.

## Docker & Connectivity

- **News store (Chroma, to be replaced by a SQL table; see `decisions.md`)**: a local directory (`CHROMA_DB_PATH`, default `./var/chroma_db`) opened in-process with `PersistentClient`. There is no Chroma server to run or to reach over the network. In the container it lives in `/app/var`, which is the Docker volume `alpaca_trader_var` (see below).
- **Containers** (`docker compose up -d`): `alpaca-trader` runs the trading loop (`run`, dry-run: real data, simulated orders; it does not use `--paper`) and `alpaca-trader-data` runs `fetch-history --days 180` once an hour. Both run `alembic upgrade head` first and share the named volume `alpaca_trader_var`, mounted at `/app/var`: that volume is the only copy of the data (`trading.db` with the state, signals, orders and the `market_bars` candles, and the log). The host's `var/` is not mounted on purpose (SQLite in WAL mode does not work reliably over a Windows bind mount, and a host copy would diverge). Both restart by themselves (`unless-stopped`) when Docker Desktop starts. `docker compose logs -f trader` follows the loop; `docker compose stop trader` is a clean stop (SIGTERM). Only `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `IS_PAPER` and `DATABASE_URL` reach the containers; nothing else from `.env`. `docker compose down` keeps the data; `docker compose down -v` deletes it. After a code change: `docker compose build && docker compose up -d`. One-off commands: `docker compose exec trader python -m trading_system status`, or `docker compose run --rm app …` when the loop is not running.
- **Database**: the trading state is the project's own SQLite file. The Go backend's containers and their SQL database are separate and are not used by this project.
- **Host services from a container**: when a real Oracle provider is added (Phase 2.2), a container reaches a service on the host through the OS-specific gateway (`host.docker.internal` on Mac/Win, `172.17.0.1` on Linux), never through `127.0.0.1`.

## Docker image

The image is built from `Dockerfile` on Python 3.11, the version the project is tested on (the
locked numpy has no wheels beyond 3.12, so a newer base image cannot install the dependencies).
It contains runtime dependencies only: no pytest, ruff, mypy or safety, and never `.env` or `var/`.

```bash
docker build -t alpaca-roundtable-trader:latest .
docker run --rm alpaca-roundtable-trader:latest python -m trading_system --help
# Keep the database and logs outside the container:
docker run --rm -e ALPACA_API_KEY=... -e ALPACA_SECRET_KEY=... -v "$PWD/var:/app/var"     alpaca-roundtable-trader:latest python -m alembic upgrade head
```

`docker compose build` produces the same image name. CI builds the image and smoke-tests it on every
push and pull request. Do not change the Python tag on its own; change it together with `pyproject.toml`
and the CI Python version.

## Simulation Limitations
- **Backtest Fills**: Conservative assumptions apply (assume stops hit first if both triggered in same bar).
- **Paper Fills**: Slippage is generally zero on limit orders in paper environments compared to live. Real market depth is not fully simulated. Survivorship bias exists as the watchlist is fixed in testing.
