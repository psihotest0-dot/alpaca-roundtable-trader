# Operations

## Installation & Setup

1. **Dependencies**: Install using `poetry install`.
2. **Configuration**: Copy `.env.example` to `.env` and fill the variables.
3. **Database Migration**: Ensure the SQLite database schema is applied before startup:
   ```bash
   poetry run alembic upgrade head
   ```

## Local Operations Commands

- **Config Validation**: `poetry run python -m trading_system validate-config` (Verifies Pydantic settings).
- **Diagnostics**: `poetry run python -m trading_system doctor [--network]` (Checks Alpaca connection).
- **Dry Run**: `poetry run python -m trading_system run --dry-run` (Executes the loop without sending API orders).
- **Paper Run**: `poetry run python -m trading_system run --paper` (Live paper trading against Alpaca).

## Control & Incidents

### Pause / Resume (Kill Switch)
- **Pause**: `poetry run python -m trading_system pause-entries`
  This acts as a manual kill switch. It stops any *new* entries from being submitted. Open positions are not automatically liquidated.
- **Resume**: `poetry run python -m trading_system resume-entries`
  *WARNING*: Before resuming, you MUST perform a manual reconciliation to ensure local state matches the broker.
- **Status**: `poetry run python -m trading_system status`

### Incident Procedures
- **Reconciliation**: If the system crashes mid-submission or enters an unknown state, it halts entries. The `reconcile_order` function attempts to recover status on restart.
- **Unprotected Positions**: The emergency monitor (`emergency_check_unprotected_positions`) liquidates any open positions lacking a covering Stop/TakeProfit order.

## Database Backup and Restore (SQLite)

- **Backup**: SQLite stores the entire state in a single file (default: `./var/trading.db`). To back up safely while running (using WAL mode):
  ```bash
  sqlite3 ./var/trading.db ".backup './var/trading_backup.db'"
  ```
- **Restore**: Simply stop the application and copy the backup file over the active database file:
  ```bash
  cp ./var/trading_backup.db ./var/trading.db
  poetry run alembic upgrade head
  ```

## Docker & Connectivity

- **Chroma & AnythingLLM Networking**: The local Chroma database MUST run on the host system outside of Docker.
  - On macOS/Linux: Usually `~/opper-data/chroma`.
  - On Windows: Usually `C:\opper-data\chroma`.
  Inside Docker (`compose.yaml`), the container resolves host endpoints using OS-specific gateways (e.g., `host.docker.internal` for Mac/Win or `172.17.0.1` for Linux). Do NOT use `127.0.0.1` inside Docker for AnythingLLM/Chroma URLs.

## Simulation Limitations
- **Backtest Fills**: Conservative assumptions apply (assume stops hit first if both triggered in same bar).
- **Paper Fills**: Slippage is generally zero on limit orders in paper environments compared to live. Real market depth is not fully simulated. Survivorship bias exists as the watchlist is fixed in testing.
