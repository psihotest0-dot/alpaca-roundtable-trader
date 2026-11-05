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

- **Config Validation**: `poetry run python -m trading_system validate-config` judges the real configuration: the Alpaca keys must be there (in `.env` or the environment; no dummy keys are made up), every setting must be in range and agree with the others (the position size not above the exposure cap, the risk of a trade not above the total risk, `STOP_LOSS_PCT` and `TAKE_PROFIT_PCT` above zero, the bar sizes the only ones supported), and a name in `.env` that no setting reads (a stale or misspelt key) is warned about, with a suggestion. Only names are looked at: no value is ever printed. Exit 0 if valid, 1 if not. `doctor` warns about the same stale keys.
- **Diagnostics**: `poetry run python -m trading_system doctor [--network]` (Checks Alpaca connection).
- **Candles**: `poetry run python -m trading_system fetch-history [--days 180] [--symbols AAPL,MSFT]` downloads or updates the stored candles (needs the Alpaca keys; run `alembic upgrade head` first). It is safe to run while the system is running, and as often as you like: it only fetches what is missing and checks that the stored prices are still current.
- **Daily bars for the strategy research** (Phase 3b): `docker compose exec trader python -m trading_system fetch-daily [--since 2016-01-01] [--symbols AAPL,SPY]` downloads or updates the daily bars of the research universe (`data/universe.py`: 145 large stocks plus SPY) into its own file `var/research.db` (setting `RESEARCH_DATABASE_URL`), never into `trading.db`. The first run takes about 30 seconds and writes about 394,000 rows; later runs fetch the last week and check the stored prices, like `fetch-history`. The bars come from the full-market feed (SIP), not from `DATA_FEED`: the free IEX feed starts in 2020 and carries 2 to 3 % of the real volume. SIP serves history for free as long as the request ends 15 minutes ago or earlier (the code holds the end back by 16 minutes). A daily bar counts as closed from the next midnight New York time. The command ends with a WARNING for a symbol whose newest bar is more than 5 days older than the others (renamed, delisted, halted: the old name just stops, as MMC did when it became MRSH). `research.db` is a cache with one table and no migrations; it is not backed up (the data can be fetched again); if the shape of `market_bars` ever changes, delete the file and fetch again. The command refuses to run if `RESEARCH_DATABASE_URL` is the same file as `DATABASE_URL`.
- **News** (Phase 5, recording only): `docker compose exec trader python -m trading_system fetch-news [--days 30] [--symbols AMZN,AAPL]` brings the news cache `var/news.db` (setting `NEWS_DATABASE_URL`) up to date from Alpaca's news API (read-only; Benzinga and fiscal.ai) for the research universe and the WATCHLIST (155 symbols). The container `alpaca-trader-news` runs it every 5 minutes (healthy while `var/news_ok` is under 30 minutes old). Each update looks back 2 days from the newest stored article to take in articles the provider changed (about 1 in 40) and carries on from there after a stop (no further back than 30 days; `--days N` fills a longer gap, Alpaca keeps its news from 2015). About 500 articles a working day, text kept as plain text cut to 2000 characters. It is a cache in a file of its own (two tables, no migrations, not in the archives of `trading.db`); the only thing in it that Alpaca cannot give back is the time we first saw an article (`first_seen_at`). When the Oracle is switched on (`ORACLE_ENABLED=true`) the trading loop reads it: each Oracle request carries up to 5 articles about the signal's stock that the cache had stored by then (`OracleRequest.news`, `oracle/news_input.py`): up to 3 from the last 24 hours and up to 2 from the rest of the last week (older than 24 hours, newer than 7 days), newest first, each with its age in hours; round-ups that name more than 5 stocks left out, the same story once whichever source or day, each cut to a one-line headline of at most 160 characters and a summary of at most 300, with addresses, markup, template tokens, invisible and control characters removed. The ids of the articles it was shown are journalled (`oracle_news` in `audit_events`) so that its verdicts can be judged against them. A cache that cannot be read means an Oracle asked without news, never a lost signal. What puts the articles into the prompt, as quoted data under a line that says so, is the provider (2.2). If a window reaches the 10,000-article limit the command prints a WARNING.
- **Measuring signals** (Phase 3b, step C2): `docker compose exec trader python -m trading_system measure-signals --signal reversal_1w [--horizons 1,2,3,5] [--regime-filter spy200] [--cost 0.001] [--symbols ...]` measures what prices do after a signal on the daily bars in `research.db` (run `fetch-daily` first): one difference to the average of all stocks per day, a Newey-West t, by year and by market regime, checked against the criteria fixed in advance. It never loads the hold-out (from 2026-05-05). Every real signal measured is written to the journal as a `research_trial`; the control `random_tenth` is not. How it measures and what it does not account for: docs/strategy-research-plan.md, section 9. Read-only on Alpaca: it does not call it at all.
- **Backtest**: `poetry run python -m trading_system backtest [--days 60]` replays the stored candles. The first days of the window are warm-up for the indicators. It takes a few seconds for 90 days (3.6 s of computing for 22 symbols; the strategy is asked only about symbols that could signal, and sees the same window of history as the live loop). The report ends with the signals that did not become a trade and why.
- **Dry Run**: `poetry run python -m trading_system run --dry-run` (Executes the loop without sending API orders).
- **Paper Run**: `poetry run python -m trading_system run --paper` (Live paper trading against Alpaca).

## Control & Incidents

### Pause / Resume (Kill Switch)
- **Pause**: `poetry run python -m trading_system pause-entries`
  This acts as a manual kill switch. It stops any *new* entries from being submitted. Open positions are not automatically liquidated.
- **Resume**: `poetry run python -m trading_system resume-entries [--force]`
  Looks at the account again before it lifts the halt: it refreshes the orders from the broker and checks that every position has an active stop and that no order we sent is unknown to the broker. If it finds a problem it lists it and the halt stays (exit 1). `--force` lifts the halt over the problems, for when you have seen them and decided to go on. If entries are not halted it does nothing.
- **Reconcile**: `poetry run python -m trading_system reconcile` is that same look, on demand and on its own: it refreshes the orders, reports the positions and open orders and the problems, and exits 0 if the account agrees with our records, 1 if not. It trades nothing, halts nothing and closes nothing.
- **Status**: `poetry run python -m trading_system status`
  Shows whether entries are halted and the daily loss baseline (start-of-day equity and the trading day it belongs to).

### Is it healthy? (Phase 6)
- `docker ps` (or Docker Desktop) shows `healthy` / `unhealthy` for each container. `docker compose exec trader python -m trading_system health` says why: it reads the heartbeat file the loop writes every tick (`HEARTBEAT_PATH`, default `var/heartbeat.json`) and exits 1 with the reasons if the loop stopped ticking, cannot read the market clock, has not looked at a bar for 20 minutes with the market open, or its last look-up had signals that broke before they became orders.
- `alpaca-trader-data` is healthy while its last successful download is under 2.5 hours old (`var/data_ok`); `alpaca-trader-backup` while an archive under 7 hours old exists in `var/backups`.
- An alert is a `CRITICAL` log line that starts with `ALERT:` (a signal broke in risk or execution; the whole look-up crashed): `docker compose logs trader | grep ALERT`. Docker does not restart an unhealthy container by itself; it only shows it.
- `docker compose exec trader python -m trading_system report [--date YYYY-MM-DD]` is the day in numbers (a New York trading day; today by default): passes over a bar, signals, how many became an order, how many broke, risk decisions with the reasons for refusals, orders (simulated or not), fills, the equity at the first and the last pass, the alerts. A quiet failure shows as `BROKE` and `ALERTS`.
- What each pass over a bar did is in the table `audit_events` (type `entries_cycle`), and the risk decision of every signal, with its reason, is in `risk_decisions`.

- The containers run as an ordinary user (`trader`, uid 10001), not as root. A **new** volume takes its ownership from the image. A volume made while the image still ran as root (the `alpaca_trader_var` of 2026-11-03 was one) needs this once, with the containers stopped and a fresh archive taken first: `docker run --rm -v alpaca_trader_var:/v alpine chown -R 10001:10001 /v`. A restored database file (see the restore steps) needs the same owner: add `&& chown 10001:10001 /app/var/trading.db` to the restore command.
- Logs: `var/alpaca_trading.log` is rotated (5 files of up to 5 MB) and only the `run` process writes it; everything also goes to the container's output (`docker compose logs`), which Docker keeps to 3 files of 10 MB per container.

### Incident Procedures
- **Reconciliation**: If the system crashes mid-submission or enters an unknown state, it halts entries. The `reconcile_order` function attempts to recover status on restart.
- **Unprotected Positions**: The check `emergency_check_unprotected_positions` runs on every tick of the runner, also while entries are halted or the market is closed. A position without a covering stop halts new entries and raises an alarm; it is not sold. Closing it at market happens only with `EMERGENCY_CLOSE_UNPROTECTED=true` (default `false`). Undo the halt with `resume-entries` after checking the account.

## Database Backup and Restore (SQLite)

- **Automatic archive (Docker setup)**: the container `alpaca-trader-backup` runs `python -m trading_system backup-db --out /backups --keep 28` at start and then every 6 hours (`BACKUP_INTERVAL_SECONDS`, `BACKUP_KEEP` in `.env`). The archives land on the PC, in `./var/backups/trading-YYYYMMDD-HHMMSS.db.gz` (UTC), outside Docker, so they survive a reset of Docker Desktop or `docker compose down -v`. Each one is a consistent snapshot made with SQLite's backup API (a plain file copy would miss what is still in the `-wal` file), it is checked (`integrity_check`, a migration revision present) before it is kept, it is written under a temporary name and renamed when complete, and old archives are removed only after a new one is safely there. A failed backup prints `Backup FAILED` (`docker compose logs backup`) and is retried at the next interval; nothing already archived is touched. One archive is about 4 MB (the candles are most of the database and compress well). Only files named `trading-YYYYMMDD-HHMMSS.db.gz` are ever deleted; anything else in the folder is left alone. A backup can also be taken by hand: `docker compose exec backup python -m trading_system backup-db --out /backups`.
- **Restore (Docker setup)**: (1) `docker compose stop trader data backup`. (2) Choose an archive in `var/backups`. (3) Put it into the volume (Git Bash: prefix with `MSYS_NO_PATHCONV=1`):
  ```bash
  docker run --rm -v alpaca_trader_var:/app/var -v "$PWD/var/backups:/backups:ro" alpine sh -c \
    "rm -f /app/var/trading.db /app/var/trading.db-wal /app/var/trading.db-shm && gunzip -c /backups/trading-YYYYMMDD-HHMMSS.db.gz > /app/var/trading.db && chown 10001:10001 /app/var/trading.db"
  ```
  (4) `docker compose up -d`, then `docker compose exec trader python -m trading_system status`, and reconcile the broker state before resuming entries. To try an archive without touching the live data, do step 3 into a scratch volume (`docker volume create alpaca_restore_test`, mount that instead of `alpaca_trader_var`), run `status` and `alembic current` on it with the image, and remove the volume. That was done for the first archive on 2026-11-03: integrity ok, revision `b6d1f4a82c35`, the daily baseline present.
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
  5. Run `reconcile` and, if the entries are halted, `resume-entries`: both look at the broker again.

## Docker & Connectivity

- **News cache**: `var/news.db` (`NEWS_DATABASE_URL`), a SQLite file of its own in the same volume, filled by the container `alpaca-trader-news` (see `fetch-news` above). Chroma is gone (2026-11-05); there is no vector store to run.
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
