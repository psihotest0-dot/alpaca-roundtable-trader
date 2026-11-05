# Architecture

## Components

The system is split into the following subdomains:
- **Domain**: Core entities and models.
- **Broker**: Alpaca interactions and API integrations.
- **Data**: Market data ingestion and processing.
- **Strategy**: Trading logic and signal generation.
- **Oracle**: Optional second opinion on a signal (`APPROVE` / `VETO` / `HOLD`), behind the `OracleProvider` interface.
- **Risk**: Position sizing and entry approval (final say).
- **Execution**: The ONLY component authorized to send orders to the broker.
- **Storage**: Persistence layer (SQLite for MVP).
- **Runtime**: Application lifecycle management.
- **Observability**: Logging, metrics, and monitoring.

## Data Flow

```
Strategy signal -> Oracle (APPROVE / VETO / HOLD) -> Risk -> Execution -> Alpaca
```

Each arrow can stop the signal. The Oracle is optional: with `ORACLE_ENABLED=false` the signal goes straight to Risk.

## Oracle

The strategy produces a signal, the Oracle says yes, no or not now, and Risk still decides. The Oracle never proposes orders, sizes or prices.

- **Contract** (`oracle/contract.py`): `OracleProvider.evaluate(OracleRequest) -> OracleDecision`. A provider does not raise: a timeout or an unreadable reply is a `HOLD` with `failed=True`, never a veto.
- **Answers**: `APPROVE` lets the signal go on to Risk. `VETO` ends that signal for good. `HOLD` means "not now": the question is asked again every cycle until the signal expires.
- **Input** (`oracle/inputs.py`): only numbers and fixed labels computed by this project (indicators of the last closed bar, the last 12 bars, the volume ratio, the live price, the account and the risk limits). There is no text from the outside world, so nothing for a prompt injection to ride on. News will be the first such text (Phase 5) and will get its own handling.
- **Record** (`oracle/recording.py`): every answer is stored in `oracle_decisions`. A final `APPROVE` or `VETO` is read back from the record, also after a restart, so the model is not asked twice about the same signal. `export-oracle-decisions` writes the record for the backtest replay.
- **Providers**: `NullOracle` (`ORACLE_PROVIDER=null`) approves everything and is the only provider today. A provider for a real model is planned (Phase 2.2): the model is to be served by a Lemonade server behind the owner's Go chat backend. Not decided yet.

## Storage

- **Trading state** lives in this project's own SQLite file, `trading.db`, migrated with Alembic. Only this project reads or writes it. Since 2026-11-03 the file is in the Docker volume `alpaca_trader_var` (`/app/var`), the only copy; the host has none. A checked, compressed archive is written to `var/backups` on the PC every 6 hours.
- **The Go backend** runs in its own Docker containers (four of them) with its own SQL database. That database belongs to the backend: it is not shared with this project and is not a store for it. How the two talk to each other (Phase 2.2) is decided when the owner shows the backend.
- **News** (Phase 5): Alpaca's news API is refreshed every 5 minutes into the cache `var/news.db` (`data/news.py`, `storage/news.py`; two tables, no chunking). When the Oracle is on, each request carries up to 5 cleaned articles about the signal's stock (`oracle/news_input.py`) and the journal records which. Chroma and the RSS pipeline were removed on 2026-11-05.
- **Market data** (Phase 3.2): the closed candles from Alpaca (15 minutes and 1 hour) are kept in the table `market_bars` of `trading.db`. `fetch-history` fills and updates it (`data/history.py`; the price policy is in `decisions.md`) and the backtest reads it. The live loop does not use it yet: every cycle still downloads 30 days of 15-minute and 90 days of hourly bars again.

## Operations (Phase 6)

Three containers from one image, on one volume (`compose.yaml`; the commands are in `operations.md`):

- `alpaca-trader` runs the trading loop in dry-run mode. Every tick it writes a heartbeat file; every pass over a closed 15-minute bar is journalled in `audit_events` (with the account's equity), every risk decision in `risk_decisions`, every alert in `audit_events` and as a `CRITICAL` log line starting with `ALERT:`.
- `alpaca-trader-data` keeps the candle store up to date once an hour.
- `alpaca-trader-backup` archives the database outside Docker.

Each has a Docker healthcheck. `trading_system health` judges the loop from its heartbeat (stopped ticking, unreadable market clock, no completed look-up for 20 minutes with the market open, broken signals in the last look-up). `trading_system report` is the day in numbers, built from the same records. `reconcile` and `resume-entries` compare the account with our records before a halt is lifted. The containers run as an ordinary user, and their logs are rotated.
