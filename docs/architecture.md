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

- **Trading state** lives in this project's own SQLite file, `var/trading.db`, migrated with Alembic. Only this project reads or writes it.
- **The Go backend** runs in its own Docker containers (four of them) with its own SQL database. That database belongs to the backend: it is not shared with this project and is not a store for it. How the two talk to each other (Phase 2.2) is decided when the owner shows the backend.
- **News** (Phase 5): today the RSS pipeline writes to a local Chroma `PersistentClient` directory (`CHROMA_DB_PATH`, default `./var/chroma_db`), and nothing reads the news into the Oracle yet. Decided on 2026-11-02, not built: Chroma goes away and the news are stored in a SQL table.
- **Market data** (Phase 3): decided on 2026-11-02, not built: the candles from Alpaca are kept in a SQL table, which the backtest and the live loop read. Today every cycle downloads 30 days of 15-minute and 90 days of hourly bars again.
