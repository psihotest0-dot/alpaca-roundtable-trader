# Architecture

## Components

The system is split into the following subdomains:
- **Domain**: Core entities and models.
- **Broker**: Alpaca interactions and API integrations.
- **Data**: Market data ingestion and processing.
- **Strategy**: Trading logic and signal generation.
- **Oracle**: Optional AnythingLLM integration for enhanced insights.
- **Risk**: Position sizing and entry approval (final say).
- **Execution**: The ONLY component authorized to send orders to the broker.
- **Storage**: Persistence layer (SQLite for MVP).
- **Runtime**: Application lifecycle management.
- **Observability**: Logging, metrics, and monitoring.

## Data Flow (Oracle)

```
Trading Application -> AnythingLLM API -> Host Chroma Server
```

- The trading application communicates only with the AnythingLLM API.
- AnythingLLM manages interactions with the Chroma server.
- The Chroma server runs directly on the host machine, independent of Docker.
