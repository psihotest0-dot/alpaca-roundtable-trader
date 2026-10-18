# RSS Pipeline Plan

## Existing Components
- Configuration: `pydantic-settings` via `src/trading_system/config.py`
- Network Client: `httpx` (async support available)
- Bot Engine / LLM: AnythingLLM Oracle Adapter (`src/trading_system/oracle/anythingllm.py`)
- Lifecycle / Scheduler: Orchestrator loop (`src/trading_system/runtime/orchestrator.py`)
- Logging: Standard Python `logging`
- Testing: `pytest` and `pytest-mock` / `respx`

## Missing Components
- RSS fetcher & normalizer (HTML cleaning)
- Text chunker
- Embedding provider
- Chroma adapter and persistence manager
- News search / retrieval service
- Integration of news context into the Oracle prompt

## Files to Create/Modify
### Create:
- `src/trading_system/data/rss_collector.py` (fetching, cleaning, normalization)
- `src/trading_system/data/chunking.py` (text chunking)
- `src/trading_system/data/embeddings.py` (interface & implementation for embedding provider)
- `src/trading_system/storage/chroma_adapter.py` (vector store integration, deduplication, recovery)
- `src/trading_system/data/retrieval.py` (search service)
- `tests/data/test_rss_pipeline.py` (unit & integration tests)
- `docs/rss_pipeline.md` (documentation)

### Modify:
- `pyproject.toml` (add `feedparser`, `beautifulsoup4`, `chromadb`)
- `src/trading_system/config.py` (add RSS pipeline configuration)
- `src/trading_system/cli.py` (add commands for single-run ingestion and testing)
- `src/trading_system/oracle/anythingllm.py` (inject retrieved news context into the prompt)
- `src/trading_system/runtime/orchestrator.py` (add periodic or background execution hook)
- `.env.example` (add configuration defaults)

## Chosen Embedding Provider
Default to `chromadb` default embedding function (all-MiniLM-L6-v2) for simplicity, or provide an interface that uses a local dummy for tests and a standard model in production. This fulfills the requirement without paid APIs.

## Strategy for Persistence, Deduplication and Recovery
- **Persistence:** Local Chroma DB stored in a configured directory (e.g., `./var/chroma_db`).
- **Deduplication:** Generate deterministic IDs based on stable hashes:
  - `article_id = stable_hash(source_id + entry_id_or_normalized_url)`
  - `content_hash = stable_hash(normalized_article_text)`
  - `chunk_id = stable_hash(article_id + content_hash + chunk_index)`
- **Recovery & Safe Activation:** Write all chunks of a new version. The active status is maintained using a simple key-value state (or just checking if all chunks count is matched). If an ingestion fails mid-way, `upsert` will overwrite the chunks or we use the deterministic IDs to safely retry.

## Point of Integration
The retrieval service will be injected into `AnythingLLMAdapter` or the Orchestrator. When `evaluate_signal` is called, the Orchestrator or Oracle will first query the search service with the trading symbol / timeframe, fetch the top `K` most relevant news chunks, and inject them into the LLM prompt context as external data.
