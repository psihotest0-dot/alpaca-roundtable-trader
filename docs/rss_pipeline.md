# RSS Pipeline Documentation

> **REMOVED on 2026-11-05.** The RSS pipeline, the chunker, the embeddings, the Chroma adapter and the search service were deleted together with the dependencies `chromadb`, `feedparser`, `beautifulsoup4`, `httpx` and `tenacity` (the code is in git history up to commit `397509e`). News now come from Alpaca's news API into the cache `var/news.db` and reach the Oracle through `oracle/news_input.py`; see `decisions.md` (2026-11-05) and `operations.md` (News). This page is kept as a record of what was built and why it was dropped.

The RSS pipeline enables the automated extraction, normalization, embedding, storage, and retrieval of news content for use within the automated trading system. **Today it only collects and stores**: nothing reads the news into the Oracle yet. Giving the Oracle a short summary of a matching item is planned for Phase 5 and needs its own protection against prompt injection (see "Limitations").

> **Decision of 2026-11-02:** Chroma, the chunker and the embeddings are to be removed. Articles will be stored in a SQL table with a `source` column, so that the Alpaca News API can use the same table, and the symbol of an item will be found with a company-name dictionary. The sections below describe what exists today.

## Architecture

1. **Collector (`AsyncRSSCollector`)**: Fetches feeds concurrently utilizing `httpx` with bounded backoff and size limits.
2. **Parser (`RSSParser` & `HTMLCleaner`)**: Parses RSS XML, extracts relevant fields, and strictly sanitizes HTML using `BeautifulSoup`.
3. **Chunker (`TextChunker`)**: Deterministically splits texts into manageable sizes with overlap, preferring sentence and word boundaries.
4. **Embedding (`EmbeddingProvider`)**: Generates vector representations of the texts. By default, it uses Chroma's built-in `all-MiniLM-L6-v2`.
5. **Storage (`ChromaAdapter`)**: Uses `chromadb.PersistentClient` for local persistence. Generates deterministic stable hashes (SHA-256) for deduplication. Supports a two-step "write-then-activate" versioning mechanism to handle partial ingestion failures cleanly.
6. **Retrieval (`NewsSearchService`)**: Performs vector-based similarity searches on the active chunks within ChromaDB, ensuring deduplicated responses per article.

## Configuration

Settings are controlled via environment variables (or `.env` file) as defined in `src/trading_system/config.py`:

- `RSS_ENABLED`: Set to `true` to enable background collection. The orchestrator starts the collector only when `ORACLE_ENABLED=true` as well. (Default: `false`)
- `RSS_FEEDS`: Comma-separated list of RSS feed URLs.
- `RSS_UPDATE_INTERVAL_MINUTES`: How often the orchestrator fetches new data. (Default: 60)
- `RSS_MAX_CONCURRENT_REQUESTS`: Bounding for concurrent HTTP requests. (Default: 5)
- `RSS_MAX_RESPONSE_BYTES`: Limit for feed payload size. (Default: 5MB)
- `RSS_MAX_ENTRIES_PER_FEED`: Limits the number of processed entries. (Default: 50)
- `CHROMA_DB_PATH`: Local path to persist Chroma DB files. (Default: `./var/chroma_db`)
- `CHUNK_SIZE`: Maximum text chunk length. (Default: 500)
- `CHUNK_OVERLAP`: Character overlap between chunks. (Default: 50)
- `RETRIEVAL_TOP_K`: Number of news articles a search returns. (Default: 5) The Oracle does not use them yet.

## Executing

### Periodic Execution
When `RSS_ENABLED=true`, the `Orchestrator` runs a background thread that executes ingestion periodically (based on `RSS_UPDATE_INTERVAL_MINUTES`).

### Single-run ingestion and search (CLI)
Not built yet. The commands `rss-ingest` and `rss-search` were planned (`rss_pipeline_plan.md`) but never added to `cli.py`. Until they exist, `RSSIngestionPipeline(settings).run()` and `NewsSearchService` are used from Python, and `tests/data/test_rss_pipeline.py` shows how.

## Error Recovery and Persistence

- **Partial Ingestion**: Chunks are inserted into Chroma with an `is_active=False` flag. Only when all chunks for an article are successfully processed and upserted will `activate_version()` be called, marking them active and deleting any older versions of the same article.
- **Failures**: HTTP requests are bounded and will retry automatically. Broken entries or malformed HTML won't crash the loop.
- **Deduplication**: Repeating an ingestion for already existing and active items is aborted early via `check_existing_version()`.

## Limitations

- Automatic ticker extraction (NER) is not natively implemented; a search matches the symbol name by vector similarity only.
- **Prompt injection.** A news item is text written by someone else. When it reaches the Oracle (Phase 5) it must be a short summary of limited length, with no URLs or commands, framed as data and not as instructions, and the Oracle's answer stays limited to `APPROVE` / `VETO` / `HOLD`. The worst a hostile item can then do is make the model approve a signal the strategy already produced, or block one; it cannot choose an order, and Risk still applies its limits. Until then the Oracle's input contains no outside text (`oracle/contract.py`).
- Embeddings use Chroma's default model out-of-the-box (`all-MiniLM-L6-v2`); if the embedding model is changed, a manual collection reset/re-index is required.
