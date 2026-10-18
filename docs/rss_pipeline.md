# RSS Pipeline Documentation

The RSS pipeline enables the automated extraction, normalization, embedding, storage, and retrieval of news content for use within the automated trading system, particularly to provide real-time external context to the AnythingLLM Oracle.

## Architecture

1. **Collector (`AsyncRSSCollector`)**: Fetches feeds concurrently utilizing `httpx` with bounded backoff and size limits.
2. **Parser (`RSSParser` & `HTMLCleaner`)**: Parses RSS XML, extracts relevant fields, and strictly sanitizes HTML using `BeautifulSoup`.
3. **Chunker (`TextChunker`)**: Deterministically splits texts into manageable sizes with overlap, preferring sentence and word boundaries.
4. **Embedding (`EmbeddingProvider`)**: Generates vector representations of the texts. By default, it uses Chroma's built-in `all-MiniLM-L6-v2`.
5. **Storage (`ChromaAdapter`)**: Uses `chromadb.PersistentClient` for local persistence. Generates deterministic stable hashes (SHA-256) for deduplication. Supports a two-step "write-then-activate" versioning mechanism to handle partial ingestion failures cleanly.
6. **Retrieval (`NewsSearchService`)**: Performs vector-based similarity searches on the active chunks within ChromaDB, ensuring deduplicated responses per article.

## Configuration

Settings are controlled via environment variables (or `.env` file) as defined in `src/trading_system/config.py`:

- `RSS_ENABLED`: Set to `true` to enable background collection and prompt injection. (Default: `false`)
- `RSS_FEEDS`: Comma-separated list of RSS feed URLs.
- `RSS_UPDATE_INTERVAL_MINUTES`: How often the orchestrator fetches new data. (Default: 60)
- `RSS_MAX_CONCURRENT_REQUESTS`: Bounding for concurrent HTTP requests. (Default: 5)
- `RSS_MAX_RESPONSE_BYTES`: Limit for feed payload size. (Default: 5MB)
- `RSS_MAX_ENTRIES_PER_FEED`: Limits the number of processed entries. (Default: 50)
- `CHROMA_DB_PATH`: Local path to persist Chroma DB files. (Default: `./var/chroma_db`)
- `CHUNK_SIZE`: Maximum text chunk length. (Default: 500)
- `CHUNK_OVERLAP`: Character overlap between chunks. (Default: 50)
- `RETRIEVAL_TOP_K`: Number of news articles to inject into the Oracle prompt. (Default: 5)

## Executing

### Periodic Execution
When `RSS_ENABLED=true`, the `Orchestrator` runs a background thread that executes ingestion periodically (based on `RSS_UPDATE_INTERVAL_MINUTES`).

### Single-run Ingestion (CLI)
You can manually trigger ingestion using the CLI:

```bash
poetry run python -m trading_system rss-ingest
```

### Searching (CLI)
You can test the retrieval pipeline using the CLI:

```bash
poetry run python -m trading_system rss-search "Apple stock rally" --top-k 3
```

## Error Recovery and Persistence

- **Partial Ingestion**: Chunks are inserted into Chroma with an `is_active=False` flag. Only when all chunks for an article are successfully processed and upserted will `activate_version()` be called, marking them active and deleting any older versions of the same article.
- **Failures**: HTTP requests are bounded and will retry automatically. Broken entries or malformed HTML won't crash the loop.
- **Deduplication**: Repeating an ingestion for already existing and active items is aborted early via `check_existing_version()`.

## Limitations

- Automatic ticker extraction (NER) is not natively implemented; currently, the Oracle query filters primarily via vector semantic search on the symbol name.
- Embeddings use Chroma's default model out-of-the-box (`all-MiniLM-L6-v2`); if the embedding model is changed, a manual collection reset/re-index is required.
