import logging
from typing import Dict

from trading_system.config import Settings
from trading_system.data.rss_collector import AsyncRSSCollector, RSSParser
from trading_system.data.chunking import TextChunker
from trading_system.data.embeddings import ChromaDefaultEmbeddingProvider
from trading_system.storage.chroma_adapter import ChromaAdapter

import threading

logger = logging.getLogger(__name__)

# Global lock to protect against overlapping executions within the same process
_INGESTION_LOCK = threading.Lock()

class RSSIngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collector = AsyncRSSCollector(settings)
        self.parser = RSSParser(settings)
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
        self.provider = ChromaDefaultEmbeddingProvider()
        self.adapter = ChromaAdapter(settings, self.provider)

    async def run(self) -> Dict[str, int]:
        report = {
            "sources_ok": 0,
            "sources_failed": 0,
            "entries_seen": 0,
            "articles_indexed": 0,
            "articles_skipped": 0,
            "articles_failed": 0,
            "chunks_written": 0,
        }

        # Ensure that only one ingestion process runs at a time within this process
        # Multi-process coordination is left to external operational limits.
        if not _INGESTION_LOCK.acquire(blocking=False):
            logger.warning("Ingestion is already running. Skipping this execution.")
            return report

        try:
            feeds = [f.strip() for f in self.settings.rss_feeds.split(",") if f.strip()]
            logger.info(f"Processing {len(feeds)} feeds...")

            for feed_url in feeds:
                logger.info(f"Fetching {feed_url}...")
                try:
                    content = await self.collector.fetch_feed(feed_url)
                    if not content:
                        logger.warning(f"Failed to fetch {feed_url}")
                        report["sources_failed"] += 1
                        continue

                    report["sources_ok"] += 1
                    entries = self.parser.parse(feed_url, content)
                    logger.info(f"Parsed {len(entries)} entries from {feed_url}")

                    for entry in entries:
                        report["entries_seen"] += 1
                        article_text = f"{entry.title}\n{entry.summary}"

                        ids = self.adapter.generate_ids(entry.source_id, entry.entry_id, article_text, 0)

                        # Check if this exact content is already active
                        if self.adapter.check_existing_version(ids["article_id"], ids["content_hash"]):
                            report["articles_skipped"] += 1
                            continue

                        # Preserve first_seen_at if an older version of this article already exists
                        existing_first_seen = self.adapter.get_first_seen_at(ids["article_id"])
                        if existing_first_seen:
                            first_seen = existing_first_seen
                        else:
                            first_seen = entry.first_seen_at.timestamp()

                        chunks = self.chunker.chunk_text(article_text)
                        if not chunks:
                            report["articles_skipped"] += 1
                            continue

                        try:
                            # Avoid blocking the event loop with synchronous embedding & chroma calls by yielding to a thread
                            import asyncio
                            embeddings = await asyncio.to_thread(self.provider.embed_documents, chunks)

                            chunk_ids = []
                            metadatas = []

                            for i, chunk in enumerate(chunks):
                                c_ids = self.adapter.generate_ids(entry.source_id, entry.entry_id, article_text, i)
                                chunk_ids.append(c_ids["chunk_id"])

                                meta = {
                                    "source_id": entry.source_id,
                                    "article_id": c_ids["article_id"],
                                    "content_hash": c_ids["content_hash"],
                                    "url": entry.url,
                                    "title": entry.title,
                                    "chunk_index": i,
                                    "chunk_count": len(chunks),
                                    "ingested_at": entry.ingested_at.timestamp(),
                                    "first_seen_at": first_seen,
                                    "embedding_model_version": self.provider.get_model_version(),
                                    "is_active": False
                                }

                                if entry.published_at:
                                    meta["published_at"] = entry.published_at.timestamp()

                                metadatas.append(meta)

                            success = await asyncio.to_thread(self.adapter.store_chunks, chunks, embeddings, metadatas, chunk_ids)
                            if success:
                                activated = await asyncio.to_thread(self.adapter.activate_version, ids["article_id"], ids["content_hash"], chunk_ids)
                                if activated:
                                    report["articles_indexed"] += 1
                                    report["chunks_written"] += len(chunks)
                                else:
                                    report["articles_failed"] += 1
                            else:
                                report["articles_failed"] += 1

                        except Exception as e:
                            logger.error(f"Failed to process entry {entry.url}: {e}")
                            report["articles_failed"] += 1

                except Exception as e:
                    logger.error(f"Failed ingesting feed {feed_url}: {e}")
                    report["sources_failed"] += 1

            logger.info(f"Ingestion complete. Report: {report}")
            return report
        finally:
            _INGESTION_LOCK.release()
