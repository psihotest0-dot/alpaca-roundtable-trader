from typing import Any
import pytest
import os
from datetime import datetime, timezone

from trading_system.config import Settings
from trading_system.data.rss_collector import RSSParser, HTMLCleaner
from trading_system.data.chunking import TextChunker
from trading_system.data.embeddings import FakeEmbeddingProvider
from trading_system.storage.chroma_adapter import ChromaAdapter
from trading_system.data.retrieval import NewsSearchService


@pytest.fixture
def settings() -> Settings:
    os.environ["ALPACA_API_KEY"] = "dummy"
    os.environ["ALPACA_SECRET_KEY"] = "dummy"
    return Settings(
        rss_enabled=True,
        rss_feeds="http://example.com/rss",
        chroma_db_path="./var/test_chroma_db"
    )

def test_html_cleaner() -> None:
    html = "<script>alert('xss');</script><p>Hello <b>World</b></p><style>.hidden { display: none; }</style>"
    cleaned = HTMLCleaner.clean(html)
    assert cleaned == "Hello World"

def test_rss_parser(settings: Settings) -> None:
    parser = RSSParser(settings)
    content = b"""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>W3Schools Home Page</title>
  <link>https://www.w3schools.com</link>
  <description>Free web building tutorials</description>
  <item>
    <title>RSS Tutorial</title>
    <link>https://www.w3schools.com/xml/xml_rss.asp</link>
    <description>New RSS tutorial on W3Schools</description>
  </item>
</channel>
</rss>"""
    entries = parser.parse("http://example.com/rss", content)
    assert len(entries) == 1
    assert entries[0].title == "RSS Tutorial"
    assert entries[0].url == "https://www.w3schools.com/xml/xml_rss.asp"
    assert entries[0].summary == "New RSS tutorial on W3Schools"
    assert entries[0].source_id == "W3Schools Home Page"

def test_text_chunker() -> None:
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    text = "This is a sentence. And another one. Finally, the last one."
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    # Check that chunks are not exceeding size limit
    for chunk in chunks:
        assert len(chunk) <= 20

def test_chroma_storage_and_retrieval(settings: Settings, tmp_path: Any) -> None:
    settings.chroma_db_path = str(tmp_path)

    provider = FakeEmbeddingProvider(dimension=3)
    adapter = ChromaAdapter(settings, provider)
    searcher = NewsSearchService(settings, adapter, provider)

    # Test Ingestion
    source_id = "test_source"
    url = "http://example.com/1"
    content = "Test content about Apple Inc."

    ids = adapter.generate_ids(source_id, url, content, 0)

    chunks = [content]
    embeddings = provider.embed_documents(chunks)
    chunk_ids = [ids["chunk_id"]]

    now = datetime.now(timezone.utc).timestamp()
    metadatas = [{
        "source_id": source_id,
        "article_id": ids["article_id"],
        "content_hash": ids["content_hash"],
        "url": url,
        "title": "Test Title",
        "chunk_index": 0,
        "chunk_count": 1,
        "ingested_at": now,
        "first_seen_at": now,
        "is_active": False
    }]

    success = adapter.store_chunks(chunks, embeddings, metadatas, chunk_ids)
    assert success is True

    # At this point, chunk is NOT active, so retrieval should return nothing
    results = searcher.search("Apple")
    assert len(results) == 0

    # Activate it
    adapter.activate_version(ids["article_id"], ids["content_hash"], chunk_ids)

    # Now it should be retrievable
    results = searcher.search("Apple")
    assert len(results) == 1
    assert results[0].article_id == ids["article_id"]
    assert results[0].text == content

    # Test deduplication
    assert adapter.check_existing_version(ids["article_id"], ids["content_hash"]) is True
