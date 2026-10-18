import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import feedparser
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trading_system.config import Settings

logger = logging.getLogger(__name__)


class AsyncRSSCollector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_response_bytes = settings.rss_max_response_bytes
        self.semaphore = asyncio.Semaphore(settings.rss_max_concurrent_requests)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def fetch_feed(self, url: str) -> Optional[bytes]:
        """Fetch RSS feed content using a bounded HTTP request."""

        # Only allow HTTP and HTTPS
        if not url.startswith("http://") and not url.startswith("https://"):
            logger.error(f"Invalid feed URL scheme: {url}")
            return None

        async with self.semaphore:
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    logger.debug(f"Fetching RSS feed: {url}")
                    # Use stream to enforce response size limits
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()

                        content = b""
                        async for chunk in response.aiter_bytes():
                            content += chunk
                            if len(content) > self.max_response_bytes:
                                logger.error(f"Feed {url} exceeded max response size of {self.max_response_bytes} bytes")
                                return None

                        return content
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    logger.error(f"Error fetching feed {url}: {e}")
                    raise # Re-raise for tenacity to handle retries


class RSSEntry(BaseModel):
    source_id: str
    feed_url: str
    entry_id: str
    url: str
    title: str
    summary: str
    published_at: Optional[datetime] = None
    first_seen_at: datetime
    ingested_at: datetime

    def __hash__(self) -> int:
        return hash((self.source_id, self.entry_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RSSEntry):
            return NotImplemented
        return self.source_id == other.source_id and self.entry_id == other.entry_id


class HTMLCleaner:
    @staticmethod
    def clean(html_content: str) -> str:
        """Cleans HTML content, removing scripts/styles and normalizing whitespace."""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        # Get text, using a space as separator to prevent words from joining
        text = soup.get_text(separator=" ", strip=True)

        # Normalize whitespace
        return " ".join(text.split())


class RSSParser:
    def __init__(self, settings: Settings):
        self.settings = settings

    def parse(self, feed_url: str, content: bytes) -> List[RSSEntry]:
        """Parses RSS feed content and extracts structured entries."""
        if not content:
            return []

        try:
            feed = feedparser.parse(content)
        except Exception as e:
            logger.error(f"Failed to parse feed {feed_url}: {e}")
            return []

        entries = []
        now = datetime.now(timezone.utc)

        # We need a stable source_id. Let's use the feed's base URL or title, or fallback to the provided url
        source_title = getattr(feed.feed, 'title', feed_url)
        source_id = source_title.strip() if source_title else feed_url

        for item in feed.entries[:self.settings.rss_max_entries_per_feed]:
            entry_id = getattr(item, 'id', getattr(item, 'link', ''))
            url = getattr(item, 'link', '')

            # Skip if we don't have enough data
            if not entry_id and not url:
                continue

            if not entry_id:
                entry_id = url

            title = getattr(item, 'title', '')
            summary = getattr(item, 'summary', getattr(item, 'description', ''))

            # Clean up fields a bit
            title = HTMLCleaner.clean(title) if title else ""
            summary = HTMLCleaner.clean(summary) if summary else ""

            published_at = None
            if hasattr(item, 'published_parsed') and item.published_parsed:
                try:
                    published_at = datetime.fromtimestamp(time.mktime(item.published_parsed), tz=timezone.utc)
                except Exception:
                    # Invalid dates should not be represented as the current date
                    pass

            entries.append(
                RSSEntry(
                    source_id=source_id,
                    feed_url=feed_url,
                    entry_id=entry_id,
                    url=url,
                    title=title,
                    summary=summary,
                    published_at=published_at,
                    first_seen_at=now,
                    ingested_at=now
                )
            )

        return entries
