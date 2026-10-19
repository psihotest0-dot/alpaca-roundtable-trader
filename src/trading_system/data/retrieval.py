import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from trading_system.config import Settings
from trading_system.storage.chroma_adapter import ChromaAdapter
from trading_system.data.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

class NewsSearchResult(BaseModel):
    title: str
    text: str
    url: str
    source_id: str
    published_at: Optional[float] = None
    article_id: str
    score: float

class NewsSearchService:
    def __init__(self, settings: Settings, chroma_adapter: ChromaAdapter, embedding_provider: EmbeddingProvider):
        self.settings = settings
        self.chroma_adapter = chroma_adapter
        self.embedding_provider = embedding_provider

    def search(self, query: str, top_k: int = 5, symbol_filter: Optional[str] = None, time_filter: Optional[float] = None) -> List[NewsSearchResult]:
        """
        Searches for news documents related to a query.
        """
        if not query:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.embed_query(query)

            # Construct where clause based on active chunks and optional filters
            conditions: List[Dict[str, Any]] = [{"is_active": True}]

            if symbol_filter:
                conditions.append({f"has_symbol_{symbol_filter}": True})

            if time_filter:
                # time_filter could be a unix timestamp we only want documents strictly newer than.
                conditions.append({"published_at": {"$gt": time_filter}})

            where_clause: Dict[str, Any] = conditions[0] if len(conditions) == 1 else {"$and": conditions}

            # Perform search in Chroma
            results = self.chroma_adapter.collection.query(
                query_embeddings=[query_embedding],  # type: ignore
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            search_results: list[NewsSearchResult] = []

            if not results or not results.get("ids") or not results["ids"][0]:
                return search_results

            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]  # type: ignore
            metadatas = results.get("metadatas", [[]])[0]  # type: ignore
            distances = results.get("distances", [[]])[0]  # type: ignore

            # Deduplicate by article_id to avoid returning multiple chunks of the same article
            seen_articles = set()

            for i, chunk_id in enumerate(ids):
                meta = metadatas[i] or {}
                article_id = meta.get("article_id", "")

                if article_id in seen_articles:
                    continue

                seen_articles.add(article_id)

                search_results.append(
                    NewsSearchResult(
                        title=str(meta.get("title", "")),
                        text=documents[i],
                        url=str(meta.get("url", "")),
                        source_id=str(meta.get("source_id", "")),
                        published_at=float(meta.get("published_at")) if meta.get("published_at") is not None else None,  # type: ignore
                        article_id=str(article_id),
                        # Distance is L2 typically, smaller is better. We can invert or return as is.
                        # Let's return distance directly and clearly define it as distance.
                        score=distances[i]
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error during news search: {e}")
            return []
