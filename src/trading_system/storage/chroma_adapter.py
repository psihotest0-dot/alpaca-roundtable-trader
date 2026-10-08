import hashlib
import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from trading_system.config import Settings

logger = logging.getLogger(__name__)


class ChromaAdapter:
    def __init__(self, settings: Settings, embedding_provider: Any = None):
        self.settings = settings

        self.client = chromadb.PersistentClient(
            path=self.settings.chroma_db_path, settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection_name = self.settings.chroma_collection_name
        self.embedding_provider = embedding_provider

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "News documents for trading context"},
        )

    @staticmethod
    def _stable_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def generate_ids(
        self, source_id: str, url_or_entry_id: str, content: str, chunk_index: int
    ) -> dict[str, str]:
        article_id = self._stable_hash(f"{source_id}_{url_or_entry_id}")
        content_hash = self._stable_hash(content)
        chunk_id = self._stable_hash(f"{article_id}_{content_hash}_{chunk_index}")

        return {"article_id": article_id, "content_hash": content_hash, "chunk_id": chunk_id}

    def store_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict[str, Any]],
        chunk_ids: list[str],
    ) -> bool:
        if not chunks:
            return True

        try:
            self.collection.upsert(
                documents=chunks,
                embeddings=embeddings,  # type: ignore
                metadatas=metadata_list,  # type: ignore
                ids=chunk_ids,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert chunks to Chroma: {e}")
            return False

    def check_existing_version(self, article_id: str, content_hash: str) -> bool:
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"article_id": article_id},
                        {"content_hash": content_hash},
                        {"is_active": True},
                    ]
                },
                limit=1,
                include=[],
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Error checking existing version in Chroma: {e}")
            return False

    def get_first_seen_at(self, article_id: str) -> Optional[float]:
        try:
            results = self.collection.get(
                where={"article_id": article_id}, limit=1, include=["metadatas"]
            )
            metas = results.get("metadatas", [])
            if metas and metas[0]:
                val = metas[0].get("first_seen_at")
                if isinstance(val, float) or isinstance(val, int):
                    return float(val)
        except Exception as e:
            logger.error(f"Failed to fetch first_seen_at for article {article_id}: {e}")
        return None

    def activate_version(
        self, article_id: str, content_hash: str, new_chunk_ids: list[str]
    ) -> bool:
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"article_id": article_id},
                        {"content_hash": {"$ne": content_hash}},  # type: ignore
                        {"is_active": True},
                    ]
                },
                include=["metadatas"],
            )
            old_ids = results.get("ids", [])

            if old_ids:
                logger.info(f"Deleting {len(old_ids)} old chunks for article {article_id}")
                self.collection.delete(ids=old_ids)

            new_results = self.collection.get(ids=new_chunk_ids, include=["metadatas"])
            new_metas = new_results.get("metadatas", [])

            if new_metas:
                updated_metas = []
                for meta in new_metas:
                    if meta:
                        m = dict(meta)
                        m["is_active"] = True
                        updated_metas.append(m)

                self.collection.update(
                    ids=new_chunk_ids,
                    metadatas=updated_metas,  # type: ignore
                )

            return True
        except Exception as e:
            logger.error(f"Error activating version in Chroma: {e}")
            return False
