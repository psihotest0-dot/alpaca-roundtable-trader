from abc import ABC, abstractmethod
from typing import List

import chromadb.utils.embedding_functions as embedding_functions


class EmbeddingProvider(ABC):
    @abstractmethod
    def get_model_version(self) -> str:
        """Returns the identifier of the underlying model being used."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of document chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding for a search query."""
        pass


class ChromaDefaultEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        # Uses the default all-MiniLM-L6-v2 model used by Chroma
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._model_version = "chroma_default_all_MiniLM_L6_v2"

    def get_model_version(self) -> str:
        return self._model_version

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not documents:
            return []
        # The default embedding function returns list of list of floats
        return self._ef(documents)  # type: ignore

    def embed_query(self, query: str) -> List[float]:
        # Returns a list containing the embedding for the single query
        res = self._ef([query])
        if not res:
            return []
        return res[0]  # type: ignore


class FakeEmbeddingProvider(EmbeddingProvider):
    """A fake provider for tests that returns deterministic dummy vectors."""

    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def get_model_version(self) -> str:
        return "fake_model_v1"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [[float(len(doc) + i) for i in range(self.dimension)] for doc in documents]

    def embed_query(self, query: str) -> List[float]:
        return [float(len(query) + i) for i in range(self.dimension)]
