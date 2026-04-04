"""Semantic memory tier implementation for knowledge storage and retrieval."""

import uuid
from datetime import datetime

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.embedder import EmbeddingProvider
from nanobot.agent.memory.types import (
    MemoryEntry,
    MemoryTier,
    RetrievalResult,
    SemanticMemoryConfig,
)


class SemanticMemoryManager:
    """Manager for semantic memory operations.

    Handles storing knowledge with embeddings and performing both
    keyword-based and similarity-based searches over the semantic memory.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        embedder: EmbeddingProvider,
        config: SemanticMemoryConfig,
    ):
        self.backend = backend
        self.embedder = embedder
        self.config = config

    async def store_knowledge(
        self,
        content: str,
        source_session: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """Store knowledge with an embedding vector.

        Args:
            content: The knowledge content to store.
            source_session: Optional session ID that created this entry.
            metadata: Optional additional metadata for the entry.

        Returns:
            The created MemoryEntry with embedding.
        """
        embeddings = await self.embedder.embed([content])
        now = datetime.now()
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            tier=MemoryTier.SEMANTIC,
            created_at=now,
            updated_at=now,
            source_session=source_session,
            metadata=metadata or {},
            embedding=embeddings[0],
        )
        await self.backend.store(entry)
        return entry

    async def search(
        self,
        query: str,
        embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """Search semantic memory.

        If an embedding is provided, passes it to the backend for semantic
        retrieval. Otherwise performs a keyword-based search.

        Args:
            query: The search query string.
            embedding: Optional pre-computed embedding vector.
            limit: Maximum number of results to return.

        Returns:
            A list of RetrievalResult objects matching the query.
        """
        return await self.backend.retrieve(
            query=query,
            tier=MemoryTier.SEMANTIC,
            limit=limit,
            embedding=embedding,
        )

    async def search_by_similarity(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """Generate an embedding for the query and search by similarity.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            A list of RetrievalResult objects sorted by semantic similarity.
        """
        embeddings = await self.embedder.embed([query])
        return await self.backend.retrieve(
            query=query,
            tier=MemoryTier.SEMANTIC,
            limit=limit,
            embedding=embeddings[0],
        )
