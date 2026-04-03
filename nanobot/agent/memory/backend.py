"""Storage backend abstraction for the memory system.

This module defines the abstract base class for memory storage backends,
supporting both synchronous and asynchronous implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .types import MemoryEntry, MemoryTier, RetrievalResult


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends.

    This class defines the interface that all memory storage backends must implement.
    It supports both synchronous and asynchronous operations, with all primary
    methods being async to accommodate various storage backends (file, database,
    vector store, etc.).

    Implementations must provide:
    - initialize(): Set up storage connections and resources
    - store(): Persist memory entries
    - retrieve(): Search and retrieve memories by query/tier/embedding
    - consolidate(): Migrate entries between memory tiers
    - delete_expired(): Clean up old entries based on age policies
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend connection.

        This method should establish connections to the underlying storage
        system (database, file system, vector store, etc.) and perform
        any necessary setup or migrations.

        Raises:
            ConnectionError: If unable to connect to the storage backend.
            RuntimeError: If initialization fails for any other reason.
        """
        pass

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in the backend.

        Args:
            entry: The MemoryEntry to store, containing content, tier,
                   timestamps, metadata, and optional embedding.

        Raises:
            ValueError: If the entry is invalid or missing required fields.
            RuntimeError: If storage operation fails.
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None
    ) -> list[RetrievalResult]:
        """Retrieve memories matching the query criteria.

        This method supports both keyword-based and semantic (embedding-based)
        retrieval. Implementations should support filtering by memory tier
        and limit the number of results returned.

        Args:
            query: The search query string for keyword matching.
            tier: Optional filter to only return entries from a specific tier.
            limit: Maximum number of results to return (default: 10).
            embedding: Optional vector embedding for semantic similarity search.

        Returns:
            A list of RetrievalResult objects containing matching entries
            with relevance scores and retrieval method information.

        Raises:
            ValueError: If query is empty or limit is invalid.
            RuntimeError: If retrieval operation fails.
        """
        pass

    @abstractmethod
    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry]
    ) -> list[MemoryEntry]:
        """Migrate entries from source tier to target tier.

        This method is used for memory consolidation, moving entries from
        shorter-term tiers (WORKING) to longer-term tiers (EPISODIC, SEMANTIC).
        Implementations may transform entries during consolidation (e.g.,
        summarization, embedding generation).

        Args:
            source_tier: The tier to migrate entries from.
            target_tier: The tier to migrate entries to.
            entries: List of entries to consolidate. Only entries matching
                     the source_tier should be processed.

        Returns:
            A list of consolidated MemoryEntry objects in the target tier.
            The returned entries may have updated IDs, timestamps, or content
            depending on the consolidation strategy.

        Raises:
            ValueError: If source_tier equals target_tier or entries are invalid.
            RuntimeError: If consolidation operation fails.
        """
        pass

    @abstractmethod
    async def delete_expired(self, max_age_days: dict[MemoryTier, int]) -> int:
        """Delete expired memory entries based on age policies.

        This method removes entries that have exceeded their maximum age
        for their respective tiers. Each tier can have a different retention
        policy.

        Args:
            max_age_days: A dictionary mapping MemoryTier to the maximum
                          age in days for entries in that tier. Only tiers
                          present in the dictionary are cleaned.

        Returns:
            The number of entries that were deleted.

        Raises:
            ValueError: If max_age_days contains invalid values.
            RuntimeError: If deletion operation fails.
        """
        pass
