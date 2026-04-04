"""ChromaDB storage backend for the memory system.

This module implements a vector-based storage backend using ChromaDB for
semantic memory retrieval with embedding support.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from ..backend import MemoryBackend
from ..types import MemoryEntry, MemoryTier, RetrievalResult


class ChromaBackend(MemoryBackend):
    """ChromaDB backend for vector-based semantic memory.

    This backend stores memories in ChromaDB with support for:
    - Vector-based semantic search using embeddings
    - Keyword-based search as fallback
    - Metadata filtering by memory tier
    - Persistent storage across sessions

    Args:
        persist_directory: Directory for ChromaDB persistence.
            Defaults to ~/.nanobot/chroma
        collection_name: Name of the ChromaDB collection.
            Defaults to "memories"
    """

    def __init__(
        self,
        persist_directory: str = "~/.nanobot/chroma",
        collection_name: str = "memories"
    ):
        if not HAS_CHROMADB:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        self.persist_directory = Path(persist_directory).expanduser()
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        """Initialize Chroma client and collection.

        Creates the persist directory if it doesn't exist and initializes
        a PersistentClient with cosine similarity as the distance metric.

        Raises:
            ConnectionError: If unable to connect to ChromaDB.
            RuntimeError: If initialization fails for any other reason.
        """
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            persist_directory=str(self.persist_directory),
            anonymized_telemetry=False
        )

        self._client = chromadb.PersistentClient(settings=settings)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in ChromaDB.

        Stores the entry with its content, metadata, and optional embedding.
        If an embedding is provided, it will be used for semantic search.
        Otherwise, ChromaDB will generate embeddings automatically if configured.

        Args:
            entry: The MemoryEntry to store.

        Raises:
            ValueError: If the entry is invalid or missing required fields.
            RuntimeError: If storage operation fails.
        """
        if self._collection is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        metadata = {
            "tier": entry.tier.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "source_session": entry.source_session or "",
            **entry.metadata
        }

        # ChromaDB requires string values in metadata
        metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                   for k, v in metadata.items()}

        self._collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[metadata],
            embeddings=[entry.embedding] if entry.embedding else None
        )

    async def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None
    ) -> list[RetrievalResult]:
        """Retrieve memories matching the query criteria.

        Supports both embedding-based semantic search and keyword-based search.
        When an embedding is provided, uses vector similarity search.
        Otherwise, falls back to keyword search using the query text.

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
        if self._collection is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if limit <= 0:
            raise ValueError("Limit must be positive")

        # Build where filter for tier if specified
        where_filter = None
        if tier:
            where_filter = {"tier": tier.value}

        if embedding:
            # Use embedding-based search
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                where=where_filter,
                include=["metadatas", "documents", "distances"]
            )
            retrieval_method = "embedding"
        else:
            # Use keyword-based search
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
                include=["metadatas", "documents", "distances"]
            )
            retrieval_method = "keyword"

        retrieval_results = []

        if not results["ids"] or not results["ids"][0]:
            return retrieval_results

        for i, entry_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]
            distance = results["distances"][0][i]

            # Convert distance to similarity score (cosine distance to similarity)
            relevance_score = max(0.0, min(1.0, 1.0 - distance))

            entry = MemoryEntry(
                id=entry_id,
                content=content,
                tier=MemoryTier(metadata["tier"]),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                updated_at=datetime.fromisoformat(metadata["updated_at"]),
                source_session=metadata.get("source_session") or None,
                metadata={k: v for k, v in metadata.items()
                         if k not in ("tier", "created_at", "updated_at", "source_session")}
            )

            retrieval_results.append(RetrievalResult(
                entry=entry,
                relevance_score=relevance_score,
                retrieval_method=retrieval_method
            ))

        return retrieval_results

    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry]
    ) -> list[MemoryEntry]:
        """Migrate entries from source tier to target tier.

        Updates the tier metadata for matching entries in ChromaDB.
        Only processes entries that match the source_tier.

        Args:
            source_tier: The tier to migrate entries from.
            target_tier: The tier to migrate entries to.
            entries: List of entries to consolidate. Only entries matching
                     the source_tier are processed.

        Returns:
            A list of consolidated MemoryEntry objects in the target tier.

        Raises:
            ValueError: If source_tier equals target_tier.
            RuntimeError: If consolidation operation fails.
        """
        if self._collection is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if source_tier == target_tier:
            raise ValueError("Source and target tiers must be different")

        consolidated = []
        now = datetime.now()

        for entry in entries:
            if entry.tier != source_tier:
                continue

            # Get existing entry to update
            try:
                existing = self._collection.get(
                    ids=[entry.id],
                    include=["metadatas", "documents", "embeddings"]
                )

                if not existing["ids"]:
                    continue

                # Update metadata with new tier
                metadata = existing["metadatas"][0]
                metadata["tier"] = target_tier.value
                metadata["updated_at"] = now.isoformat()
                metadata["consolidated_from"] = source_tier.value

                # Re-add with updated metadata
                self._collection.update(
                    ids=[entry.id],
                    metadatas=[metadata]
                )

                # Create updated entry object
                new_entry = MemoryEntry(
                    id=entry.id,
                    content=entry.content,
                    tier=target_tier,
                    created_at=entry.created_at,
                    updated_at=now,
                    source_session=entry.source_session,
                    metadata={**entry.metadata, "consolidated_from": source_tier.value},
                    embedding=entry.embedding,
                )
                consolidated.append(new_entry)

            except Exception:
                # Entry not found or other error, skip
                continue

        return consolidated

    async def delete_expired(
        self,
        max_age_days: dict[MemoryTier, int]
    ) -> int:
        """Delete expired memory entries based on age policies.

        Removes entries that have exceeded their maximum age for their respective tiers.
        Each tier can have a different retention policy.

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
        if self._collection is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        total_deleted = 0
        now = datetime.now()

        for tier, days in max_age_days.items():
            if days <= 0:
                raise ValueError(f"max_age_days for {tier.value} must be positive")

            cutoff_date = now - timedelta(days=days)

            # Get all entries for this tier
            results = self._collection.get(
                where={"tier": tier.value},
                include=["metadatas"]
            )

            if not results["ids"]:
                continue

            ids_to_delete = []
            for i, entry_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                created_at = datetime.fromisoformat(metadata["created_at"])

                if created_at < cutoff_date:
                    ids_to_delete.append(entry_id)

            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                total_deleted += len(ids_to_delete)

        return total_deleted
