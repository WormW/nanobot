"""File system storage backend for the memory system.

This module implements a file-based storage backend that persists memory entries
to the local filesystem using JSON and JSONL formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..backend import MemoryBackend
from ..types import MemoryEntry, MemoryTier, RetrievalResult


class FileSystemBackend(MemoryBackend):
    """File system storage backend for memory entries.

    This backend stores memories to the filesystem with the following structure:
    - Working memory: JSONL format (one entry per line), organized by session_id
      at base_path/working/{session_id}.jsonl
    - Episodic/Semantic: Individual JSON files, organized by entry_id
      at base_path/{tier}/{entry_id}.json

    Args:
        base_path: The root directory for all memory storage.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    async def initialize(self) -> None:
        """Initialize the storage backend by creating directory structure.

        Creates the following directories under base_path:
        - working/ for working memory entries
        - episodic/ for episodic memory entries
        - semantic/ for semantic memory entries
        """
        for tier in MemoryTier:
            tier_path = self.base_path / tier.value
            tier_path.mkdir(parents=True, exist_ok=True)

    def _entry_to_dict(self, entry: MemoryEntry) -> dict:
        """Convert a MemoryEntry to a dictionary for serialization.

        Args:
            entry: The MemoryEntry to convert.

        Returns:
            A dictionary representation of the entry with ISO-formatted datetimes.
        """
        return {
            "id": entry.id,
            "content": entry.content,
            "tier": entry.tier.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "source_session": entry.source_session,
            "metadata": entry.metadata,
            "embedding": entry.embedding,
        }

    def _dict_to_entry(self, data: dict) -> MemoryEntry:
        """Convert a dictionary to a MemoryEntry.

        Args:
            data: The dictionary to convert.

        Returns:
            A MemoryEntry reconstructed from the dictionary.
        """
        return MemoryEntry(
            id=data["id"],
            content=data["content"],
            tier=MemoryTier(data["tier"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            source_session=data.get("source_session"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in the filesystem.

        Working memory entries are stored as JSONL (one entry per line) in
        base_path/working/{session_id}.jsonl. Episodic and semantic entries
        are stored as individual JSON files in base_path/{tier}/{entry_id}.json.

        Args:
            entry: The MemoryEntry to store.

        Raises:
            ValueError: If a working memory entry has no source_session.
            RuntimeError: If the storage operation fails.
        """
        if entry.tier == MemoryTier.WORKING:
            if not entry.source_session:
                raise ValueError("Working memory entries must have a source_session")
            file_path = self.base_path / "working" / f"{entry.source_session}.jsonl"
            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(self._entry_to_dict(entry), f)
                f.write("\n")
        else:
            tier_path = self.base_path / entry.tier.value
            file_path = tier_path / f"{entry.id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._entry_to_dict(entry), f, indent=2)

    async def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None,
    ) -> list[RetrievalResult]:
        """Retrieve memories matching the query criteria.

        Reads files from tier directories and returns entries that match
        the query (simple substring match for now). All returned entries
        have a relevance_score of 1.0.

        Args:
            query: The search query string for keyword matching.
                If empty, returns all entries in the specified tier.
            tier: Optional filter to only return entries from a specific tier.
            limit: Maximum number of results to return (default: 10).
            embedding: Optional vector embedding for semantic similarity search
                (not currently implemented).

        Returns:
            A list of RetrievalResult objects containing matching entries
            with relevance scores of 1.0.

        Raises:
            ValueError: If limit is invalid.
            RuntimeError: If retrieval operation fails.
        """
        # Empty query matches all entries in the tier
        if limit <= 0:
            raise ValueError("Limit must be positive")

        results = []
        tiers_to_search = [tier] if tier else list(MemoryTier)

        for t in tiers_to_search:
            if t is None:
                continue

            if t == MemoryTier.WORKING:
                # Search all JSONL files in working directory
                working_path = self.base_path / "working"
                if working_path.exists():
                    for jsonl_file in working_path.glob("*.jsonl"):
                        with open(jsonl_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                data = json.loads(line)
                                if not query or query.lower() in data["content"].lower():
                                    results.append(
                                        RetrievalResult(
                                            entry=self._dict_to_entry(data),
                                            relevance_score=1.0,
                                            retrieval_method="keyword",
                                        )
                                    )
                                    if len(results) >= limit:
                                        return results
            else:
                # Search individual JSON files in tier directory
                tier_path = self.base_path / t.value
                if tier_path.exists():
                    for json_file in tier_path.glob("*.json"):
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if not query or query.lower() in data["content"].lower():
                                results.append(
                                    RetrievalResult(
                                        entry=self._dict_to_entry(data),
                                        relevance_score=1.0,
                                        retrieval_method="keyword",
                                    )
                                )
                                if len(results) >= limit:
                                    return results

        return results

    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """Migrate entries from source tier to target tier.

        Copies entries to the target tier with updated metadata. The original
        entries remain in the source tier until delete_expired is called.

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
        if source_tier == target_tier:
            raise ValueError("Source and target tiers must be different")

        consolidated = []
        for entry in entries:
            if entry.tier != source_tier:
                continue

            # Create new entry with updated tier and timestamp
            new_entry = MemoryEntry(
                id=entry.id,
                content=entry.content,
                tier=target_tier,
                created_at=entry.created_at,
                updated_at=datetime.now(),
                source_session=entry.source_session,
                metadata={**entry.metadata, "consolidated_from": source_tier.value},
                embedding=entry.embedding,
            )

            # Store in target tier
            await self.store(new_entry)
            consolidated.append(new_entry)

        return consolidated

    async def delete_expired(self, max_age_days: dict[MemoryTier, int]) -> int:
        """Delete expired memory entries based on age policies.

        This is currently a placeholder implementation that returns 0.
        Full implementation would check file modification times and remove
        entries older than the specified max_age_days for each tier.

        Args:
            max_age_days: A dictionary mapping MemoryTier to the maximum
                          age in days for entries in that tier.

        Returns:
            The number of entries that were deleted (currently always 0).

        Raises:
            ValueError: If max_age_days contains invalid values.
            RuntimeError: If deletion operation fails.
        """
        # Placeholder implementation
        return 0
