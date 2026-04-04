"""SQLite storage backend for the memory system.

This module implements a SQLite-based storage backend that persists memory entries
to a local SQLite database with full support for querying, filtering, and management.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..backend import MemoryBackend
from ..types import MemoryEntry, MemoryTier, RetrievalResult


class SQLiteBackend(MemoryBackend):
    """SQLite storage backend for memory entries.

    This backend stores memories in a SQLite database with the following schema:
    - memories table with columns for id, content, tier, timestamps, metadata, and embedding

    Args:
        db_path: Path to the SQLite database file. Defaults to ~/.nanobot/memory.db
    """

    def __init__(self, db_path: str = "~/.nanobot/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self._connection: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection.

        Returns:
            An active SQLite connection with row factory configured.
        """
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    async def initialize(self) -> None:
        """Initialize the storage backend by creating tables.

        Creates the memories table with the following columns:
        - id: TEXT PRIMARY KEY
        - content: TEXT
        - tier: TEXT
        - created_at: TIMESTAMP
        - updated_at: TIMESTAMP
        - source_session: TEXT
        - metadata: TEXT (JSON serialized)
        - embedding: TEXT (JSON serialized list, optional)
        """
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                source_session TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)
        """)
        conn.commit()

    def _entry_to_row(self, entry: MemoryEntry) -> dict:
        """Convert a MemoryEntry to a database row dictionary.

        Args:
            entry: The MemoryEntry to convert.

        Returns:
            A dictionary representation suitable for database insertion.
        """
        return {
            "id": entry.id,
            "content": entry.content,
            "tier": entry.tier.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "source_session": entry.source_session,
            "metadata": json.dumps(entry.metadata),
            "embedding": json.dumps(entry.embedding) if entry.embedding else None,
        }

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry.

        Args:
            row: The SQLite row to convert.

        Returns:
            A MemoryEntry reconstructed from the row.
        """
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        embedding = json.loads(row["embedding"]) if row["embedding"] else None

        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            tier=MemoryTier(row["tier"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_session=row["source_session"],
            metadata=metadata,
            embedding=embedding,
        )

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in the database.

        Uses INSERT OR REPLACE to handle both new entries and updates.

        Args:
            entry: The MemoryEntry to store.

        Raises:
            RuntimeError: If storage operation fails.
        """
        conn = self._get_connection()
        row = self._entry_to_row(entry)

        conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, content, tier, created_at, updated_at, source_session, metadata, embedding)
            VALUES (:id, :content, :tier, :created_at, :updated_at, :source_session, :metadata, :embedding)
            """,
            row,
        )
        conn.commit()

    async def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None,
    ) -> list[RetrievalResult]:
        """Retrieve memories matching the query criteria.

        Performs keyword search using SQL LIKE operator. Supports filtering by tier
        and limiting results. Results are ordered by created_at DESC.

        Args:
            query: The search query string for keyword matching.
            tier: Optional filter to only return entries from a specific tier.
            limit: Maximum number of results to return (default: 10).
            embedding: Optional vector embedding for semantic similarity search
                (not currently implemented).

        Returns:
            A list of RetrievalResult objects containing matching entries
            with relevance scores of 1.0 for keyword matches.

        Raises:
            ValueError: If query is empty or limit is invalid.
            RuntimeError: If retrieval operation fails.
        """
        if limit <= 0:
            raise ValueError("Limit must be positive")

        conn = self._get_connection()
        results = []

        # Build query dynamically based on parameters
        if tier:
            sql = """
                SELECT * FROM memories
                WHERE tier = ? AND content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (tier.value, f"%{query}%", limit)
        else:
            sql = """
                SELECT * FROM memories
                WHERE content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (f"%{query}%", limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        for row in rows:
            entry = self._row_to_entry(row)
            results.append(
                RetrievalResult(
                    entry=entry,
                    relevance_score=1.0,
                    retrieval_method="keyword",
                )
            )

        return results

    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """Migrate entries from source tier to target tier.

        Updates the tier column for matching entries in the database.
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
        if source_tier == target_tier:
            raise ValueError("Source and target tiers must be different")

        conn = self._get_connection()
        consolidated = []
        now = datetime.now()

        for entry in entries:
            if entry.tier != source_tier:
                continue

            # Update the tier in the database
            conn.execute(
                """
                UPDATE memories
                SET tier = ?, updated_at = ?, metadata = ?
                WHERE id = ?
                """,
                (
                    target_tier.value,
                    now.isoformat(),
                    json.dumps({**entry.metadata, "consolidated_from": source_tier.value}),
                    entry.id,
                ),
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

        conn.commit()
        return consolidated

    async def delete_expired(
        self, max_age_days: dict[MemoryTier, int]
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
        conn = self._get_connection()
        total_deleted = 0

        for tier, days in max_age_days.items():
            if days <= 0:
                raise ValueError(f"max_age_days for {tier.value} must be positive")

            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE tier = ? AND datetime(created_at) < datetime('now', '-{} days')
                """.format(days),
                (tier.value,),
            )
            total_deleted += cursor.rowcount

        conn.commit()
        return total_deleted
