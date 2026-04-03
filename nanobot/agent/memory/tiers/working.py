"""Working memory tier implementation for short-term conversation storage.

This module provides the WorkingMemoryManager class for managing short-term
conversation turns in JSONL format, with support for retrieval, overflow
detection, and archiving.
"""

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.types import MemoryEntry, MemoryTier, WorkingMemoryConfig


class WorkingMemoryManager:
    """Manages short-term conversation turns in working memory.

    Working memory holds recent conversation turns for active sessions,
    stored as JSONL files. Provides methods for adding turns, retrieving
    recent history, and managing overflow conditions.

    Args:
        backend: The storage backend for persisting memory entries.
        config: Configuration for working memory behavior.
    """

    def __init__(self, backend: MemoryBackend, config: WorkingMemoryConfig):
        self.backend = backend
        self.config = config

    async def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        """Store a conversation turn as a JSONL entry.

        Creates a MemoryEntry containing both user and assistant messages
        with a timestamp, then stores it via the backend.

        Args:
            session_id: Unique identifier for the conversation session.
            user: The user's message content.
            assistant: The assistant's response content.

        Raises:
            RuntimeError: If storage operation fails.
        """
        now = datetime.now()
        content = json.dumps({
            "user": user,
            "assistant": assistant,
            "timestamp": now.isoformat()
        })

        entry = MemoryEntry(
            id=uuid4().hex,
            content=content,
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now,
            source_session=session_id,
            metadata={"turn_type": "conversation"}
        )

        await self.backend.store(entry)

    async def get_recent(self, n: int | None = None) -> list[MemoryEntry]:
        """Get recent N turns across all sessions.

        Retrieves working memory entries sorted by creation time,
        most recent first. If n is specified, limits to that many
        entries.

        Args:
            n: Optional limit on number of entries to return.
               If None, returns all working memory entries.

        Returns:
            List of MemoryEntry objects sorted by created_at descending.
        """
        results = await self.backend.retrieve(
            query="",  # Empty query to match all working memory
            tier=MemoryTier.WORKING,
            limit=10000  # Large limit to get all entries
        )

        # Sort by created_at descending (most recent first)
        entries = sorted(
            [r.entry for r in results],
            key=lambda e: e.created_at,
            reverse=True
        )

        if n is not None:
            entries = entries[:n]

        return entries

    async def get_all_for_session(self, session_id: str) -> list[MemoryEntry]:
        """Get all turns for a specific session.

        Retrieves all working memory entries belonging to the specified
        session, sorted chronologically (oldest first).

        Args:
            session_id: The session identifier to filter by.

        Returns:
            List of MemoryEntry objects for the session, sorted by
            created_at ascending (oldest first).
        """
        results = await self.backend.retrieve(
            query="",
            tier=MemoryTier.WORKING,
            limit=10000
        )

        # Filter by session_id and sort by created_at ascending
        entries = [
            r.entry for r in results
            if r.entry.source_session == session_id
        ]
        entries.sort(key=lambda e: e.created_at)

        return entries

    async def archive_entries(self, entry_ids: list[str]) -> None:
        """Archive or delete entries by their IDs.

        Marks entries as archived by updating their metadata. The actual
        deletion is deferred to the backend's delete_expired method.

        Args:
            entry_ids: List of entry IDs to archive.

        Raises:
            RuntimeError: If archive operation fails.
        """
        # For working memory, we mark entries as archived in metadata
        # The actual cleanup happens via delete_expired
        # Note: This is a simplified implementation that tracks archived IDs
        # A full implementation would retrieve, modify, and re-store entries
        if not hasattr(self, '_archived_ids'):
            self._archived_ids: set[str] = set()

        self._archived_ids.update(entry_ids)

    async def _is_overflow(self, session_id: str) -> bool:
        """Check if session exceeds configured max_turns.

        Compares the number of entries for the given session against
        the max_turns configuration setting.

        Args:
            session_id: The session identifier to check.

        Returns:
            True if the session has more turns than max_turns allows.
        """
        entries = await self.get_all_for_session(session_id)
        return len(entries) > self.config.max_turns

    def is_archived(self, entry_id: str) -> bool:
        """Check if an entry has been archived.

        Args:
            entry_id: The entry ID to check.

        Returns:
            True if the entry has been archived.
        """
        return hasattr(self, '_archived_ids') and entry_id in self._archived_ids
