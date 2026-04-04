"""Episodic memory tier implementation for the layered memory system.

This module provides the EpisodicMemoryManager, which manages conversation
summaries and recent interaction history.
"""

import json
import uuid
from datetime import datetime

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.types import (
    MemoryEntry,
    MemoryTier,
    EpisodicMemoryConfig,
    RetrievalResult,
)


class EpisodicMemoryManager:
    """Manages episodic memory: conversation summaries and interaction history.

    Args:
        backend: The storage backend for persisting and retrieving entries.
        config: Configuration for the episodic memory tier.
    """

    def __init__(self, backend: MemoryBackend, config: EpisodicMemoryConfig) -> None:
        self.backend = backend
        self.config = config

    async def create_summary(
        self,
        session_id: str,
        turns: list[MemoryEntry],
    ) -> MemoryEntry:
        """Generate a summary from working memory turns.

        Combines user messages from the provided turns into a simple text
        summary. Full LLM-based summarization will be integrated later.

        Args:
            session_id: The session identifier to associate with the summary.
            turns: A list of MemoryEntry objects from working memory.

        Returns:
            A MemoryEntry representing the episodic summary.
        """
        turn_contents = []
        for turn in turns:
            data = json.loads(turn.content)
            user_message = data.get("user", "")
            turn_contents.append(f"User: {user_message[:100]}")

        summary_text = (
            f"Conversation summary for session {session_id}: "
            + "; ".join(turn_contents[:3])
        )

        now = datetime.now()
        entry = MemoryEntry(
            id=f"episodic-{uuid.uuid4().hex}",
            content=summary_text,
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session=session_id,
            metadata={"turn_count": len(turns)},
        )

        await self.backend.store(entry)
        return entry

    async def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        """Search episodic memory by keyword.

        Args:
            query: The search query string.
            limit: Maximum number of results to return (default: 5).

        Returns:
            A list of RetrievalResult objects matching the query.
        """
        return await self.backend.retrieve(
            query=query,
            tier=MemoryTier.EPISODIC,
            limit=limit,
        )

    async def get_for_session(self, session_id: str) -> list[MemoryEntry]:
        """Get all episodic entries for a session.

        Args:
            session_id: The session identifier to filter by.

        Returns:
            A list of MemoryEntry objects associated with the session.
        """
        results = await self.backend.retrieve(
            query=session_id,
            tier=MemoryTier.EPISODIC,
            limit=self.config.max_entries,
        )
        return [r.entry for r in results if r.entry.source_session == session_id]
