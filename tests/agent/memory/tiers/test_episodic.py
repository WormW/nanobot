"""Unit tests for EpisodicMemoryManager.

This module tests the EpisodicMemoryManager class and its methods
for creating summaries, searching, and retrieving session entries.
"""

import json
import pytest
from datetime import datetime
from typing import Optional

from nanobot.agent.memory.types import MemoryEntry, MemoryTier, RetrievalResult, EpisodicMemoryConfig
from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.tiers.episodic import EpisodicMemoryManager


class MockBackend(MemoryBackend):
    """Mock implementation of MemoryBackend for episodic memory testing."""

    def __init__(self):
        self.entries: list[MemoryEntry] = []
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def store(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    async def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None
    ) -> list[RetrievalResult]:
        results = []
        for entry in self.entries:
            if tier is not None and entry.tier != tier:
                continue
            if query.lower() in entry.content.lower():
                results.append(RetrievalResult(
                    entry=entry,
                    relevance_score=0.9,
                    retrieval_method="keyword"
                ))
            elif entry.source_session and query.lower() in entry.source_session.lower():
                results.append(RetrievalResult(
                    entry=entry,
                    relevance_score=0.85,
                    retrieval_method="keyword"
                ))
        return results[:limit]

    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry]
    ) -> list[MemoryEntry]:
        consolidated = []
        for entry in entries:
            if entry.tier == source_tier:
                new_entry = MemoryEntry(
                    id=entry.id,
                    content=entry.content,
                    tier=target_tier,
                    created_at=entry.created_at,
                    updated_at=datetime.now(),
                    source_session=entry.source_session,
                    metadata=entry.metadata.copy(),
                    embedding=entry.embedding
                )
                consolidated.append(new_entry)
        return consolidated

    async def delete_expired(self, max_age_days: dict[MemoryTier, int]) -> int:
        now = datetime.now()
        to_delete = [
            entry for entry in self.entries
            if entry.tier in max_age_days and (now - entry.created_at).days > max_age_days[entry.tier]
        ]
        for entry in to_delete:
            self.entries.remove(entry)
        return len(to_delete)


@pytest.fixture
def mock_backend():
    """Fixture providing a MockBackend instance."""
    return MockBackend()


@pytest.fixture
def config():
    """Fixture providing an EpisodicMemoryConfig instance."""
    return EpisodicMemoryConfig(
        summary_model="default",
        max_entries=100,
        consolidation_batch=5,
    )


@pytest.fixture
def manager(mock_backend, config):
    """Fixture providing an EpisodicMemoryManager instance."""
    return EpisodicMemoryManager(backend=mock_backend, config=config)


@pytest.fixture
def sample_turns():
    """Fixture providing sample working memory turns."""
    now = datetime.now()
    return [
        MemoryEntry(
            id="turn-1",
            content=json.dumps({"user": "Hello, how are you?"}),
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now,
            source_session="session-001",
        ),
        MemoryEntry(
            id="turn-2",
            content=json.dumps({"user": "What is the weather today?"}),
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now,
            source_session="session-001",
        ),
    ]


class TestEpisodicMemoryManager:
    """Tests for EpisodicMemoryManager."""

    @pytest.mark.asyncio
    async def test_create_summary_generates_proper_memory_entry(self, manager, mock_backend, sample_turns):
        """Test that create_summary generates a proper MemoryEntry with EPISODIC tier."""
        session_id = "session-001"
        entry = await manager.create_summary(session_id, sample_turns)

        assert entry.tier == MemoryTier.EPISODIC
        assert entry.source_session == session_id
        assert entry.content.startswith(f"Conversation summary for session {session_id}:")
        assert "Hello, how are you?" in entry.content
        assert "What is the weather today?" in entry.content
        assert entry.metadata["turn_count"] == 2
        assert entry.id.startswith("episodic-")

        # Verify it was stored in the backend
        assert len(mock_backend.entries) == 1
        assert mock_backend.entries[0] == entry

    @pytest.mark.asyncio
    async def test_create_summary_truncates_user_messages(self, manager):
        """Test that create_summary truncates user messages to 100 characters."""
        now = datetime.now()
        long_message = "A" * 200
        turns = [
            MemoryEntry(
                id="turn-1",
                content=json.dumps({"user": long_message}),
                tier=MemoryTier.WORKING,
                created_at=now,
                updated_at=now,
                source_session="session-001",
            )
        ]

        entry = await manager.create_summary("session-001", turns)
        assert "User: " + "A" * 100 in entry.content
        assert "A" * 101 not in entry.content.split("User: ")[1]

    @pytest.mark.asyncio
    async def test_create_summary_limits_to_three_turns(self, manager):
        """Test that create_summary only includes up to three turns."""
        now = datetime.now()
        turns = [
            MemoryEntry(
                id=f"turn-{i}",
                content=json.dumps({"user": f"Message {i}"}),
                tier=MemoryTier.WORKING,
                created_at=now,
                updated_at=now,
                source_session="session-001",
            )
            for i in range(5)
        ]

        entry = await manager.create_summary("session-001", turns)
        assert "Message 0" in entry.content
        assert "Message 1" in entry.content
        assert "Message 2" in entry.content
        assert "Message 3" not in entry.content
        assert "Message 4" not in entry.content
        assert entry.metadata["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_search_returns_matching_entries(self, manager, mock_backend):
        """Test that search returns entries matching the query."""
        now = datetime.now()

        # Store matching entry
        matching_entry = MemoryEntry(
            id="episodic-1",
            content="Conversation summary for session abc: User: Hello there",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session="session-abc",
        )

        # Store non-matching entry
        non_matching_entry = MemoryEntry(
            id="episodic-2",
            content="Conversation summary for session xyz: User: Goodbye",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session="session-xyz",
        )

        # Store working memory entry that matches query but wrong tier
        working_entry = MemoryEntry(
            id="working-1",
            content="User: Hello there",
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now,
            source_session="session-abc",
        )

        await mock_backend.store(matching_entry)
        await mock_backend.store(non_matching_entry)
        await mock_backend.store(working_entry)

        results = await manager.search("Hello")

        assert len(results) == 1
        assert results[0].entry.id == "episodic-1"
        assert results[0].entry.tier == MemoryTier.EPISODIC

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, manager, mock_backend):
        """Test that search respects the limit parameter."""
        now = datetime.now()

        for i in range(5):
            entry = MemoryEntry(
                id=f"episodic-{i}",
                content=f"Conversation summary: User: Test message {i}",
                tier=MemoryTier.EPISODIC,
                created_at=now,
                updated_at=now,
            )
            await mock_backend.store(entry)

        results = await manager.search("Test", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_uses_episodic_tier_filter(self, manager, mock_backend):
        """Test that search only returns EPISODIC tier entries."""
        now = datetime.now()

        episodic_entry = MemoryEntry(
            id="episodic-1",
            content="Conversation summary: User: Hello",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
        )

        semantic_entry = MemoryEntry(
            id="semantic-1",
            content="Fact: Hello is a greeting",
            tier=MemoryTier.SEMANTIC,
            created_at=now,
            updated_at=now,
        )

        await mock_backend.store(episodic_entry)
        await mock_backend.store(semantic_entry)

        results = await manager.search("Hello")
        assert len(results) == 1
        assert results[0].entry.tier == MemoryTier.EPISODIC

    @pytest.mark.asyncio
    async def test_get_for_session_filters_correctly(self, manager, mock_backend):
        """Test that get_for_session returns only entries for the specified session."""
        now = datetime.now()

        session_a_entry = MemoryEntry(
            id="episodic-a",
            content="Conversation summary for session alpha: User: Hello",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session="session-alpha",
        )

        session_b_entry = MemoryEntry(
            id="episodic-b",
            content="Conversation summary for session beta: User: Goodbye",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session="session-beta",
        )

        await mock_backend.store(session_a_entry)
        await mock_backend.store(session_b_entry)

        results = await manager.get_for_session("session-alpha")

        assert len(results) == 1
        assert results[0].id == "episodic-a"
        assert results[0].source_session == "session-alpha"

    @pytest.mark.asyncio
    async def test_get_for_session_uses_config_max_entries(self, manager, mock_backend, config):
        """Test that get_for_session uses config.max_entries as the retrieval limit."""
        now = datetime.now()

        # Create more entries than the default mock retrieve limit but within config.max_entries
        for i in range(config.max_entries + 1):
            entry = MemoryEntry(
                id=f"episodic-{i}",
                content=f"Conversation summary for session bulk: User: Message {i}",
                tier=MemoryTier.EPISODIC,
                created_at=now,
                updated_at=now,
                source_session="session-bulk",
            )
            await mock_backend.store(entry)

        results = await manager.get_for_session("session-bulk")
        assert len(results) == config.max_entries

    @pytest.mark.asyncio
    async def test_get_for_session_excludes_false_positives(self, manager, mock_backend):
        """Test that get_for_session excludes entries where query matches but source_session differs."""
        now = datetime.now()

        # Entry where content contains the session_id but source_session is different
        false_positive = MemoryEntry(
            id="episodic-fp",
            content="Conversation summary for session target: User: Hello",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session="session-other",
        )

        true_match = MemoryEntry(
            id="episodic-tm",
            content="Conversation summary for session target: User: Hello again",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
            source_session="session-target",
        )

        await mock_backend.store(false_positive)
        await mock_backend.store(true_match)

        results = await manager.get_for_session("session-target")

        assert len(results) == 1
        assert results[0].id == "episodic-tm"
