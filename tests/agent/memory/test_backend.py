"""Unit tests for MemoryBackend abstract base class.

This module tests the MemoryBackend interface using a MockBackend implementation.
"""

import pytest
from datetime import datetime
from typing import Optional

from nanobot.agent.memory.types import MemoryTier, MemoryEntry, RetrievalResult
from nanobot.agent.memory.backend import MemoryBackend


class MockBackend(MemoryBackend):
    """Mock implementation of MemoryBackend for testing purposes."""

    def __init__(self):
        self.entries: list[MemoryEntry] = []
        self.initialized = False
        self.deleted_count = 0

    async def initialize(self) -> None:
        """Initialize the mock storage."""
        self.initialized = True

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        self.entries.append(entry)

    async def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None
    ) -> list[RetrievalResult]:
        """Retrieve memories matching the query."""
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
        return results[:limit]

    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry]
    ) -> list[MemoryEntry]:
        """Consolidate entries from source tier to target tier."""
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
        """Delete expired entries based on tier-specific age limits."""
        now = datetime.now()
        to_delete = []

        for entry in self.entries:
            if entry.tier in max_age_days:
                age_days = (now - entry.created_at).days
                if age_days > max_age_days[entry.tier]:
                    to_delete.append(entry)

        for entry in to_delete:
            self.entries.remove(entry)

        self.deleted_count += len(to_delete)
        return len(to_delete)


@pytest.fixture
def mock_backend():
    """Fixture providing a MockBackend instance."""
    return MockBackend()


@pytest.fixture
def sample_entry():
    """Fixture providing a sample MemoryEntry."""
    now = datetime.now()
    return MemoryEntry(
        id="test-001",
        content="Test memory content",
        tier=MemoryTier.WORKING,
        created_at=now,
        updated_at=now,
        source_session="session-001",
        metadata={"key": "value"}
    )


class TestMockBackend:
    """Tests for the MockBackend implementation."""

    @pytest.mark.asyncio
    async def test_initialize(self, mock_backend):
        """Test that initialize sets the initialized flag."""
        assert not mock_backend.initialized
        await mock_backend.initialize()
        assert mock_backend.initialized

    @pytest.mark.asyncio
    async def test_store(self, mock_backend, sample_entry):
        """Test storing a memory entry."""
        assert len(mock_backend.entries) == 0
        await mock_backend.store(sample_entry)
        assert len(mock_backend.entries) == 1
        assert mock_backend.entries[0] == sample_entry

    @pytest.mark.asyncio
    async def test_retrieve_by_keyword(self, mock_backend, sample_entry):
        """Test retrieving memories by keyword."""
        await mock_backend.store(sample_entry)

        results = await mock_backend.retrieve("Test")
        assert len(results) == 1
        assert results[0].entry.id == sample_entry.id
        assert results[0].relevance_score == 0.9
        assert results[0].retrieval_method == "keyword"

    @pytest.mark.asyncio
    async def test_retrieve_with_tier_filter(self, mock_backend):
        """Test retrieving memories with tier filter."""
        now = datetime.now()

        working_entry = MemoryEntry(
            id="working-001",
            content="Working memory content",
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now
        )

        episodic_entry = MemoryEntry(
            id="episodic-001",
            content="Episodic memory content",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now
        )

        await mock_backend.store(working_entry)
        await mock_backend.store(episodic_entry)

        # Filter by WORKING tier
        results = await mock_backend.retrieve("memory", tier=MemoryTier.WORKING)
        assert len(results) == 1
        assert results[0].entry.tier == MemoryTier.WORKING

        # Filter by EPISODIC tier
        results = await mock_backend.retrieve("memory", tier=MemoryTier.EPISODIC)
        assert len(results) == 1
        assert results[0].entry.tier == MemoryTier.EPISODIC

    @pytest.mark.asyncio
    async def test_retrieve_limit(self, mock_backend):
        """Test that retrieve respects the limit parameter."""
        now = datetime.now()

        for i in range(5):
            entry = MemoryEntry(
                id=f"entry-{i}",
                content=f"Test content {i}",
                tier=MemoryTier.WORKING,
                created_at=now,
                updated_at=now
            )
            await mock_backend.store(entry)

        results = await mock_backend.retrieve("Test", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self, mock_backend, sample_entry):
        """Test retrieving with no matching results."""
        await mock_backend.store(sample_entry)

        results = await mock_backend.retrieve("nonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_consolidate(self, mock_backend):
        """Test consolidating entries between tiers."""
        now = datetime.now()

        working_entry = MemoryEntry(
            id="consolidate-001",
            content="Content to consolidate",
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now
        )

        await mock_backend.store(working_entry)

        consolidated = await mock_backend.consolidate(
            source_tier=MemoryTier.WORKING,
            target_tier=MemoryTier.EPISODIC,
            entries=[working_entry]
        )

        assert len(consolidated) == 1
        assert consolidated[0].tier == MemoryTier.EPISODIC
        assert consolidated[0].id == working_entry.id
        assert consolidated[0].content == working_entry.content

    @pytest.mark.asyncio
    async def test_consolidate_only_matching_tier(self, mock_backend):
        """Test that consolidate only processes entries from source tier."""
        now = datetime.now()

        working_entry = MemoryEntry(
            id="working-001",
            content="Working content",
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now
        )

        episodic_entry = MemoryEntry(
            id="episodic-001",
            content="Episodic content",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now
        )

        consolidated = await mock_backend.consolidate(
            source_tier=MemoryTier.WORKING,
            target_tier=MemoryTier.SEMANTIC,
            entries=[working_entry, episodic_entry]
        )

        assert len(consolidated) == 1
        assert consolidated[0].id == "working-001"

    @pytest.mark.asyncio
    async def test_delete_expired(self, mock_backend):
        """Test deleting expired entries."""
        from datetime import timedelta

        now = datetime.now()
        old_time = now - timedelta(days=10)
        recent_time = now - timedelta(days=1)

        old_entry = MemoryEntry(
            id="old-001",
            content="Old entry",
            tier=MemoryTier.WORKING,
            created_at=old_time,
            updated_at=old_time
        )

        recent_entry = MemoryEntry(
            id="recent-001",
            content="Recent entry",
            tier=MemoryTier.WORKING,
            created_at=recent_time,
            updated_at=recent_time
        )

        await mock_backend.store(old_entry)
        await mock_backend.store(recent_entry)

        deleted = await mock_backend.delete_expired({MemoryTier.WORKING: 5})

        assert deleted == 1
        assert len(mock_backend.entries) == 1
        assert mock_backend.entries[0].id == "recent-001"

    @pytest.mark.asyncio
    async def test_delete_expired_tier_specific(self, mock_backend):
        """Test that delete_expired respects tier-specific limits."""
        from datetime import timedelta

        now = datetime.now()
        old_time = now - timedelta(days=20)

        old_working = MemoryEntry(
            id="old-working",
            content="Old working entry",
            tier=MemoryTier.WORKING,
            created_at=old_time,
            updated_at=old_time
        )

        old_episodic = MemoryEntry(
            id="old-episodic",
            content="Old episodic entry",
            tier=MemoryTier.EPISODIC,
            created_at=old_time,
            updated_at=old_time
        )

        await mock_backend.store(old_working)
        await mock_backend.store(old_episodic)

        # Only WORKING has a limit, EPISODIC is not in the config
        deleted = await mock_backend.delete_expired({MemoryTier.WORKING: 10})

        assert deleted == 1
        assert mock_backend.entries[0].tier == MemoryTier.EPISODIC


class TestMemoryBackendAbstract:
    """Tests verifying MemoryBackend cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """Test that MemoryBackend is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MemoryBackend()

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement all abstract methods."""
        class IncompleteBackend(MemoryBackend):
            pass

        with pytest.raises(TypeError):
            IncompleteBackend()
