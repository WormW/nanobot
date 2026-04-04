"""Tests for SQLite storage backend."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nanobot.agent.memory.storage.sqlite import SQLiteBackend
from nanobot.agent.memory.types import MemoryEntry, MemoryTier, RetrievalResult


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        yield str(db_path)


@pytest.fixture
async def backend(temp_db_path):
    """Create and initialize a SQLiteBackend for testing."""
    backend = SQLiteBackend(db_path=temp_db_path)
    await backend.initialize()
    return backend


@pytest.mark.asyncio
async def test_initialize_creates_database(temp_db_path):
    """Test that initialize creates the database and tables."""
    backend = SQLiteBackend(db_path=temp_db_path)

    # Database should not exist yet
    assert not Path(temp_db_path).exists()

    # Initialize should create the database
    await backend.initialize()

    # Database should now exist
    assert Path(temp_db_path).exists()


@pytest.mark.asyncio
async def test_store_and_retrieve(backend):
    """Test storing and retrieving a memory entry."""
    entry = MemoryEntry(
        id="test-1",
        content="This is a test memory",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-1",
        metadata={"key": "value"},
    )

    await backend.store(entry)

    # Retrieve the entry
    results = await backend.retrieve(query="test")

    assert len(results) == 1
    assert results[0].entry.id == "test-1"
    assert results[0].entry.content == "This is a test memory"
    assert results[0].entry.tier == MemoryTier.WORKING
    assert results[0].entry.source_session == "session-1"
    assert results[0].entry.metadata == {"key": "value"}
    assert results[0].relevance_score == 1.0
    assert results[0].retrieval_method == "keyword"


@pytest.mark.asyncio
async def test_retrieve_with_tier_filter(backend):
    """Test retrieving memories with tier filtering."""
    # Store entries in different tiers
    working_entry = MemoryEntry(
        id="working-1",
        content="Working memory content",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-1",
    )
    episodic_entry = MemoryEntry(
        id="episodic-1",
        content="Episodic memory content",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    semantic_entry = MemoryEntry(
        id="semantic-1",
        content="Semantic memory content",
        tier=MemoryTier.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    await backend.store(working_entry)
    await backend.store(episodic_entry)
    await backend.store(semantic_entry)

    # Retrieve all entries
    all_results = await backend.retrieve(query="memory")
    assert len(all_results) == 3

    # Retrieve only working tier
    working_results = await backend.retrieve(
        query="memory", tier=MemoryTier.WORKING
    )
    assert len(working_results) == 1
    assert working_results[0].entry.tier == MemoryTier.WORKING

    # Retrieve only episodic tier
    episodic_results = await backend.retrieve(
        query="memory", tier=MemoryTier.EPISODIC
    )
    assert len(episodic_results) == 1
    assert episodic_results[0].entry.tier == MemoryTier.EPISODIC

    # Retrieve only semantic tier
    semantic_results = await backend.retrieve(
        query="memory", tier=MemoryTier.SEMANTIC
    )
    assert len(semantic_results) == 1
    assert semantic_results[0].entry.tier == MemoryTier.SEMANTIC


@pytest.mark.asyncio
async def test_retrieve_limit(backend):
    """Test that retrieve respects the limit parameter."""
    # Store multiple entries
    for i in range(5):
        entry = MemoryEntry(
            id=f"entry-{i}",
            content=f"Content {i}",
            tier=MemoryTier.WORKING,
            created_at=datetime.now() + timedelta(seconds=i),
            updated_at=datetime.now(),
            source_session="session-1",
        )
        await backend.store(entry)

    # Retrieve with limit
    results = await backend.retrieve(query="Content", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_retrieve_order_by_created_at(backend):
    """Test that retrieve returns results ordered by created_at DESC."""
    # Store entries with different timestamps
    for i in range(3):
        entry = MemoryEntry(
            id=f"entry-{i}",
            content="Test content",
            tier=MemoryTier.WORKING,
            created_at=datetime.now() + timedelta(seconds=i),
            updated_at=datetime.now(),
            source_session="session-1",
        )
        await backend.store(entry)

    results = await backend.retrieve(query="Test")
    assert len(results) == 3
    # Should be in reverse order (newest first)
    assert results[0].entry.id == "entry-2"
    assert results[1].entry.id == "entry-1"
    assert results[2].entry.id == "entry-0"


@pytest.mark.asyncio
async def test_retrieve_invalid_limit(backend):
    """Test that retrieve raises ValueError for invalid limit."""
    with pytest.raises(ValueError, match="Limit must be positive"):
        await backend.retrieve(query="test", limit=0)

    with pytest.raises(ValueError, match="Limit must be positive"):
        await backend.retrieve(query="test", limit=-1)


@pytest.mark.asyncio
async def test_store_updates_existing_entry(backend):
    """Test that store updates an existing entry with the same ID."""
    entry = MemoryEntry(
        id="test-1",
        content="Original content",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-1",
    )

    await backend.store(entry)

    # Update the entry
    updated_entry = MemoryEntry(
        id="test-1",
        content="Updated content",
        tier=MemoryTier.EPISODIC,
        created_at=entry.created_at,
        updated_at=datetime.now(),
        source_session="session-2",
    )

    await backend.store(updated_entry)

    # Should only have one entry with updated content
    results = await backend.retrieve(query="content")
    assert len(results) == 1
    assert results[0].entry.content == "Updated content"
    assert results[0].entry.tier == MemoryTier.EPISODIC
    assert results[0].entry.source_session == "session-2"


@pytest.mark.asyncio
async def test_consolidate_updates_tier(backend):
    """Test that consolidate updates the tier for entries."""
    # Create entries in working tier
    entries = []
    for i in range(3):
        entry = MemoryEntry(
            id=f"entry-{i}",
            content=f"Content {i}",
            tier=MemoryTier.WORKING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="session-1",
        )
        await backend.store(entry)
        entries.append(entry)

    # Add an entry from a different tier (should be ignored)
    other_entry = MemoryEntry(
        id="entry-other",
        content="Other content",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await backend.store(other_entry)
    entries.append(other_entry)

    # Consolidate from WORKING to EPISODIC
    consolidated = await backend.consolidate(
        source_tier=MemoryTier.WORKING,
        target_tier=MemoryTier.EPISODIC,
        entries=entries,
    )

    # Should have consolidated 3 entries (not the episodic one)
    assert len(consolidated) == 3
    for entry in consolidated:
        assert entry.tier == MemoryTier.EPISODIC
        assert entry.metadata.get("consolidated_from") == "working"

    # Verify in database
    working_results = await backend.retrieve(
        query="Content", tier=MemoryTier.WORKING
    )
    assert len(working_results) == 0

    episodic_results = await backend.retrieve(
        query="Content", tier=MemoryTier.EPISODIC
    )
    assert len(episodic_results) == 4  # 3 consolidated + 1 original


@pytest.mark.asyncio
async def test_consolidate_same_tier_raises(backend):
    """Test that consolidate raises ValueError when source equals target."""
    entry = MemoryEntry(
        id="test-1",
        content="Test",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    with pytest.raises(ValueError, match="Source and target tiers must be different"):
        await backend.consolidate(
            source_tier=MemoryTier.WORKING,
            target_tier=MemoryTier.WORKING,
            entries=[entry],
        )


@pytest.mark.asyncio
async def test_delete_expired_removes_old_entries(backend):
    """Test that delete_expired removes entries older than specified days."""
    now = datetime.now()

    # Create entries with different ages
    old_working = MemoryEntry(
        id="old-working",
        content="Old working memory",
        tier=MemoryTier.WORKING,
        created_at=now - timedelta(days=10),
        updated_at=now,
        source_session="session-1",
    )
    recent_working = MemoryEntry(
        id="recent-working",
        content="Recent working memory",
        tier=MemoryTier.WORKING,
        created_at=now - timedelta(days=1),
        updated_at=now,
        source_session="session-1",
    )
    old_episodic = MemoryEntry(
        id="old-episodic",
        content="Old episodic memory",
        tier=MemoryTier.EPISODIC,
        created_at=now - timedelta(days=30),
        updated_at=now,
    )

    await backend.store(old_working)
    await backend.store(recent_working)
    await backend.store(old_episodic)

    # Delete entries older than 5 days for working, 20 days for episodic
    deleted = await backend.delete_expired(
        max_age_days={
            MemoryTier.WORKING: 5,
            MemoryTier.EPISODIC: 20,
        }
    )

    assert deleted == 2  # old_working and old_episodic

    # Verify remaining entries
    results = await backend.retrieve(query="memory")
    assert len(results) == 1
    assert results[0].entry.id == "recent-working"


@pytest.mark.asyncio
async def test_delete_expired_invalid_days(backend):
    """Test that delete_expired raises ValueError for invalid days."""
    with pytest.raises(ValueError, match="must be positive"):
        await backend.delete_expired(
            max_age_days={MemoryTier.WORKING: 0}
        )

    with pytest.raises(ValueError, match="must be positive"):
        await backend.delete_expired(
            max_age_days={MemoryTier.WORKING: -1}
        )


@pytest.mark.asyncio
async def test_store_with_embedding(backend):
    """Test storing and retrieving entries with embeddings."""
    entry = MemoryEntry(
        id="test-embedding",
        content="Test with embedding",
        tier=MemoryTier.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    )

    await backend.store(entry)

    results = await backend.retrieve(query="embedding")
    assert len(results) == 1
    assert results[0].entry.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.asyncio
async def test_store_with_complex_metadata(backend):
    """Test storing and retrieving entries with complex metadata."""
    entry = MemoryEntry(
        id="test-metadata",
        content="Test with metadata",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
            "boolean": True,
        },
    )

    await backend.store(entry)

    results = await backend.retrieve(query="metadata")
    assert len(results) == 1
    assert results[0].entry.metadata["nested"]["key"] == "value"
    assert results[0].entry.metadata["list"] == [1, 2, 3]
    assert results[0].entry.metadata["number"] == 42
    assert results[0].entry.metadata["boolean"] is True


@pytest.mark.asyncio
async def test_default_db_path():
    """Test that default database path is in home directory."""
    backend = SQLiteBackend()
    assert ".nanobot" in str(backend.db_path)
    assert "memory.db" in str(backend.db_path)
