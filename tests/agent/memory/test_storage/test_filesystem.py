"""Unit tests for FileSystemBackend.

Tests cover initialization, store, retrieve, consolidate, and delete_expired operations.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from nanobot.agent.memory.storage import FileSystemBackend
from nanobot.agent.memory.types import MemoryEntry, MemoryTier, RetrievalResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
async def backend(temp_dir):
    """Create and initialize a FileSystemBackend for testing."""
    fs = FileSystemBackend(temp_dir)
    await fs.initialize()
    return fs


@pytest.mark.asyncio
async def test_initialize_creates_directories(temp_dir):
    """Test that initialize creates the expected directory structure."""
    fs = FileSystemBackend(temp_dir)
    await fs.initialize()

    assert (Path(temp_dir) / "working").exists()
    assert (Path(temp_dir) / "episodic").exists()
    assert (Path(temp_dir) / "semantic").exists()


@pytest.mark.asyncio
async def test_store_working_memory(backend, temp_dir):
    """Test storing a working memory entry."""
    entry = MemoryEntry(
        id="test-1",
        content="Test working memory content",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-123",
        metadata={"key": "value"},
    )

    await backend.store(entry)

    # Check file was created
    file_path = Path(temp_dir) / "working" / "session-123.jsonl"
    assert file_path.exists()

    # Check content
    with open(file_path, "r") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["id"] == "test-1"
        assert data["content"] == "Test working memory content"
        assert data["tier"] == "working"
        assert data["source_session"] == "session-123"


@pytest.mark.asyncio
async def test_store_working_memory_without_session_raises(backend):
    """Test that storing working memory without source_session raises ValueError."""
    entry = MemoryEntry(
        id="test-1",
        content="Test content",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session=None,
    )

    with pytest.raises(ValueError, match="source_session"):
        await backend.store(entry)


@pytest.mark.asyncio
async def test_store_episodic_memory(backend, temp_dir):
    """Test storing an episodic memory entry."""
    entry = MemoryEntry(
        id="ep-1",
        content="Episodic memory content",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-456",
    )

    await backend.store(entry)

    # Check file was created
    file_path = Path(temp_dir) / "episodic" / "ep-1.json"
    assert file_path.exists()

    # Check content
    with open(file_path, "r") as f:
        data = json.load(f)
        assert data["id"] == "ep-1"
        assert data["content"] == "Episodic memory content"
        assert data["tier"] == "episodic"


@pytest.mark.asyncio
async def test_store_semantic_memory(backend, temp_dir):
    """Test storing a semantic memory entry."""
    entry = MemoryEntry(
        id="sem-1",
        content="Semantic knowledge",
        tier=MemoryTier.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"category": "fact"},
    )

    await backend.store(entry)

    # Check file was created
    file_path = Path(temp_dir) / "semantic" / "sem-1.json"
    assert file_path.exists()

    # Check content
    with open(file_path, "r") as f:
        data = json.load(f)
        assert data["id"] == "sem-1"
        assert data["content"] == "Semantic knowledge"
        assert data["tier"] == "semantic"


@pytest.mark.asyncio
async def test_retrieve_from_working_memory(backend):
    """Test retrieving entries from working memory."""
    entry = MemoryEntry(
        id="test-1",
        content="Hello world test",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-1",
    )
    await backend.store(entry)

    results = await backend.retrieve("Hello")

    assert len(results) == 1
    assert results[0].entry.id == "test-1"
    assert results[0].entry.content == "Hello world test"
    assert results[0].relevance_score == 1.0
    assert results[0].retrieval_method == "keyword"


@pytest.mark.asyncio
async def test_retrieve_from_episodic_memory(backend):
    """Test retrieving entries from episodic memory."""
    entry = MemoryEntry(
        id="ep-1",
        content="Meeting notes from yesterday",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await backend.store(entry)

    results = await backend.retrieve("Meeting")

    assert len(results) == 1
    assert results[0].entry.id == "ep-1"
    assert results[0].entry.content == "Meeting notes from yesterday"


@pytest.mark.asyncio
async def test_retrieve_from_semantic_memory(backend):
    """Test retrieving entries from semantic memory."""
    entry = MemoryEntry(
        id="sem-1",
        content="Python is a programming language",
        tier=MemoryTier.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await backend.store(entry)

    results = await backend.retrieve("Python")

    assert len(results) == 1
    assert results[0].entry.id == "sem-1"


@pytest.mark.asyncio
async def test_retrieve_with_tier_filter(backend):
    """Test retrieving entries with tier filter."""
    # Store entries in different tiers
    working_entry = MemoryEntry(
        id="w-1",
        content="Test content",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="s1",
    )
    episodic_entry = MemoryEntry(
        id="e-1",
        content="Test content",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    await backend.store(working_entry)
    await backend.store(episodic_entry)

    # Retrieve only from working
    results = await backend.retrieve("Test", tier=MemoryTier.WORKING)
    assert len(results) == 1
    assert results[0].entry.tier == MemoryTier.WORKING

    # Retrieve only from episodic
    results = await backend.retrieve("Test", tier=MemoryTier.EPISODIC)
    assert len(results) == 1
    assert results[0].entry.tier == MemoryTier.EPISODIC


@pytest.mark.asyncio
async def test_retrieve_with_limit(backend):
    """Test that retrieve respects the limit parameter."""
    # Store multiple entries
    for i in range(5):
        entry = MemoryEntry(
            id=f"ep-{i}",
            content=f"Content {i}",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await backend.store(entry)

    results = await backend.retrieve("Content", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_retrieve_empty_query_raises(backend):
    """Test that empty query raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        await backend.retrieve("")


@pytest.mark.asyncio
async def test_retrieve_invalid_limit_raises(backend):
    """Test that invalid limit raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        await backend.retrieve("test", limit=0)

    with pytest.raises(ValueError, match="positive"):
        await backend.retrieve("test", limit=-1)


@pytest.mark.asyncio
async def test_retrieve_case_insensitive(backend):
    """Test that retrieve is case insensitive."""
    entry = MemoryEntry(
        id="test-1",
        content="HELLO World",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await backend.store(entry)

    results = await backend.retrieve("hello")
    assert len(results) == 1

    results = await backend.retrieve("WORLD")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_consolidate_working_to_episodic(backend, temp_dir):
    """Test consolidating entries from working to episodic memory."""
    entry = MemoryEntry(
        id="cons-1",
        content="Content to consolidate",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-1",
    )
    await backend.store(entry)

    consolidated = await backend.consolidate(
        MemoryTier.WORKING, MemoryTier.EPISODIC, [entry]
    )

    assert len(consolidated) == 1
    assert consolidated[0].tier == MemoryTier.EPISODIC
    assert consolidated[0].metadata["consolidated_from"] == "working"

    # Verify file was created in episodic
    file_path = Path(temp_dir) / "episodic" / "cons-1.json"
    assert file_path.exists()


@pytest.mark.asyncio
async def test_consolidate_episodic_to_semantic(backend, temp_dir):
    """Test consolidating entries from episodic to semantic memory."""
    entry = MemoryEntry(
        id="cons-2",
        content="Knowledge to extract",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await backend.store(entry)

    consolidated = await backend.consolidate(
        MemoryTier.EPISODIC, MemoryTier.SEMANTIC, [entry]
    )

    assert len(consolidated) == 1
    assert consolidated[0].tier == MemoryTier.SEMANTIC

    # Verify file was created in semantic
    file_path = Path(temp_dir) / "semantic" / "cons-2.json"
    assert file_path.exists()


@pytest.mark.asyncio
async def test_consolidate_same_tier_raises(backend):
    """Test that consolidating to same tier raises ValueError."""
    entry = MemoryEntry(
        id="test-1",
        content="Test",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="s1",
    )

    with pytest.raises(ValueError, match="different"):
        await backend.consolidate(MemoryTier.WORKING, MemoryTier.WORKING, [entry])


@pytest.mark.asyncio
async def test_consolidate_skips_wrong_source_tier(backend):
    """Test that consolidate skips entries not matching source_tier."""
    working_entry = MemoryEntry(
        id="w-1",
        content="Working",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="s1",
    )
    episodic_entry = MemoryEntry(
        id="e-1",
        content="Episodic",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Try to consolidate both as if they were working
    consolidated = await backend.consolidate(
        MemoryTier.WORKING, MemoryTier.EPISODIC, [working_entry, episodic_entry]
    )

    # Only the working entry should be consolidated
    assert len(consolidated) == 1
    assert consolidated[0].id == "w-1"


@pytest.mark.asyncio
async def test_delete_expired_returns_zero(backend):
    """Test that delete_expired returns 0 (placeholder implementation)."""
    result = await backend.delete_expired({MemoryTier.WORKING: 1})
    assert result == 0


@pytest.mark.asyncio
async def test_datetime_serialization(backend, temp_dir):
    """Test that datetime fields are properly serialized and deserialized."""
    now = datetime(2024, 1, 15, 10, 30, 0)
    entry = MemoryEntry(
        id="dt-test",
        content="Test datetime",
        tier=MemoryTier.EPISODIC,
        created_at=now,
        updated_at=now,
    )

    await backend.store(entry)

    # Retrieve and verify datetime is preserved
    results = await backend.retrieve("datetime")
    assert len(results) == 1
    assert results[0].entry.created_at == now
    assert results[0].entry.updated_at == now


@pytest.mark.asyncio
async def test_working_memory_jsonl_format(backend, temp_dir):
    """Test that working memory uses JSONL format (one entry per line)."""
    session_id = "multi-session"

    for i in range(3):
        entry = MemoryEntry(
            id=f"entry-{i}",
            content=f"Content {i}",
            tier=MemoryTier.WORKING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=session_id,
        )
        await backend.store(entry)

    file_path = Path(temp_dir) / "working" / f"{session_id}.jsonl"

    with open(file_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 3
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["id"] == f"entry-{i}"


@pytest.mark.asyncio
async def test_retrieve_no_matches(backend):
    """Test retrieve returns empty list when no matches found."""
    entry = MemoryEntry(
        id="test-1",
        content="Unique content xyz",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await backend.store(entry)

    results = await backend.retrieve("nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_entry_with_embedding(backend, temp_dir):
    """Test storing and retrieving entries with embeddings."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    entry = MemoryEntry(
        id="emb-1",
        content="Embedded content",
        tier=MemoryTier.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        embedding=embedding,
    )

    await backend.store(entry)

    # Verify embedding is preserved
    results = await backend.retrieve("Embedded")
    assert len(results) == 1
    assert results[0].entry.embedding == embedding
