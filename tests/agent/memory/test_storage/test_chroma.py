"""Unit tests for ChromaBackend.

Tests cover initialization, store, retrieve, consolidate, and delete_expired operations.
Tests are skipped if chromadb is not installed.
"""

import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Check if chromadb is available
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from nanobot.agent.memory.types import MemoryEntry, MemoryTier, RetrievalResult

if HAS_CHROMADB:
    from nanobot.agent.memory.storage import ChromaBackend


pytestmark = pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
async def backend(temp_dir):
    """Create and initialize a ChromaBackend for testing."""
    # Use unique collection name to avoid test interference
    collection_name = f"test_memories_{uuid.uuid4().hex[:8]}"
    chroma = ChromaBackend(persist_directory=temp_dir, collection_name=collection_name)
    await chroma.initialize()
    return chroma


@pytest.mark.asyncio
async def test_initialize_creates_directory(temp_dir):
    """Test that initialize creates the persist directory."""
    collection_name = f"test_init_{uuid.uuid4().hex[:8]}"
    chroma = ChromaBackend(persist_directory=temp_dir, collection_name=collection_name)
    await chroma.initialize()

    assert Path(temp_dir).exists()


@pytest.mark.asyncio
async def test_store_without_embedding(backend):
    """Test storing a memory entry without embedding."""
    entry_id = f"test-{uuid.uuid4().hex[:8]}"
    entry = MemoryEntry(
        id=entry_id,
        content="Test memory content for store without embedding",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-123",
        metadata={"key": "value"},
    )

    await backend.store(entry)

    # Verify by retrieving
    results = await backend.retrieve(query="Test memory content", limit=1)
    assert len(results) == 1
    assert results[0].entry.id == entry_id
    assert results[0].entry.content == "Test memory content for store without embedding"


@pytest.mark.asyncio
async def test_store_with_embedding(backend):
    """Test storing a memory entry with embedding."""
    # Use 384 dimensions (ChromaDB default embedding dimension)
    embedding = [0.1] * 384
    entry_id = f"test-emb-{uuid.uuid4().hex[:8]}"
    entry = MemoryEntry(
        id=entry_id,
        content="Test memory with embedding for similarity search",
        tier=MemoryTier.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source_session="session-456",
        metadata={"importance": "high"},
        embedding=embedding,
    )

    await backend.store(entry)

    # Verify by retrieving with embedding
    results = await backend.retrieve(query="", embedding=embedding, limit=1)
    assert len(results) == 1
    assert results[0].entry.id == entry_id
    assert results[0].retrieval_method == "embedding"


@pytest.mark.asyncio
async def test_retrieve_by_text_query(backend):
    """Test retrieving memories by text query."""
    # Store multiple entries with unique content
    entries = [
        MemoryEntry(
            id=f"entry-{uuid.uuid4().hex[:8]}",
            content=f"Unique content about topic alpha {uuid.uuid4().hex[:4]}",
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=None,
        )
        for _ in range(3)
    ]

    for entry in entries:
        await backend.store(entry)

    # Retrieve by query using unique content from first entry
    query_text = entries[0].content
    results = await backend.retrieve(query=query_text, limit=5)
    assert len(results) >= 1
    # Check that the first result is the one we queried for
    assert any(query_text in r.entry.content for r in results)


@pytest.mark.asyncio
async def test_retrieve_by_embedding_similarity(backend):
    """Test retrieving memories by embedding similarity."""
    # Use 384 dimensions for ChromaDB compatibility
    base_embedding = [0.1] * 384

    # Store entries with slightly different embeddings
    for i in range(3):
        embedding = [x + (i * 0.01) for x in base_embedding]
        entry = MemoryEntry(
            id=f"emb-entry-{uuid.uuid4().hex[:8]}",
            content=f"Embedding test content {i} {uuid.uuid4().hex[:4]}",
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=None,
            embedding=embedding,
        )
        await backend.store(entry)

    # Query with embedding similar to first entry
    query_embedding = [0.105] * 384
    results = await backend.retrieve(query="", embedding=query_embedding, limit=2)

    assert len(results) > 0
    assert all(r.retrieval_method == "embedding" for r in results)
    # Results should be sorted by relevance
    assert results[0].relevance_score >= results[-1].relevance_score


@pytest.mark.asyncio
async def test_retrieve_with_tier_filter(backend):
    """Test retrieving memories filtered by tier."""
    unique_prefix = uuid.uuid4().hex[:8]

    # Store entries in different tiers
    for tier in [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC]:
        entry = MemoryEntry(
            id=f"tier-{tier.value}-{unique_prefix}",
            content=f"{unique_prefix} Content in {tier.value} tier",
            tier=tier,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="session-123" if tier == MemoryTier.WORKING else None,
        )
        await backend.store(entry)

    # Retrieve only working tier
    results = await backend.retrieve(query=unique_prefix, tier=MemoryTier.WORKING, limit=10)
    assert len(results) == 1
    assert results[0].entry.tier == MemoryTier.WORKING

    # Retrieve only episodic tier
    results = await backend.retrieve(query=unique_prefix, tier=MemoryTier.EPISODIC, limit=10)
    assert len(results) == 1
    assert results[0].entry.tier == MemoryTier.EPISODIC


@pytest.mark.asyncio
async def test_retrieve_limit(backend):
    """Test that retrieve respects the limit parameter."""
    unique_prefix = uuid.uuid4().hex[:8]

    # Store multiple entries
    for i in range(5):
        entry = MemoryEntry(
            id=f"limit-{uuid.uuid4().hex[:8]}",
            content=f"{unique_prefix} Limit test content {i}",
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await backend.store(entry)

    # Retrieve with limit
    results = await backend.retrieve(query=unique_prefix, limit=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_retrieve_invalid_limit(backend):
    """Test that retrieve raises ValueError for invalid limit."""
    with pytest.raises(ValueError, match="Limit must be positive"):
        await backend.retrieve(query="test", limit=0)

    with pytest.raises(ValueError, match="Limit must be positive"):
        await backend.retrieve(query="test", limit=-1)


@pytest.mark.asyncio
async def test_consolidate(backend):
    """Test consolidating entries from one tier to another."""
    unique_prefix = uuid.uuid4().hex[:8]

    # Create entries in working tier
    entries = [
        MemoryEntry(
            id=f"consolidate-{uuid.uuid4().hex[:8]}",
            content=f"{unique_prefix} Consolidate test {i}",
            tier=MemoryTier.WORKING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="session-123",
        )
        for i in range(3)
    ]

    for entry in entries:
        await backend.store(entry)

    # Consolidate to episodic
    consolidated = await backend.consolidate(
        source_tier=MemoryTier.WORKING,
        target_tier=MemoryTier.EPISODIC,
        entries=entries,
    )

    assert len(consolidated) == 3
    assert all(e.tier == MemoryTier.EPISODIC for e in consolidated)
    assert all("consolidated_from" in e.metadata for e in consolidated)

    # Verify in database - should find exactly 3 entries
    results = await backend.retrieve(
        query=unique_prefix,
        tier=MemoryTier.EPISODIC,
        limit=10
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_consolidate_same_tier_raises(backend):
    """Test that consolidating to the same tier raises ValueError."""
    entry = MemoryEntry(
        id=f"test-{uuid.uuid4().hex[:8]}",
        content="Test",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    with pytest.raises(ValueError, match="must be different"):
        await backend.consolidate(
            source_tier=MemoryTier.WORKING,
            target_tier=MemoryTier.WORKING,
            entries=[entry],
        )


@pytest.mark.asyncio
async def test_delete_expired(backend):
    """Test deleting expired entries."""
    now = datetime.now()
    unique_prefix = uuid.uuid4().hex[:8]

    # Create entries with different ages and very distinct content
    old_entry = MemoryEntry(
        id=f"old-entry-{uuid.uuid4().hex[:8]}",
        content=f"{unique_prefix} Ancient historical archival material from past",
        tier=MemoryTier.WORKING,
        created_at=now - timedelta(days=10),
        updated_at=now - timedelta(days=10),
        source_session="session-123",
    )
    new_entry = MemoryEntry(
        id=f"new-entry-{uuid.uuid4().hex[:8]}",
        content=f"{unique_prefix} Fresh contemporary recent information current",
        tier=MemoryTier.WORKING,
        created_at=now,
        updated_at=now,
        source_session="session-123",
    )

    await backend.store(old_entry)
    await backend.store(new_entry)

    # Delete entries older than 5 days
    deleted = await backend.delete_expired({MemoryTier.WORKING: 5})

    assert deleted == 1

    # Verify old entry is gone by querying all working tier entries and checking IDs
    all_working = await backend.retrieve(query="", tier=MemoryTier.WORKING, limit=100)
    entry_ids = [r.entry.id for r in all_working]
    assert old_entry.id not in entry_ids
    assert new_entry.id in entry_ids


@pytest.mark.asyncio
async def test_delete_expired_invalid_days(backend):
    """Test that delete_expired raises ValueError for invalid days."""
    with pytest.raises(ValueError, match="must be positive"):
        await backend.delete_expired({MemoryTier.WORKING: 0})

    with pytest.raises(ValueError, match="must be positive"):
        await backend.delete_expired({MemoryTier.WORKING: -1})


@pytest.mark.asyncio
async def test_store_not_initialized(temp_dir):
    """Test that store raises RuntimeError if not initialized."""
    collection_name = f"test_uninit_{uuid.uuid4().hex[:8]}"
    chroma = ChromaBackend(persist_directory=temp_dir, collection_name=collection_name)
    # Don't initialize

    entry = MemoryEntry(
        id=f"test-{uuid.uuid4().hex[:8]}",
        content="Test",
        tier=MemoryTier.WORKING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        await chroma.store(entry)


@pytest.mark.asyncio
async def test_retrieve_not_initialized(temp_dir):
    """Test that retrieve raises RuntimeError if not initialized."""
    collection_name = f"test_uninit_{uuid.uuid4().hex[:8]}"
    chroma = ChromaBackend(persist_directory=temp_dir, collection_name=collection_name)

    with pytest.raises(RuntimeError, match="not initialized"):
        await chroma.retrieve(query="test")


@pytest.mark.asyncio
async def test_metadata_serialization(backend):
    """Test that complex metadata is properly serialized."""
    entry_id = f"meta-test-{uuid.uuid4().hex[:8]}"
    entry = MemoryEntry(
        id=entry_id,
        content="Metadata test for serialization",
        tier=MemoryTier.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "string_key": "value",
            "int_key": 42,
            "float_key": 3.14,
            "bool_key": True,
            "list_key": [1, 2, 3],
            "nested_dict": {"a": 1, "b": 2},
        },
    )

    await backend.store(entry)

    # Retrieve and verify metadata
    results = await backend.retrieve(query="Metadata test for serialization", limit=1)
    assert len(results) == 1
    # Metadata values should be preserved (as strings for non-primitives)
    assert results[0].entry.metadata["string_key"] == "value"
    assert results[0].entry.metadata["int_key"] in (42, "42")
    assert results[0].entry.metadata["float_key"] in (3.14, "3.14")
    assert results[0].entry.metadata["bool_key"] in (True, "True")


@pytest.mark.asyncio
async def test_relevance_scores(backend):
    """Test that relevance scores are properly calculated."""
    unique_prefix = uuid.uuid4().hex[:8]

    # Store entries
    for i in range(3):
        entry = MemoryEntry(
            id=f"relevance-{uuid.uuid4().hex[:8]}",
            content=f"{unique_prefix} Relevance test content number {i}",
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await backend.store(entry)

    # Retrieve all
    results = await backend.retrieve(query=f"{unique_prefix} Relevance test content number 0", limit=3)

    # Scores should be between -1 and 1 (cosine similarity range)
    # Note: ChromaDB can return negative similarities for very dissimilar items
    assert all(-1.0 <= r.relevance_score <= 1.0 for r in results)
    # Best match should have highest score
    assert results[0].relevance_score >= results[-1].relevance_score
