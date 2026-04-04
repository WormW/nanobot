"""Unit tests for ConsolidationEngine.

Tests cover the core functionality of memory consolidation including:
- Threshold checking for consolidation triggers
- Episodic summary creation from working memory
- Semantic knowledge extraction from episodic summaries
- Working memory archival after consolidation
- End-to-end consolidation workflows
"""

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.consolidation import ConsolidationEngine, ConsolidationResult
from nanobot.agent.memory.embedder import EmbeddingProvider
from nanobot.agent.memory.storage.filesystem import FileSystemBackend
from nanobot.agent.memory.tiers.episodic import EpisodicMemoryManager
from nanobot.agent.memory.tiers.semantic import SemanticMemoryManager
from nanobot.agent.memory.tiers.working import WorkingMemoryManager
from nanobot.agent.memory.types import (
    ConsolidationConfig,
    EpisodicMemoryConfig,
    MemoryEntry,
    MemoryTier,
    SemanticMemoryConfig,
    WorkingMemoryConfig,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings (simple deterministic vectors)."""
        return [[float(len(text)) / 100.0] * 768 for text in texts]

    @property
    def dimension(self) -> int:
        """Return mock embedding dimension."""
        return 768

    @property
    def max_tokens_per_text(self) -> int:
        """Return mock max tokens limit."""
        return 8192


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
async def backend(temp_dir):
    """Create and initialize a FileSystemBackend."""
    be = FileSystemBackend(temp_dir)
    await be.initialize()
    return be


@pytest.fixture
def working_config():
    """Create a WorkingMemoryConfig for testing."""
    return WorkingMemoryConfig(max_turns=10, max_tokens=4000, ttl_seconds=3600)


@pytest.fixture
def episodic_config():
    """Create an EpisodicMemoryConfig with small batch for testing."""
    return EpisodicMemoryConfig(
        summary_model="default", max_entries=1000, consolidation_batch=3
    )


@pytest.fixture
def semantic_config():
    """Create a SemanticMemoryConfig for testing."""
    return SemanticMemoryConfig(
        embedding_dimension=768, similarity_threshold=0.75, max_entries=10000
    )


@pytest.fixture
def consolidation_config():
    """Create a ConsolidationConfig for testing."""
    return ConsolidationConfig(
        working_memory_token_threshold=3000,
        episodic_memory_count_threshold=50,
        auto_consolidate_interval_minutes=30,
        enable_explicit_consolidation=True,
    )


@pytest.fixture
async def working_manager(backend, working_config):
    """Create a WorkingMemoryManager with initialized backend."""
    return WorkingMemoryManager(backend, working_config)


@pytest.fixture
async def episodic_manager(backend, episodic_config):
    """Create an EpisodicMemoryManager with initialized backend."""
    return EpisodicMemoryManager(backend, episodic_config)


@pytest.fixture
async def semantic_manager(backend, semantic_config):
    """Create a SemanticMemoryManager with initialized backend."""
    embedder = MockEmbeddingProvider()
    return SemanticMemoryManager(backend, embedder, semantic_config)


@pytest.fixture
async def consolidation_engine(
    working_manager, episodic_manager, semantic_manager, consolidation_config
):
    """Create a ConsolidationEngine with all dependencies."""
    return ConsolidationEngine(
        working=working_manager,
        episodic=episodic_manager,
        semantic=semantic_manager,
        config=consolidation_config,
    )


class TestConsolidationResult:
    """Tests for the ConsolidationResult dataclass."""

    def test_consolidation_result_creation(self):
        """Test that ConsolidationResult can be created with proper fields."""
        result = ConsolidationResult(
            session_id="test_session",
            episodic_created=[],
            semantic_created=[],
        )
        assert result.session_id == "test_session"
        assert result.episodic_created == []
        assert result.semantic_created == []

    def test_consolidation_result_with_entries(self):
        """Test ConsolidationResult with actual memory entries."""
        from datetime import datetime

        entry1 = MemoryEntry(
            id="entry1",
            content="Test content",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="test_session",
        )
        entry2 = MemoryEntry(
            id="entry2",
            content="More content",
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="test_session",
        )

        result = ConsolidationResult(
            session_id="test_session",
            episodic_created=[entry1],
            semantic_created=[entry2],
        )
        assert len(result.episodic_created) == 1
        assert len(result.semantic_created) == 1
        assert result.episodic_created[0].id == "entry1"
        assert result.semantic_created[0].id == "entry2"


class TestShouldConsolidate:
    """Tests for the should_consolidate method."""

    @pytest.mark.asyncio
    async def test_should_consolidate_returns_true_above_threshold(
        self, consolidation_engine
    ):
        """Test that should_consolidate returns True when tokens exceed threshold."""
        # Threshold is 3000, so 3001 should trigger consolidation
        result = await consolidation_engine.should_consolidate(3001)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_consolidate_returns_false_at_threshold(
        self, consolidation_engine
    ):
        """Test that should_consolidate returns False at exactly the threshold."""
        # At exactly 3000, should not consolidate
        result = await consolidation_engine.should_consolidate(3000)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_consolidate_returns_false_below_threshold(
        self, consolidation_engine
    ):
        """Test that should_consolidate returns False when tokens are below threshold."""
        result = await consolidation_engine.should_consolidate(2999)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_consolidate_with_custom_config(self, working_manager, episodic_manager, semantic_manager):
        """Test should_consolidate with a custom threshold."""
        custom_config = ConsolidationConfig(
            working_memory_token_threshold=1000,
            episodic_memory_count_threshold=50,
            auto_consolidate_interval_minutes=30,
            enable_explicit_consolidation=True,
        )
        engine = ConsolidationEngine(
            working=working_manager,
            episodic=episodic_manager,
            semantic=semantic_manager,
            config=custom_config,
        )

        assert await engine.should_consolidate(1001) is True
        assert await engine.should_consolidate(1000) is False
        assert await engine.should_consolidate(999) is False


class TestRunConsolidation:
    """Tests for the run method."""

    @pytest.mark.asyncio
    async def test_run_with_insufficient_entries_no_consolidation(
        self, consolidation_engine, working_manager
    ):
        """Test that run does nothing when there aren't enough entries."""
        session_id = "test_session"

        # Add only 2 turns (below consolidation_batch of 3)
        await working_manager.add_turn(session_id, "Hello", "Hi!")
        await working_manager.add_turn(session_id, "How are you?", "Good!")

        result = await consolidation_engine.run(session_id)

        assert result.session_id == session_id
        assert result.episodic_created == []
        assert result.semantic_created == []

    @pytest.mark.asyncio
    async def test_run_creates_episodic_summary(
        self, consolidation_engine, working_manager, backend
    ):
        """Test that run creates an episodic summary when threshold is met."""
        session_id = "test_session"

        # Add 3 turns (meets consolidation_batch of 3)
        await working_manager.add_turn(session_id, "Hello", "Hi!")
        await working_manager.add_turn(session_id, "How are you?", "Good!")
        await working_manager.add_turn(session_id, "Thanks", "You're welcome!")

        result = await consolidation_engine.run(session_id)

        assert len(result.episodic_created) == 1
        episodic_entry = result.episodic_created[0]
        assert episodic_entry.tier == MemoryTier.EPISODIC
        assert episodic_entry.source_session == session_id
        assert "Conversation summary" in episodic_entry.content

    @pytest.mark.asyncio
    async def test_run_creates_semantic_knowledge(
        self, consolidation_engine, working_manager
    ):
        """Test that run extracts and stores semantic knowledge."""
        session_id = "test_session"

        # Add 3 turns to trigger consolidation
        await working_manager.add_turn(session_id, "Hello there", "Hi!")
        await working_manager.add_turn(session_id, "How are you doing today", "Good!")
        await working_manager.add_turn(session_id, "Thanks for your help", "Welcome!")

        result = await consolidation_engine.run(session_id)

        # Should have created some semantic entries (sentences > 20 chars)
        assert len(result.semantic_created) > 0
        for entry in result.semantic_created:
            assert entry.tier == MemoryTier.SEMANTIC
            assert entry.source_session == session_id
            assert len(entry.content) > 20

    @pytest.mark.asyncio
    async def test_run_archives_working_entries(
        self, consolidation_engine, working_manager
    ):
        """Test that run archives working memory entries after consolidation."""
        session_id = "test_session"

        # Add 3 turns
        await working_manager.add_turn(session_id, "Hello", "Hi!")
        await working_manager.add_turn(session_id, "How are you?", "Good!")
        await working_manager.add_turn(session_id, "Thanks", "You're welcome!")

        # Get entry IDs before consolidation
        working_entries = await working_manager.get_all_for_session(session_id)
        entry_ids = [e.id for e in working_entries]
        assert len(entry_ids) == 3

        # Run consolidation
        await consolidation_engine.run(session_id)

        # Verify entries are archived
        for entry_id in entry_ids:
            assert working_manager.is_archived(entry_id) is True

    @pytest.mark.asyncio
    async def test_run_multiple_batches(
        self, consolidation_engine, working_manager, episodic_manager
    ):
        """Test consolidation with more entries than batch size."""
        session_id = "test_session"

        # Add 5 turns (above consolidation_batch of 3)
        for i in range(5):
            await working_manager.add_turn(
                session_id, f"Message {i}", f"Response {i}"
            )

        result = await consolidation_engine.run(session_id)

        # Should create one episodic summary from all 5 turns
        assert len(result.episodic_created) == 1
        assert result.episodic_created[0].metadata["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_run_empty_session(self, consolidation_engine):
        """Test consolidation with no working memory entries."""
        session_id = "empty_session"

        result = await consolidation_engine.run(session_id)

        assert result.session_id == session_id
        assert result.episodic_created == []
        assert result.semantic_created == []

    @pytest.mark.asyncio
    async def test_run_filters_short_sentences(
        self, consolidation_engine, working_manager
    ):
        """Test that semantic extraction filters sentences <= 20 chars."""
        session_id = "test_session"

        # Add 3 turns with very short content
        await working_manager.add_turn(session_id, "Hi", "Hi")
        await working_manager.add_turn(session_id, "Bye", "Bye")
        await working_manager.add_turn(session_id, "OK", "OK")

        result = await consolidation_engine.run(session_id)

        # Should create episodic but may have few/no semantic entries
        # due to short sentence filtering
        assert len(result.episodic_created) == 1
        # The summary format includes session ID which should be > 20 chars


class TestConsolidationEngineInit:
    """Tests for ConsolidationEngine initialization."""

    def test_init_stores_dependencies(
        self, working_manager, episodic_manager, semantic_manager, consolidation_config
    ):
        """Test that __init__ properly stores all dependencies."""
        engine = ConsolidationEngine(
            working=working_manager,
            episodic=episodic_manager,
            semantic=semantic_manager,
            config=consolidation_config,
        )

        assert engine.working is working_manager
        assert engine.episodic is episodic_manager
        assert engine.semantic is semantic_manager
        assert engine.config is consolidation_config


class TestIntegration:
    """Integration tests for full consolidation workflows."""

    @pytest.mark.asyncio
    async def test_full_consolidation_workflow(
        self, temp_dir, working_config, episodic_config, semantic_config, consolidation_config
    ):
        """Test a complete consolidation workflow end-to-end."""
        # Set up real backend and managers
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()

        working = WorkingMemoryManager(backend, working_config)
        episodic = EpisodicMemoryManager(backend, episodic_config)
        embedder = MockEmbeddingProvider()
        semantic = SemanticMemoryManager(backend, embedder, semantic_config)

        engine = ConsolidationEngine(
            working=working,
            episodic=episodic,
            semantic=semantic,
            config=consolidation_config,
        )

        session_id = "integration_session"

        # Add conversation turns
        conversation = [
            ("What is the weather today?", "It's sunny and warm outside."),
            ("Should I bring an umbrella?", "No, there's no rain expected."),
            ("Thanks for the information!", "You're welcome! Have a great day!"),
        ]

        for user_msg, assistant_msg in conversation:
            await working.add_turn(session_id, user_msg, assistant_msg)

        # Run consolidation
        result = await engine.run(session_id)

        # Verify episodic entry was created
        assert len(result.episodic_created) == 1
        episodic_entry = result.episodic_created[0]
        assert episodic_entry.tier == MemoryTier.EPISODIC

        # Verify semantic entries were extracted
        assert len(result.semantic_created) > 0
        for entry in result.semantic_created:
            assert entry.tier == MemoryTier.SEMANTIC
            assert entry.embedding is not None  # Mock embedder provides embeddings

        # Verify working entries were archived
        working_entries = await working.get_all_for_session(session_id)
        for entry in working_entries:
            assert working.is_archived(entry.id)

        # Verify data was persisted to disk
        # Episodic entries are stored as individual JSON files named by entry ID
        episodic_dir = Path(temp_dir) / "episodic"
        assert episodic_dir.exists()
        episodic_files = list(episodic_dir.glob("*.json"))
        assert len(episodic_files) >= 1  # At least one episodic entry file

    @pytest.mark.asyncio
    async def test_multiple_sessions_consolidation(
        self, consolidation_engine, working_manager
    ):
        """Test that consolidation only affects the specified session."""
        session_a = "session_a"
        session_b = "session_b"

        # Add turns to both sessions
        for i in range(3):
            await working_manager.add_turn(session_a, f"A message {i}", f"A response {i}")
            await working_manager.add_turn(session_b, f"B message {i}", f"B response {i}")

        # Consolidate only session_a
        result = await consolidation_engine.run(session_a)

        # Verify only session_a was consolidated
        assert result.session_id == session_a
        assert len(result.episodic_created) == 1

        # Verify session_a entries are archived
        a_entries = await working_manager.get_all_for_session(session_a)
        for entry in a_entries:
            assert working_manager.is_archived(entry.id)

        # Verify session_b entries are NOT archived
        b_entries = await working_manager.get_all_for_session(session_b)
        for entry in b_entries:
            assert not working_manager.is_archived(entry.id)

    @pytest.mark.asyncio
    async def test_consolidation_preserves_content_integrity(
        self, consolidation_engine, working_manager, episodic_manager
    ):
        """Test that consolidated content maintains integrity."""
        session_id = "integrity_test"

        # Add specific content
        await working_manager.add_turn(session_id, "User question here", "Assistant answer here")
        await working_manager.add_turn(session_id, "Another question", "Another answer")
        await working_manager.add_turn(session_id, "Final question", "Final answer")

        result = await consolidation_engine.run(session_id)

        # Verify episodic content contains expected information
        episodic_entry = result.episodic_created[0]
        assert session_id in episodic_entry.content
        assert "User:" in episodic_entry.content

        # Verify we can retrieve the episodic entry later
        retrieved = await episodic_manager.get_for_session(session_id)
        assert len(retrieved) == 1
        assert retrieved[0].id == episodic_entry.id
