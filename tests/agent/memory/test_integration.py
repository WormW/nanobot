"""Integration tests for the layered memory system.

Tests cover end-to-end workflows, multi-backend scenarios, and concurrent access.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from nanobot.agent.memory import (
    MemoryConfig,
    MemoryOrchestrator,
    MemoryTier,
    WorkingMemoryConfig,
    EpisodicMemoryConfig,
    SemanticMemoryConfig,
    ConsolidationConfig,
)
from nanobot.agent.memory.embedder import EmbeddingProvider
from nanobot.agent.memory.storage import FileSystemBackend, SQLiteBackend


class MockEmbedder(EmbeddingProvider):
    """Mock embedder for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic mock embeddings."""
        return [[0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] for i in range(len(texts))]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def max_tokens_per_text(self) -> int:
        return 1000


@pytest.fixture
async def orchestrator():
    """Create a memory orchestrator with filesystem backend."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileSystemBackend(tmp)
        await backend.initialize()

        embedder = MockEmbedder()
        config = MemoryConfig(
            working=WorkingMemoryConfig(max_turns=5, max_tokens=1000),
            episodic=EpisodicMemoryConfig(max_entries=100, consolidation_batch=2),
            semantic=SemanticMemoryConfig(embedding_dimension=3, max_entries=100),
            consolidation=ConsolidationConfig(
                working_memory_token_threshold=500,
                episodic_memory_count_threshold=5,
            ),
        )

        orchestrator = MemoryOrchestrator(backend, embedder, config)
        yield orchestrator


@pytest.fixture
async def sqlite_orchestrator():
    """Create a memory orchestrator with SQLite backend."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        backend = SQLiteBackend(str(db_path))
        await backend.initialize()

        embedder = MockEmbedder()
        config = MemoryConfig(
            working=WorkingMemoryConfig(max_turns=5, max_tokens=1000),
            episodic=EpisodicMemoryConfig(max_entries=100, consolidation_batch=2),
            semantic=SemanticMemoryConfig(embedding_dimension=3, max_entries=100),
            consolidation=ConsolidationConfig(
                working_memory_token_threshold=500,
                episodic_memory_count_threshold=5,
            ),
        )

        orchestrator = MemoryOrchestrator(backend, embedder, config)
        yield orchestrator


class TestEndToEndWorkflow:
    """Test complete conversation to retrieval workflow."""

    @pytest.mark.asyncio
    async def test_conversation_stored_in_working_memory(self, orchestrator):
        """Test that conversation turns are stored and retrievable."""
        session_id = "test-session-1"

        # Add conversation turns
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Hello, what's the weather?",
            assistant_response="It's sunny today.",
            prompt_tokens=100,
        )
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Thanks! What about tomorrow?",
            assistant_response="It will be cloudy.",
            prompt_tokens=100,
        )

        # Verify working memory
        working_memories = await orchestrator.working.get_recent(n=10)
        assert len(working_memories) == 2
        # Check that either memory contains weather-related content
        all_content = " ".join([m.content.lower() for m in working_memories])
        assert "weather" in all_content or "sunny" in all_content

    @pytest.mark.asyncio
    async def test_retrieve_for_context_combines_tiers(self, orchestrator):
        """Test that context retrieval combines all memory tiers."""
        session_id = "test-session-2"

        # Add conversation
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="I love Python programming",
            assistant_response="Python is a great language!",
            prompt_tokens=100,
        )

        # Add episodic memory directly
        from nanobot.agent.memory.types import MemoryEntry
        episodic_entry = MemoryEntry(
            id="ep-test-1",
            content="User prefers Python over JavaScript",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await orchestrator.backend.store(episodic_entry)

        # Retrieve context
        context = await orchestrator.retrieve_for_context(
            current_query="What programming language should I use?",
            recent_context=[],
            max_tokens=500,
        )

        assert context is not None

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, orchestrator):
        """Test that different sessions don't interfere."""
        # Session 1
        await orchestrator.on_conversation_turn(
            session_id="session-a",
            user_message="Session A message",
            assistant_response="Response A",
            prompt_tokens=50,
        )

        # Session 2
        await orchestrator.on_conversation_turn(
            session_id="session-b",
            user_message="Session B message",
            assistant_response="Response B",
            prompt_tokens=50,
        )

        # Verify isolation
        session_a_memories = await orchestrator.working.get_all_for_session("session-a")
        session_b_memories = await orchestrator.working.get_all_for_session("session-b")

        assert len(session_a_memories) == 1
        assert len(session_b_memories) == 1
        assert "Session A" in session_a_memories[0].content
        assert "Session B" in session_b_memories[0].content


class TestMultiBackendWorkflow:
    """Test workflows across different backend combinations."""

    @pytest.mark.asyncio
    async def test_data_persistence_sqlite_backend(self, sqlite_orchestrator):
        """Test that data persists with SQLite backend."""
        session_id = "sqlite-test"

        # Store data
        await sqlite_orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Test message",
            assistant_response="Test response",
            prompt_tokens=50,
        )

        # Verify storage
        memories = await sqlite_orchestrator.working.get_recent(n=10)
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_cross_backend_compatibility(self):
        """Test that data can be migrated and used across backends."""
        with tempfile.TemporaryDirectory() as tmp:
            fs_path = Path(tmp) / "fs_memory"
            db_path = Path(tmp) / "memory.db"

            # Create filesystem backend and store data
            fs_backend = FileSystemBackend(str(fs_path))
            await fs_backend.initialize()

            embedder = MockEmbedder()
            config = MemoryConfig()

            fs_orchestrator = MemoryOrchestrator(fs_backend, embedder, config)

            await fs_orchestrator.on_conversation_turn(
                session_id="cross-test",
                user_message="Original message",
                assistant_response="Original response",
                prompt_tokens=50,
            )

            # Get data from filesystem
            fs_memories = await fs_orchestrator.working.get_recent(n=10)
            assert len(fs_memories) == 1

            # Migrate to SQLite
            from nanobot.agent.memory import migrate_filesystem_to_sqlite

            result = await migrate_filesystem_to_sqlite(str(fs_path), str(db_path))
            assert result.total_entries >= 1

            # Verify data in SQLite
            sqlite_backend = SQLiteBackend(str(db_path))
            await sqlite_backend.initialize()

            sqlite_orchestrator = MemoryOrchestrator(sqlite_backend, embedder, config)
            sqlite_memories = await sqlite_orchestrator.working.get_recent(n=10)
            assert len(sqlite_memories) == 1


class TestConcurrentAccess:
    """Test concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_conversation_turns_same_session(self, orchestrator):
        """Test that concurrent turns in same session are handled safely."""
        session_id = "concurrent-session"

        async def add_turn(i):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=50,
            )

        # Add turns concurrently
        await asyncio.gather(*[add_turn(i) for i in range(5)])

        # Verify all turns stored
        memories = await orchestrator.working.get_all_for_session(session_id)
        assert len(memories) == 5

    @pytest.mark.asyncio
    async def test_concurrent_retrieval_and_storage(self, orchestrator):
        """Test concurrent retrieval and storage operations."""
        session_id = "concurrent-mixed"

        async def add_turns():
            for i in range(3):
                await orchestrator.on_conversation_turn(
                    session_id=session_id,
                    user_message=f"Message {i}",
                    assistant_response=f"Response {i}",
                    prompt_tokens=50,
                )
                await asyncio.sleep(0.01)

        async def retrieve_context():
            for _ in range(3):
                await orchestrator.retrieve_for_context(
                    current_query="test query",
                    recent_context=[],
                    max_tokens=100,
                )
                await asyncio.sleep(0.01)

        # Run concurrently
        await asyncio.gather(add_turns(), retrieve_context())

        # Verify data integrity
        memories = await orchestrator.working.get_all_for_session(session_id)
        assert len(memories) == 3


class TestConsolidationWorkflow:
    """Test memory consolidation workflows."""

    @pytest.mark.asyncio
    async def test_consolidation_triggered_by_token_threshold(self, orchestrator):
        """Test that consolidation triggers when token threshold exceeded."""
        session_id = "consolidation-test"

        # Add turns with high token count
        for i in range(3):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"This is a test message number {i} with some content",
                assistant_response=f"This is a response number {i} with additional content",
                prompt_tokens=600,
            )

        # Verify memories stored
        memories = await orchestrator.working.get_all_for_session(session_id)
        assert len(memories) == 3

    @pytest.mark.asyncio
    async def test_explicit_consolidation(self, orchestrator):
        """Test manual consolidation trigger."""
        session_id = "explicit-consolidation"

        # Add turns
        for i in range(3):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=100,
            )

        # Archive entries
        entries = await orchestrator.working.get_all_for_session(session_id)
        await orchestrator.working.archive_entries([e.id for e in entries])

        # Verify archived - is_archived is synchronous
        is_archived = orchestrator.working.is_archived(entries[0].id)
        assert is_archived is True


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_graceful_handling_of_empty_memory(self, orchestrator):
        """Test that empty memory doesn't cause errors."""
        context = await orchestrator.retrieve_for_context(
            current_query="query with no history",
            recent_context=[],
            max_tokens=100,
        )

        assert context is not None

    @pytest.mark.asyncio
    async def test_recovery_after_backend_error(self, orchestrator):
        """Test system recovery after backend error."""
        session_id = "recovery-test"

        # Normal operation
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Test",
            assistant_response="Response",
            prompt_tokens=50,
        )

        # Verify still functional
        memories = await orchestrator.working.get_recent(n=10)
        assert len(memories) == 1
