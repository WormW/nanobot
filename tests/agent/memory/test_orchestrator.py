"""Unit tests for MemoryOrchestrator.

Tests cover the core functionality of the memory orchestrator including:
- Initialization of tier managers and consolidation engine
- Conversation turn processing and storage
- Consolidation triggering based on token thresholds
- Context retrieval from all memory tiers
- Integration of working, episodic, and semantic memories
"""

import tempfile

import pytest

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.embedder import EmbeddingProvider
from nanobot.agent.memory.orchestrator import MemoryOrchestrator
from nanobot.agent.memory.storage.filesystem import FileSystemBackend
from nanobot.agent.memory.context_builder import RetrievalContext
from nanobot.agent.memory.types import (
    ConsolidationConfig,
    EpisodicMemoryConfig,
    MemoryConfig,
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
def embedder():
    """Create a MockEmbeddingProvider."""
    return MockEmbeddingProvider()


@pytest.fixture
def memory_config():
    """Create a MemoryConfig for testing."""
    return MemoryConfig(
        working=WorkingMemoryConfig(max_turns=5, max_tokens=1000, ttl_seconds=3600),
        episodic=EpisodicMemoryConfig(
            summary_model="default", max_entries=100, consolidation_batch=3
        ),
        semantic=SemanticMemoryConfig(
            embedding_dimension=768, similarity_threshold=0.75, max_entries=1000
        ),
        consolidation=ConsolidationConfig(
            working_memory_token_threshold=100,
            episodic_memory_count_threshold=50,
            auto_consolidate_interval_minutes=30,
            enable_explicit_consolidation=True,
        ),
    )


@pytest.fixture
async def orchestrator(backend, embedder, memory_config):
    """Create a MemoryOrchestrator with all dependencies."""
    return MemoryOrchestrator(
        backend=backend,
        embedder=embedder,
        config=memory_config,
    )


class TestMemoryOrchestratorInit:
    """Tests for MemoryOrchestrator initialization."""

    @pytest.mark.asyncio
    async def test_init_stores_dependencies(self, backend, embedder, memory_config):
        """Test that __init__ properly stores all dependencies."""
        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=memory_config,
        )

        assert orchestrator.backend is backend
        assert orchestrator.embedder is embedder
        assert orchestrator.config is memory_config

    @pytest.mark.asyncio
    async def test_init_creates_working_manager(self, backend, embedder, memory_config):
        """Test that __init__ creates WorkingMemoryManager."""
        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=memory_config,
        )

        from nanobot.agent.memory.tiers.working import WorkingMemoryManager

        assert isinstance(orchestrator.working, WorkingMemoryManager)
        assert orchestrator.working.config == memory_config.working

    @pytest.mark.asyncio
    async def test_init_creates_episodic_manager(self, backend, embedder, memory_config):
        """Test that __init__ creates EpisodicMemoryManager."""
        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=memory_config,
        )

        from nanobot.agent.memory.tiers.episodic import EpisodicMemoryManager

        assert isinstance(orchestrator.episodic, EpisodicMemoryManager)
        assert orchestrator.episodic.config == memory_config.episodic

    @pytest.mark.asyncio
    async def test_init_creates_semantic_manager(self, backend, embedder, memory_config):
        """Test that __init__ creates SemanticMemoryManager."""
        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=memory_config,
        )

        from nanobot.agent.memory.tiers.semantic import SemanticMemoryManager

        assert isinstance(orchestrator.semantic, SemanticMemoryManager)
        assert orchestrator.semantic.config == memory_config.semantic
        assert orchestrator.semantic.embedder is embedder

    @pytest.mark.asyncio
    async def test_init_creates_consolidation_engine(
        self, backend, embedder, memory_config
    ):
        """Test that __init__ creates ConsolidationEngine."""
        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=memory_config,
        )

        from nanobot.agent.memory.consolidation import ConsolidationEngine

        assert isinstance(orchestrator.consolidator, ConsolidationEngine)
        assert orchestrator.consolidator.working is orchestrator.working
        assert orchestrator.consolidator.episodic is orchestrator.episodic
        assert orchestrator.consolidator.semantic is orchestrator.semantic
        assert orchestrator.consolidator.config == memory_config.consolidation

    @pytest.mark.asyncio
    async def test_init_creates_context_builder(self, backend, embedder, memory_config):
        """Test that __init__ creates MixedContextBuilder."""
        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=memory_config,
        )

        from nanobot.agent.memory.context_builder import MixedContextBuilder

        assert isinstance(orchestrator.context_builder, MixedContextBuilder)


class TestOnConversationTurn:
    """Tests for the on_conversation_turn method."""

    @pytest.mark.asyncio
    async def test_stores_turn_in_working_memory(self, orchestrator):
        """Test that conversation turn is stored in working memory."""
        session_id = "test_session"
        user_message = "Hello"
        assistant_response = "Hi there!"
        prompt_tokens = 50  # Below threshold

        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            prompt_tokens=prompt_tokens,
        )

        # Verify turn was stored
        entries = await orchestrator.working.get_all_for_session(session_id)
        assert len(entries) == 1

        import json

        content = json.loads(entries[0].content)
        assert content["user"] == user_message
        assert content["assistant"] == assistant_response

    @pytest.mark.asyncio
    async def test_does_not_trigger_consolidation_below_threshold(self, orchestrator):
        """Test that consolidation is not triggered when below token threshold."""
        session_id = "test_session"
        prompt_tokens = 50  # Below threshold of 100

        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Hello",
            assistant_response="Hi!",
            prompt_tokens=prompt_tokens,
        )

        # Verify no consolidation occurred (entries not archived)
        entries = await orchestrator.working.get_all_for_session(session_id)
        assert len(entries) == 1
        assert not orchestrator.working.is_archived(entries[0].id)

    @pytest.mark.asyncio
    async def test_triggers_consolidation_above_threshold(self, orchestrator):
        """Test that consolidation is triggered when above token threshold."""
        session_id = "test_session"

        # Add enough turns to meet consolidation batch
        for i in range(3):
            await orchestrator.working.add_turn(
                session_id, f"Message {i}", f"Response {i}"
            )

        # Get entry IDs before consolidation
        entries_before = await orchestrator.working.get_all_for_session(session_id)
        original_ids = {e.id for e in entries_before}
        assert len(original_ids) == 3

        # Trigger consolidation with high token count
        prompt_tokens = 150  # Above threshold of 100

        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Trigger consolidation",
            assistant_response="OK",
            prompt_tokens=prompt_tokens,
        )

        # Verify consolidation occurred (original entries archived)
        # The new turn is added before consolidation, so all 4 entries get archived
        all_entries = await orchestrator.working.get_all_for_session(session_id)
        archived_count = sum(
            1 for e in all_entries if orchestrator.working.is_archived(e.id)
        )
        # All entries for the session are archived during consolidation
        assert archived_count == 4

    @pytest.mark.asyncio
    async def test_multiple_turns_same_session(self, orchestrator):
        """Test adding multiple turns to the same session."""
        session_id = "test_session"

        for i in range(5):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=50,
            )

        entries = await orchestrator.working.get_all_for_session(session_id)
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_turns_different_sessions(self, orchestrator):
        """Test adding turns to different sessions."""
        await orchestrator.on_conversation_turn(
            session_id="session_a",
            user_message="Hello A",
            assistant_response="Hi A",
            prompt_tokens=50,
        )
        await orchestrator.on_conversation_turn(
            session_id="session_b",
            user_message="Hello B",
            assistant_response="Hi B",
            prompt_tokens=50,
        )

        entries_a = await orchestrator.working.get_all_for_session("session_a")
        entries_b = await orchestrator.working.get_all_for_session("session_b")

        assert len(entries_a) == 1
        assert len(entries_b) == 1


class TestRetrieveForContext:
    """Tests for the retrieve_for_context method."""

    @pytest.mark.asyncio
    async def test_returns_retrieval_context(self, orchestrator):
        """Test that retrieve_for_context returns a RetrievalContext."""
        context = await orchestrator.retrieve_for_context(
            current_query="test query",
            recent_context=[],
            max_tokens=1000,
        )

        assert isinstance(context, RetrievalContext)

    @pytest.mark.asyncio
    async def test_includes_working_memory(self, orchestrator):
        """Test that retrieval includes recent working memory."""
        # Add some working memory entries
        await orchestrator.working.add_turn("session_1", "Hello", "Hi!")
        await orchestrator.working.add_turn("session_1", "How are you?", "Good!")

        context = await orchestrator.retrieve_for_context(
            current_query="hello",
            recent_context=[],
            max_tokens=1000,
        )

        # Working memory should be included in the natural section
        assert context.has_content()
        assert "你们刚才聊到" in context.natural_section

    @pytest.mark.asyncio
    async def test_searches_episodic_memory(self, orchestrator):
        """Test that retrieval searches episodic memory."""
        # Create an episodic entry
        from datetime import datetime
        from nanobot.agent.memory.types import MemoryEntry, MemoryTier

        episodic_entry = MemoryEntry(
            id="episodic-1",
            content="Previous conversation about Python programming",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="session_1",
        )
        await orchestrator.backend.store(episodic_entry)

        context = await orchestrator.retrieve_for_context(
            current_query="Python",
            recent_context=[],
            max_tokens=1000,
        )

        # Should have debug sources showing episodic retrieval
        assert len(context.debug_sources) > 0
        episodic_sources = [
            s for s in context.debug_sources if s["tier"] == "episodic"
        ]
        assert len(episodic_sources) >= 1

    @pytest.mark.asyncio
    async def test_searches_semantic_memory(self, orchestrator):
        """Test that retrieval searches semantic memory with embedding."""
        # Store some knowledge
        await orchestrator.semantic.store_knowledge(
            content="Python is a programming language",
            source_session="session_1",
        )

        context = await orchestrator.retrieve_for_context(
            current_query="programming language",
            recent_context=[],
            max_tokens=1000,
        )

        # Should have debug sources showing semantic retrieval
        assert len(context.debug_sources) > 0
        semantic_sources = [
            s for s in context.debug_sources if s["tier"] == "semantic"
        ]
        assert len(semantic_sources) >= 1

    @pytest.mark.asyncio
    async def test_combines_all_sources(self, orchestrator):
        """Test that retrieval combines working, episodic, and semantic sources."""
        # Add working memory
        await orchestrator.working.add_turn("session_1", "Hello", "Hi!")

        # Add episodic memory
        from datetime import datetime
        from nanobot.agent.memory.types import MemoryEntry, MemoryTier

        episodic_entry = MemoryEntry(
            id="episodic-1",
            content="Previous greeting conversation",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session="session_1",
        )
        await orchestrator.backend.store(episodic_entry)

        # Add semantic memory
        await orchestrator.semantic.store_knowledge(
            content="Greetings are social interactions",
            source_session="session_1",
        )

        context = await orchestrator.retrieve_for_context(
            current_query="greeting",
            recent_context=[],
            max_tokens=1000,
        )

        # Should have content from multiple sources
        assert context.has_content()
        assert len(context.debug_sources) >= 2  # At least episodic and semantic

    @pytest.mark.asyncio
    async def test_respects_max_tokens(self, orchestrator):
        """Test that retrieval respects max_tokens parameter."""
        # Add some working memory
        await orchestrator.working.add_turn("session_1", "Hello", "Hi!")

        context = await orchestrator.retrieve_for_context(
            current_query="test",
            recent_context=[],
            max_tokens=100,  # Small limit
        )

        # Should still return a valid context
        assert isinstance(context, RetrievalContext)

    @pytest.mark.asyncio
    async def test_empty_query_returns_context(self, orchestrator):
        """Test that empty query still returns a valid context."""
        context = await orchestrator.retrieve_for_context(
            current_query="",
            recent_context=[],
            max_tokens=1000,
        )

        assert isinstance(context, RetrievalContext)


class TestIntegration:
    """Integration tests for MemoryOrchestrator workflows."""

    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self, temp_dir, embedder):
        """Test a complete conversation workflow with consolidation."""
        config = MemoryConfig(
            working=WorkingMemoryConfig(max_turns=10, max_tokens=1000, ttl_seconds=3600),
            episodic=EpisodicMemoryConfig(
                summary_model="default", max_entries=100, consolidation_batch=3
            ),
            semantic=SemanticMemoryConfig(
                embedding_dimension=768, similarity_threshold=0.75, max_entries=1000
            ),
            consolidation=ConsolidationConfig(
                working_memory_token_threshold=50,
                episodic_memory_count_threshold=50,
                auto_consolidate_interval_minutes=30,
                enable_explicit_consolidation=True,
            ),
        )

        backend = FileSystemBackend(temp_dir)
        await backend.initialize()

        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=config,
        )

        session_id = "integration_session"

        # Simulate conversation turns
        conversation = [
            ("Hello!", "Hi there! How can I help?"),
            ("What's the weather?", "It's sunny today!"),
            ("Thanks!", "You're welcome!"),
        ]

        # Add turns below threshold first
        for user_msg, assistant_msg in conversation:
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=user_msg,
                assistant_response=assistant_msg,
                prompt_tokens=30,  # Below threshold
            )

        # Verify all turns are in working memory
        working_entries = await orchestrator.working.get_all_for_session(session_id)
        assert len(working_entries) == 3

        # Trigger consolidation with high token count
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Trigger consolidation",
            assistant_response="OK",
            prompt_tokens=100,  # Above threshold
        )

        # Verify consolidation occurred
        archived_count = sum(
            1 for e in working_entries if orchestrator.working.is_archived(e.id)
        )
        assert archived_count == 3

        # Verify episodic memory was created
        episodic_entries = await orchestrator.episodic.get_for_session(session_id)
        assert len(episodic_entries) >= 1

        # Verify semantic memory was created
        semantic_results = await orchestrator.semantic.search("weather", limit=10)
        assert len(semantic_results) > 0

    @pytest.mark.asyncio
    async def test_retrieve_after_consolidation(self, temp_dir, embedder):
        """Test retrieving context after consolidation has occurred."""
        config = MemoryConfig(
            working=WorkingMemoryConfig(max_turns=10, max_tokens=1000, ttl_seconds=3600),
            episodic=EpisodicMemoryConfig(
                summary_model="default", max_entries=100, consolidation_batch=2
            ),
            semantic=SemanticMemoryConfig(
                embedding_dimension=768, similarity_threshold=0.75, max_entries=1000
            ),
            consolidation=ConsolidationConfig(
                working_memory_token_threshold=50,
                episodic_memory_count_threshold=50,
                auto_consolidate_interval_minutes=30,
                enable_explicit_consolidation=True,
            ),
        )

        backend = FileSystemBackend(temp_dir)
        await backend.initialize()

        orchestrator = MemoryOrchestrator(
            backend=backend,
            embedder=embedder,
            config=config,
        )

        session_id = "retrieve_test"

        # Add and consolidate some memories
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="I love Python programming",
            assistant_response="Python is a great language!",
            prompt_tokens=100,  # Trigger consolidation
        )
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Tell me more",
            assistant_response="Python is versatile and easy to learn.",
            prompt_tokens=100,  # Trigger consolidation
        )

        # Retrieve context
        context = await orchestrator.retrieve_for_context(
            current_query="Python programming",
            recent_context=[],
            max_tokens=1000,
        )

        # Should have content from consolidated memories
        assert context.has_content()

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, orchestrator):
        """Test that multiple sessions remain isolated."""
        session_a = "session_a"
        session_b = "session_b"

        # Add turns to both sessions
        for i in range(3):
            await orchestrator.on_conversation_turn(
                session_id=session_a,
                user_message=f"A message {i}",
                assistant_response=f"A response {i}",
                prompt_tokens=50,
            )
            await orchestrator.on_conversation_turn(
                session_id=session_b,
                user_message=f"B message {i}",
                assistant_response=f"B response {i}",
                prompt_tokens=50,
            )

        # Verify isolation
        entries_a = await orchestrator.working.get_all_for_session(session_a)
        entries_b = await orchestrator.working.get_all_for_session(session_b)

        assert len(entries_a) == 3
        assert len(entries_b) == 3
        assert all(e.source_session == session_a for e in entries_a)
        assert all(e.source_session == session_b for e in entries_b)

    @pytest.mark.asyncio
    async def test_context_builder_integration(self, orchestrator):
        """Test that context builder properly formats retrieved memories."""
        # Add working memory
        await orchestrator.working.add_turn("session_1", "Hello", "Hi!")

        # Retrieve context
        context = await orchestrator.retrieve_for_context(
            current_query="hello",
            recent_context=[],
            max_tokens=1000,
        )

        # Verify context structure
        assert isinstance(context.natural_section, str)
        assert isinstance(context.structured_facts, list)
        assert isinstance(context.debug_sources, list)

        # Verify to_system_prompt_addition works
        prompt_addition = context.to_system_prompt_addition()
        assert isinstance(prompt_addition, str)
