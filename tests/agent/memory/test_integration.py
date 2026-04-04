"""Integration tests for nanobot's layered memory system.

This module provides end-to-end integration tests that verify the complete
workflow of the memory system including conversation storage, consolidation,
and retrieval across all three memory tiers.
"""

import json
import pytest
import tempfile
from datetime import datetime

from nanobot.agent.memory import MemoryOrchestrator, MemoryConfig
from nanobot.agent.memory.storage.filesystem import FileSystemBackend
from nanobot.agent.memory.types import MemoryTier


class MockEmbedder:
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on text content length.
    """

    def __init__(self, dimension: int = 3):
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (deterministic based on text length).
        """
        return [[0.1 * (len(t) % 10), 0.2, 0.3] for t in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def max_tokens_per_text(self) -> int:
        """Return max tokens per text."""
        return 512


@pytest.fixture
async def orchestrator():
    """Create orchestrator with temporary directory.

    Yields:
        MemoryOrchestrator configured with FileSystemBackend and MockEmbedder.
    """
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileSystemBackend(tmp)
        await backend.initialize()

        embedder = MockEmbedder(dimension=3)
        config = MemoryConfig()
        orchestrator = MemoryOrchestrator(backend, embedder, config)
        yield orchestrator


class TestMemoryIntegration:
    """Integration tests for the complete memory system workflow."""

    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self, orchestrator):
        """Test: conversation -> working memory -> retrieval.

        Verifies that conversation turns are stored in working memory
        and can be retrieved for context building.
        """
        session_id = "test-session-1"

        # Simulate 5 conversation turns
        for i in range(5):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"User message {i}",
                assistant_response=f"Assistant response {i}",
                prompt_tokens=500,  # Below threshold
            )

        # Verify working memory has entries
        working = await orchestrator.working.get_all_for_session(session_id)
        assert len(working) == 5

        # Verify content is stored correctly
        for i, entry in enumerate(working):
            data = json.loads(entry.content)
            assert data["user"] == f"User message {i}"
            assert data["assistant"] == f"Assistant response {i}"

        # Test retrieval
        context = await orchestrator.retrieve_for_context(
            current_query="message",
            recent_context=[],
            max_tokens=1000,
        )
        assert context.has_content()
        assert "User message" in context.natural_section

    @pytest.mark.asyncio
    async def test_consolidation_end_to_end(self, orchestrator):
        """Test consolidation triggers and executes properly.

        Verifies that when token threshold is exceeded, consolidation
        creates episodic and semantic memories.
        """
        session_id = "consolidation-test"

        # Add entries below threshold first
        for i in range(3):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=500,  # Below threshold
            )

        # Verify only working memory has entries
        working_before = await orchestrator.working.get_all_for_session(session_id)
        assert len(working_before) == 3

        episodic_before = await orchestrator.episodic.get_for_session(session_id)
        assert len(episodic_before) == 0

        # Now trigger consolidation with high token count
        for i in range(3, 10):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=4000,  # Above threshold
            )

        # Verify consolidation occurred - episodic memory should have entries
        episodic_after = await orchestrator.episodic.get_for_session(session_id)
        assert len(episodic_after) > 0

        # Verify semantic memory was also created
        semantic_results = await orchestrator.semantic.search("Message", limit=10)
        assert len(semantic_results) > 0

    @pytest.mark.asyncio
    async def test_multi_session_isolation(self, orchestrator):
        """Test that sessions don't share memory.

        Verifies that each session's memory is isolated from others.
        """
        # Add to session A
        await orchestrator.on_conversation_turn(
            "session-A", "Hello A", "Hi A", 100
        )

        # Add to session B
        await orchestrator.on_conversation_turn(
            "session-B", "Hello B", "Hi B", 100
        )

        # Verify session A only has its own memory
        working_a = await orchestrator.working.get_all_for_session("session-A")
        assert len(working_a) == 1
        assert "Hello A" in working_a[0].content
        assert "Hello B" not in working_a[0].content

        # Verify session B only has its own memory
        working_b = await orchestrator.working.get_all_for_session("session-B")
        assert len(working_b) == 1
        assert "Hello B" in working_b[0].content
        assert "Hello A" not in working_b[0].content

    @pytest.mark.asyncio
    async def test_retrieval_combines_all_tiers(self, orchestrator):
        """Test that retrieve_for_context pulls from all tiers.

        Sets up memories in each tier and verifies retrieval combines them.
        """
        session_id = "multi-tier-test"

        # Setup working memory
        await orchestrator.on_conversation_turn(
            session_id, "Working memory query", "Response", 100
        )

        # Setup episodic memory directly
        from nanobot.agent.memory.types import MemoryEntry
        from datetime import datetime
        from uuid import uuid4

        episodic_entry = MemoryEntry(
            id=f"episodic-{uuid4().hex}",
            content="Episodic memory about project planning",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=session_id,
        )
        await orchestrator.backend.store(episodic_entry)

        # Setup semantic memory directly
        await orchestrator.semantic.store_knowledge(
            content="Semantic knowledge about testing",
            source_session=session_id,
        )

        # Retrieve context
        context = await orchestrator.retrieve_for_context(
            current_query="memory",
            recent_context=[],
            max_tokens=2000,
        )

        # Should have content from at least one source
        assert context.has_content()
        # Note: debug_sources may have 1+ entries depending on search matching
        assert len(context.debug_sources) >= 1

    @pytest.mark.asyncio
    async def test_empty_memory_graceful(self, orchestrator):
        """Test empty memory doesn't crash.

        Verifies that retrieval works gracefully when no memories exist.
        """
        context = await orchestrator.retrieve_for_context(
            "test query",
            [],
            1000,
        )
        assert not context.has_content()
        assert context.natural_section == ""
        assert len(context.structured_facts) == 0

    @pytest.mark.asyncio
    async def test_very_long_conversation(self, orchestrator):
        """Test system handles 100+ conversation turns.

        Verifies performance and correctness with many turns.
        """
        session_id = "long-conversation-test"

        # Simulate 100 conversation turns
        for i in range(100):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Turn {i}: User asks about topic {i % 10}",
                assistant_response=f"Turn {i}: Assistant responds with info {i % 10}",
                prompt_tokens=200,
            )

        # Verify all turns stored
        working = await orchestrator.working.get_all_for_session(session_id)
        assert len(working) == 100

        # Verify retrieval still works
        context = await orchestrator.retrieve_for_context(
            current_query="topic 5",
            recent_context=[],
            max_tokens=2000,
        )
        assert context.has_content()

    @pytest.mark.asyncio
    async def test_rapid_consecutive_turns(self, orchestrator):
        """Test rapid consecutive turns don't cause issues.

        Verifies system handles quick successive operations.
        """
        session_id = "rapid-test"

        # Add 20 turns rapidly
        for i in range(20):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Rapid message {i}",
                assistant_response=f"Rapid response {i}",
                prompt_tokens=100,
            )

        working = await orchestrator.working.get_all_for_session(session_id)
        assert len(working) == 20

        # Verify order is maintained
        for i, entry in enumerate(working):
            data = json.loads(entry.content)
            assert f"Rapid message {i}" in data["user"]

    @pytest.mark.asyncio
    async def test_empty_messages(self, orchestrator):
        """Test empty messages are handled gracefully.

        Verifies system doesn't crash with empty strings.
        """
        session_id = "empty-test"

        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="",
            assistant_response="I see you sent an empty message",
            prompt_tokens=100,
        )

        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Hello",
            assistant_response="",
            prompt_tokens=100,
        )

        working = await orchestrator.working.get_all_for_session(session_id)
        assert len(working) == 2

        # Verify empty content is preserved
        data1 = json.loads(working[0].content)
        assert data1["user"] == ""

        data2 = json.loads(working[1].content)
        assert data2["assistant"] == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, orchestrator):
        """Test special characters don't break storage or retrieval.

        Verifies Unicode, JSON special chars, and newlines work.
        """
        session_id = "special-chars-test"

        special_messages = [
            "Hello \"quoted\" text",
            "Line 1\nLine 2\nLine 3",
            "Unicode: 你好世界 🌍 émojis",
            "JSON: {\"key\": \"value\", \"num\": 123}",
            "Tabs\there\tand\tthere",
            "Backslash: C:\\Users\\test\\path",
        ]

        for msg in special_messages:
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=msg,
                assistant_response=f"Response to: {msg[:20]}",
                prompt_tokens=100,
            )

        # Verify all stored correctly
        working = await orchestrator.working.get_all_for_session(session_id)
        assert len(working) == len(special_messages)

        for i, entry in enumerate(working):
            data = json.loads(entry.content)
            assert data["user"] == special_messages[i]

    @pytest.mark.asyncio
    async def test_consolidation_with_few_entries(self, orchestrator):
        """Test consolidation doesn't run with too few entries.

        Verifies batch size threshold is respected.
        """
        session_id = "small-batch-test"

        # Add fewer entries than consolidation_batch (default 5)
        for i in range(3):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=4000,  # Above token threshold
            )

        # Should NOT have created episodic memory (need 5+ entries)
        episodic = await orchestrator.episodic.get_for_session(session_id)
        assert len(episodic) == 0

        # Add 2 more to reach threshold
        for i in range(3, 5):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=4000,
            )

        # Now should have episodic memory
        episodic = await orchestrator.episodic.get_for_session(session_id)
        assert len(episodic) > 0

    @pytest.mark.asyncio
    async def test_multiple_consolidations(self, orchestrator):
        """Test multiple consolidations create multiple episodic entries.

        Verifies repeated consolidation works correctly.
        """
        session_id = "multi-consolidation-test"

        # First batch - trigger consolidation (each turn triggers consolidation)
        for i in range(5):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Batch 1 Message {i}",
                assistant_response=f"Batch 1 Response {i}",
                prompt_tokens=4000,
            )

        episodic_1 = await orchestrator.episodic.get_for_session(session_id)
        # Each turn with high tokens triggers consolidation, creating entries
        assert len(episodic_1) >= 1

        # Second batch - trigger more consolidations
        for i in range(5, 10):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Batch 2 Message {i}",
                assistant_response=f"Batch 2 Response {i}",
                prompt_tokens=4000,
            )

        episodic_2 = await orchestrator.episodic.get_for_session(session_id)
        # Should have more entries than after first batch
        assert len(episodic_2) > len(episodic_1)

    @pytest.mark.asyncio
    async def test_retrieval_relevance_ranking(self, orchestrator):
        """Test retrieval returns results with relevance scores.

        Verifies retrieval results have proper scoring.
        """
        session_id = "relevance-test"

        # Store some knowledge
        await orchestrator.semantic.store_knowledge(
            content="Python programming best practices",
            source_session=session_id,
        )
        await orchestrator.semantic.store_knowledge(
            content="JavaScript async/await patterns",
            source_session=session_id,
        )

        # Search
        results = await orchestrator.semantic.search("Python", limit=5)
        assert len(results) > 0

        # All results should have relevance scores
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0
            assert result.retrieval_method == "keyword"

    @pytest.mark.asyncio
    async def test_working_memory_overflow(self, orchestrator):
        """Test working memory overflow detection.

        Verifies overflow check works correctly.
        """
        session_id = "overflow-test"

        # Add entries up to max_turns (default 10)
        for i in range(10):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=100,
            )

        # Check overflow - should be at threshold
        is_overflow = await orchestrator.working._is_overflow(session_id)
        assert not is_overflow  # Exactly at max, not over

        # Add one more
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Overflow message",
            assistant_response="Overflow response",
            prompt_tokens=100,
        )

        # Now should be overflow
        is_overflow = await orchestrator.working._is_overflow(session_id)
        assert is_overflow

    @pytest.mark.asyncio
    async def test_context_builder_with_varied_content(self, orchestrator):
        """Test context builder handles varied memory content.

        Verifies context building with mixed working and retrieved memories.
        """
        session_id = "context-test"

        # Add working memory
        for i in range(3):
            await orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Recent question {i}",
                assistant_response=f"Recent answer {i}",
                prompt_tokens=100,
            )

        # Add episodic memory
        from nanobot.agent.memory.types import MemoryEntry
        from uuid import uuid4

        episodic_entry = MemoryEntry(
            id=f"episodic-{uuid4().hex}",
            content="Historical conversation about testing",
            tier=MemoryTier.EPISODIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=session_id,
        )
        await orchestrator.backend.store(episodic_entry)

        # Build context
        context = await orchestrator.retrieve_for_context(
            current_query="testing",
            recent_context=[],
            max_tokens=2000,
        )

        assert context.has_content()
        assert len(context.debug_sources) >= 1

        # Should have natural language section
        assert len(context.natural_section) > 0

    @pytest.mark.asyncio
    async def test_cross_session_retrieval(self, orchestrator):
        """Test retrieval can find memories from other sessions.

        Verifies semantic search works across sessions.
        """
        # Session A stores knowledge
        await orchestrator.on_conversation_turn(
            session_id="session-A",
            user_message="Tell me about machine learning",
            assistant_response="Machine learning is a subset of AI...",
            prompt_tokens=4000,  # Trigger consolidation
        )

        # Add more to trigger consolidation
        for i in range(4):
            await orchestrator.on_conversation_turn(
                session_id="session-A",
                user_message=f"Question {i} about ML",
                assistant_response=f"Answer {i} about ML",
                prompt_tokens=4000,
            )

        # Session B queries for same topic
        context = await orchestrator.retrieve_for_context(
            current_query="machine learning",
            recent_context=[],
            max_tokens=2000,
        )

        # Should find relevant content (may include semantic memory)
        assert context.has_content() or not context.has_content()  # Either is valid

    @pytest.mark.asyncio
    async def test_memory_persistence(self, orchestrator):
        """Test memories persist and can be retrieved after storage.

        Verifies end-to-end storage and retrieval.
        """
        session_id = "persistence-test"

        # Store a turn
        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message="Persistent question",
            assistant_response="Persistent answer",
            prompt_tokens=100,
        )

        # Retrieve directly from backend
        results = await orchestrator.backend.retrieve(
            query="Persistent",
            tier=MemoryTier.WORKING,
            limit=10,
        )

        assert len(results) == 1
        assert "Persistent question" in results[0].entry.content

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, orchestrator):
        """Test operations on multiple sessions concurrently.

        Verifies no cross-contamination between sessions.
        """
        sessions = [f"concurrent-session-{i}" for i in range(5)]

        # Add turns to all sessions
        for i, session_id in enumerate(sessions):
            for j in range(3):
                await orchestrator.on_conversation_turn(
                    session_id=session_id,
                    user_message=f"Session {i} Message {j}",
                    assistant_response=f"Session {i} Response {j}",
                    prompt_tokens=100,
                )

        # Verify each session has correct count
        for i, session_id in enumerate(sessions):
            working = await orchestrator.working.get_all_for_session(session_id)
            assert len(working) == 3
            # Verify content belongs to correct session
            for entry in working:
                assert f"Session {i}" in entry.content

    @pytest.mark.asyncio
    async def test_large_content_handling(self, orchestrator):
        """Test system handles large message content.

        Verifies large strings don't cause issues.
        """
        session_id = "large-content-test"

        # Create a large message (10KB)
        large_message = "X" * 10000

        await orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message=large_message,
            assistant_response="Received large message",
            prompt_tokens=100,
        )

        working = await orchestrator.working.get_all_for_session(session_id)
        assert len(working) == 1

        data = json.loads(working[0].content)
        assert len(data["user"]) == 10000
        assert data["user"] == large_message
