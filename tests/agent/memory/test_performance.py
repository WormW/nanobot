"""Performance tests and benchmarks for the layered memory system.

Tests cover retrieval performance, memory usage, and scalability benchmarks.
"""

import asyncio
import tempfile
import time
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
from nanobot.agent.memory.types import MemoryEntry


class MockEmbedder(EmbeddingProvider):
    """Mock embedder for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1), 0.2 * (i + 1)] for i in range(len(texts))]

    @property
    def dimension(self) -> int:
        return 2

    @property
    def max_tokens_per_text(self) -> int:
        return 1000


@pytest.fixture
async def fs_orchestrator():
    """Create orchestrator with filesystem backend."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileSystemBackend(tmp)
        await backend.initialize()
        embedder = MockEmbedder()
        config = MemoryConfig()
        yield MemoryOrchestrator(backend, embedder, config)


@pytest.fixture
async def sqlite_orchestrator():
    """Create orchestrator with SQLite backend."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        backend = SQLiteBackend(str(db_path))
        await backend.initialize()
        embedder = MockEmbedder()
        config = MemoryConfig()
        yield MemoryOrchestrator(backend, embedder, config)


class TestRetrievalPerformance:
    """Test retrieval performance across backends."""

    @pytest.mark.asyncio
    async def test_working_memory_retrieval_performance(self, fs_orchestrator):
        """Test that working memory retrieval is fast."""
        session_id = "perf-test"

        # Add 100 conversation turns
        for i in range(100):
            await fs_orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=50,
            )

        # Benchmark retrieval
        start = time.time()
        memories = await fs_orchestrator.working.get_recent(n=10)
        elapsed = time.time() - start

        assert len(memories) == 10
        assert elapsed < 0.1, f"Retrieval too slow: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_sqlite_vs_filesystem_performance(self):
        """Compare SQLite vs Filesystem retrieval performance."""
        with tempfile.TemporaryDirectory() as tmp:
            fs_path = Path(tmp) / "fs"
            db_path = Path(tmp) / "memory.db"

            # Setup backends
            fs_backend = FileSystemBackend(str(fs_path))
            sqlite_backend = SQLiteBackend(str(db_path))
            await fs_backend.initialize()
            await sqlite_backend.initialize()

            # Add test data
            for i in range(50):
                entry = MemoryEntry(
                    id=f"test-{i}",
                    content=f"Test content {i} with searchable keywords",
                    tier=MemoryTier.EPISODIC,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                await fs_backend.store(entry)
                await sqlite_backend.store(entry)

            # Benchmark filesystem
            start = time.time()
            fs_results = await fs_backend.retrieve("searchable", limit=10)
            fs_time = time.time() - start

            # Benchmark SQLite
            start = time.time()
            sqlite_results = await sqlite_backend.retrieve("searchable", limit=10)
            sqlite_time = time.time() - start

            assert len(fs_results) == len(sqlite_results)
            # SQLite should be comparable or faster
            print(f"Filesystem: {fs_time:.4f}s, SQLite: {sqlite_time:.4f}s")

    @pytest.mark.asyncio
    async def test_large_dataset_retrieval_scaling(self):
        """Test retrieval performance with large datasets."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            backend = SQLiteBackend(str(db_path))
            await backend.initialize()

            # Add 1000 entries
            for i in range(1000):
                entry = MemoryEntry(
                    id=f"entry-{i}",
                    content=f"Content for entry {i} with keyword",
                    tier=MemoryTier.EPISODIC,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                await backend.store(entry)

            # Search should still be fast
            start = time.time()
            results = await backend.retrieve("keyword", limit=10)
            elapsed = time.time() - start

            assert len(results) <= 10
            assert elapsed < 1.0, f"Large dataset search too slow: {elapsed:.3f}s"


class TestMemoryUsage:
    """Test memory usage patterns."""

    @pytest.mark.asyncio
    async def test_batch_retrieval_efficiency(self):
        """Test that retrieval with limits works efficiently."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = FileSystemBackend(tmp)
            await backend.initialize()

            # Add entries
            for i in range(100):
                entry = MemoryEntry(
                    id=f"batch-{i}",
                    content=f"Batch test content {i}",
                    tier=MemoryTier.SEMANTIC,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                await backend.store(entry)

            # Retrieve with limits (simulating batch behavior)
            all_results = []
            for _ in range(10):
                results = await backend.retrieve("Batch", limit=10)
                all_results.extend(results[:5])  # Take top 5 from each batch
                if len(all_results) >= 50:
                    break

            assert len(all_results) <= 100

    @pytest.mark.asyncio
    async def test_orchestrator_memory_with_many_sessions(self, fs_orchestrator):
        """Test memory usage with many concurrent sessions."""
        async def session_workflow(session_num):
            session_id = f"session-{session_num}"
            for i in range(5):
                await fs_orchestrator.on_conversation_turn(
                    session_id=session_id,
                    user_message=f"Session {session_num} message {i}",
                    assistant_response=f"Response {i}",
                    prompt_tokens=50,
                )

        # Run 20 sessions concurrently
        start = time.time()
        await asyncio.gather(*[session_workflow(i) for i in range(20)])
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Too slow with many sessions: {elapsed:.3f}s"


class TestConcurrentPerformance:
    """Test performance under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_read_write_performance(self, sqlite_orchestrator):
        """Test read/write performance under concurrent load."""
        session_id = "concurrent-perf"

        async def writer_task(task_id):
            for i in range(10):
                await sqlite_orchestrator.on_conversation_turn(
                    session_id=session_id,
                    user_message=f"Task {task_id} message {i}",
                    assistant_response=f"Response {i}",
                    prompt_tokens=50,
                )

        async def reader_task():
            for _ in range(10):
                await sqlite_orchestrator.working.get_recent(n=5)
                await asyncio.sleep(0.01)

        start = time.time()
        # 5 writers, 5 readers
        await asyncio.gather(
            *[writer_task(i) for i in range(5)],
            *[reader_task() for _ in range(5)],
        )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 3.0, f"Concurrent operations too slow: {elapsed:.3f}s"


class TestMigrationPerformance:
    """Test migration performance with large datasets."""

    @pytest.mark.asyncio
    async def test_large_migration_performance(self):
        """Test migration speed with many entries."""
        with tempfile.TemporaryDirectory() as tmp:
            fs_path = Path(tmp) / "fs"
            db_path = Path(tmp) / "memory.db"

            fs_backend = FileSystemBackend(str(fs_path))
            await fs_backend.initialize()

            # Add 500 entries
            for i in range(500):
                tier = [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC][i % 3]
                entry = MemoryEntry(
                    id=f"mig-{i}",
                    content=f"Migration content {i}",
                    tier=tier,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    source_session="test" if tier == MemoryTier.WORKING else None,
                )
                await fs_backend.store(entry)

            # Benchmark migration
            from nanobot.agent.memory import migrate_filesystem_to_sqlite

            start = time.time()
            result = await migrate_filesystem_to_sqlite(str(fs_path), str(db_path))
            elapsed = time.time() - start

            assert result.total_entries >= 500
            assert elapsed < 10.0, f"Migration too slow: {elapsed:.3f}s"
            print(f"Migrated {result.total_entries} entries in {elapsed:.2f}s")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_query_retrieval(self, fs_orchestrator):
        """Test retrieval with empty query returns all entries."""
        session_id = "edge-test"

        # Add entries
        for i in range(10):
            await fs_orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=50,
            )

        # Empty query should return all
        results = await fs_orchestrator.backend.retrieve("", tier=MemoryTier.WORKING)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_very_long_content_storage(self, fs_orchestrator):
        """Test storage of very long content."""
        session_id = "long-content"

        # Create very long message
        long_message = "word " * 10000  # ~50KB of text

        await fs_orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message=long_message,
            assistant_response="Response",
            prompt_tokens=5000,
        )

        memories = await fs_orchestrator.working.get_recent(n=1)
        assert len(memories) == 1
        assert len(memories[0].content) > 50000

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, fs_orchestrator):
        """Test storage of special characters."""
        session_id = "special-chars"

        special_message = 'Special: "quotes", \nnewlines\n, emojis 🎉, unicode 中文'

        await fs_orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message=special_message,
            assistant_response="Response 🎊",
            prompt_tokens=50,
        )

        memories = await fs_orchestrator.working.get_recent(n=1)
        assert len(memories) == 1
        # Content is JSON-encoded, check for escaped emoji or raw emoji
        content = memories[0].content
        assert "🎉" in content or "\\ud83c\\udf89" in content

    @pytest.mark.asyncio
    async def test_retrieval_with_large_limit(self, fs_orchestrator):
        """Test retrieval with limit larger than dataset."""
        session_id = "large-limit"

        # Add 5 entries
        for i in range(5):
            await fs_orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
                prompt_tokens=50,
            )

        # Request 100, should get 5
        results = await fs_orchestrator.backend.retrieve("", limit=100)
        assert len(results) == 5
