"""Unit tests for SemanticMemoryManager."""

import pytest
from datetime import datetime

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.embedder import EmbeddingProvider
from nanobot.agent.memory.tiers.semantic import SemanticMemoryManager
from nanobot.agent.memory.types import (
    MemoryEntry,
    MemoryTier,
    RetrievalResult,
    SemanticMemoryConfig,
)


class MockEmbedder(EmbeddingProvider):
    """Mock embedder for testing SemanticMemoryManager."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def max_tokens_per_text(self) -> int:
        return 512


class MockBackend(MemoryBackend):
    """Mock backend for testing SemanticMemoryManager."""

    def __init__(self):
        self.entries: list[MemoryEntry] = []
        self.initialized = False
        self.last_retrieve_params: dict | None = None

    async def initialize(self) -> None:
        self.initialized = True

    async def store(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    async def retrieve(
        self,
        query: str,
        tier: MemoryTier | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        self.last_retrieve_params = {
            "query": query,
            "tier": tier,
            "limit": limit,
            "embedding": embedding,
        }
        results = []
        for entry in self.entries:
            if tier is not None and entry.tier != tier:
                continue
            if embedding is not None and entry.embedding == embedding:
                results.append(
                    RetrievalResult(
                        entry=entry,
                        relevance_score=0.95,
                        retrieval_method="semantic",
                    )
                )
            elif query.lower() in entry.content.lower():
                results.append(
                    RetrievalResult(
                        entry=entry,
                        relevance_score=0.8,
                        retrieval_method="keyword",
                    )
                )
        return results[:limit]

    async def consolidate(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        return entries

    async def delete_expired(self, max_age_days: dict[MemoryTier, int]) -> int:
        return 0


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


@pytest.fixture
def config():
    return SemanticMemoryConfig()


@pytest.fixture
def manager(mock_backend, mock_embedder, config):
    return SemanticMemoryManager(
        backend=mock_backend,
        embedder=mock_embedder,
        config=config,
    )


class TestSemanticMemoryManager:
    """Tests for SemanticMemoryManager."""

    @pytest.mark.asyncio
    async def test_store_knowledge(self, manager, mock_backend):
        """Test storing knowledge creates a semantic memory entry."""
        entry = await manager.store_knowledge("Python is a programming language.")

        assert len(mock_backend.entries) == 1
        assert entry.tier == MemoryTier.SEMANTIC
        assert entry.content == "Python is a programming language."
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert isinstance(entry.id, str)
        assert len(entry.id) > 0
        assert entry.metadata == {}
        assert entry.source_session is None

    @pytest.mark.asyncio
    async def test_store_knowledge_with_metadata_and_session(self, manager):
        """Test storing knowledge with metadata and source session."""
        entry = await manager.store_knowledge(
            content="Important fact",
            source_session="session-123",
            metadata={"category": "facts", "priority": "high"},
        )

        assert entry.source_session == "session-123"
        assert entry.metadata == {"category": "facts", "priority": "high"}

    @pytest.mark.asyncio
    async def test_store_knowledge_sets_timestamps(self, manager):
        """Test that stored knowledge has current timestamps."""
        before = datetime.now()
        entry = await manager.store_knowledge("Time-sensitive info")
        after = datetime.now()

        assert before <= entry.created_at <= after
        assert before <= entry.updated_at <= after
        assert entry.created_at == entry.updated_at

    @pytest.mark.asyncio
    async def test_search_keyword(self, manager, mock_backend):
        """Test keyword search without embedding."""
        await manager.store_knowledge("Hello world")
        await manager.store_knowledge("Goodbye world")

        results = await manager.search("hello")

        assert len(results) == 1
        assert results[0].entry.content == "Hello world"
        assert results[0].retrieval_method == "keyword"
        assert mock_backend.last_retrieve_params["tier"] == MemoryTier.SEMANTIC
        assert mock_backend.last_retrieve_params["embedding"] is None
        assert mock_backend.last_retrieve_params["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_with_provided_embedding(self, manager, mock_backend):
        """Test search with a pre-computed embedding."""
        await manager.store_knowledge("Test content")

        custom_embedding = [0.9, 0.8, 0.7]
        results = await manager.search("query", embedding=custom_embedding, limit=3)

        assert mock_backend.last_retrieve_params["embedding"] == custom_embedding
        assert mock_backend.last_retrieve_params["limit"] == 3
        assert mock_backend.last_retrieve_params["tier"] == MemoryTier.SEMANTIC

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, manager, mock_backend):
        """Test that search respects the limit parameter."""
        for i in range(5):
            await manager.store_knowledge(f"Content {i}")

        results = await manager.search("Content", limit=2)

        assert len(results) == 2
        assert mock_backend.last_retrieve_params["limit"] == 2

    @pytest.mark.asyncio
    async def test_search_by_similarity_generates_embedding(self, manager, mock_backend):
        """Test search_by_similarity generates an embedding for the query."""
        await manager.store_knowledge("Semantic search topic")

        results = await manager.search_by_similarity("search query", limit=3)

        assert mock_backend.last_retrieve_params["embedding"] == [0.1, 0.2, 0.3]
        assert mock_backend.last_retrieve_params["tier"] == MemoryTier.SEMANTIC
        assert mock_backend.last_retrieve_params["limit"] == 3
        assert mock_backend.last_retrieve_params["query"] == "search query"

    @pytest.mark.asyncio
    async def test_search_by_similarity_uses_semantic_retrieval(self, manager):
        """Test that search_by_similarity triggers semantic retrieval method."""
        await manager.store_knowledge("Exact match content")

        results = await manager.search_by_similarity("Exact match content")

        assert len(results) == 1
        assert results[0].retrieval_method == "semantic"
        assert results[0].relevance_score == 0.95

    def test_manager_initialization(self, mock_backend, mock_embedder, config):
        """Test that the manager stores dependencies correctly."""
        manager = SemanticMemoryManager(
            backend=mock_backend,
            embedder=mock_embedder,
            config=config,
        )

        assert manager.backend is mock_backend
        assert manager.embedder is mock_embedder
        assert manager.config is config
