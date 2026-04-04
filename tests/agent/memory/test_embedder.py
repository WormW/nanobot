"""Unit tests for EmbeddingProvider abstract class."""

import pytest
from abc import ABC
from typing import List

from nanobot.agent.memory.embedder import EmbeddingProvider


class MockEmbedder(EmbeddingProvider):
    """Mock implementation of EmbeddingProvider for testing."""

    def __init__(self, dimension: int = 384, max_tokens: int = 512):
        self._dimension = dimension
        self._max_tokens = max_tokens

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for testing.

        Returns deterministic embeddings based on text content.
        Each embedding is a vector of the configured dimension
        with values derived from the text hash for consistency.
        """
        embeddings = []
        for text in texts:
            # Generate deterministic mock embedding based on text
            # Using hash to create consistent values for the same text
            hash_val = hash(text)
            embedding = [
                float((hash_val + i * 31) % 1000) / 1000.0
                for i in range(self._dimension)
            ]
            embeddings.append(embedding)
        return embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def max_tokens_per_text(self) -> int:
        """Return the maximum tokens per text."""
        return self._max_tokens


class TestMockEmbedder:
    """Tests for MockEmbedder implementation."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = MockEmbedder(dimension=384, max_tokens=512)
        texts = ["Hello world"]

        result = await embedder.embed(texts)

        assert len(result) == 1
        assert len(result[0]) == 384
        assert all(isinstance(v, float) for v in result[0])

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        embedder = MockEmbedder(dimension=384, max_tokens=512)
        texts = ["First text", "Second text", "Third text"]

        result = await embedder.embed(texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 384
            assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_embed_deterministic(self):
        """Test that same text produces same embedding."""
        embedder = MockEmbedder(dimension=384, max_tokens=512)
        texts = ["Test text"]

        result1 = await embedder.embed(texts)
        result2 = await embedder.embed(texts)

        assert result1[0] == result2[0]

    @pytest.mark.asyncio
    async def test_embed_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        embedder = MockEmbedder(dimension=384, max_tokens=512)
        texts = ["Text one", "Text two"]

        result = await embedder.embed(texts)

        assert result[0] != result[1]

    def test_dimension_property(self):
        """Test dimension property returns correct value."""
        embedder = MockEmbedder(dimension=384, max_tokens=512)
        assert embedder.dimension == 384

    def test_max_tokens_per_text_property(self):
        """Test max_tokens_per_text property returns correct value."""
        embedder = MockEmbedder(dimension=384, max_tokens=512)
        assert embedder.max_tokens_per_text == 512

    def test_custom_dimension(self):
        """Test embedder with custom dimension."""
        embedder = MockEmbedder(dimension=768, max_tokens=1024)
        assert embedder.dimension == 768
        assert embedder.max_tokens_per_text == 1024

    @pytest.mark.asyncio
    async def test_custom_dimension_embedding(self):
        """Test embedding with custom dimension."""
        embedder = MockEmbedder(dimension=768, max_tokens=1024)
        texts = ["Test"]

        result = await embedder.embed(texts)

        assert len(result[0]) == 768


class TestEmbeddingProviderABC:
    """Tests for EmbeddingProvider abstract base class."""

    def test_is_abstract_base_class(self):
        """Test that EmbeddingProvider is an abstract base class."""
        assert issubclass(EmbeddingProvider, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_subclass_must_implement_embed(self):
        """Test that subclasses must implement embed method."""
        class IncompleteEmbedder(EmbeddingProvider):
            @property
            def dimension(self) -> int:
                return 384

            @property
            def max_tokens_per_text(self) -> int:
                return 512

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_subclass_must_implement_dimension(self):
        """Test that subclasses must implement dimension property."""
        class IncompleteEmbedder(EmbeddingProvider):
            async def embed(self, texts):
                return []

            @property
            def max_tokens_per_text(self) -> int:
                return 512

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_subclass_must_implement_max_tokens(self):
        """Test that subclasses must implement max_tokens_per_text property."""
        class IncompleteEmbedder(EmbeddingProvider):
            async def embed(self, texts):
                return []

            @property
            def dimension(self) -> int:
                return 384

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that complete subclasses can be instantiated."""
        class CompleteEmbedder(EmbeddingProvider):
            async def embed(self, texts):
                return [[0.0] * 384 for _ in texts]

            @property
            def dimension(self) -> int:
                return 384

            @property
            def max_tokens_per_text(self) -> int:
                return 512

        embedder = CompleteEmbedder()
        assert embedder.dimension == 384
        assert embedder.max_tokens_per_text == 512


class TestEmbeddingProviderInterface:
    """Tests for EmbeddingProvider interface contract."""

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_lists(self):
        """Test that embed returns List[List[float]]."""
        embedder = MockEmbedder()
        result = await embedder.embed(["text"])

        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding empty list returns empty list."""
        embedder = MockEmbedder()
        result = await embedder.embed([])

        assert result == []

    def test_dimension_is_int(self):
        """Test that dimension returns an integer."""
        embedder = MockEmbedder()
        assert isinstance(embedder.dimension, int)

    def test_max_tokens_is_int(self):
        """Test that max_tokens_per_text returns an integer."""
        embedder = MockEmbedder()
        assert isinstance(embedder.max_tokens_per_text, int)
