"""Embedding provider abstraction for the memory system.

This module defines the abstract base class for embedding providers,
which are responsible for generating vector embeddings from text content.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """嵌入向量生成抽象 (Abstract base class for embedding generation).

    This abstract base class defines the interface for embedding providers
    that convert text into vector representations for semantic search and
    similarity comparisons in the memory system.

    Implementations must provide:
    - embed(): Batch generation of embeddings for text inputs
    - dimension: The size of the embedding vectors
    - max_tokens_per_text: Maximum token limit per text input

    Example:
        class OpenAIEmbedder(EmbeddingProvider):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                # Call OpenAI API to generate embeddings
                return embeddings

            @property
            def dimension(self) -> int:
                return 1536  # OpenAI text-embedding-3-small

            @property
            def max_tokens_per_text(self) -> int:
                return 8192
    """

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入 (Batch generate embeddings for texts).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, where each vector is a list of floats.
            The outer list has the same length as the input texts list.
            Each inner list has length equal to the dimension property.

        Raises:
            ValueError: If any text exceeds max_tokens_per_text.
            RuntimeError: If embedding generation fails.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入维度 (Embedding dimension).

        Returns:
            The dimensionality of the embedding vectors produced by this provider.
            Common values: 384 (small), 768 (medium), 1536 (OpenAI).
        """
        pass

    @property
    @abstractmethod
    def max_tokens_per_text(self) -> int:
        """单次最大 token 数 (Maximum tokens per text).

        Returns:
            The maximum number of tokens that can be processed in a single
            text input. Texts exceeding this limit may be truncated or rejected.
        """
        pass
