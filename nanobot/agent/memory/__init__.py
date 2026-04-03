"""Memory system for Nanobot agent.

This module provides a hierarchical memory system with three tiers:
- Working Memory: Short-term, contextually relevant information
- Episodic Memory: Recent conversation history and patterns
- Semantic Memory: Consolidated long-term knowledge
"""

# Re-export from existing memory store module
from nanobot.agent.memory.store import (
    MemoryConsolidator,
    MemoryStore,
)

# New types for hierarchical memory system
from nanobot.agent.memory.types import (
    ConsolidationConfig,
    EpisodicMemoryConfig,
    MemoryConfig,
    MemoryEntry,
    MemoryTier,
    RetrievalResult,
    SemanticMemoryConfig,
    WorkingMemoryConfig,
)

# Embedding provider
from nanobot.agent.memory.embedder import EmbeddingProvider

__all__ = [
    # Existing
    "MemoryStore",
    "MemoryConsolidator",
    # New types
    "MemoryTier",
    "MemoryEntry",
    "RetrievalResult",
    "WorkingMemoryConfig",
    "EpisodicMemoryConfig",
    "SemanticMemoryConfig",
    "ConsolidationConfig",
    "MemoryConfig",
    # Embedding
    "EmbeddingProvider",
]
