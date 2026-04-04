"""Base types and configuration classes for the memory system.

This module defines the core data structures used throughout the memory system,
including memory tiers, entry types, retrieval results, and configuration classes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MemoryTier(str, Enum):
    """Enumeration of memory tiers in the hierarchical memory system.

    The memory system organizes information into three tiers:
    - WORKING: Short-term, contextually relevant information for current task
    - EPISODIC: Recent conversation history and interaction patterns
    - SEMANTIC: Consolidated knowledge and long-term facts
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the system.

    Attributes:
        id: Unique identifier for the memory entry
        content: The actual content/text of the memory
        tier: The memory tier this entry belongs to
        created_at: Timestamp when the entry was created
        updated_at: Timestamp when the entry was last updated
        source_session: Optional session ID that created this entry
        metadata: Additional key-value metadata for the entry
        embedding: Optional vector embedding for semantic search
    """

    id: str
    content: str
    tier: MemoryTier
    created_at: datetime
    updated_at: datetime
    source_session: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation.

    Attributes:
        entry: The retrieved memory entry
        relevance_score: Float score indicating relevance (0.0 to 1.0)
        retrieval_method: String identifier of the retrieval method used
    """

    entry: MemoryEntry
    relevance_score: float
    retrieval_method: str


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory tier.

    Working memory holds short-term, contextually relevant information
    for the current task or conversation turn.

    Attributes:
        max_turns: Maximum number of conversation turns to retain
        max_tokens: Maximum token count for working memory content
        ttl_seconds: Time-to-live in seconds before automatic cleanup
    """

    max_turns: int = 10
    max_tokens: int = 4000
    ttl_seconds: int = 3600


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory tier.

    Episodic memory stores recent conversation history and interaction
    patterns, typically used for maintaining context across sessions.

    Attributes:
        summary_model: Model identifier for generating summaries
        max_entries: Maximum number of episodic entries to store
        consolidation_batch: Number of entries to process per consolidation batch
    """

    summary_model: str = "default"
    max_entries: int = 1000
    consolidation_batch: int = 5


@dataclass
class SemanticMemoryConfig:
    """Configuration for semantic memory tier.

    Semantic memory holds consolidated knowledge, facts, and long-term
    information that has been extracted from episodic memories.

    Attributes:
        embedding_dimension: Dimension of the embedding vectors
        similarity_threshold: Minimum similarity score for retrieval (0.0 to 1.0)
        max_entries: Maximum number of semantic entries to store
    """

    embedding_dimension: int = 768
    similarity_threshold: float = 0.75
    max_entries: int = 10000


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation process.

    Consolidation moves information from working to episodic to semantic
    memory based on various thresholds and triggers.

    Attributes:
        working_memory_token_threshold: Token count trigger for consolidation
        episodic_memory_count_threshold: Entry count trigger for consolidation
        auto_consolidate_interval_minutes: Time interval for automatic consolidation
        enable_explicit_consolidation: Whether to allow manual consolidation triggers
    """

    working_memory_token_threshold: int = 3000
    episodic_memory_count_threshold: int = 50
    auto_consolidate_interval_minutes: int = 30
    enable_explicit_consolidation: bool = True


@dataclass
class MemoryConfig:
    """Top-level configuration container for the memory system.

    This class aggregates all configuration settings for the memory system,
    including settings for each memory tier and consolidation behavior.

    Attributes:
        working: Configuration for working memory tier
        episodic: Configuration for episodic memory tier
        semantic: Configuration for semantic memory tier
        consolidation: Configuration for consolidation process
    """

    working: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    episodic: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    semantic: SemanticMemoryConfig = field(default_factory=SemanticMemoryConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
