"""Unit tests for memory system base types and configuration classes."""

import pytest
from dataclasses import fields
from datetime import datetime
from typing import Optional

from nanobot.agent.memory.types import (
    MemoryTier,
    MemoryEntry,
    RetrievalResult,
    WorkingMemoryConfig,
    EpisodicMemoryConfig,
    SemanticMemoryConfig,
    ConsolidationConfig,
    MemoryConfig,
)


class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.EPISODIC.value == "episodic"
        assert MemoryTier.SEMANTIC.value == "semantic"

    def test_enum_members(self):
        """Test that all expected members exist."""
        assert hasattr(MemoryTier, 'WORKING')
        assert hasattr(MemoryTier, 'EPISODIC')
        assert hasattr(MemoryTier, 'SEMANTIC')


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_required_fields(self):
        """Test creating MemoryEntry with required fields."""
        now = datetime.now()
        entry = MemoryEntry(
            id="test-id-123",
            content="Test memory content",
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now,
        )
        assert entry.id == "test-id-123"
        assert entry.content == "Test memory content"
        assert entry.tier == MemoryTier.WORKING
        assert entry.created_at == now
        assert entry.updated_at == now

    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        now = datetime.now()
        entry = MemoryEntry(
            id="test-id",
            content="Content",
            tier=MemoryTier.EPISODIC,
            created_at=now,
            updated_at=now,
        )
        assert entry.source_session is None
        assert entry.metadata == {}
        assert entry.embedding is None

    def test_optional_fields_custom(self):
        """Test setting optional fields to custom values."""
        now = datetime.now()
        embedding = [0.1, 0.2, 0.3, 0.4]
        entry = MemoryEntry(
            id="test-id",
            content="Content",
            tier=MemoryTier.SEMANTIC,
            created_at=now,
            updated_at=now,
            source_session="session-456",
            metadata={"key": "value", "count": 42},
            embedding=embedding,
        )
        assert entry.source_session == "session-456"
        assert entry.metadata == {"key": "value", "count": 42}
        assert entry.embedding == embedding

    def test_all_tiers(self):
        """Test MemoryEntry with all tier types."""
        now = datetime.now()
        for tier in [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC]:
            entry = MemoryEntry(
                id=f"id-{tier.value}",
                content=f"Content for {tier.value}",
                tier=tier,
                created_at=now,
                updated_at=now,
            )
            assert entry.tier == tier


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_creation(self):
        """Test creating RetrievalResult."""
        now = datetime.now()
        entry = MemoryEntry(
            id="test-id",
            content="Test content",
            tier=MemoryTier.WORKING,
            created_at=now,
            updated_at=now,
        )
        result = RetrievalResult(
            entry=entry,
            relevance_score=0.85,
            retrieval_method="semantic_search",
        )
        assert result.entry == entry
        assert result.relevance_score == 0.85
        assert result.retrieval_method == "semantic_search"

    def test_different_methods(self):
        """Test RetrievalResult with different retrieval methods."""
        now = datetime.now()
        entry = MemoryEntry(
            id="test-id",
            content="Test content",
            tier=MemoryTier.SEMANTIC,
            created_at=now,
            updated_at=now,
        )
        methods = ["exact_match", "fuzzy_search", "embedding_similarity", "hybrid"]
        for method in methods:
            result = RetrievalResult(
                entry=entry,
                relevance_score=0.75,
                retrieval_method=method,
            )
            assert result.retrieval_method == method


class TestWorkingMemoryConfig:
    """Tests for WorkingMemoryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WorkingMemoryConfig()
        assert config.max_turns == 10
        assert config.max_tokens == 4000
        assert config.ttl_seconds == 3600

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WorkingMemoryConfig(
            max_turns=20,
            max_tokens=8000,
            ttl_seconds=7200,
        )
        assert config.max_turns == 20
        assert config.max_tokens == 8000
        assert config.ttl_seconds == 7200


class TestEpisodicMemoryConfig:
    """Tests for EpisodicMemoryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EpisodicMemoryConfig()
        assert config.summary_model == "default"
        assert config.max_entries == 1000
        assert config.consolidation_batch == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EpisodicMemoryConfig(
            summary_model="gpt-4",
            max_entries=2000,
            consolidation_batch=10,
        )
        assert config.summary_model == "gpt-4"
        assert config.max_entries == 2000
        assert config.consolidation_batch == 10


class TestSemanticMemoryConfig:
    """Tests for SemanticMemoryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SemanticMemoryConfig()
        assert config.embedding_dimension == 768
        assert config.similarity_threshold == 0.75
        assert config.max_entries == 10000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SemanticMemoryConfig(
            embedding_dimension=1536,
            similarity_threshold=0.85,
            max_entries=50000,
        )
        assert config.embedding_dimension == 1536
        assert config.similarity_threshold == 0.85
        assert config.max_entries == 50000


class TestConsolidationConfig:
    """Tests for ConsolidationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConsolidationConfig()
        assert config.working_memory_token_threshold == 3000
        assert config.episodic_memory_count_threshold == 50
        assert config.auto_consolidate_interval_minutes == 30
        assert config.enable_explicit_consolidation is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConsolidationConfig(
            working_memory_token_threshold=5000,
            episodic_memory_count_threshold=100,
            auto_consolidate_interval_minutes=60,
            enable_explicit_consolidation=False,
        )
        assert config.working_memory_token_threshold == 5000
        assert config.episodic_memory_count_threshold == 100
        assert config.auto_consolidate_interval_minutes == 60
        assert config.enable_explicit_consolidation is False


class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MemoryConfig()
        assert isinstance(config.working, WorkingMemoryConfig)
        assert isinstance(config.episodic, EpisodicMemoryConfig)
        assert isinstance(config.semantic, SemanticMemoryConfig)
        assert isinstance(config.consolidation, ConsolidationConfig)

        # Verify defaults are applied
        assert config.working.max_turns == 10
        assert config.episodic.max_entries == 1000
        assert config.semantic.embedding_dimension == 768
        assert config.consolidation.auto_consolidate_interval_minutes == 30

    def test_custom_sub_configs(self):
        """Test MemoryConfig with custom sub-configurations."""
        config = MemoryConfig(
            working=WorkingMemoryConfig(max_turns=50, max_tokens=10000),
            semantic=SemanticMemoryConfig(embedding_dimension=1536),
        )
        assert config.working.max_turns == 50
        assert config.working.max_tokens == 10000
        assert config.semantic.embedding_dimension == 1536
        # Other configs should use defaults
        assert config.episodic.max_entries == 1000
        assert config.consolidation.enable_explicit_consolidation is True
