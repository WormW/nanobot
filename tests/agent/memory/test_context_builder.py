"""Unit tests for context builder module."""

from datetime import datetime

import pytest

from nanobot.agent.memory.context_builder import (
    MixedContextBuilder,
    RetrievalContext,
)
from nanobot.agent.memory.types import MemoryEntry, MemoryTier, RetrievalResult


class TestRetrievalContext:
    """Tests for RetrievalContext dataclass."""

    def test_empty_context_has_no_content(self):
        """Test that empty context returns False for has_content."""
        context = RetrievalContext()
        assert not context.has_content()

    def test_context_with_natural_section_has_content(self):
        """Test that context with natural section returns True."""
        context = RetrievalContext(natural_section="Some content")
        assert context.has_content()

    def test_context_with_facts_has_content(self):
        """Test that context with structured facts returns True."""
        context = RetrievalContext(structured_facts=[{"category": "fact", "content": "test"}])
        assert context.has_content()

    def test_to_system_prompt_addition_empty(self):
        """Test empty context returns empty string."""
        context = RetrievalContext()
        assert context.to_system_prompt_addition() == ""

    def test_to_system_prompt_addition_natural_only(self):
        """Test natural section only in prompt addition."""
        context = RetrievalContext(natural_section="你们刚才聊到测试。")
        result = context.to_system_prompt_addition()
        assert "你们刚才聊到测试。" in result

    def test_to_system_prompt_addition_with_facts(self):
        """Test structured facts in prompt addition."""
        context = RetrievalContext(
            natural_section="你们刚才聊到测试。",
            structured_facts=[
                {"category": "semantic", "content": "重要事实", "relevance": 0.8}
            ]
        )
        result = context.to_system_prompt_addition()
        assert "你们刚才聊到测试。" in result
        assert "[semantic]" in result
        assert "重要事实" in result
        assert "0.80" in result


class TestMixedContextBuilder:
    """Tests for MixedContextBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a MixedContextBuilder instance."""
        return MixedContextBuilder()

    @pytest.fixture
    def sample_datetime(self):
        """Create a sample datetime for testing."""
        return datetime(2024, 1, 1, 12, 0, 0)

    def test_build_empty_inputs(self, builder):
        """Test build with empty inputs returns empty context."""
        context = builder.build([], [])
        assert not context.has_content()
        assert context.natural_section == ""
        assert context.structured_facts == []
        assert context.debug_sources == []

    def test_build_with_working_memories(self, builder, sample_datetime):
        """Test build with working memories generates natural paragraph."""
        working = [
            MemoryEntry(
                id="w1",
                content="用户询问天气",
                tier=MemoryTier.WORKING,
                created_at=sample_datetime,
                updated_at=sample_datetime,
            ),
            MemoryEntry(
                id="w2",
                content="系统回答晴天",
                tier=MemoryTier.WORKING,
                created_at=sample_datetime,
                updated_at=sample_datetime,
            ),
        ]

        context = builder.build(working, [])
        assert context.has_content()
        assert "你们刚才聊到" in context.natural_section
        assert "用户询问天气" in context.natural_section
        assert "随后" in context.natural_section
        assert "系统回答晴天" in context.natural_section

    def test_build_with_high_priority_results(self, builder, sample_datetime):
        """Test build with high priority results includes them in natural section."""
        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id="h1",
                    content="之前讨论过项目计划",
                    tier=MemoryTier.EPISODIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.9,
                retrieval_method="semantic_search",
            )
        ]

        context = builder.build([], retrieved)
        assert context.has_content()
        assert "我记得之前你们聊过" in context.natural_section
        assert "之前讨论过项目计划" in context.natural_section

    def test_build_with_medium_priority_results(self, builder, sample_datetime):
        """Test build with medium priority results includes them as structured facts."""
        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id="m1",
                    content="用户喜欢Python",
                    tier=MemoryTier.SEMANTIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.75,
                retrieval_method="semantic_search",
            )
        ]

        context = builder.build([], retrieved)
        assert len(context.structured_facts) == 1
        fact = context.structured_facts[0]
        assert fact["category"] == "semantic"
        assert fact["content"] == "用户喜欢Python"
        assert fact["relevance"] == 0.75
        assert fact["source_id"] == "m1"

    def test_deduplicate_and_rank(self, builder, sample_datetime):
        """Test deduplication keeps highest relevance for duplicate entries."""
        entry = MemoryEntry(
            id="dup",
            content="重复内容",
            tier=MemoryTier.SEMANTIC,
            created_at=sample_datetime,
            updated_at=sample_datetime,
        )

        results = [
            RetrievalResult(entry=entry, relevance_score=0.6, retrieval_method="method1"),
            RetrievalResult(entry=entry, relevance_score=0.9, retrieval_method="method2"),
            RetrievalResult(entry=entry, relevance_score=0.7, retrieval_method="method3"),
        ]

        deduplicated = builder._deduplicate_and_rank(results)
        assert len(deduplicated) == 1
        assert deduplicated[0].relevance_score == 0.9

    def test_categorize_fact(self, builder, sample_datetime):
        """Test categorization of different memory tiers."""
        episodic_entry = MemoryEntry(
            id="e1",
            content="episodic memory",
            tier=MemoryTier.EPISODIC,
            created_at=sample_datetime,
            updated_at=sample_datetime,
        )
        semantic_entry = MemoryEntry(
            id="s1",
            content="semantic memory",
            tier=MemoryTier.SEMANTIC,
            created_at=sample_datetime,
            updated_at=sample_datetime,
        )
        working_entry = MemoryEntry(
            id="w1",
            content="working memory",
            tier=MemoryTier.WORKING,
            created_at=sample_datetime,
            updated_at=sample_datetime,
        )

        assert builder._categorize_fact(episodic_entry) == "episodic"
        assert builder._categorize_fact(semantic_entry) == "semantic"
        assert builder._categorize_fact(working_entry) == "fact"

    def test_build_creates_debug_sources(self, builder, sample_datetime):
        """Test build creates debug source information."""
        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id="r1",
                    content="测试内容",
                    tier=MemoryTier.EPISODIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.88,
                retrieval_method="keyword_search",
            )
        ]

        context = builder.build([], retrieved)
        assert len(context.debug_sources) == 1
        source = context.debug_sources[0]
        assert source["id"] == "r1"
        assert source["tier"] == "episodic"
        assert source["relevance"] == 0.88
        assert source["method"] == "keyword_search"

    def test_build_limits_structured_facts_to_5(self, builder, sample_datetime):
        """Test build limits structured facts to maximum 5 entries."""
        # Create 10 medium priority results
        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id=f"m{i}",
                    content=f"事实{i}",
                    tier=MemoryTier.SEMANTIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.75,  # Medium priority
                retrieval_method="semantic_search",
            )
            for i in range(10)
        ]

        context = builder.build([], retrieved)
        assert len(context.structured_facts) == 5

    def test_build_integrates_working_and_retrieved(self, builder, sample_datetime):
        """Test build properly integrates working memories and retrieved results."""
        working = [
            MemoryEntry(
                id="w1",
                content="当前话题",
                tier=MemoryTier.WORKING,
                created_at=sample_datetime,
                updated_at=sample_datetime,
            )
        ]

        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id="h1",
                    content="高优先级历史",
                    tier=MemoryTier.EPISODIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.9,
                retrieval_method="semantic_search",
            ),
            RetrievalResult(
                entry=MemoryEntry(
                    id="m1",
                    content="中等优先级事实",
                    tier=MemoryTier.SEMANTIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.8,
                retrieval_method="semantic_search",
            ),
        ]

        context = builder.build(working, retrieved)

        # Natural section should have both working and high priority
        assert "你们刚才聊到" in context.natural_section
        assert "当前话题" in context.natural_section
        assert "我记得之前你们聊过" in context.natural_section
        assert "高优先级历史" in context.natural_section

        # Structured facts should have medium priority
        assert len(context.structured_facts) == 1
        assert context.structured_facts[0]["content"] == "中等优先级事实"

    def test_build_sorts_working_by_time(self, builder):
        """Test working memories are sorted chronologically."""
        working = [
            MemoryEntry(
                id="w2",
                content="第二条",
                tier=MemoryTier.WORKING,
                created_at=datetime(2024, 1, 1, 12, 30, 0),
                updated_at=datetime(2024, 1, 1, 12, 30, 0),
            ),
            MemoryEntry(
                id="w1",
                content="第一条",
                tier=MemoryTier.WORKING,
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                updated_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
        ]

        context = builder.build(working, [])
        # Should be sorted: 第一条, 随后 第二条
        assert context.natural_section.index("第一条") < context.natural_section.index("第二条")

    def test_build_sorts_high_priority_by_relevance(self, builder, sample_datetime):
        """Test high priority results are sorted by relevance."""
        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id="h2",
                    content="较低相关",
                    tier=MemoryTier.EPISODIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.86,
                retrieval_method="semantic_search",
            ),
            RetrievalResult(
                entry=MemoryEntry(
                    id="h1",
                    content="最高相关",
                    tier=MemoryTier.EPISODIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.95,
                retrieval_method="semantic_search",
            ),
        ]

        context = builder.build([], retrieved)
        # Higher relevance should appear first
        assert context.natural_section.index("最高相关") < context.natural_section.index("较低相关")

    def test_build_ignores_low_priority(self, builder, sample_datetime):
        """Test that low priority results (< 0.7) are ignored."""
        retrieved = [
            RetrievalResult(
                entry=MemoryEntry(
                    id="low",
                    content="低优先级",
                    tier=MemoryTier.SEMANTIC,
                    created_at=sample_datetime,
                    updated_at=sample_datetime,
                ),
                relevance_score=0.5,
                retrieval_method="semantic_search",
            )
        ]

        context = builder.build([], retrieved)
        assert not context.has_content()
        assert len(context.structured_facts) == 0
        assert context.natural_section == ""
