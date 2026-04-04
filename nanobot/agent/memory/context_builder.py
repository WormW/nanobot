"""Context builder for memory retrieval results.

This module provides classes for building and formatting retrieval context
from working memories and retrieved results for use in system prompts.
"""

from dataclasses import dataclass, field
from typing import Optional

from nanobot.agent.memory.types import MemoryEntry, MemoryTier, RetrievalResult
from nanobot.agent.memory.utils import calculate_memory_tokens, truncate_to_tokens


@dataclass
class RetrievalContext:
    """Container for formatted retrieval context.

    This dataclass holds different representations of retrieved memory context,
    including natural language descriptions, structured facts, and debug information.

    Attributes:
        natural_section: Natural language paragraph describing memories
        structured_facts: List of structured facts with category/content/relevance
        debug_sources: List of debug information about sources
    """

    natural_section: str = ""
    structured_facts: list[dict] = field(default_factory=list)
    debug_sources: list[dict] = field(default_factory=list)

    def has_content(self) -> bool:
        """Check if any content exists in this context.

        Returns:
            True if natural_section is non-empty or structured_facts has items.
        """
        return bool(self.natural_section) or len(self.structured_facts) > 0

    def to_system_prompt_addition(self) -> str:
        """Generate system prompt text from this context.

        Formats the natural section and structured facts into a string
        suitable for addition to a system prompt.

        Returns:
            Formatted system prompt text.
        """
        parts = []

        if self.natural_section:
            parts.append(self.natural_section)

        if self.structured_facts:
            parts.append("\n相关背景信息：")
            for fact in self.structured_facts:
                category = fact.get("category", "fact")
                content = fact.get("content", "")
                relevance = fact.get("relevance", 0.0)
                parts.append(f"  [{category}] {content} (相关度: {relevance:.2f})")

        return "\n".join(parts) if parts else ""


class MixedContextBuilder:
    """Builds mixed context from working memories and retrieved results.

    This class combines working memory (recent conversation context) with
    retrieved episodic and semantic memories to create a comprehensive
    context for the system prompt.
    """

    def build(
        self,
        working_memories: list[MemoryEntry],
        retrieved_results: list[RetrievalResult],
        max_tokens: int = 2000,
    ) -> RetrievalContext:
        """Build retrieval context from working memories and retrieved results.

        Args:
            working_memories: List of working memory entries (recent turns)
            retrieved_results: List of retrieval results from search
            max_tokens: Maximum tokens for the context

        Returns:
            RetrievalContext containing formatted context information.
        """
        # Deduplicate and rank results
        deduplicated = self._deduplicate_and_rank(retrieved_results)

        # Classify by relevance score
        high_priority = [r for r in deduplicated if r.relevance_score > 0.85]
        medium_priority = [r for r in deduplicated if 0.7 <= r.relevance_score <= 0.85]

        # Generate natural paragraph from working memories + high priority
        natural_section = self._generate_natural_paragraph(working_memories, high_priority)

        # Build structured facts from medium priority (up to 5)
        structured_facts = []
        for result in medium_priority[:5]:
            category = self._categorize_fact(result.entry)
            structured_facts.append({
                "category": category,
                "content": result.entry.content,
                "relevance": result.relevance_score,
                "source_id": result.entry.id,
            })

        # Build debug sources
        debug_sources = []
        for result in deduplicated:
            debug_sources.append({
                "id": result.entry.id,
                "tier": result.entry.tier.value,
                "relevance": result.relevance_score,
                "method": result.retrieval_method,
            })

        # Create context and apply token limit
        context = RetrievalContext(
            natural_section=natural_section,
            structured_facts=structured_facts,
            debug_sources=debug_sources,
        )

        # Truncate to fit within max_tokens if needed
        context = self._apply_token_limit(context, max_tokens)

        return context

    def _apply_token_limit(
        self,
        context: RetrievalContext,
        max_tokens: int,
    ) -> RetrievalContext:
        """Apply token limit to context, prioritizing high relevance content.

        Args:
            context: The retrieval context to limit
            max_tokens: Maximum tokens allowed

        Returns:
            RetrievalContext with content truncated to fit within token budget.
        """
        current_tokens = calculate_memory_tokens(
            context.natural_section,
            context.structured_facts,
        )
        if current_tokens <= max_tokens:
            return context

        # Calculate token budget allocation
        # Natural section gets 60%, structured facts get 40%
        natural_budget = int(max_tokens * 0.6)
        facts_budget = max_tokens - natural_budget

        # Truncate natural section
        truncated_natural = truncate_to_tokens(context.natural_section, natural_budget)

        # Truncate structured facts - sort by relevance and keep highest first
        sorted_facts = sorted(
            context.structured_facts,
            key=lambda f: f.get("relevance", 0.0),
            reverse=True,
        )

        truncated_facts = []
        current_fact_tokens = 0
        for fact in sorted_facts:
            fact_tokens = len(fact.get("content", "")) // 4
            if current_fact_tokens + fact_tokens <= facts_budget:
                truncated_facts.append(fact)
                current_fact_tokens += fact_tokens
            else:
                # Try to fit a truncated version of this fact
                remaining_tokens = facts_budget - current_fact_tokens
                if remaining_tokens > 10:  # Only add if we have meaningful space
                    truncated_content = truncate_to_tokens(
                        fact.get("content", ""), remaining_tokens
                    )
                    truncated_fact = fact.copy()
                    truncated_fact["content"] = truncated_content
                    truncated_facts.append(truncated_fact)
                break

        return RetrievalContext(
            natural_section=truncated_natural,
            structured_facts=truncated_facts,
            debug_sources=context.debug_sources,  # Debug sources don't count toward token limit
        )

    def _generate_natural_paragraph(
        self,
        working: list[MemoryEntry],
        high_priority: list[RetrievalResult],
    ) -> str:
        """Generate natural language paragraph from working memories and high priority results.

        Args:
            working: Working memory entries (recent conversation turns)
            high_priority: High relevance retrieval results (score > 0.85)

        Returns:
            Natural language paragraph describing the context.
        """
        parts = []

        # Add working memory content (recent conversation)
        if working:
            # Sort by creation time to maintain chronological order
            sorted_working = sorted(working, key=lambda e: e.created_at)
            working_contents = [e.content for e in sorted_working if e.content]
            if working_contents:
                parts.append("你们刚才聊到" + "，随后".join(working_contents) + "。")

        # Add high priority historical context
        if high_priority:
            # Sort by relevance score descending
            sorted_high = sorted(high_priority, key=lambda r: r.relevance_score, reverse=True)
            historical_contents = [r.entry.content for r in sorted_high if r.entry.content]
            if historical_contents:
                if parts:
                    parts.append("我记得之前你们聊过" + "。另外还聊过".join(historical_contents) + "。")
                else:
                    parts.append("我记得之前你们聊过" + "。另外还聊过".join(historical_contents) + "。")

        return "".join(parts)

    def _deduplicate_and_rank(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Deduplicate results by entry ID and rank by relevance.

        Args:
            results: List of retrieval results potentially containing duplicates

        Returns:
            Deduplicated list sorted by relevance score descending.
        """
        seen_ids = set()
        unique_results = []

        # Sort by relevance first (highest first)
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        for result in sorted_results:
            if result.entry.id not in seen_ids:
                seen_ids.add(result.entry.id)
                unique_results.append(result)

        return unique_results

    def _categorize_fact(self, entry: MemoryEntry) -> str:
        """Categorize a memory entry by type.

        Args:
            entry: Memory entry to categorize

        Returns:
            Category string: "episodic", "semantic", or "fact"
        """
        if entry.tier == MemoryTier.EPISODIC:
            return "episodic"
        elif entry.tier == MemoryTier.SEMANTIC:
            return "semantic"
        else:
            return "fact"
