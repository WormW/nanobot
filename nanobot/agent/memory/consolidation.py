"""Memory consolidation engine for the layered memory system.

This module provides the ConsolidationEngine class, which orchestrates the
movement of memories from working memory to episodic memory, and from episodic
to semantic memory. It handles batch processing, threshold checking, and
coordination between the different memory tiers.
"""

from dataclasses import dataclass

from nanobot.agent.memory.tiers.working import WorkingMemoryManager
from nanobot.agent.memory.tiers.episodic import EpisodicMemoryManager
from nanobot.agent.memory.tiers.semantic import SemanticMemoryManager
from nanobot.agent.memory.types import MemoryEntry, ConsolidationConfig


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation operation.

    Attributes:
        session_id: The session ID that was consolidated.
        episodic_created: List of episodic memory entries created.
        semantic_created: List of semantic memory entries created.
    """

    session_id: str
    episodic_created: list[MemoryEntry]
    semantic_created: list[MemoryEntry]


class ConsolidationEngine:
    """Engine for consolidating memories across tiers.

    The ConsolidationEngine manages the flow of information from working memory
    to episodic memory (conversation summaries), and from episodic to semantic
    memory (extracted knowledge). It uses configurable thresholds to determine
    when consolidation should occur.

    Args:
        working: Manager for working memory tier.
        episodic: Manager for episodic memory tier.
        semantic: Manager for semantic memory tier.
        config: Configuration for consolidation behavior.
    """

    def __init__(
        self,
        working: WorkingMemoryManager,
        episodic: EpisodicMemoryManager,
        semantic: SemanticMemoryManager,
        config: ConsolidationConfig,
    ):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.config = config

    async def should_consolidate(self, prompt_tokens: int) -> bool:
        """Check if consolidation should be triggered based on token count.

        Compares the provided token count against the configured threshold
        to determine if working memory should be consolidated.

        Args:
            prompt_tokens: The number of tokens in the current prompt.

        Returns:
            True if prompt_tokens exceeds the working_memory_token_threshold.
        """
        return prompt_tokens > self.config.working_memory_token_threshold

    async def run(self, session_id: str) -> ConsolidationResult:
        """Run consolidation for a specific session.

        This method performs the following steps:
        1. Retrieves all working memory entries for the session.
        2. If the entry count meets or exceeds the consolidation_batch threshold,
           creates an episodic summary from the working entries.
        3. Extracts semantic knowledge from the episodic summary by splitting
           the content into sentences and storing each as knowledge.
        4. Archives the consolidated working memory entries.

        Args:
            session_id: The session identifier to consolidate.

        Returns:
            A ConsolidationResult containing the created episodic and semantic
            memory entries.
        """
        episodic_created: list[MemoryEntry] = []
        semantic_created: list[MemoryEntry] = []

        # Step 1: Get all working entries for the session
        working_entries = await self.working.get_all_for_session(session_id)

        # Step 2: Check if we have enough entries to consolidate
        if len(working_entries) >= self.episodic.config.consolidation_batch:
            # Create episodic summary from working entries
            summary_entry = await self.episodic.create_summary(
                session_id, working_entries
            )
            episodic_created.append(summary_entry)

            # Step 3: Extract semantic knowledge from the summary
            # For now, split by sentences and store each as knowledge
            sentences = summary_entry.content.split(".")
            for sentence in sentences:
                stripped = sentence.strip()
                if len(stripped) > 20:
                    semantic_entry = await self.semantic.store_knowledge(
                        stripped, session_id
                    )
                    semantic_created.append(semantic_entry)

            # Step 4: Archive consolidated working entries
            entry_ids = [entry.id for entry in working_entries]
            await self.working.archive_entries(entry_ids)

        return ConsolidationResult(
            session_id=session_id,
            episodic_created=episodic_created,
            semantic_created=semantic_created,
        )
