"""Memory orchestrator for coordinating the layered memory system.

This module provides the MemoryOrchestrator class, which serves as the central
coordinator for the memory system. It manages the three memory tiers (working,
episodic, semantic) and handles consolidation triggers and context retrieval.
"""

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.embedder import EmbeddingProvider
from nanobot.agent.memory.types import MemoryConfig
from nanobot.agent.memory.tiers.working import WorkingMemoryManager
from nanobot.agent.memory.tiers.episodic import EpisodicMemoryManager
from nanobot.agent.memory.tiers.semantic import SemanticMemoryManager
from nanobot.agent.memory.consolidation import ConsolidationEngine
from nanobot.agent.memory.context_builder import MixedContextBuilder, RetrievalContext
from nanobot.agent.memory.utils import AsyncLockManager


class MemoryOrchestrator:
    """Central coordinator for the layered memory system.

    The MemoryOrchestrator manages the three memory tiers and coordinates
    consolidation and retrieval operations. It serves as the main entry point
    for the memory system, handling conversation turns and providing context
    for queries.

    Args:
        backend: The storage backend for persisting memory entries.
        embedder: The embedding provider for semantic search.
        config: The memory system configuration.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        embedder: EmbeddingProvider,
        config: MemoryConfig,
    ):
        self.backend = backend
        self.embedder = embedder
        self.config = config

        # Initialize tier managers
        self.working = WorkingMemoryManager(backend, config.working)
        self.episodic = EpisodicMemoryManager(backend, config.episodic)
        self.semantic = SemanticMemoryManager(backend, embedder, config.semantic)

        # Initialize consolidation engine
        self.consolidator = ConsolidationEngine(
            working=self.working,
            episodic=self.episodic,
            semantic=self.semantic,
            config=config.consolidation,
        )

        # Initialize context builder
        self.context_builder = MixedContextBuilder()

        # Initialize lock manager for session isolation
        self._lock_manager = AsyncLockManager()

    async def on_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        prompt_tokens: int,
    ) -> None:
        """Process a conversation turn and trigger consolidation if needed.

        This method stores the conversation turn in working memory and checks
        if consolidation should be triggered based on the prompt token count.

        Args:
            session_id: Unique identifier for the conversation session.
            user_message: The user's message content.
            assistant_response: The assistant's response content.
            prompt_tokens: The number of tokens in the current prompt.
        """
        # Use lock for session isolation
        lock_key = f"conversation_turn_{session_id}"
        async with self._lock_manager.get_lock(lock_key):
            # 1. Store turn in working memory
            await self.working.add_turn(session_id, user_message, assistant_response)

            # 2. Check if consolidation is needed
            should_consolidate = await self.consolidator.should_consolidate(prompt_tokens)

            # 3. If needed, run consolidation
            if should_consolidate:
                await self.consolidator.run(session_id)

    async def retrieve_for_context(
        self,
        current_query: str,
        recent_context: list[dict],
        max_tokens: int,
    ) -> RetrievalContext:
        """Retrieve relevant memories for context building.

        This method retrieves memories from all three tiers:
        - Working memory: Recent conversation turns
        - Episodic memory: Keyword search for relevant summaries
        - Semantic memory: Similarity search for relevant knowledge

        Args:
            current_query: The current query to search for.
            recent_context: Recent context (reserved for future use).
            max_tokens: Maximum tokens for the retrieval context.

        Returns:
            RetrievalContext containing formatted context information.
        """
        # Use lock for session isolation during retrieval
        lock_key = "retrieve_for_context"
        async with self._lock_manager.get_lock(lock_key):
            # 1. Get recent working memory
            working_memories = await self.working.get_recent(
                n=self.config.working.max_turns
            )

            # 2. Search episodic memory (keyword search)
            episodic_results = await self.episodic.search(
                query=current_query,
                limit=5,
            )

            # 3. Search semantic memory (with embedding)
            query_embedding = await self.embedder.embed([current_query])
            semantic_results = await self.semantic.search(
                query=current_query,
                embedding=query_embedding[0],
                limit=5,
            )

            # 4. Combine all retrieved results
            all_retrieved = episodic_results + semantic_results

            # 5. Use MixedContextBuilder to build RetrievalContext
            context = self.context_builder.build(
                working_memories=working_memories,
                retrieved_results=all_retrieved,
                max_tokens=max_tokens,
            )

            return context
