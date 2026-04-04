"""MemoryHook for integrating memory system with AgentLoop.

This module provides the MemoryHook class, which extends AgentHook to
integrate the memory system with the agent's lifecycle. It handles
memory retrieval before iterations and memory storage after iterations.
"""

from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.memory.orchestrator import MemoryOrchestrator
from nanobot.agent.memory.context_builder import RetrievalContext


class MemoryHook(AgentHook):
    """Hook for integrating memory system with AgentLoop.

    This hook extends AgentHook to provide memory functionality:
    - before_iteration: Retrieves relevant memories and injects them into the system prompt
    - after_iteration: Saves the conversation turn to memory

    Args:
        orchestrator: The MemoryOrchestrator instance to use for memory operations
    """

    def __init__(self, orchestrator: MemoryOrchestrator):
        self.orchestrator = orchestrator

    async def before_iteration(self, context: AgentHookContext) -> None:
        """Inject relevant memories into the system prompt before iteration.

        This method:
        1. Extracts the last user message from context.messages
        2. Calls orchestrator.retrieve_for_context() to get relevant memories
        3. If memories have content, injects them into the system message

        Args:
            context: The AgentHookContext containing messages and other state
        """
        try:
            if not context.messages:
                logger.debug("No messages in context, skipping memory retrieval")
                return

            # Extract last user message
            last_user_msg = self._extract_last_user_message(context.messages)
            if not last_user_msg:
                logger.debug("No user message found, skipping memory retrieval")
                return

            # Retrieve relevant memories
            retrieval_context = await self.orchestrator.retrieve_for_context(
                current_query=last_user_msg,
                recent_context=context.messages[-5:] if len(context.messages) >= 5 else context.messages,
                max_tokens=1500,
            )

            # Inject memory if content exists
            if retrieval_context.has_content():
                memory_prompt = retrieval_context.to_system_prompt_addition()
                self._inject_memory_to_system(context, memory_prompt)
                logger.debug(
                    "Memory injected with {} debug sources",
                    len(retrieval_context.debug_sources),
                )
        except Exception as e:
            logger.warning(f"Memory injection failed: {e}")
            # Continue without memory injection

    async def after_iteration(self, context: AgentHookContext) -> None:
        """Save the conversation turn to memory after iteration.

        This method:
        1. Extracts the session_id from context
        2. Gets the user message and assistant response
        3. Calls orchestrator.on_conversation_turn() to save the turn

        Args:
            context: The AgentHookContext containing response and other state
        """
        try:
            session_id = self._extract_session_id(context)
            if not session_id:
                logger.debug("No session ID found, skipping memory storage")
                return

            if not context.response:
                logger.debug("No response in context, skipping memory storage")
                return

            # Extract user message and assistant response
            user_message = self._extract_last_user_message(context.messages)
            assistant_response = context.response.content or ""

            if not user_message:
                logger.debug("No user message found, skipping memory storage")
                return

            # Calculate prompt tokens from usage
            prompt_tokens = self._calculate_prompt_tokens(context)

            # Save the conversation turn
            await self.orchestrator.on_conversation_turn(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                prompt_tokens=prompt_tokens,
            )
            logger.debug("Conversation turn saved to memory for session {}", session_id)
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")
            # Continue without saving

    def _extract_last_user_message(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract the last user message from the message list.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            The content of the last user message, or None if not found
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None

    def _extract_session_id(self, context: AgentHookContext) -> str | None:
        """Extract the session ID from the context.

        The session ID is extracted from context.session.session_key if available.

        Args:
            context: The AgentHookContext containing session information

        Returns:
            The session ID string, or None if not found
        """
        # Try to get session_key from context.session if it exists
        if hasattr(context, "session") and context.session is not None:
            session = context.session
            if hasattr(session, "session_key"):
                return session.session_key
            if hasattr(session, "id"):
                return session.id
            # If session is a dict-like object
            if isinstance(session, dict):
                return session.get("session_key") or session.get("id")
        return None

    def _inject_memory_to_system(
        self,
        context: AgentHookContext,
        memory_prompt: str,
    ) -> None:
        """Inject memory prompt into the system message.

        If a system message exists, appends the memory prompt to it.
        If no system message exists, inserts a new one at the start.

        Args:
            context: The AgentHookContext containing messages
            memory_prompt: The memory prompt text to inject
        """
        # Find existing system message
        for msg in context.messages:
            if msg.get("role") == "system":
                # Append memory prompt to existing system message
                original_content = msg.get("content", "")
                msg["content"] = f"{original_content}\n\n{memory_prompt}"
                return

        # No system message found, insert new one at start
        context.messages.insert(0, {"role": "system", "content": memory_prompt})

    def _calculate_prompt_tokens(self, context: AgentHookContext) -> int:
        """Calculate the prompt tokens from context usage.

        Args:
            context: The AgentHookContext containing usage information

        Returns:
            The number of prompt tokens, or 0 if not available
        """
        if context.usage and "prompt_tokens" in context.usage:
            return context.usage["prompt_tokens"]
        if context.usage and "input_tokens" in context.usage:
            return context.usage["input_tokens"]
        return 0
