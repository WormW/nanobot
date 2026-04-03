"""Unit tests for MemoryHook module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.memory.hook import MemoryHook
from nanobot.agent.memory.orchestrator import MemoryOrchestrator
from nanobot.agent.memory.context_builder import RetrievalContext
from nanobot.agent.memory.types import MemoryEntry, MemoryTier
from nanobot.providers.base import LLMResponse


class TestMemoryHook:
    """Tests for MemoryHook class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock MemoryOrchestrator."""
        return MagicMock(spec=MemoryOrchestrator)

    @pytest.fixture
    def memory_hook(self, mock_orchestrator):
        """Create a MemoryHook instance with mock orchestrator."""
        return MemoryHook(orchestrator=mock_orchestrator)

    @pytest.fixture
    def sample_context(self):
        """Create a sample AgentHookContext."""
        return AgentHookContext(
            iteration=1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )

    @pytest.fixture
    def sample_context_with_response(self):
        """Create a sample AgentHookContext with response."""
        # Use a mock object that mimics AgentHookContext but allows adding session
        context = MagicMock(spec=AgentHookContext)
        context.iteration = 1
        context.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        context.response = LLMResponse(content="I'm doing well, thank you!")
        context.usage = {"prompt_tokens": 50, "completion_tokens": 10}
        context.session = MagicMock()
        context.session.session_key = "test-session-123"
        return context

    class TestBeforeIteration:
        """Tests for before_iteration method."""

        @pytest.mark.asyncio
        async def test_before_iteration_empty_messages(self, memory_hook, mock_orchestrator):
            """Test before_iteration with empty messages."""
            context = AgentHookContext(iteration=1, messages=[])

            await memory_hook.before_iteration(context)

            mock_orchestrator.retrieve_for_context.assert_not_called()

        @pytest.mark.asyncio
        async def test_before_iteration_no_user_message(self, memory_hook, mock_orchestrator):
            """Test before_iteration with no user message."""
            context = AgentHookContext(
                iteration=1,
                messages=[{"role": "system", "content": "You are helpful."}],
            )

            await memory_hook.before_iteration(context)

            mock_orchestrator.retrieve_for_context.assert_not_called()

        @pytest.mark.asyncio
        async def test_before_iteration_with_user_message(self, memory_hook, mock_orchestrator, sample_context):
            """Test before_iteration retrieves memory for user message."""
            # Setup mock return value
            mock_context = RetrievalContext(
                natural_section="你们刚才聊到测试。",
                structured_facts=[{"category": "fact", "content": "test", "relevance": 0.8}],
            )
            mock_orchestrator.retrieve_for_context = AsyncMock(return_value=mock_context)

            await memory_hook.before_iteration(sample_context)

            mock_orchestrator.retrieve_for_context.assert_called_once_with(
                current_query="Hello, how are you?",
                recent_context=sample_context.messages,
                max_tokens=1500,
            )

        @pytest.mark.asyncio
        async def test_before_iteration_injects_memory(self, memory_hook, mock_orchestrator, sample_context):
            """Test before_iteration injects memory into system message."""
            mock_context = RetrievalContext(
                natural_section="你们刚才聊到测试。",
            )
            mock_orchestrator.retrieve_for_context = AsyncMock(return_value=mock_context)

            await memory_hook.before_iteration(sample_context)

            # Check that system message was updated
            assert "你们刚才聊到测试。" in sample_context.messages[0]["content"]

        @pytest.mark.asyncio
        async def test_before_iteration_creates_system_message(self, memory_hook, mock_orchestrator):
            """Test before_iteration creates system message if none exists."""
            context = AgentHookContext(
                iteration=1,
                messages=[{"role": "user", "content": "Hello"}],
            )
            mock_context = RetrievalContext(
                natural_section="相关背景信息",
            )
            mock_orchestrator.retrieve_for_context = AsyncMock(return_value=mock_context)

            await memory_hook.before_iteration(context)

            # Check that system message was created
            assert context.messages[0]["role"] == "system"
            assert "相关背景信息" in context.messages[0]["content"]

        @pytest.mark.asyncio
        async def test_before_iteration_no_content_no_injection(self, memory_hook, mock_orchestrator, sample_context):
            """Test before_iteration does not inject when no content."""
            mock_context = RetrievalContext()  # Empty context
            mock_orchestrator.retrieve_for_context = AsyncMock(return_value=mock_context)

            original_content = sample_context.messages[0]["content"]
            await memory_hook.before_iteration(sample_context)

            # Check that system message was not modified
            assert sample_context.messages[0]["content"] == original_content

        @pytest.mark.asyncio
        async def test_before_iteration_handles_exception(self, memory_hook, mock_orchestrator, sample_context):
            """Test before_iteration handles exceptions gracefully."""
            mock_orchestrator.retrieve_for_context = AsyncMock(side_effect=Exception("Test error"))

            # Should not raise
            await memory_hook.before_iteration(sample_context)

    class TestAfterIteration:
        """Tests for after_iteration method."""

        @pytest.mark.asyncio
        async def test_after_iteration_no_session_id(self, memory_hook, mock_orchestrator):
            """Test after_iteration with no session ID."""
            context = AgentHookContext(
                iteration=1,
                messages=[{"role": "user", "content": "Hello"}],
                response=LLMResponse(content="Hi!"),
            )

            await memory_hook.after_iteration(context)

            mock_orchestrator.on_conversation_turn.assert_not_called()

        @pytest.mark.asyncio
        async def test_after_iteration_no_response(self, memory_hook, mock_orchestrator):
            """Test after_iteration with no response."""
            context = MagicMock(spec=AgentHookContext)
            context.messages = [{"role": "user", "content": "Hello"}]
            context.session = MagicMock()
            context.session.session_key = "test-session"
            context.response = None

            await memory_hook.after_iteration(context)

            mock_orchestrator.on_conversation_turn.assert_not_called()

        @pytest.mark.asyncio
        async def test_after_iteration_no_user_message(self, memory_hook, mock_orchestrator):
            """Test after_iteration with no user message."""
            context = MagicMock(spec=AgentHookContext)
            context.messages = [{"role": "system", "content": "You are helpful."}]
            context.response = LLMResponse(content="Hi!")
            context.session = MagicMock()
            context.session.session_key = "test-session"

            await memory_hook.after_iteration(context)

            mock_orchestrator.on_conversation_turn.assert_not_called()

        @pytest.mark.asyncio
        async def test_after_iteration_saves_turn(self, memory_hook, mock_orchestrator, sample_context_with_response):
            """Test after_iteration saves conversation turn."""
            await memory_hook.after_iteration(sample_context_with_response)

            mock_orchestrator.on_conversation_turn.assert_called_once_with(
                session_id="test-session-123",
                user_message="Hello, how are you?",
                assistant_response="I'm doing well, thank you!",
                prompt_tokens=50,
            )

        @pytest.mark.asyncio
        async def test_after_iteration_handles_exception(self, memory_hook, mock_orchestrator, sample_context_with_response):
            """Test after_iteration handles exceptions gracefully."""
            mock_orchestrator.on_conversation_turn = AsyncMock(side_effect=Exception("Test error"))

            # Should not raise
            await memory_hook.after_iteration(sample_context_with_response)

        @pytest.mark.asyncio
        async def test_after_iteration_uses_input_tokens(self, memory_hook, mock_orchestrator):
            """Test after_iteration uses input_tokens if prompt_tokens not available."""
            context = MagicMock(spec=AgentHookContext)
            context.messages = [{"role": "user", "content": "Hello"}]
            context.response = LLMResponse(content="Hi!")
            context.usage = {"input_tokens": 100, "completion_tokens": 5}
            context.session = MagicMock()
            context.session.session_key = "test-session"

            await memory_hook.after_iteration(context)

            mock_orchestrator.on_conversation_turn.assert_called_once_with(
                session_id="test-session",
                user_message="Hello",
                assistant_response="Hi!",
                prompt_tokens=100,
            )

    class TestExtractLastUserMessage:
        """Tests for _extract_last_user_message helper."""

        def test_extract_last_user_message_found(self, memory_hook):
            """Test extracting last user message."""
            messages = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Last message"},
            ]

            result = memory_hook._extract_last_user_message(messages)

            assert result == "Last message"

        def test_extract_last_user_message_not_found(self, memory_hook):
            """Test extracting when no user message exists."""
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "Hello"},
            ]

            result = memory_hook._extract_last_user_message(messages)

            assert result is None

        def test_extract_last_user_message_empty_content(self, memory_hook):
            """Test extracting with empty content."""
            messages = [
                {"role": "user", "content": "   "},
                {"role": "user", "content": "Valid message"},
            ]

            result = memory_hook._extract_last_user_message(messages)

            assert result == "Valid message"

        def test_extract_last_user_message_non_string_content(self, memory_hook):
            """Test extracting with non-string content."""
            messages = [
                {"role": "user", "content": ["list", "content"]},
            ]

            result = memory_hook._extract_last_user_message(messages)

            assert result is None

    class TestExtractSessionId:
        """Tests for _extract_session_id helper."""

        def test_extract_session_id_from_session_key(self, memory_hook):
            """Test extracting session_id from session_key."""
            context = MagicMock(spec=AgentHookContext)
            context.session = MagicMock()
            context.session.session_key = "session-123"

            result = memory_hook._extract_session_id(context)

            assert result == "session-123"

        def test_extract_session_id_from_id(self, memory_hook):
            """Test extracting session_id from id attribute."""
            context = MagicMock(spec=AgentHookContext)
            context.session = MagicMock()
            context.session.id = "session-456"
            # No session_key attribute
            del context.session.session_key

            result = memory_hook._extract_session_id(context)

            assert result == "session-456"

        def test_extract_session_id_from_dict(self, memory_hook):
            """Test extracting session_id from dict session."""
            context = MagicMock(spec=AgentHookContext)
            context.session = {"session_key": "session-789"}

            result = memory_hook._extract_session_id(context)

            assert result == "session-789"

        def test_extract_session_id_no_session(self, memory_hook):
            """Test extracting when no session exists."""
            context = MagicMock(spec=AgentHookContext)
            context.session = None

            result = memory_hook._extract_session_id(context)

            assert result is None

        def test_extract_session_id_no_attributes(self, memory_hook):
            """Test extracting when session has no id attributes."""
            context = MagicMock(spec=AgentHookContext)
            context.session = MagicMock()
            # Remove both session_key and id
            del context.session.session_key
            del context.session.id

            result = memory_hook._extract_session_id(context)

            assert result is None

    class TestInjectMemoryToSystem:
        """Tests for _inject_memory_to_system helper."""

        def test_inject_to_existing_system_message(self, memory_hook):
            """Test injecting memory to existing system message."""
            context = MagicMock(spec=AgentHookContext)
            context.messages = [
                {"role": "system", "content": "Original system prompt."},
                {"role": "user", "content": "Hello"},
            ]

            memory_hook._inject_memory_to_system(context, "Memory content")

            assert "Original system prompt." in context.messages[0]["content"]
            assert "Memory content" in context.messages[0]["content"]
            assert len(context.messages) == 2

        def test_inject_creates_new_system_message(self, memory_hook):
            """Test injecting memory creates new system message if none exists."""
            context = MagicMock(spec=AgentHookContext)
            context.messages = [
                {"role": "user", "content": "Hello"},
            ]

            memory_hook._inject_memory_to_system(context, "Memory content")

            assert context.messages[0]["role"] == "system"
            assert context.messages[0]["content"] == "Memory content"
            assert len(context.messages) == 2

        def test_inject_preserves_other_messages(self, memory_hook):
            """Test injecting memory preserves other messages."""
            context = MagicMock(spec=AgentHookContext)
            context.messages = [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second"},
            ]

            memory_hook._inject_memory_to_system(context, "Memory")

            assert context.messages[1]["role"] == "user"
            assert context.messages[1]["content"] == "First"
            assert context.messages[2]["role"] == "assistant"
            assert context.messages[3]["role"] == "user"
            assert context.messages[3]["content"] == "Second"

    class TestCalculatePromptTokens:
        """Tests for _calculate_prompt_tokens helper."""

        def test_calculate_from_prompt_tokens(self, memory_hook):
            """Test calculating from prompt_tokens."""
            context = MagicMock(spec=AgentHookContext)
            context.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            result = memory_hook._calculate_prompt_tokens(context)

            assert result == 100

        def test_calculate_from_input_tokens(self, memory_hook):
            """Test calculating from input_tokens."""
            context = MagicMock(spec=AgentHookContext)
            context.usage = {"input_tokens": 200, "output_tokens": 100}

            result = memory_hook._calculate_prompt_tokens(context)

            assert result == 200

        def test_calculate_no_usage(self, memory_hook):
            """Test calculating when no usage exists."""
            context = MagicMock(spec=AgentHookContext)
            context.usage = {}

            result = memory_hook._calculate_prompt_tokens(context)

            assert result == 0

        def test_calculate_none_usage(self, memory_hook):
            """Test calculating when usage is None."""
            context = MagicMock(spec=AgentHookContext)
            context.usage = None

            result = memory_hook._calculate_prompt_tokens(context)

            assert result == 0
