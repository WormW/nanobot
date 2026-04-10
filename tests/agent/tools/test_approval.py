"""Tests for tool approval system."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.approval import (
    PendingApproval,
    ToolApprovalHook,
    ToolApprovalManager,
    approval_manager,
    format_approval_message,
)
from nanobot.agent.hook import AgentHookContext
from nanobot.providers.base import ToolCallRequest


@pytest.fixture
def clean_approval_manager():
    """Provide a clean approval manager for tests."""
    manager = ToolApprovalManager()
    yield manager
    # Cleanup after test
    manager._pending.clear()
    manager._session_approvals.clear()


@pytest.fixture
def mock_tool_call():
    """Create a mock tool call request."""
    return ToolCallRequest(
        id="tool-123",
        name="exec",
        arguments={"command": "ls -la"},
    )


class TestToolApprovalManager:
    """Tests for ToolApprovalManager."""

    def test_create_approval(self, clean_approval_manager, mock_tool_call):
        """Test creating a pending approval."""
        pending = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            timeout_seconds=60.0,
        )

        assert pending.tool_call == mock_tool_call
        assert pending.session_key == "test-session"
        assert pending.chat_id == "12345"
        assert pending.channel == "telegram"
        assert pending.timeout_seconds == 60.0
        assert pending.approval_id is not None
        assert pending.approved is False

    def test_get_pending(self, clean_approval_manager, mock_tool_call):
        """Test retrieving a pending approval."""
        pending = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
        )

        retrieved = clean_approval_manager.get_pending(pending.approval_id)
        assert retrieved == pending

        # Non-existent approval returns None
        assert clean_approval_manager.get_pending("nonexistent") is None

    def test_approve(self, clean_approval_manager, mock_tool_call):
        """Test approving a pending request."""
        pending = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
        )

        # Not approved yet
        assert not pending.approved
        assert not pending.event.is_set()

        # Approve it
        result = clean_approval_manager.approve(pending.approval_id)
        assert result is True
        assert pending.approved is True
        assert pending.event.is_set()

    def test_approve_nonexistent(self, clean_approval_manager):
        """Test approving a non-existent approval returns False."""
        result = clean_approval_manager.approve("nonexistent")
        assert result is False

    def test_reject(self, clean_approval_manager, mock_tool_call):
        """Test rejecting a pending request."""
        pending = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
        )

        # Reject it
        result = clean_approval_manager.reject(pending.approval_id)
        assert result is True
        assert pending.approved is False  # Rejected, not approved
        assert pending.event.is_set()

    def test_cleanup(self, clean_approval_manager, mock_tool_call):
        """Test cleaning up a pending approval."""
        pending = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
        )

        approval_id = pending.approval_id
        assert clean_approval_manager.get_pending(approval_id) is not None

        clean_approval_manager.cleanup(approval_id)
        assert clean_approval_manager.get_pending(approval_id) is None

    def test_get_session_approvals(self, clean_approval_manager, mock_tool_call):
        """Test getting all approvals for a session."""
        # Create multiple approvals for same session
        pending1 = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="session-1",
            chat_id="12345",
            channel="telegram",
        )
        pending2 = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="session-1",
            chat_id="12345",
            channel="telegram",
        )
        # Different session
        pending3 = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="session-2",
            chat_id="67890",
            channel="telegram",
        )

        session1_approvals = clean_approval_manager.get_session_approvals("session-1")
        assert len(session1_approvals) == 2
        assert pending1 in session1_approvals
        assert pending2 in session1_approvals

    def test_cancel_session_approvals(self, clean_approval_manager, mock_tool_call):
        """Test canceling all approvals for a session."""
        # Create approvals for a session
        pending1 = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="session-1",
            chat_id="12345",
            channel="telegram",
        )
        pending2 = clean_approval_manager.create_approval(
            tool_call=mock_tool_call,
            session_key="session-1",
            chat_id="12345",
            channel="telegram",
        )

        # Cancel them
        count = clean_approval_manager.cancel_session_approvals("session-1")
        assert count == 2

        # Both should be rejected and cleaned up
        assert pending1.approved is False
        assert pending1.event.is_set()
        assert pending2.approved is False
        assert pending2.event.is_set()


class TestToolApprovalHook:
    """Tests for ToolApprovalHook."""

    @pytest.mark.asyncio
    async def test_no_approval_needed(self, clean_approval_manager):
        """Test that safe tools don't require approval."""
        request_callback = AsyncMock()
        hook = ToolApprovalHook(
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            request_callback=request_callback,
        )

        # Override the approval manager
        hook._approval_manager = clean_approval_manager

        # Create context with safe tool (read_file doesn't require approval)
        context = AgentHookContext(
            iteration=0,
            messages=[],
            tool_calls=[
                ToolCallRequest(
                    id="tool-1",
                    name="read_file",
                    arguments={"path": "/tmp/test.txt"},
                )
            ],
        )

        await hook.before_execute_tools(context)

        # Request callback should not be called for safe tools
        request_callback.assert_not_called()
        # Tool calls should remain unchanged
        assert len(context.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_approval_required_and_approved(self, clean_approval_manager):
        """Test that dangerous tools require and wait for approval."""
        request_callback = AsyncMock()
        hook = ToolApprovalHook(
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            request_callback=request_callback,
            timeout_seconds=1.0,  # Longer timeout for test
            approval_manager=clean_approval_manager,
        )

        tool_call = ToolCallRequest(
            id="tool-1",
            name="exec",
            arguments={"command": "rm -rf /"},
        )
        context = AgentHookContext(
            iteration=0,
            messages=[],
            tool_calls=[tool_call],
        )

        # Schedule approval after a short delay
        async def approve_later():
            await asyncio.sleep(0.01)
            for approval_id, pending in clean_approval_manager._pending.items():
                clean_approval_manager.approve(approval_id)
                break

        # Run both tasks
        await asyncio.gather(
            hook.before_execute_tools(context),
            approve_later(),
        )

        # Request callback should be called
        request_callback.assert_called_once()
        # Tool should still be in context (approved)
        assert len(context.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_approval_timeout(self, clean_approval_manager):
        """Test that approval times out if no response."""
        request_callback = AsyncMock()
        hook = ToolApprovalHook(
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            request_callback=request_callback,
            timeout_seconds=0.01,  # Very short timeout
            approval_manager=clean_approval_manager,
        )

        tool_call = ToolCallRequest(
            id="tool-1",
            name="exec",
            arguments={"command": "rm -rf /"},
        )
        context = AgentHookContext(
            iteration=0,
            messages=[],
            tool_calls=[tool_call],
        )

        # Should raise RuntimeError on timeout
        with pytest.raises(RuntimeError, match="超时"):
            await hook.before_execute_tools(context)

    @pytest.mark.asyncio
    async def test_approval_rejected(self, clean_approval_manager):
        """Test that rejected approval raises error."""
        request_callback = AsyncMock()
        hook = ToolApprovalHook(
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            request_callback=request_callback,
            timeout_seconds=1.0,
            approval_manager=clean_approval_manager,
        )

        tool_call = ToolCallRequest(
            id="tool-1",
            name="exec",
            arguments={"command": "rm -rf /"},
        )
        context = AgentHookContext(
            iteration=0,
            messages=[],
            tool_calls=[tool_call],
        )

        # Schedule rejection after a short delay
        async def reject_later():
            await asyncio.sleep(0.01)
            for approval_id, pending in clean_approval_manager._pending.items():
                clean_approval_manager.reject(approval_id)
                break

        # Run both tasks
        with pytest.raises(RuntimeError, match="拒绝"):
            await asyncio.gather(
                hook.before_execute_tools(context),
                reject_later(),
            )


class TestFormatApprovalMessage:
    """Tests for format_approval_message function."""

    def test_format_with_dict_args(self):
        """Test formatting with dictionary arguments."""
        tool_call = ToolCallRequest(
            id="tool-1",
            name="exec",
            arguments={"command": "ls -la", "timeout": 60},
        )
        pending = PendingApproval(
            approval_id="abc123",
            tool_call=tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            timeout_seconds=60.0,
        )

        message = format_approval_message(pending)

        assert "exec" in message
        assert "abc123" in message
        assert "ls -la" in message
        assert "60" in message  # timeout value
        assert "确认" in message
        assert "取消" in message

    def test_format_with_long_args(self):
        """Test that long arguments are truncated."""
        long_command = "echo " + "x" * 500
        tool_call = ToolCallRequest(
            id="tool-1",
            name="exec",
            arguments={"command": long_command},
        )
        pending = PendingApproval(
            approval_id="abc123",
            tool_call=tool_call,
            session_key="test-session",
            chat_id="12345",
            channel="telegram",
            timeout_seconds=60.0,
        )

        message = format_approval_message(pending)

        # Long command should be truncated
        assert "..." in message
        assert len(message) < 800


class TestExecToolApproval:
    """Tests that exec tool requires approval."""

    def test_exec_tool_requires_approval(self):
        """Test that ExecTool has require_approval=True."""
        from nanobot.agent.tools.shell import ExecTool

        tool = ExecTool()
        assert tool.require_approval is True


class TestFilesystemToolApproval:
    """Tests that filesystem tools require approval."""

    def test_write_file_requires_approval(self, tmp_path):
        """Test that WriteFileTool has require_approval=True."""
        from nanobot.agent.tools.filesystem import WriteFileTool

        tool = WriteFileTool(workspace=tmp_path)
        assert tool.require_approval is True

    def test_edit_file_requires_approval(self, tmp_path):
        """Test that EditFileTool has require_approval=True."""
        from nanobot.agent.tools.filesystem import EditFileTool

        tool = EditFileTool(workspace=tmp_path)
        assert tool.require_approval is True
