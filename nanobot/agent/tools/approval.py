"""Tool approval system for dangerous operations.

Requires user confirmation before executing tools marked as `require_approval`.
Supports Telegram confirmation messages with timeout.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.providers.base import ToolCallRequest


@dataclass
class PendingApproval:
    """A pending tool execution awaiting user confirmation."""

    approval_id: str
    tool_call: ToolCallRequest
    session_key: str
    chat_id: str
    channel: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool = False
    timeout_seconds: float = 60.0


class ToolApprovalManager:
    """Manages pending tool approvals across all sessions."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}
        self._session_approvals: dict[str, set[str]] = {}  # session_key -> approval_ids

    def create_approval(
        self,
        tool_call: ToolCallRequest,
        session_key: str,
        chat_id: str,
        channel: str,
        timeout_seconds: float = 60.0,
    ) -> PendingApproval:
        """Create a new pending approval request."""
        approval_id = str(uuid.uuid4())[:8]
        pending = PendingApproval(
            approval_id=approval_id,
            tool_call=tool_call,
            session_key=session_key,
            chat_id=chat_id,
            channel=channel,
            timeout_seconds=timeout_seconds,
        )
        self._pending[approval_id] = pending
        self._session_approvals.setdefault(session_key, set()).add(approval_id)
        return pending

    def get_pending(self, approval_id: str) -> PendingApproval | None:
        """Get a pending approval by ID."""
        return self._pending.get(approval_id)

    def approve(self, approval_id: str) -> bool:
        """Mark an approval as approved. Returns True if found."""
        pending = self._pending.get(approval_id)
        if pending is None:
            return False
        pending.approved = True
        pending.event.set()
        logger.info("Tool approval {} approved", approval_id)
        return True

    def reject(self, approval_id: str) -> bool:
        """Mark an approval as rejected. Returns True if found."""
        pending = self._pending.get(approval_id)
        if pending is None:
            return False
        pending.approved = False
        pending.event.set()
        logger.info("Tool approval {} rejected", approval_id)
        return True

    def cleanup(self, approval_id: str) -> None:
        """Remove a pending approval from tracking."""
        pending = self._pending.pop(approval_id, None)
        if pending:
            self._session_approvals.get(pending.session_key, set()).discard(approval_id)

    def get_session_approvals(self, session_key: str) -> list[PendingApproval]:
        """Get all pending approvals for a session."""
        approval_ids = self._session_approvals.get(session_key, set())
        return [self._pending[aid] for aid in approval_ids if aid in self._pending]

    def cancel_session_approvals(self, session_key: str) -> int:
        """Cancel all pending approvals for a session. Returns count cancelled."""
        approval_ids = list(self._session_approvals.get(session_key, set()))
        count = 0
        for aid in approval_ids:
            pending = self._pending.get(aid)
            if pending:
                pending.approved = False
                pending.event.set()
                count += 1
            self.cleanup(aid)
        return count


# Global approval manager instance
approval_manager = ToolApprovalManager()


# Callback type for sending approval requests
ApprovalRequestCallback = Callable[[PendingApproval], Coroutine[Any, Any, None]]


class ToolApprovalHook(AgentHook):
    """Hook that intercepts dangerous tool calls and requires user approval.

    Usage:
        hook = ToolApprovalHook(
            session_key="user:123",
            chat_id="456",
            channel="telegram",
            request_callback=send_telegram_confirmation,
        )
    """

    def __init__(
        self,
        session_key: str,
        chat_id: str,
        channel: str,
        request_callback: ApprovalRequestCallback,
        timeout_seconds: float = 60.0,
        approval_manager: ToolApprovalManager | None = None,
    ):
        super().__init__()
        self.session_key = session_key
        self.chat_id = chat_id
        self.channel = channel
        self.request_callback = request_callback
        self.timeout_seconds = timeout_seconds
        self._approval_manager = approval_manager or globals()["approval_manager"]

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Check for tools requiring approval and wait for confirmation."""
        from nanobot.agent.tools.registry import ToolRegistry

        # Get tools from context (they should be accessible via the spec)
        # We need to inspect the tool_calls and check if any require approval
        approved_tool_calls: list[ToolCallRequest] = []

        for tool_call in context.tool_calls:
            # Check if this tool requires approval
            # We need to look up the tool to check its require_approval property
            requires_approval = self._tool_requires_approval(tool_call.name)

            if not requires_approval:
                approved_tool_calls.append(tool_call)
                continue

            # Create approval request
            pending = self._approval_manager.create_approval(
                tool_call=tool_call,
                session_key=self.session_key,
                chat_id=self.chat_id,
                channel=self.channel,
                timeout_seconds=self.timeout_seconds,
            )

            # Send approval request
            try:
                await self.request_callback(pending)
            except Exception as e:
                logger.error("Failed to send approval request: {}", e)
                self._approval_manager.cleanup(pending.approval_id)
                raise RuntimeError(f"无法发送审批请求: {e}")

            # Wait for approval with timeout
            logger.info(
                "Waiting for approval {} for tool {} (timeout: {}s)",
                pending.approval_id,
                tool_call.name,
                self.timeout_seconds,
            )

            try:
                await asyncio.wait_for(
                    pending.event.wait(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._approval_manager.cleanup(pending.approval_id)
                raise RuntimeError(
                    f"工具 {tool_call.name} 执行超时：用户在 {self.timeout_seconds} 秒内未确认"
                )

            self._approval_manager.cleanup(pending.approval_id)

            if not pending.approved:
                raise RuntimeError(f"工具 {tool_call.name} 已被用户拒绝")

            approved_tool_calls.append(tool_call)

        # Update context with only approved tools
        context.tool_calls = approved_tool_calls

    def _tool_requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires approval based on name patterns.

        This is a fallback when we can't access the actual tool instance.
        """
        # List of dangerous tools that always require approval
        dangerous_patterns = [
            "exec",
            "shell",
            "spawn",
            "write_file",
            "edit_file",
            "delete",
            "remove",
            "torrent",
            "download",
        ]
        return any(pattern in tool_name.lower() for pattern in dangerous_patterns)


def format_approval_message(pending: PendingApproval) -> str:
    """Format a user-friendly approval request message."""
    tool_name = pending.tool_call.name
    args = pending.tool_call.arguments

    # Format arguments for display
    args_str = ""
    if isinstance(args, dict):
        # Filter out very long values
        display_args = {}
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 100:
                v_str = v_str[:100] + "..."
            display_args[k] = v_str
        args_str = "\n".join(f"  {k}: {v}" for k, v in display_args.items())
    else:
        args_str = str(args)[:200]

    message = f"""⚠️ **需要确认的危险操作**

工具: `{tool_name}`
参数:
{args_str}

请回复以下选项之一:
• **确认** - 批准执行此操作
• **取消** - 拒绝执行此操作

⏰ 超时时间: {int(pending.timeout_seconds)} 秒
🔑 审批ID: `{pending.approval_id}`"""

    return message
