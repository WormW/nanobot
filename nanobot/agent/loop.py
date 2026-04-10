"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.follow_up import evaluate_follow_up
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import MemoryConsolidator, Dream
from nanobot.agent.registry import AgentRegistry
from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.delegate import DelegateTool
from nanobot.agent.tools.discuss import DiscussTool
from nanobot.agent.tools.manage_agents import ManageAgentsTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GlobTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import image_placeholder_text, truncate_text as truncate_text_fn
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebToolsConfig
    from nanobot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"

class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean):]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    async def after_iteration(self, context: AgentHookContext) -> None:
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)

class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        config: "Config | None" = None,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig, WebToolsConfig

        defaults = AgentDefaults()
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.web_search_config = self.web_config.search
        self.web_proxy = self.web_config.proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(workspace, timezone=timezone)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._sticky_agents: dict[str, str] = {}  # session_key -> agent_name (sticky routing)
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._follow_up_cooldowns: dict[str, float] = {}  # session_key -> last follow-up time
        self._config = config  # Store config for later use

        # Named agent registry (always created so runtime registration works)
        named_configs = config.agents.named if config else {}
        self.agent_registry = AgentRegistry(
            workspace=workspace,
            named_configs=named_configs,
            main_model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=self.web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=self.restrict_to_workspace,
            extra_skill_paths=config.skills.extra_paths if config else None,
        )
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = MemoryConsolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                sandbox=self.exec_config.sandbox,
                path_append=self.exec_config.path_append,
                allowed_env_keys=self.exec_config.allowed_env_keys,
            ))
        if self.web_config.enable:
            self.tools.register(WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy))
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

        # Named agent tools
        if self.agent_registry:
            agent_names = lambda: [a.name for a in self.agent_registry.list_agents()]
            self.tools.register(DelegateTool(
                run_callback=self._run_named_agent,
                list_callback=agent_names,
            ))
            self.tools.register(DiscussTool(
                run_callback=self._run_named_agent,
                list_callback=agent_names,
            ))
            self.tools.register(ManageAgentsTool(registry=self.agent_registry))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from nanobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> tuple[str | None, list[str], list[dict], str]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hooks: list[AgentHook] = [loop_hook]

        # Add approval hook for Telegram channel
        if channel == "telegram":
            from nanobot.agent.tools.approval import ToolApprovalHook, format_approval_message
            from nanobot.channels.telegram import TelegramChannel

            async def send_telegram_approval(pending):
                # Get the bus to send the message
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=channel,
                        chat_id=chat_id,
                        content=format_approval_message(pending),
                        metadata={"parse_mode": "Markdown"},
                    )
                )

            approval_hook = ToolApprovalHook(
                session_key=session.key if session else f"{channel}:{chat_id}",
                chat_id=chat_id,
                channel=channel,
                request_callback=send_telegram_approval,
            )
            hooks.append(approval_hook)

        if self._extra_hooks:
            hooks.extend(self._extra_hooks)

        hook: AgentHook = CompositeHook(hooks) if len(hooks) > 1 else hooks[0]

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            workspace=self.workspace,
            session_key=session.key if session else None,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            progress_callback=on_progress,
            checkpoint_callback=_checkpoint,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            # Compute the effective session key before dispatching
            # This ensures /stop command can find tasks correctly when unified session is enabled
            effective_key = UNIFIED_SESSION_KEY if self._unified_session and not msg.session_key_override else msg.session_key
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(lambda t, k=effective_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        if self._unified_session and not msg.session_key_override:
            msg = dataclasses.replace(msg, session_key_override=UNIFIED_SESSION_KEY)
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        meta = dict(msg.metadata or {})
                        meta["_stream_delta"] = True
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata=meta,
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        meta = dict(msg.metadata or {})
                        meta["_stream_end"] = True
                        meta["_resuming"] = resuming
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata=meta,
                        ))
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            await self.consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
            )
            final_content, _, all_msgs, _ = await self._run_agent_loop(
                messages, session=session, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        # BTW (By The Way) - side question without polluting context
        if raw.lower().startswith("/btw ") or raw.lower() == "/btw":
            side_question = raw[5:].strip() if raw.lower().startswith("/btw ") else ""
            if not side_question:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Usage: /btw <your side question>",
                )
            return await self._handle_btw(msg, side_question, session)

        # @mention routing to named agents (with sticky routing)
        if self.agent_registry:
            match = self.agent_registry.match_mention(raw)
            if match:
                agent_name, stripped_msg = match
                if agent_name in self.agent_registry._RESERVED_NAMES:
                    # @main / @nanobot → clear sticky, fall through to main agent
                    self._sticky_agents.pop(key, None)
                    msg = InboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content=stripped_msg, sender_id=msg.sender_id,
                        media=msg.media, metadata=msg.metadata,
                    )
                else:
                    # @agent_name → set sticky, route to named agent
                    self._sticky_agents[key] = agent_name
                    return await self._process_named_agent_message(
                        msg, agent_name, stripped_msg, on_progress=on_progress,
                    )
            elif key in self._sticky_agents:
                # No @mention but sticky is set → route to sticky agent
                return await self._process_named_agent_message(
                    msg, self._sticky_agents[key], raw, on_progress=on_progress,
                )

        await self.consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs, stop_reason = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            session=session,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
        )

        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        self._save_turn(session, all_msgs, 1 + len(history))
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

        # Schedule follow-up question (non-blocking)
        if (
            self._config
            and self._config.agents.follow_up.enabled
            and final_content
            and msg.channel not in {"cli", "system"}
        ):
            self._schedule_background(
                self._maybe_send_follow_up(
                    msg.channel, msg.chat_id, final_content, session,
                )
            )

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason != "error":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=meta,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": name,
                "content": "Error: Task interrupted before this tool finished.",
                "timestamp": datetime.now().isoformat(),
            })

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_runtime_checkpoint(session)
        return True

    async def _handle_btw(
        self,
        msg: InboundMessage,
        side_question: str,
        session: Session,
    ) -> OutboundMessage:
        """Handle a BTW (By The Way) side question.
        
        BTW allows users to ask a quick side question without polluting the
        main session context. The question and answer are not saved to history.
        """
        logger.info("BTW query from {}:{}: {}", msg.channel, msg.sender_id, side_question[:80])
        
        # Build context snapshot from current session
        history = session.get_history(max_messages=10)  # Last 10 messages for context
        
        # Create BTW prompt with context
        btw_prompt = (
            "[BTW - Side Question]\n\n"
            "The user has asked a quick side question during an ongoing conversation. "
            "Answer this question using the conversation context below, but do NOT "
            "continue or complete any unfinished task from the main conversation. "
            "This is a standalone question.\n\n"
            f"## Conversation Context (for reference only)\n"
            f"{self._format_history_for_btw(history)}\n\n"
            f"## Side Question\n{side_question}\n\n"
            "Provide a brief, helpful answer."
        )
        
        try:
            # Call LLM without tools - this is a one-shot query
            response = await self.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering a quick side question."},
                    {"role": "user", "content": btw_prompt},
                ],
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
            )
            
            answer = response.content or "(No response)"
            
            # Return with special metadata marking this as BTW response
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=answer,
                metadata={
                    "_btw": True,  # Mark as BTW response
                    "_ephemeral": True,  # Indicates this is temporary
                },
            )
        except Exception as e:
            logger.exception("BTW query failed")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Sorry, I couldn't answer that side question: {e}",
                metadata={"_btw": True, "_error": True},
            )
    
    def _format_history_for_btw(self, history: list[dict]) -> str:
        """Format conversation history for BTW context."""
        lines = []
        for m in history[-10:]:  # Last 10 messages
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, str):
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _maybe_send_follow_up(
        self,
        channel: str,
        chat_id: str,
        last_response: str,
        session: Session,
    ) -> None:
        """Evaluate and optionally send a follow-up question after responding."""
        import random

        cfg = self._config.agents.follow_up

        # Cooldown check
        key = f"{channel}:{chat_id}"
        last_time = self._follow_up_cooldowns.get(key, 0)
        if time.monotonic() - last_time < cfg.cooldown_s:
            return

        # Frequency limiter
        if random.random() > cfg.max_frequency:
            return

        await asyncio.sleep(cfg.delay_s)

        # Build conversation summary from recent messages
        history = session.get_history(max_messages=6)
        tail = "\n".join(
            f"{m['role']}: {str(m.get('content', ''))[:200]}"
            for m in history[-6:]
            if isinstance(m.get('content'), str)
        )

        model = cfg.model or self.model
        should_ask, question = await evaluate_follow_up(
            tail, last_response, self.provider, model,
        )

        if should_ask and question:
            self._follow_up_cooldowns[key] = time.monotonic()
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel, chat_id=chat_id, content=question,
            ))

    async def _process_named_agent_message(
        self,
        msg: InboundMessage,
        agent_name: str,
        content: str,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message routed directly to a named agent via @mention."""
        agent = self.agent_registry.get(agent_name)
        if not agent:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Agent '{agent_name}' not found.",
            )

        logger.info("Routing @{} message from {}:{}", agent_name, msg.channel, msg.chat_id)
        result = await self._run_named_agent(
            agent_name, content, msg.channel, msg.chat_id,
            media=msg.media if msg.media else None,
        )
        meta = dict(msg.metadata or {})
        meta["_agent"] = agent_name
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=result, metadata=meta,
        )

    async def _run_named_agent(
        self,
        agent_name: str,
        task: str,
        channel: str,
        chat_id: str,
        media: list[str] | None = None,
    ) -> str:
        """Run a named agent with its own session, context, and tools. Used by delegate/discuss/mention."""
        from nanobot.agent.memory import MemoryConsolidator

        agent = self.agent_registry.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found."

        model = self.agent_registry.get_model(agent)
        provider = self._get_provider_for_model(model)
        session_key = f"{channel}:{chat_id}:agent:{agent_name}"
        session = self.sessions.get_or_create(session_key)

        # Memory consolidation for this agent's session
        consolidator = MemoryConsolidator(
            workspace=agent.workspace,
            provider=provider,
            model=model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=agent.context.build_messages,
            get_tool_definitions=agent.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        await consolidator.maybe_consolidate_by_tokens(session)

        history = session.get_history(max_messages=0)
        messages = agent.context.build_messages(
            history=history,
            current_message=task,
            media=media,
            channel=channel,
            chat_id=chat_id,
        )

        # Run using AgentRunner
        runner = self.agent_registry.get_runner(agent, provider)
        max_iterations = agent.config.max_iterations or 40

        result = await runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=agent.tools,
            model=model,
            max_iterations=max_iterations,
            error_message="Agent encountered an error.",
            concurrent_tools=True,
        ))

        # Save turn to the agent's session
        self._save_turn(session, result.messages, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(consolidator.maybe_consolidate_by_tokens(session))

        final_content = result.final_content or f"Agent '{agent_name}' completed."
        logger.info("[{}] completed, response: {}", agent_name, final_content[:120])
        return final_content

    def _get_provider_for_model(self, model: str) -> "LLMProvider":
        """Get the appropriate provider for a model, creating a new one if needed."""
        main_prefix = self.model.split("/", 1)[0] if "/" in self.model else ""
        agent_prefix = model.split("/", 1)[0] if "/" in model else ""

        if agent_prefix == main_prefix or not agent_prefix:
            return self.provider

        # Different provider prefix — try to build a new provider
        if self._config:
            saved_model = self._config.agents.defaults.model
            self._config.agents.defaults.model = model
            try:
                from nanobot.cli.commands import _make_provider
                provider = _make_provider(self._config)
                return provider
            except Exception as e:
                logger.warning("Failed to create provider for model {}: {}, falling back", model, e)
            finally:
                self._config.agents.defaults.model = saved_model

        return self.provider

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        hooks: list[AgentHook] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)

        # Merge runtime hooks with extra hooks
        prev_hooks = self._extra_hooks
        if hooks:
            self._extra_hooks = list(prev_hooks) + list(hooks)

        try:
            return await self._process_message(
                msg, session_key=session_key, on_progress=on_progress,
                on_stream=on_stream, on_stream_end=on_stream_end,
            )
        finally:
            self._extra_hooks = prev_hooks
