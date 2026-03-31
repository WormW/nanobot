"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show current model or switch to a new one."""
    loop = ctx.loop
    msg = ctx.msg
    arg = ctx.args.strip()

    if not arg:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"Current model: {loop.model}",
            metadata={"render_as": "text"},
        )

    new_model = arg
    old_model = loop.model

    # Determine if we need to rebuild the provider (different provider prefix)
    old_prefix = old_model.split("/", 1)[0] if "/" in old_model else ""
    new_prefix = new_model.split("/", 1)[0] if "/" in new_model else ""

    if old_prefix != new_prefix and loop._config:
        saved_model = loop._config.agents.defaults.model
        loop._config.agents.defaults.model = new_model
        try:
            from nanobot.cli.commands import _make_provider
            new_provider = _make_provider(loop._config)
            loop.provider = new_provider
            loop.subagents.provider = new_provider
            loop.memory_consolidator.provider = new_provider
        except Exception as e:
            loop._config.agents.defaults.model = saved_model
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Failed to switch model: {e}",
            )

    loop.model = new_model
    loop.subagents.model = new_model
    loop.memory_consolidator.model = new_model

    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content=f"Switched model: {old_model} → {new_model}",
        metadata={"render_as": "text"},
    )


async def cmd_models(ctx: CommandContext) -> OutboundMessage:
    """List configured providers or show details for a specific provider."""
    loop = ctx.loop
    msg = ctx.msg
    arg = ctx.args.strip()

    if not loop._config:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Config not available.",
        )

    from nanobot.providers.registry import PROVIDERS

    providers_cfg = loop._config.providers

    if not arg:
        lines = ["Configured providers:"]
        for spec in PROVIDERS:
            p = getattr(providers_cfg, spec.name, None)
            if not p:
                continue
            if spec.is_oauth or spec.is_local or p.api_key:
                base = f" ({p.api_base})" if p.api_base else ""
                lines.append(f"  {spec.label}{base}")
        # Extras providers
        for name, ep in providers_cfg.extras.items():
            if ep.api_key:
                api_type = ep.api or "openai"
                model_count = len(ep.models)
                base = f" ({ep.api_base})" if ep.api_base else ""
                lines.append(f"  {name} (extras, {api_type}, {model_count} models){base}")
        if len(lines) == 1:
            lines.append("  (none)")
        lines.append(f"\nCurrent model: {loop.model}")
        lines.append("Use /models <provider> to see models.")
        lines.append("Use /model <provider>/<model_name> to switch.")
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="\n".join(lines),
            metadata={"render_as": "text"},
        )

    # Show details for a specific provider
    query = arg.lower().replace("-", "_")

    # Check extras providers first
    for key, ep in providers_cfg.extras.items():
        if key.lower().replace("-", "_") == query:
            if not ep.api_key:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Provider '{arg}' is not configured (no API key).",
                )
            api_type = ep.api or "openai"
            lines = [
                f"Provider: {key} (extras)",
                f"  API: {api_type}",
                f"  API base: {ep.api_base or '(not set)'}",
            ]
            if ep.models:
                lines.append(f"  Models ({len(ep.models)}):")
                for mc in ep.models:
                    img = "img" if mc.supports_image else "no-img"
                    think = ", think" if mc.supports_reasoning else ""
                    lines.append(f"    {mc.id}  ctx={mc.context_window}  {img}{think}")
            else:
                lines.append("  Models: (none declared, any model name accepted)")
            lines.append(f"\nUse /model {key}/<model_name> to switch.")
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="\n".join(lines),
                metadata={"render_as": "text"},
            )

    # Check registry providers
    for spec in PROVIDERS:
        if spec.name == query or spec.display_name.lower().replace(" ", "_") == query:
            p = getattr(providers_cfg, spec.name, None)
            if not p or not (spec.is_oauth or spec.is_local or p.api_key):
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Provider '{arg}' is not configured (no API key).",
                )
            lines = [
                f"Provider: {spec.label}",
                f"  Backend: {spec.backend}",
                f"  API base: {p.api_base or spec.default_api_base or '(default)'}",
                f"  Gateway: {'yes' if spec.is_gateway else 'no'}",
            ]
            if p.models:
                lines.append(f"  Models ({len(p.models)}):")
                for mc in p.models:
                    img = "img" if mc.supports_image else "no-img"
                    think = ", think" if mc.supports_reasoning else ""
                    lines.append(f"    {mc.id}  ctx={mc.context_window}  {img}{think}")
            lines.append(f"\nUse /model {spec.name}/<model_name> to switch.")
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="\n".join(lines),
                metadata={"render_as": "text"},
            )

    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content=f"Provider '{arg}' not found. Use /models to list available providers.",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/model — Show current model (or /model <name> to switch)",
        "/models — List configured providers",
        "/status — Show bot status",
        "/help — Show available commands",
    ]
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/help", cmd_help)
    router.prefix("/models ", cmd_models)
    router.prefix("/model ", cmd_model)
    router.exact("/models", cmd_models)
    router.exact("/model", cmd_model)
