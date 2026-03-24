from pathlib import Path
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.context_engine import TurnCapture
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.viking import VikingContextProvider
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ChannelsConfig, Config, VikingConfig
from nanobot.providers.base import LLMResponse


def _make_config() -> Config:
    return Config.model_validate({
        "channels": ChannelsConfig().model_dump(),
        "viking": VikingConfig(enabled=True).model_dump(),
    })


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="Hello", tool_calls=[]))
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        channels_config=ChannelsConfig(),
        config=_make_config(),
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def test_viking_normalizes_findresult_style_objects() -> None:
    provider = VikingContextProvider(VikingConfig(enabled=True))
    hit = MagicMock(
        uri="viking://user/default/memories/preferences/m1",
        title="Preference",
        score=0.92,
        summary="User prefers Chinese replies",
        snippet="User prefers Chinese replies on Telegram",
    )
    raw = MagicMock(memories=[hit], resources=[], skills=[], query_results=[])

    items = provider._normalize_search_results(raw)

    assert len(items) == 1
    assert items[0].uri == "viking://user/default/memories/preferences/m1"
    assert items[0].score == 0.92
    assert "Chinese replies" in items[0].summary


@pytest.mark.asyncio
async def test_process_message_injects_recalled_viking_context(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    fake_viking = MagicMock()
    fake_viking.is_ready = True
    fake_viking.recall = AsyncMock(return_value="memory hit")
    fake_viking.capture = AsyncMock()
    loop._context_engine = fake_viking

    captured = {}
    original_build = loop.context.build_messages

    def _capture_messages(*args, **kwargs):
        msgs = original_build(*args, **kwargs)
        captured["before_inject"] = msgs
        return msgs

    loop.context.build_messages = _capture_messages  # type: ignore[method-assign]

    seen = {}

    async def _run_agent_loop(initial_messages, on_progress=None, agent_name="main"):
        seen["messages"] = initial_messages
        return "Hello", [], initial_messages, False

    loop._run_agent_loop = _run_agent_loop  # type: ignore[method-assign]
    loop._schedule_background = lambda coro: asyncio.get_event_loop().create_task(coro)  # type: ignore[method-assign]

    msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="Where did we leave off?")
    result = await loop._process_message(msg)

    assert result is not None
    assert seen["messages"][1]["role"] == "system"
    assert "memory hit" in seen["messages"][1]["content"]
    assert seen["messages"][2]["role"] == "user"


@pytest.mark.asyncio
async def test_build_context_capture_prefers_same_target_message_tool_output(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    fake_engine = MagicMock()
    fake_engine.is_ready = True
    loop._context_engine = fake_engine

    from nanobot.session.manager import Session

    session = Session(key="telegram:c1")
    message_tool = loop.tools.get("message")
    assert isinstance(message_tool, MessageTool)
    message_tool._last_sent_message = OutboundMessage(
        channel="telegram",
        chat_id="c1",
        content="Real delivered text",
    )
    msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="Hi")

    capture = loop._build_context_capture(session, msg, "Done")

    assert capture == TurnCapture(
        session_id="telegram:c1",
        channel="telegram",
        chat_id="c1",
        user_text="Hi",
        assistant_text="Real delivered text",
    )


@pytest.mark.asyncio
async def test_capture_context_turn_uses_engine(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    fake_engine = MagicMock()
    fake_engine.is_ready = True
    fake_engine.capture = AsyncMock()
    loop._context_engine = fake_engine

    turn = TurnCapture(
        session_id="telegram:c1",
        channel="telegram",
        chat_id="c1",
        user_text="Hi",
        assistant_text="Hello",
    )
    await loop._capture_context_turn(turn)

    fake_engine.capture.assert_awaited_once_with(turn)
