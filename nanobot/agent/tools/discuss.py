"""Discuss tool for collaborative discussion with named agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool


@dataclass
class DiscussTool(Tool):
    """Discuss a topic with a named agent (collaborative, not task-based)."""

    run_callback: Callable[[str, str], Awaitable[str]] | None = None
    list_callback: Callable[[], list[str]] | None = None

    name: str = "discuss"
    description: str = (
        "Discuss a topic with a named agent. "
        "This is a collaborative conversation where the agent provides input, "
        "feedback, or ideas. Unlike `delegate`, this does not assign a task to complete."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        agents = self.list_callback() if self.list_callback else []
        agent_list = ", ".join(f'"{a}"' for a in agents) if agents else "(none configured)"
        return {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": f"Name of the agent to discuss with. Available: {agent_list}",
                },
                "topic": {
                    "type": "string",
                    "description": "The topic to discuss. Be clear about what perspective or input you want.",
                },
            },
            "required": ["agent", "topic"],
        }

    async def execute(self, agent: str, topic: str) -> str:
        """Discuss topic with named agent."""
        if not self.run_callback:
            return "Error: discuss tool not configured"
        # Add a prefix to differentiate from delegation
        prompt = f"[Discussion] {topic}\n\nThis is a collaborative discussion. Share your thoughts and perspective."
        return await self.run_callback(agent, prompt)

    def set_callback(
        self,
        run_callback: Callable[[str, str], Awaitable[str]] | None = None,
        list_callback: Callable[[], list[str]] | None = None,
    ) -> None:
        """Set callbacks after initialization."""
        if run_callback:
            self.run_callback = run_callback
        if list_callback:
            self.list_callback = list_callback
