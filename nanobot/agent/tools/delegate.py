"""Delegate tool for routing tasks to named agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool


@dataclass
class DelegateTool(Tool):
    """Delegate a task to a named agent."""

    run_callback: Callable[[str, str], Awaitable[str]] | None = None
    list_callback: Callable[[], list[str]] | None = None

    name: str = "delegate"
    description: str = (
        "Delegate a task to a named agent. "
        "The agent will process the task with its own context and tools, "
        "then return the result. Use this to distribute work across specialized agents."
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
                    "description": f"Name of the agent to delegate to. Available: {agent_list}",
                },
                "task": {
                    "type": "string",
                    "description": "The task to delegate. Be specific about what you want the agent to do.",
                },
            },
            "required": ["agent", "task"],
        }

    async def execute(self, agent: str, task: str) -> str:
        """Delegate task to named agent."""
        if not self.run_callback:
            return "Error: delegate tool not configured"
        return await self.run_callback(agent, task)

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
