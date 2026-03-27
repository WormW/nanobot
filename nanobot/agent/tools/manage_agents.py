"""Manage agents tool for runtime agent registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.registry import AgentRegistry


@dataclass
class ManageAgentsTool(Tool):
    """Register, update, or unregister named agents at runtime."""

    registry: "AgentRegistry | None" = None

    name: str = "manage_agents"
    description: str = (
        "Manage named agents at runtime. "
        "You can register new agents, update existing ones, or unregister agents. "
        "Each agent has its own workspace, identity, and can have custom tools."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["register", "unregister", "list", "get"],
                    "description": (
                        "register = create/update an agent; "
                        "unregister = remove an agent; "
                        "list = show all agents; "
                        "get = show details of a specific agent"
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Agent name (required for register, unregister, get)",
                },
                "identity": {
                    "type": "string",
                    "description": "System prompt / identity for the agent (for register)",
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Alternative names for the agent (for register)",
                },
                "model": {
                    "type": "string",
                    "description": "Override model for this agent, e.g. 'openai/gpt-4o' (for register)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        identity: str | None = None,
        aliases: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        """Execute agent management action."""
        if not self.registry:
            return "Error: Agent registry not configured"

        if action == "list":
            agents = self.registry.list_agents()
            if not agents:
                return "No named agents registered."
            lines = ["Registered agents:"]
            for agent in agents:
                alias_str = f" (aliases: {', '.join(agent.config.aliases)})" if agent.config.aliases else ""
                model_str = f" [model: {agent.config.model}]" if agent.config.model else ""
                lines.append(f"- {agent.name}{alias_str}{model_str}")
            return "\n".join(lines)

        if action == "get":
            if not name:
                return "Error: 'name' is required for get action"
            agent = self.registry.get(name)
            if not agent:
                return f"Agent '{name}' not found."
            lines = [
                f"Agent: {agent.name}",
                f"Identity: {agent.config.identity or '(default)'}"[:200],
            ]
            if agent.config.aliases:
                lines.append(f"Aliases: {', '.join(agent.config.aliases)}")
            if agent.config.model:
                lines.append(f"Model: {agent.config.model}")
            lines.append(f"Workspace: {agent.workspace}")
            return "\n".join(lines)

        if action == "unregister":
            if not name:
                return "Error: 'name' is required for unregister action"
            success = self.registry.unregister(name)
            return f"Agent '{name}' unregistered." if success else f"Agent '{name}' not found."

        if action == "register":
            if not name:
                return "Error: 'name' is required for register action"
            from nanobot.config.schema import NamedAgentConfig

            config = NamedAgentConfig(
                identity=identity or "",
                aliases=aliases or [],
                model=model,
            )
            try:
                agent = self.registry.register(name, config)
                alias_str = f" (aliases: {', '.join(agent.config.aliases)})" if agent.config.aliases else ""
                return f"Agent '{agent.name}'{alias_str} registered successfully."
            except ValueError as e:
                return f"Error: {e}"

        return f"Error: Unknown action '{action}'"
