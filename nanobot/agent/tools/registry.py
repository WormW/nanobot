"""Tool registry for dynamic tool management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.config.schema import Config


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)
            
            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def discover_tool_plugins(registry: ToolRegistry, config: Config) -> None:
    """Discover and register external tool plugins via entry_points."""
    from importlib.metadata import entry_points

    for ep in entry_points(group="nanobot.tools"):
        try:
            register_fn = ep.load()
            register_fn(registry, config)
            logger.info("Tool plugin '{}' registered", ep.name)
        except Exception as e:
            logger.warning("Failed to load tool plugin '{}': {}", ep.name, e)


def discover_skill_tools(registry: ToolRegistry, config: Config, workspace: str | Any) -> None:
    """Discover and register native skill tools from workspace/skills and configured extra skill paths."""
    import importlib.util
    from pathlib import Path

    seen_names: set[str] = set()
    skill_roots = [Path(str(workspace)) / "skills"]
    skill_roots.extend(Path(p).expanduser() for p in config.skills.extra_paths)

    for skills_dir in skill_roots:
        if not skills_dir.is_dir():
            continue

        for init_path in sorted(skills_dir.glob("*/tools/__init__.py")):
            skill_name = init_path.parent.parent.name
            if skill_name in seen_names:
                continue
            try:
                pkg_name = f"nanobot_skill_{skill_name}".replace("-", "_")
                tools_dir = str(init_path.parent)
                spec = importlib.util.spec_from_file_location(
                    pkg_name, str(init_path),
                    submodule_search_locations=[tools_dir],
                )
                mod = importlib.util.module_from_spec(spec)
                import sys
                sys.modules[pkg_name] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "register"):
                    before = len(registry)
                    mod.register(registry, config)
                    added = len(registry) - before
                    seen_names.add(skill_name)
                    logger.info("Skill '{}' registered {} tool(s)", skill_name, added)
                else:
                    logger.debug("Skill '{}' tools/__init__.py has no register()", skill_name)
            except Exception as e:
                logger.warning("Failed to load skill tools '{}': {}", skill_name, e)
