"""Workspace binding service for binding Telegram chats to coding workspaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock
from loguru import logger


@dataclass
class WorkspaceBinding:
    """Binding between a Telegram chat and a workspace."""

    chat_id: str
    workspace_name: str
    channel: str = "telegram"
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkspaceBindingService:
    """Service for managing workspace bindings.

    Allows Telegram chats to bind to specific workspaces (tmux sessions)
    for direct message forwarding without intent classification.
    """

    _SUFFIX = "-coding"
    _ALLOWED_CHANNELS = {"telegram"}  # Currently only Telegram is supported

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._lock = FileLock(str(store_path) + ".lock")
        self._bindings: dict[str, WorkspaceBinding] = {}
        self._load()

    def _load(self) -> None:
        """Load bindings from disk."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                for item in data.get("bindings", []):
                    binding = WorkspaceBinding(
                        chat_id=item["chat_id"],
                        workspace_name=item["workspace_name"],
                        channel=item.get("channel", "telegram"),
                        metadata=item.get("metadata", {}),
                    )
                    self._bindings[binding.chat_id] = binding
                logger.info("Loaded {} workspace bindings", len(self._bindings))
            except Exception as e:
                logger.warning("Failed to load workspace bindings: {}", e)
                self._bindings = {}

    def _save(self) -> None:
        """Save bindings to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bindings": [
                {
                    "chat_id": b.chat_id,
                    "workspace_name": b.workspace_name,
                    "channel": b.channel,
                    "metadata": b.metadata,
                }
                for b in self._bindings.values()
            ]
        }
        with self._lock:
            self.store_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _validate_binding(
        self, chat_id: str, workspace_name: str, channel: str
    ) -> tuple[bool, str]:
        """Validate binding request.

        Returns (is_valid, error_message).
        """
        if channel not in self._ALLOWED_CHANNELS:
            return False, f"Channel '{channel}' does not support workspace binding"

        # Check if workspace name ends with -coding suffix
        if not workspace_name.endswith(self._SUFFIX):
            return (
                False,
                f"Workspace name must end with '{self._SUFFIX}' suffix",
            )

        # For Telegram, we need to verify it's a group chat
        # Group chat IDs are negative numbers
        if channel == "telegram":
            try:
                chat_id_int = int(chat_id)
                if chat_id_int >= 0:
                    return (
                        False,
                        "Only group chats can be bound to workspaces. "
                        f"Chat ID {chat_id} appears to be a private chat.",
                    )
            except ValueError:
                return False, f"Invalid chat ID: {chat_id}"

        return True, ""

    def bind(
        self,
        chat_id: str,
        workspace_name: str,
        channel: str = "telegram",
    ) -> tuple[bool, str]:
        """Bind a chat to a workspace.

        Returns (success, message).
        """
        # Validate
        is_valid, error = self._validate_binding(chat_id, workspace_name, channel)
        if not is_valid:
            return False, error

        # Create binding
        binding = WorkspaceBinding(
            chat_id=chat_id,
            workspace_name=workspace_name,
            channel=channel,
        )

        self._bindings[chat_id] = binding
        self._save()

        logger.info(
            "Bound {} chat {} to workspace {}",
            channel,
            chat_id,
            workspace_name,
        )
        return True, f"✅ Successfully bound to workspace `{workspace_name}`"

    def unbind(self, chat_id: str) -> tuple[bool, str]:
        """Unbind a chat from its workspace.

        Returns (success, message).
        """
        if chat_id not in self._bindings:
            return False, "❌ This chat is not bound to any workspace"

        binding = self._bindings.pop(chat_id)
        self._save()

        logger.info(
            "Unbound {} chat {} from workspace {}",
            binding.channel,
            chat_id,
            binding.workspace_name,
        )
        return (
            True,
            f"✅ Unbound from workspace `{binding.workspace_name}`",
        )

    def get_binding(self, chat_id: str) -> WorkspaceBinding | None:
        """Get the binding for a chat, if any."""
        return self._bindings.get(chat_id)

    def is_bound(self, chat_id: str) -> bool:
        """Check if a chat is bound to a workspace."""
        return chat_id in self._bindings

    def get_workspace(self, chat_id: str) -> str | None:
        """Get the workspace name for a bound chat."""
        binding = self._bindings.get(chat_id)
        return binding.workspace_name if binding else None

    def list_bindings(self) -> list[WorkspaceBinding]:
        """List all bindings."""
        return list(self._bindings.values())

    def get_bound_chats(self, workspace_name: str) -> list[str]:
        """Get all chat IDs bound to a specific workspace."""
        return [
            b.chat_id
            for b in self._bindings.values()
            if b.workspace_name == workspace_name
        ]
