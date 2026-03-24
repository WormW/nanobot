"""Context engine interfaces for recall/capture backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class TurnCapture:
    """Conversation turn to archive into a context engine."""

    session_id: str
    channel: str
    chat_id: str
    user_text: str
    assistant_text: str


class ContextEngine(ABC):
    """Abstract recall/capture backend."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the engine is initialized and usable."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine."""

    @abstractmethod
    async def close(self) -> None:
        """Close engine resources."""

    @abstractmethod
    async def recall(self, *, session_id: str, query: str) -> str:
        """Return prompt-safe recalled context for a turn."""

    @abstractmethod
    async def capture(self, turn: TurnCapture) -> None:
        """Archive a completed turn into the engine."""
