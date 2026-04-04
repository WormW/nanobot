"""Utility functions for memory system."""

import asyncio
from datetime import datetime
from typing import Any


def calculate_tokens(text: str) -> int:
    """Rough estimate of tokens in text (1 token ≈ 4 chars)."""
    return len(text) // 4


def calculate_memory_tokens(natural_section: str, structured_facts: list[dict]) -> int:
    """Calculate total tokens in retrieval context.

    Args:
        natural_section: Natural language section text
        structured_facts: List of structured fact dictionaries with 'content' key

    Returns:
        Total estimated token count
    """
    tokens = calculate_tokens(natural_section)
    for fact in structured_facts:
        tokens += calculate_tokens(fact.get("content", ""))
    return tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


class AsyncLockManager:
    """Manages async locks per session."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def acquire(self, key: str):
        """Context manager for lock acquisition."""
        lock = self.get_lock(key)
        await lock.acquire()
        return lock

    def release(self, key: str):
        """Release lock for key."""
        if key in self._locks:
            self._locks[key].release()


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON, return default on error."""
    import json
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def format_datetime(dt: datetime) -> str:
    """Format datetime for storage."""
    return dt.isoformat()


def parse_datetime(s: str) -> datetime:
    """Parse datetime from storage."""
    from datetime import datetime
    return datetime.fromisoformat(s)
