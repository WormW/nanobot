"""Storage backends for the memory system.

This package provides various storage backend implementations for persisting
memory entries, including file system and database backends.
"""

from .filesystem import FileSystemBackend

__all__ = ["FileSystemBackend"]
