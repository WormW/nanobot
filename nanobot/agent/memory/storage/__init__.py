"""Storage backends for the memory system.

This package provides various storage backend implementations for persisting
memory entries, including file system and database backends.
"""

from .filesystem import FileSystemBackend
from .sqlite import SQLiteBackend

try:
    from .chroma import ChromaBackend
except ImportError:
    ChromaBackend = None

__all__ = ["FileSystemBackend", "SQLiteBackend"]
if ChromaBackend:
    __all__.append("ChromaBackend")
