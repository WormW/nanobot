"""Memory backfill service for importing history.jsonl to external memory systems.

This service provides incremental import of agent conversation history from
history.jsonl files to external memory storage systems like OpenViking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

# Type alias for memory sink (external storage write function)
MemorySink = Callable[[list[dict[str, Any]]], Coroutine[Any, Any, int]]


@dataclass
class BackfillOffset:
    """Offset tracking for incremental backfill."""

    session: str
    last_cursor: int = 0
    last_sync_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_imported: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "last_cursor": self.last_cursor,
            "last_sync_at": self.last_sync_at,
            "total_imported": self.total_imported,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackfillOffset:
        return cls(
            session=data["session"],
            last_cursor=data.get("last_cursor", 0),
            last_sync_at=data.get("last_sync_at", datetime.now().isoformat()),
            total_imported=data.get("total_imported", 0),
        )


@dataclass
class BackfillEntry:
    """A single entry to be backfilled."""

    cursor: int
    timestamp: str
    content: str
    session: str
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_memory_entry(self) -> dict[str, Any]:
        """Convert to a memory entry format suitable for external storage."""
        return {
            "id": f"{self.session}:{self.cursor}",
            "session": self.session,
            "timestamp": self.timestamp,
            "content": self.content,
            "type": "agent_turn",
            "metadata": {
                "source": "history.jsonl",
                "cursor": self.cursor,
                **self.raw_data.get("metadata", {}),
            },
        }


@dataclass
class BackfillResult:
    """Result of a backfill operation."""

    session: str
    entries_scanned: int
    entries_imported: int
    entries_filtered: int
    last_cursor: int
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class MemoryBackfillService:
    """Service for backfilling memory history to external storage.

    Supports incremental import with offset tracking to avoid duplicates.
    Designed to work with external memory systems like OpenViking.
    """

    _OFFSET_FILE = ".backfill_offset.json"
    _BATCH_SIZE = 100

    def __init__(
        self,
        workspace: Path,
        memory_sink: MemorySink | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ):
        self.workspace = Path(workspace).expanduser()
        self.memory_sink = memory_sink
        self.filter_fn = filter_fn or self._default_filter
        self._offset_file = self.workspace / self._OFFSET_FILE
        self._offsets: dict[str, BackfillOffset] = {}
        self._load_offsets()

    def _load_offsets(self) -> None:
        """Load sync offsets from disk."""
        if self._offset_file.exists():
            try:
                data = json.loads(self._offset_file.read_text(encoding="utf-8"))
                for session, offset_data in data.get("offsets", {}).items():
                    self._offsets[session] = BackfillOffset.from_dict(offset_data)
                logger.info(
                    "Loaded backfill offsets for {} sessions",
                    len(self._offsets),
                )
            except Exception as e:
                logger.warning("Failed to load backfill offsets: {}", e)
                self._offsets = {}

    def _save_offsets(self) -> None:
        """Save sync offsets to disk."""
        try:
            self._offset_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "offsets": {
                    session: offset.to_dict()
                    for session, offset in self._offsets.items()
                }
            }
            self._offset_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save backfill offsets: {}", e)

    def _get_offset(self, session: str) -> BackfillOffset:
        """Get or create offset for a session."""
        if session not in self._offsets:
            self._offsets[session] = BackfillOffset(session=session)
        return self._offsets[session]

    @staticmethod
    def _default_filter(entry: dict[str, Any]) -> bool:
        """Default filter: include only agent_turn entries.

        Filters out:
        - system/dream/consolidation entries
        - Empty content
        - Entries with specific skip markers
        """
        content = entry.get("content", "")
        if not content or not isinstance(content, str):
            return False

        # Skip system entries
        skip_prefixes = (
            "[SYSTEM]",
            "[DREAM]",
            "[CONSOLIDATION]",
            "[RAW]",  # Raw dumps are usually temporary
        )
        if any(content.startswith(p) for p in skip_prefixes):
            return False

        # Skip entries with skip markers in metadata
        metadata = entry.get("metadata", {})
        if metadata.get("skip_backfill"):
            return False

        return True

    def _read_history_entries(
        self,
        history_file: Path,
        offset: BackfillOffset,
    ) -> list[BackfillEntry]:
        """Read entries from history.jsonl after the given offset."""
        entries: list[BackfillEntry] = []

        if not history_file.exists():
            return entries

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        raw = json.loads(line)
                        cursor = raw.get("cursor", 0)

                        # Skip already imported entries
                        if cursor <= offset.last_cursor:
                            continue

                        entry = BackfillEntry(
                            cursor=cursor,
                            timestamp=raw.get("timestamp", ""),
                            content=raw.get("content", ""),
                            session=offset.session,
                            raw_data=raw,
                        )
                        entries.append(entry)
                    except json.JSONDecodeError:
                        logger.debug("Skipping invalid JSON line in {}", history_file)
                        continue

        except Exception as e:
            logger.error("Error reading history file {}: {}", history_file, e)

        return entries

    async def _write_to_sink(
        self,
        entries: list[BackfillEntry],
    ) -> tuple[int, list[str]]:
        """Write entries to the memory sink.

        Returns (count_imported, errors).
        """
        if not entries:
            return 0, []

        if self.memory_sink is None:
            # No sink configured - just log what would be imported
            logger.info(
                "Would import {} entries (no memory sink configured)",
                len(entries),
            )
            return len(entries), []

        memory_entries = [e.to_memory_entry() for e in entries]

        try:
            imported = await self.memory_sink(memory_entries)
            return imported, []
        except Exception as e:
            error_msg = f"Memory sink error: {e}"
            logger.error(error_msg)
            return 0, [error_msg]

    async def backfill_session(
        self,
        session: str,
        dry_run: bool = False,
    ) -> BackfillResult:
        """Backfill a single session's history.

        Args:
            session: Session identifier (e.g., "telegram:-123456")
            dry_run: If True, only preview what would be imported

        Returns:
            BackfillResult with statistics
        """
        offset = self._get_offset(session)
        history_file = self._get_history_file(session)

        logger.info(
            "Backfilling session {} from cursor {} (dry_run={})",
            session,
            offset.last_cursor,
            dry_run,
        )

        # Read entries
        entries = self._read_history_entries(history_file, offset)
        entries_scanned = len(entries)

        # Filter entries
        filtered_entries = [e for e in entries if self.filter_fn(e.raw_data)]
        entries_filtered = entries_scanned - len(filtered_entries)

        if dry_run:
            logger.info(
                "Dry run for {}: {} entries would be imported ({} filtered)",
                session,
                len(filtered_entries),
                entries_filtered,
            )
            # Update offset tracking for preview
            if filtered_entries:
                max_cursor = max(e.cursor for e in filtered_entries)
                offset.last_cursor = max_cursor

            return BackfillResult(
                session=session,
                entries_scanned=entries_scanned,
                entries_imported=0,  # Not actually imported in dry run
                entries_filtered=entries_filtered,
                last_cursor=offset.last_cursor,
            )

        # Batch import
        total_imported = 0
        errors: list[str] = []
        max_cursor = offset.last_cursor

        for i in range(0, len(filtered_entries), self._BATCH_SIZE):
            batch = filtered_entries[i : i + self._BATCH_SIZE]
            imported, batch_errors = await self._write_to_sink(batch)
            total_imported += imported
            errors.extend(batch_errors)

            if batch:
                batch_max = max(e.cursor for e in batch)
                max_cursor = max(max_cursor, batch_max)

            if batch_errors:
                # Stop on error to avoid skipping entries
                logger.error("Stopping backfill due to sink error")
                break

        # Update offset
        if max_cursor > offset.last_cursor:
            offset.last_cursor = max_cursor
            offset.last_sync_at = datetime.now().isoformat()
            offset.total_imported += total_imported
            self._save_offsets()

        logger.info(
            "Backfilled session {}: {} entries imported, cursor now {}",
            session,
            total_imported,
            offset.last_cursor,
        )

        return BackfillResult(
            session=session,
            entries_scanned=entries_scanned,
            entries_imported=total_imported,
            entries_filtered=entries_filtered,
            last_cursor=offset.last_cursor,
            errors=errors,
        )

    def _get_history_file(self, session: str) -> Path:
        """Get the history.jsonl file path for a session.

        Session format can be:
        - "telegram:<chat_id>" -> workspace/memory/history.jsonl
        - "cli:<name>" -> workspace/memory/history.jsonl
        - "workspace:<name>" -> workspace/<name>/memory/history.jsonl
        """
        # Parse session key
        if session.startswith("workspace:"):
            # Workspace-specific session
            workspace_name = session.split(":", 1)[1]
            return self.workspace / workspace_name / "memory" / "history.jsonl"

        # Default: use main workspace memory
        return self.workspace / "memory" / "history.jsonl"

    def list_sessions(self) -> list[str]:
        """List all sessions with history files available for backfill."""
        sessions: list[str] = []

        # Check main workspace
        main_history = self.workspace / "memory" / "history.jsonl"
        if main_history.exists():
            # Add known sessions that use main workspace
            # These would typically come from session tracking
            for offset in self._offsets.values():
                sessions.append(offset.session)

        # Check workspace-specific histories
        for ws_dir in self.workspace.iterdir():
            if ws_dir.is_dir() and ws_dir.name.endswith("-coding"):
                history_file = ws_dir / "memory" / "history.jsonl"
                if history_file.exists():
                    session_key = f"workspace:{ws_dir.name}"
                    if session_key not in sessions:
                        sessions.append(session_key)

        return sorted(set(sessions))

    async def backfill_all(
        self,
        dry_run: bool = False,
    ) -> list[BackfillResult]:
        """Backfill all available sessions.

        Args:
            dry_run: If True, only preview what would be imported

        Returns:
            List of BackfillResult for each session
        """
        sessions = self.list_sessions()
        logger.info("Found {} sessions to backfill", len(sessions))

        results: list[BackfillResult] = []
        for session in sessions:
            try:
                result = await self.backfill_session(session, dry_run=dry_run)
                results.append(result)
            except Exception as e:
                logger.error("Failed to backfill session {}: {}", session, e)
                results.append(
                    BackfillResult(
                        session=session,
                        entries_scanned=0,
                        entries_imported=0,
                        entries_filtered=0,
                        last_cursor=0,
                        errors=[str(e)],
                    )
                )

        return results

    def get_status(self, session: str | None = None) -> dict[str, Any]:
        """Get backfill status for one or all sessions."""
        if session:
            offset = self._offsets.get(session)
            if not offset:
                return {"session": session, "status": "not_found"}
            return {
                "session": session,
                "status": "active",
                **offset.to_dict(),
            }

        return {
            "total_sessions": len(self._offsets),
            "sessions": {
                s: o.to_dict() for s, o in self._offsets.items()
            },
        }
