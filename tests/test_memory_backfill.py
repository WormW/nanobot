"""Tests for memory backfill service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.memory_backfill.service import (
    BackfillEntry,
    BackfillOffset,
    BackfillResult,
    MemoryBackfillService,
)


@pytest.fixture
def backfill_service(tmp_path: Path):
    """Provide a memory backfill service with temp workspace."""
    return MemoryBackfillService(workspace=tmp_path)


@pytest.fixture
def sample_history(tmp_path: Path):
    """Create a sample history.jsonl file."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    history_file = memory_dir / "history.jsonl"

    entries = [
        {"cursor": 1, "timestamp": "2024-01-01 10:00", "content": "Hello"},
        {"cursor": 2, "timestamp": "2024-01-01 10:05", "content": "World"},
        {"cursor": 3, "timestamp": "2024-01-01 10:10", "content": "[SYSTEM] System message"},
        {"cursor": 4, "timestamp": "2024-01-01 10:15", "content": "User question"},
        {"cursor": 5, "timestamp": "2024-01-01 10:20", "content": "[DREAM] Dream entry"},
    ]

    with open(history_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return history_file


class TestBackfillOffset:
    """Tests for BackfillOffset."""

    def test_creation(self):
        offset = BackfillOffset(session="test-session", last_cursor=10)
        assert offset.session == "test-session"
        assert offset.last_cursor == 10
        assert offset.total_imported == 0

    def test_to_dict(self):
        offset = BackfillOffset(session="test", last_cursor=5, total_imported=100)
        data = offset.to_dict()
        assert data["session"] == "test"
        assert data["last_cursor"] == 5
        assert data["total_imported"] == 100

    def test_from_dict(self):
        data = {
            "session": "test",
            "last_cursor": 10,
            "last_sync_at": "2024-01-01T00:00:00",
            "total_imported": 50,
        }
        offset = BackfillOffset.from_dict(data)
        assert offset.session == "test"
        assert offset.last_cursor == 10
        assert offset.total_imported == 50


class TestBackfillEntry:
    """Tests for BackfillEntry."""

    def test_creation(self):
        entry = BackfillEntry(
            cursor=1,
            timestamp="2024-01-01 10:00",
            content="Hello",
            session="test",
        )
        assert entry.cursor == 1
        assert entry.content == "Hello"

    def test_to_memory_entry(self):
        entry = BackfillEntry(
            cursor=1,
            timestamp="2024-01-01 10:00",
            content="Hello",
            session="test-session",
            raw_data={"metadata": {"key": "value"}},
        )
        memory = entry.to_memory_entry()
        assert memory["id"] == "test-session:1"
        assert memory["session"] == "test-session"
        assert memory["content"] == "Hello"
        assert memory["type"] == "agent_turn"
        assert memory["metadata"]["cursor"] == 1
        assert memory["metadata"]["key"] == "value"


class TestMemoryBackfillService:
    """Tests for MemoryBackfillService."""

    def test_load_save_offsets(self, tmp_path: Path):
        service = MemoryBackfillService(workspace=tmp_path)

        # Manually set an offset
        service._offsets["test-session"] = BackfillOffset(
            session="test-session",
            last_cursor=10,
            total_imported=100,
        )
        service._save_offsets()

        # Load in a new service instance
        service2 = MemoryBackfillService(workspace=tmp_path)
        assert "test-session" in service2._offsets
        assert service2._offsets["test-session"].last_cursor == 10
        assert service2._offsets["test-session"].total_imported == 100

    def test_default_filter(self):
        # Regular entries pass
        assert MemoryBackfillService._default_filter({"content": "Hello"})

        # System entries are filtered
        assert not MemoryBackfillService._default_filter({"content": "[SYSTEM] message"})
        assert not MemoryBackfillService._default_filter({"content": "[DREAM] entry"})
        assert not MemoryBackfillService._default_filter({"content": "[CONSOLIDATION] data"})

        # Empty content filtered
        assert not MemoryBackfillService._default_filter({"content": ""})
        assert not MemoryBackfillService._default_filter({})

        # Skip marker filtered
        assert not MemoryBackfillService._default_filter(
            {"content": "test", "metadata": {"skip_backfill": True}}
        )

    @pytest.mark.asyncio
    async def test_backfill_session_dry_run(self, backfill_service, sample_history):
        result = await backfill_service.backfill_session(
            session="test-session",
            dry_run=True,
        )

        assert result.session == "test-session"
        assert result.entries_scanned == 5
        assert result.entries_filtered == 2  # SYSTEM and DREAM entries
        assert result.entries_imported == 0  # Dry run doesn't import
        assert result.success

    @pytest.mark.asyncio
    async def test_backfill_session_incremental(self, backfill_service, sample_history):
        # First backfill
        result1 = await backfill_service.backfill_session("test-session")
        assert result1.entries_imported == 3  # 5 - 2 filtered

        # Second backfill should find nothing new to import
        # (cursor=4 entry is read but already filtered)
        result2 = await backfill_service.backfill_session("test-session")
        assert result2.entries_imported == 0
        # entries_scanned may be 0 or 1 depending on whether last cursor is inclusive
        assert result2.entries_scanned <= 1

    @pytest.mark.asyncio
    async def test_backfill_session_with_sink(self, tmp_path: Path, sample_history):
        imported_entries = []

        async def mock_sink(entries):
            imported_entries.extend(entries)
            return len(entries)

        service = MemoryBackfillService(
            workspace=tmp_path,
            memory_sink=mock_sink,
        )

        result = await service.backfill_session("test-session")

        assert result.entries_imported == 3
        assert len(imported_entries) == 3
        assert imported_entries[0]["content"] == "Hello"
        assert imported_entries[1]["content"] == "World"

    def test_list_sessions(self, tmp_path: Path):
        # Create main workspace history
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "history.jsonl").write_text('{"cursor": 1}\n')

        # Create a workspace-specific history
        ws_dir = tmp_path / "myproject-coding"
        ws_memory = ws_dir / "memory"
        ws_memory.mkdir(parents=True)
        (ws_memory / "history.jsonl").write_text('{"cursor": 1}\n')

        # Create an offset for main workspace
        service = MemoryBackfillService(workspace=tmp_path)
        service._offsets["telegram:-123"] = BackfillOffset(
            session="telegram:-123", last_cursor=0
        )
        service._save_offsets()

        sessions = service.list_sessions()

        assert "telegram:-123" in sessions
        assert "workspace:myproject-coding" in sessions

    def test_get_history_file_main_workspace(self, backfill_service):
        path = backfill_service._get_history_file("telegram:-123")
        assert path.name == "history.jsonl"
        assert "memory" in str(path)

    def test_get_history_file_workspace_session(self, backfill_service):
        path = backfill_service._get_history_file("workspace:myproject-coding")
        assert path.name == "history.jsonl"
        assert "myproject-coding" in str(path)

    def test_persistence(self, tmp_path: Path):
        service1 = MemoryBackfillService(workspace=tmp_path)

        # Add offset
        service1._offsets["test"] = BackfillOffset(
            session="test", last_cursor=5, total_imported=10
        )
        service1._save_offsets()

        # Create new instance
        service2 = MemoryBackfillService(workspace=tmp_path)
        assert service2._offsets["test"].last_cursor == 5
        assert service2._offsets["test"].total_imported == 10

    def test_get_status_single_session(self, backfill_service):
        backfill_service._offsets["test-sess"] = BackfillOffset(
            session="test-sess", last_cursor=10, total_imported=100
        )

        status = backfill_service.get_status("test-sess")
        assert status["status"] == "active"
        assert status["last_cursor"] == 10

    def test_get_status_all_sessions(self, backfill_service):
        backfill_service._offsets["s1"] = BackfillOffset(session="s1")
        backfill_service._offsets["s2"] = BackfillOffset(session="s2")

        status = backfill_service.get_status()
        assert status["total_sessions"] == 2
        assert "s1" in status["sessions"]
        assert "s2" in status["sessions"]
