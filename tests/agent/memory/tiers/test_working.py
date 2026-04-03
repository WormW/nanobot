"""Unit tests for WorkingMemoryManager.

Tests cover the core functionality of working memory management including:
- Adding conversation turns
- Retrieving recent entries
- Session-specific queries
- Overflow detection
- Entry archiving
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.storage.filesystem import FileSystemBackend
from nanobot.agent.memory.tiers.working import WorkingMemoryManager
from nanobot.agent.memory.types import MemoryEntry, MemoryTier, WorkingMemoryConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
async def backend(temp_dir):
    """Create and initialize a FileSystemBackend."""
    be = FileSystemBackend(temp_dir)
    await be.initialize()
    return be


@pytest.fixture
def config():
    """Create a WorkingMemoryConfig with small max_turns for testing."""
    return WorkingMemoryConfig(max_turns=3, max_tokens=1000, ttl_seconds=3600)


@pytest.fixture
async def manager(backend, config):
    """Create a WorkingMemoryManager with initialized backend."""
    return WorkingMemoryManager(backend, config)


class TestAddTurn:
    """Tests for the add_turn method."""

    @pytest.mark.asyncio
    async def test_add_turn_creates_entry(self, temp_dir, config):
        """Test that add_turn creates a properly formatted entry."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "Hello", "Hi there!")

        # Verify file was created
        working_file = Path(temp_dir) / "working" / "session_1.jsonl"
        assert working_file.exists()

        # Read and verify content
        lines = working_file.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["tier"] == "working"
        assert data["source_session"] == "session_1"

        # Verify content JSON structure
        content = json.loads(data["content"])
        assert content["user"] == "Hello"
        assert content["assistant"] == "Hi there!"
        assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_add_turn_multiple_same_session(self, temp_dir, config):
        """Test adding multiple turns to the same session."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "Hello", "Hi!")
        await manager.add_turn("session_1", "How are you?", "I'm good!")
        await manager.add_turn("session_1", "Great!", "Thanks!")

        working_file = Path(temp_dir) / "working" / "session_1.jsonl"
        lines = working_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Verify each turn
        for i, line in enumerate(lines):
            data = json.loads(line)
            content = json.loads(data["content"])
            assert "user" in content
            assert "assistant" in content
            assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_add_turn_different_sessions(self, temp_dir, config):
        """Test adding turns to different sessions."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_a", "Hello A", "Response A")
        await manager.add_turn("session_b", "Hello B", "Response B")
        await manager.add_turn("session_a", "Follow up A", "Follow response A")

        # Verify separate files
        working_dir = Path(temp_dir) / "working"
        assert (working_dir / "session_a.jsonl").exists()
        assert (working_dir / "session_b.jsonl").exists()

        # Verify session_a has 2 entries
        lines_a = (working_dir / "session_a.jsonl").read_text().strip().split("\n")
        assert len(lines_a) == 2

        # Verify session_b has 1 entry
        lines_b = (working_dir / "session_b.jsonl").read_text().strip().split("\n")
        assert len(lines_b) == 1

    @pytest.mark.asyncio
    async def test_add_turn_generates_unique_ids(self, temp_dir, config):
        """Test that each turn gets a unique ID."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "Hello", "Hi!")
        await manager.add_turn("session_1", "Again", "Yes!")

        working_file = Path(temp_dir) / "working" / "session_1.jsonl"
        lines = working_file.read_text().strip().split("\n")

        ids = [json.loads(line)["id"] for line in lines]
        assert len(ids) == len(set(ids))  # All unique
        assert all(len(id_) == 32 for id_ in ids)  # hex format

    @pytest.mark.asyncio
    async def test_add_turn_timestamps(self, temp_dir, config):
        """Test that timestamps are properly set."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        before = datetime.now()
        await manager.add_turn("session_1", "Hello", "Hi!")
        after = datetime.now()

        working_file = Path(temp_dir) / "working" / "session_1.jsonl"
        lines = working_file.read_text().strip().split("\n")
        data = json.loads(lines[0])

        created_at = datetime.fromisoformat(data["created_at"])
        assert before <= created_at <= after


class TestGetRecent:
    """Tests for the get_recent method."""

    @pytest.mark.asyncio
    async def test_get_recent_returns_all_by_default(self, temp_dir, config):
        """Test that get_recent returns all entries when n is None."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "Hello", "Hi!")
        await manager.add_turn("session_2", "Hey", "Hello!")

        entries = await manager.get_recent()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_get_recent_limits_to_n(self, temp_dir, config):
        """Test that get_recent respects the n parameter."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "First", "Response 1")
        await manager.add_turn("session_1", "Second", "Response 2")
        await manager.add_turn("session_1", "Third", "Response 3")

        entries = await manager.get_recent(n=2)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_get_recent_orders_by_time_descending(self, temp_dir, config):
        """Test that get_recent returns entries in reverse chronological order."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "First", "Response 1")
        await manager.add_turn("session_1", "Second", "Response 2")
        await manager.add_turn("session_1", "Third", "Response 3")

        entries = await manager.get_recent()

        # Should be most recent first
        for i in range(len(entries) - 1):
            assert entries[i].created_at >= entries[i + 1].created_at

    @pytest.mark.asyncio
    async def test_get_recent_empty(self, temp_dir, config):
        """Test that get_recent returns empty list when no entries."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        entries = await manager.get_recent()
        assert entries == []

    @pytest.mark.asyncio
    async def test_get_recent_returns_memory_entries(self, temp_dir, config):
        """Test that get_recent returns proper MemoryEntry objects."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "Hello", "Hi!")

        entries = await manager.get_recent()
        assert len(entries) == 1

        entry = entries[0]
        assert isinstance(entry, MemoryEntry)
        assert entry.tier == MemoryTier.WORKING
        assert entry.source_session == "session_1"

        # Verify content can be parsed
        content = json.loads(entry.content)
        assert content["user"] == "Hello"
        assert content["assistant"] == "Hi!"


class TestGetAllForSession:
    """Tests for the get_all_for_session method."""

    @pytest.mark.asyncio
    async def test_get_all_for_session_filters_correctly(self, temp_dir, config):
        """Test that get_all_for_session returns only entries for that session."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_a", "Hello A", "Response A")
        await manager.add_turn("session_b", "Hello B", "Response B")
        await manager.add_turn("session_a", "Follow up A", "Follow A")

        entries = await manager.get_all_for_session("session_a")
        assert len(entries) == 2
        assert all(e.source_session == "session_a" for e in entries)

    @pytest.mark.asyncio
    async def test_get_all_for_session_orders_ascending(self, temp_dir, config):
        """Test that get_all_for_session returns entries in chronological order."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.add_turn("session_1", "First", "Response 1")
        await manager.add_turn("session_1", "Second", "Response 2")
        await manager.add_turn("session_1", "Third", "Response 3")

        entries = await manager.get_all_for_session("session_1")

        # Should be oldest first
        for i in range(len(entries) - 1):
            assert entries[i].created_at <= entries[i + 1].created_at

    @pytest.mark.asyncio
    async def test_get_all_for_session_empty(self, temp_dir, config):
        """Test that get_all_for_session returns empty for non-existent session."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        entries = await manager.get_all_for_session("nonexistent")
        assert entries == []


class TestIsOverflow:
    """Tests for the _is_overflow method."""

    @pytest.mark.asyncio
    async def test_is_overflow_false_when_under_limit(self, temp_dir):
        """Test that _is_overflow returns False when under max_turns."""
        config = WorkingMemoryConfig(max_turns=5)
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        # Add 3 turns (under limit of 5)
        for i in range(3):
            await manager.add_turn("session_1", f"Turn {i}", f"Response {i}")

        assert await manager._is_overflow("session_1") is False

    @pytest.mark.asyncio
    async def test_is_overflow_true_when_over_limit(self, temp_dir):
        """Test that _is_overflow returns True when over max_turns."""
        config = WorkingMemoryConfig(max_turns=3)
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        # Add 4 turns (over limit of 3)
        for i in range(4):
            await manager.add_turn("session_1", f"Turn {i}", f"Response {i}")

        assert await manager._is_overflow("session_1") is True

    @pytest.mark.asyncio
    async def test_is_overflow_at_exact_limit(self, temp_dir):
        """Test that _is_overflow returns False at exactly max_turns."""
        config = WorkingMemoryConfig(max_turns=3)
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        # Add exactly 3 turns (at limit of 3)
        for i in range(3):
            await manager.add_turn("session_1", f"Turn {i}", f"Response {i}")

        assert await manager._is_overflow("session_1") is False

    @pytest.mark.asyncio
    async def test_is_overflow_per_session(self, temp_dir):
        """Test that _is_overflow checks per-session, not globally."""
        config = WorkingMemoryConfig(max_turns=3)
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        # Add 4 turns to session_1 (overflow)
        for i in range(4):
            await manager.add_turn("session_1", f"Turn {i}", f"Response {i}")

        # Add 2 turns to session_2 (no overflow)
        for i in range(2):
            await manager.add_turn("session_2", f"Turn {i}", f"Response {i}")

        assert await manager._is_overflow("session_1") is True
        assert await manager._is_overflow("session_2") is False


class TestArchiveEntries:
    """Tests for the archive_entries method."""

    @pytest.mark.asyncio
    async def test_archive_entries_tracks_ids(self, temp_dir, config):
        """Test that archive_entries tracks entry IDs."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        entry_ids = ["id_1", "id_2", "id_3"]
        await manager.archive_entries(entry_ids)

        for entry_id in entry_ids:
            assert manager.is_archived(entry_id) is True

    @pytest.mark.asyncio
    async def test_archive_entries_appends_to_existing(self, temp_dir, config):
        """Test that archive_entries appends to existing archived IDs."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.archive_entries(["id_1", "id_2"])
        await manager.archive_entries(["id_3"])

        assert manager.is_archived("id_1") is True
        assert manager.is_archived("id_2") is True
        assert manager.is_archived("id_3") is True

    @pytest.mark.asyncio
    async def test_archive_entries_empty_list(self, temp_dir, config):
        """Test that archive_entries handles empty list."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        await manager.archive_entries([])
        # Should not raise and should have empty archived set
        assert not hasattr(manager, '_archived_ids') or len(manager._archived_ids) == 0

    @pytest.mark.asyncio
    async def test_is_archived_false_for_non_archived(self, temp_dir, config):
        """Test that is_archived returns False for non-archived entries."""
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        assert manager.is_archived("never_archived") is False


class TestIntegration:
    """Integration tests for WorkingMemoryManager workflows."""

    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self, temp_dir):
        """Test a full conversation workflow."""
        config = WorkingMemoryConfig(max_turns=3)
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        session_id = "chat_123"

        # Simulate a conversation
        conversation = [
            ("Hello!", "Hi there! How can I help?"),
            ("What's the weather?", "It's sunny today!"),
            ("Thanks!", "You're welcome!"),
        ]

        for user_msg, assistant_msg in conversation:
            await manager.add_turn(session_id, user_msg, assistant_msg)

        # Verify all turns are stored
        entries = await manager.get_all_for_session(session_id)
        assert len(entries) == 3

        # Verify conversation flow is preserved
        for i, (user_msg, assistant_msg) in enumerate(conversation):
            content = json.loads(entries[i].content)
            assert content["user"] == user_msg
            assert content["assistant"] == assistant_msg

        # Verify not yet overflowing
        assert await manager._is_overflow(session_id) is False

        # Add one more turn to trigger overflow
        await manager.add_turn(session_id, "One more", "Sure!")
        assert await manager._is_overflow(session_id) is True

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, temp_dir):
        """Test that multiple sessions remain isolated."""
        config = WorkingMemoryConfig(max_turns=10)
        backend = FileSystemBackend(temp_dir)
        await backend.initialize()
        manager = WorkingMemoryManager(backend, config)

        # Add turns to multiple sessions
        sessions = {
            "user_a": [("Hi A", "Hello A"), ("How A?", "Good A")],
            "user_b": [("Hi B", "Hello B")],
            "user_c": [("Hi C", "Hello C"), ("How C?", "Good C"), ("Thanks C", "Bye C")],
        }

        for session_id, turns in sessions.items():
            for user_msg, assistant_msg in turns:
                await manager.add_turn(session_id, user_msg, assistant_msg)

        # Verify each session's entries
        for session_id, turns in sessions.items():
            entries = await manager.get_all_for_session(session_id)
            assert len(entries) == len(turns)

        # Verify get_recent returns all entries across sessions
        all_entries = await manager.get_recent()
        total_turns = sum(len(turns) for turns in sessions.values())
        assert len(all_entries) == total_turns
