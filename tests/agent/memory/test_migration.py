"""Tests for memory migration utilities."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from nanobot.agent.memory.migration import (
    MemoryMigrator,
    MigrationResult,
    MigrationReport,
    create_backup,
    migrate_filesystem_to_sqlite,
)
from nanobot.agent.memory.storage import FileSystemBackend, SQLiteBackend
from nanobot.agent.memory.types import MemoryEntry, MemoryTier


@pytest.fixture
async def fs_backend():
    """Create a filesystem backend with test data."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileSystemBackend(tmp)
        await backend.initialize()
        yield backend


@pytest.fixture
async def sqlite_backend():
    """Create a SQLite backend."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        backend = SQLiteBackend(str(db_path))
        await backend.initialize()
        yield backend


@pytest.fixture
async def populated_fs_backend():
    """Create a filesystem backend with test data."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileSystemBackend(tmp)
        await backend.initialize()

        # Add working memory entries
        for i in range(3):
            entry = MemoryEntry(
                id=f"working-{i}",
                content=f"Working memory content {i}",
                tier=MemoryTier.WORKING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source_session="test-session",
            )
            await backend.store(entry)

        # Add episodic memory entries
        for i in range(2):
            entry = MemoryEntry(
                id=f"episodic-{i}",
                content=f"Episodic memory content {i}",
                tier=MemoryTier.EPISODIC,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            await backend.store(entry)

        # Add semantic memory entry
        entry = MemoryEntry(
            id="semantic-0",
            content="Semantic knowledge content",
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await backend.store(entry)

        yield backend


class TestMigrationResult:
    """Test MigrationResult dataclass."""

    def test_default_values(self):
        """Test default values for MigrationResult."""
        result = MigrationResult()
        assert result.total_entries == 0
        assert result.migrated[MemoryTier.WORKING] == 0
        assert result.migrated[MemoryTier.EPISODIC] == 0
        assert result.migrated[MemoryTier.SEMANTIC] == 0
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.skipped == 0

    def test_custom_values(self):
        """Test custom values for MigrationResult."""
        result = MigrationResult(
            total_entries=10,
            migrated={MemoryTier.WORKING: 5, MemoryTier.EPISODIC: 3, MemoryTier.SEMANTIC: 2},
            errors=["error1"],
            duration_seconds=5.5,
            skipped=1,
        )
        assert result.total_entries == 10
        assert result.migrated[MemoryTier.WORKING] == 5
        assert result.errors == ["error1"]
        assert result.duration_seconds == 5.5
        assert result.skipped == 1


class TestMigrationReport:
    """Test MigrationReport dataclass."""

    def test_migration_report(self):
        """Test MigrationReport creation."""
        by_tier = {MemoryTier.WORKING: 5, MemoryTier.EPISODIC: 3, MemoryTier.SEMANTIC: 2}
        report = MigrationReport(
            source_entries=10,
            by_tier=by_tier,
            estimated_time_seconds=15.0,
            warnings=["warning1"],
        )
        assert report.source_entries == 10
        assert report.by_tier == by_tier
        assert report.estimated_time_seconds == 15.0
        assert report.warnings == ["warning1"]


class TestMemoryMigrator:
    """Test MemoryMigrator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, fs_backend, sqlite_backend):
        """Test migrator initialization."""
        migrator = MemoryMigrator(fs_backend, sqlite_backend)
        assert migrator.source == fs_backend
        assert migrator.target == sqlite_backend
        assert migrator.batch_size == 100

    @pytest.mark.asyncio
    async def test_dry_run(self, populated_fs_backend, sqlite_backend):
        """Test dry run preview."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)
        report = await migrator.dry_run()

        assert report.source_entries == 6  # 3 working + 2 episodic + 1 semantic
        assert report.by_tier[MemoryTier.WORKING] == 3
        assert report.by_tier[MemoryTier.EPISODIC] == 2
        assert report.by_tier[MemoryTier.SEMANTIC] == 1
        assert report.estimated_time_seconds > 0
        assert report.warnings == []

    @pytest.mark.asyncio
    async def test_migrate_tier(self, populated_fs_backend, sqlite_backend):
        """Test migrating a single tier."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)

        count = await migrator.migrate_tier(MemoryTier.EPISODIC)

        assert count == 2

        # Verify entries in target
        results = await sqlite_backend.retrieve("", tier=MemoryTier.EPISODIC)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_migrate_all(self, populated_fs_backend, sqlite_backend):
        """Test migrating all tiers."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)

        result = await migrator.migrate_all()

        assert result.total_entries == 6
        assert result.migrated[MemoryTier.WORKING] == 3
        assert result.migrated[MemoryTier.EPISODIC] == 2
        assert result.migrated[MemoryTier.SEMANTIC] == 1
        assert result.errors == []

        # Verify entries in target
        for tier in MemoryTier:
            results = await sqlite_backend.retrieve("", tier=tier)
            assert len(results) == result.migrated[tier]

    @pytest.mark.asyncio
    async def test_migrate_skips_duplicates(self, populated_fs_backend, sqlite_backend):
        """Test that migration skips duplicate entries."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)

        # First migration
        result1 = await migrator.migrate_all()
        assert result1.skipped == 0

        # Second migration should skip all
        result2 = await migrator.migrate_all()
        assert result2.skipped == 6

    @pytest.mark.asyncio
    async def test_progress_callback(self, populated_fs_backend, sqlite_backend):
        """Test progress callback functionality."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)

        progress_calls = []

        def callback(current, total, tier):
            progress_calls.append((current, total, tier))

        migrator.set_progress_callback(callback)
        await migrator.migrate_all()

        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_validate_success(self, populated_fs_backend, sqlite_backend):
        """Test validation after successful migration."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)

        await migrator.migrate_all()
        is_valid = await migrator.validate()

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_failure(self, populated_fs_backend, sqlite_backend):
        """Test validation when entries are missing."""
        migrator = MemoryMigrator(populated_fs_backend, sqlite_backend)

        # Don't migrate - validation should fail
        is_valid = await migrator.validate()

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_migrate_empty_source(self, fs_backend, sqlite_backend):
        """Test migrating from empty source."""
        migrator = MemoryMigrator(fs_backend, sqlite_backend)

        result = await migrator.migrate_all()

        assert result.total_entries == 0
        assert all(count == 0 for count in result.migrated.values())


class TestCreateBackup:
    """Test create_backup function."""

    @pytest.mark.asyncio
    async def test_backup_file(self):
        """Test backing up a file."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.txt"
            source.write_text("test content")

            backup_path = await create_backup(source, suffix=".bak")

            assert backup_path.exists()
            assert backup_path.read_text() == "test content"
            assert ".bak." in backup_path.name

    @pytest.mark.asyncio
    async def test_backup_directory(self):
        """Test backing up a directory."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source_dir"
            source.mkdir()
            (source / "file.txt").write_text("content")

            backup_path = await create_backup(source, suffix=".bak")

            assert backup_path.exists()
            assert backup_path.is_dir()
            assert (backup_path / "file.txt").read_text() == "content"

    @pytest.mark.asyncio
    async def test_backup_nonexistent_raises(self):
        """Test that backing up nonexistent path raises error."""
        with tempfile.TemporaryDirectory() as tmp:
            nonexistent = Path(tmp) / "nonexistent"

            with pytest.raises(ValueError, match="does not exist"):
                await create_backup(nonexistent)


class TestConvenienceFunctions:
    """Test convenience migration functions."""

    @pytest.mark.asyncio
    async def test_migrate_filesystem_to_sqlite(self):
        """Test filesystem to SQLite migration convenience function."""
        with tempfile.TemporaryDirectory() as tmp:
            fs_path = Path(tmp) / "fs_memory"
            db_path = Path(tmp) / "memory.db"

            # Create source with data
            fs = FileSystemBackend(str(fs_path))
            await fs.initialize()

            entry = MemoryEntry(
                id="test-1",
                content="Test content",
                tier=MemoryTier.SEMANTIC,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            await fs.store(entry)

            # Migrate
            result = await migrate_filesystem_to_sqlite(str(fs_path), str(db_path))

            assert result.total_entries == 1
            assert result.migrated[MemoryTier.SEMANTIC] == 1
            assert result.errors == []

            # Verify target
            sqlite = SQLiteBackend(str(db_path))
            await sqlite.initialize()
            results = await sqlite.retrieve("", tier=MemoryTier.SEMANTIC)
            assert len(results) == 1
