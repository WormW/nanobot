"""Data migration utilities for memory system.

This module provides tools for migrating memory data between different
storage backends (FileSystem, SQLite, ChromaDB).
"""

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory.backend import MemoryBackend
from nanobot.agent.memory.types import MemoryEntry, MemoryTier


@dataclass
class MigrationResult:
    """Result of a memory migration operation.

    Attributes:
        total_entries: Total number of entries processed
        migrated: Count per tier that was successfully migrated
        errors: List of error messages for failed entries
        duration_seconds: Time taken for migration
        skipped: Number of entries skipped (duplicates, etc.)
    """
    total_entries: int = 0
    migrated: dict[MemoryTier, int] = field(default_factory=lambda: {
        MemoryTier.WORKING: 0,
        MemoryTier.EPISODIC: 0,
        MemoryTier.SEMANTIC: 0
    })
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    skipped: int = 0


@dataclass
class MigrationReport:
    """Preview report for dry-run migrations.

    Attributes:
        source_entries: Total entries in source
        by_tier: Entry counts per tier
        estimated_time_seconds: Estimated migration duration
        warnings: Any warnings about the migration
    """
    source_entries: int
    by_tier: dict[MemoryTier, int]
    estimated_time_seconds: float
    warnings: list[str] = field(default_factory=list)


class MemoryMigrator:
    """Migrates memory data between storage backends.

    Supports migration from any MemoryBackend implementation to another,
    with progress tracking, error handling, and validation.

    Example:
        source = FileSystemBackend("~/.nanobot/memory")
        target = SQLiteBackend("~/.nanobot/memory.db")

        migrator = MemoryMigrator(source, target)
        result = await migrator.migrate_all()

        if result.errors:
            print(f"Migration completed with {len(result.errors)} errors")
    """

    def __init__(
        self,
        source: MemoryBackend,
        target: MemoryBackend,
        batch_size: int = 100
    ):
        """Initialize migrator with source and target backends.

        Args:
            source: Backend to migrate data from
            target: Backend to migrate data to
            batch_size: Number of entries to process per batch
        """
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self._progress_callback: Any = None

    def set_progress_callback(self, callback: Any) -> None:
        """Set a callback for progress updates.

        Args:
            callback: Function called with (current, total, tier) during migration
        """
        self._progress_callback = callback

    async def dry_run(self) -> MigrationReport:
        """Preview migration without executing.

        Returns:
            MigrationReport with statistics and estimates
        """
        warnings = []

        # Get entry counts from source
        by_tier = {tier: 0 for tier in MemoryTier}
        total = 0

        for tier in MemoryTier:
            try:
                # Query with empty string to get all entries of tier
                results = await self.source.retrieve(
                    query="",
                    tier=tier,
                    limit=100000
                )
                by_tier[tier] = len(results)
                total += len(results)
            except Exception as e:
                warnings.append(f"Could not query {tier.value}: {e}")

        # Estimate time (rough: 10ms per entry)
        estimated_time = total * 0.01

        return MigrationReport(
            source_entries=total,
            by_tier=by_tier,
            estimated_time_seconds=estimated_time,
            warnings=warnings
        )

    async def migrate_all(self) -> MigrationResult:
        """Migrate all memories from source to target.

        Returns:
            MigrationResult with statistics and any errors
        """
        start_time = datetime.now()
        result = MigrationResult()

        logger.info("Starting memory migration")

        for tier in MemoryTier:
            try:
                count = await self._migrate_tier(tier, result)
                result.migrated[tier] = count
                logger.info(f"Migrated {count} {tier.value} entries")
            except Exception as e:
                error_msg = f"Failed to migrate {tier.value}: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Migration completed: {sum(result.migrated.values())} entries "
            f"in {result.duration_seconds:.2f}s"
        )

        return result

    async def migrate_tier(self, tier: MemoryTier) -> int:
        """Migrate only a specific memory tier.

        Args:
            tier: The memory tier to migrate

        Returns:
            Number of entries migrated
        """
        result = MigrationResult()
        return await self._migrate_tier(tier, result)

    async def _migrate_tier(
        self,
        tier: MemoryTier,
        result: MigrationResult
    ) -> int:
        """Internal method to migrate a single tier.

        Args:
            tier: Memory tier to migrate
            result: Result object to update

        Returns:
            Number of entries successfully migrated
        """
        # Retrieve all entries from source
        source_results = await self.source.retrieve(
            query="",
            tier=tier,
            limit=100000
        )

        entries = [r.entry for r in source_results]
        result.total_entries += len(entries)

        migrated_count = 0

        # Process in batches
        for i in range(0, len(entries), self.batch_size):
            batch = entries[i:i + self.batch_size]

            for entry in batch:
                try:
                    # Check if entry already exists in target (by ID)
                    existing = await self._check_exists(entry.id, tier)
                    if existing:
                        result.skipped += 1
                        continue

                    # Store in target
                    await self.target.store(entry)
                    migrated_count += 1

                except Exception as e:
                    error_msg = f"Failed to migrate entry {entry.id}: {e}"
                    result.errors.append(error_msg)
                    logger.warning(error_msg)

            # Report progress
            if self._progress_callback:
                await self._report_progress(i + len(batch), len(entries), tier)

            # Small delay to prevent overwhelming the target
            await asyncio.sleep(0.001)

        return migrated_count

    async def _check_exists(self, entry_id: str, tier: MemoryTier) -> bool:
        """Check if an entry already exists in target.

        Args:
            entry_id: ID of entry to check
            tier: Tier to check in

        Returns:
            True if entry exists
        """
        try:
            # Query all entries in the tier and check for ID match
            results = await self.target.retrieve(
                query="",
                tier=tier,
                limit=100000
            )
            return any(r.entry.id == entry_id for r in results)
        except Exception:
            return False

    async def _report_progress(
        self,
        current: int,
        total: int,
        tier: MemoryTier
    ) -> None:
        """Report migration progress.

        Args:
            current: Current number of entries processed
            total: Total number of entries to process
            tier: Current tier being migrated
        """
        if self._progress_callback:
            if asyncio.iscoroutinefunction(self._progress_callback):
                await self._progress_callback(current, total, tier)
            else:
                self._progress_callback(current, total, tier)

    async def validate(self, sample_size: int = 100) -> bool:
        """Validate migration by comparing sample of entries.

        Args:
            sample_size: Number of entries to validate

        Returns:
            True if validation passes
        """
        logger.info(f"Validating migration with sample size {sample_size}")

        all_valid = True

        for tier in MemoryTier:
            try:
                # Get sample from source
                source_results = await self.source.retrieve(
                    query="",
                    tier=tier,
                    limit=sample_size
                )

                # Get all target entries for this tier to check existence
                target_results_all = await self.target.retrieve(
                    query="",
                    tier=tier,
                    limit=sample_size
                )
                target_by_id = {r.entry.id: r.entry for r in target_results_all}

                for result in source_results:
                    entry = result.entry
                    # Check if entry exists in target
                    target_entry = target_by_id.get(entry.id)

                    if not target_entry:
                        logger.error(f"Entry {entry.id} missing in target")
                        all_valid = False
                    elif target_entry.content != entry.content:
                        logger.error(f"Content mismatch for {entry.id}")
                        all_valid = False

            except Exception as e:
                logger.error(f"Validation failed for {tier.value}: {e}")
                all_valid = False

        return all_valid


async def create_backup(path: Path, suffix: str = ".backup") -> Path:
    """Create a backup of the memory storage.

    Args:
        path: Path to backup
        suffix: Suffix for backup name

    Returns:
        Path to backup location
    """
    backup_path = Path(str(path) + suffix + "." + datetime.now().strftime("%Y%m%d_%H%M%S"))

    if path.is_file():
        shutil.copy2(path, backup_path)
    elif path.is_dir():
        shutil.copytree(path, backup_path)
    else:
        raise ValueError(f"Path does not exist: {path}")

    logger.info(f"Created backup at {backup_path}")
    return backup_path


# Convenience functions for common migrations

async def migrate_filesystem_to_sqlite(
    fs_path: str,
    db_path: str,
    progress_callback: Any = None
) -> MigrationResult:
    """Migrate from FileSystemBackend to SQLiteBackend.

    Args:
        fs_path: Path to filesystem memory directory
        db_path: Path to SQLite database file
        progress_callback: Optional progress callback

    Returns:
        MigrationResult
    """
    from nanobot.agent.memory.storage import FileSystemBackend, SQLiteBackend

    source = FileSystemBackend(fs_path)
    target = SQLiteBackend(db_path)

    await source.initialize()
    await target.initialize()

    migrator = MemoryMigrator(source, target)
    if progress_callback:
        migrator.set_progress_callback(progress_callback)

    return await migrator.migrate_all()


async def migrate_sqlite_to_chroma(
    db_path: str,
    chroma_path: str,
    progress_callback: Any = None
) -> MigrationResult:
    """Migrate from SQLiteBackend to ChromaBackend.

    Args:
        db_path: Path to SQLite database
        chroma_path: Path to ChromaDB directory
        progress_callback: Optional progress callback

    Returns:
        MigrationResult
    """
    from nanobot.agent.memory.storage import SQLiteBackend
    from nanobot.agent.memory.storage.chroma import ChromaBackend

    source = SQLiteBackend(db_path)
    target = ChromaBackend(chroma_path)

    await source.initialize()
    await target.initialize()

    migrator = MemoryMigrator(source, target)
    if progress_callback:
        migrator.set_progress_callback(progress_callback)

    return await migrator.migrate_all()
