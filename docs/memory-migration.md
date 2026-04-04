# Memory Data Migration Guide

This guide explains how to migrate memory data between different storage backends.

## Overview

Nanobot supports multiple storage backends for the layered memory system:

- **FileSystem** (default): JSON/JSONL files in `~/.nanobot/workspace/memory/`
- **SQLite**: Single database file at `~/.nanobot/memory.db`
- **ChromaDB**: Vector database for semantic search (requires embedding provider)

## When to Migrate

You may want to migrate data when:

- Switching from file-based to database storage for better performance
- Moving to ChromaDB for vector search capabilities
- Consolidating multiple agent memory stores
- Backing up or archiving memory data

## Migration Commands

### 1. Preview Migration (Dry Run)

Before executing a migration, preview what will be migrated:

```bash
nanobot memory migrate \
  --from filesystem \
  --from-path ~/.nanobot/workspace/memory \
  --to sqlite \
  --to-path ~/.nanobot/memory.db \
  --dry-run
```

Output shows:
- Entry counts per tier (working, episodic, semantic)
- Total entries to migrate
- Estimated migration time

### 2. Execute Migration

Migrate from FileSystem to SQLite:

```bash
nanobot memory migrate \
  --from filesystem \
  --from-path ~/.nanobot/workspace/memory \
  --to sqlite \
  --to-path ~/.nanobot/memory.db
```

Options:
- `--tier {working|episodic|semantic}`: Migrate only specific tier
- `--no-backup`: Skip automatic backup creation
- `--no-validate`: Skip post-migration validation

### 3. Migrate Specific Tier

Migrate only semantic memories (useful when enabling vector search):

```bash
nanobot memory migrate \
  --from sqlite \
  --from-path ~/.nanobot/memory.db \
  --to chroma \
  --to-path ~/.nanobot/chroma_db \
  --tier semantic
```

### 4. Create Backup

Manually create a backup before migration:

```bash
# Auto-named backup
nanobot memory backup \
  --path ~/.nanobot/workspace/memory

# Custom output path
nanobot memory backup \
  --path ~/.nanobot/memory.db \
  --output ~/backups/memory-$(date +%Y%m%d).db
```

### 5. View Statistics

Check memory statistics before/after migration:

```bash
# FileSystem backend
nanobot memory stats \
  --backend filesystem \
  --path ~/.nanobot/workspace/memory

# SQLite backend
nanobot memory stats \
  --backend sqlite \
  --path ~/.nanobot/memory.db
```

## Migration Scenarios

### Scenario 1: FileSystem → SQLite (Performance)

Best for: Improving query performance with large memory stores

```bash
# 1. Preview
nanobot memory migrate --from filesystem --to sqlite --dry-run

# 2. Execute with backup
nanobot memory migrate --from filesystem --to sqlite

# 3. Verify
nanobot memory stats --backend sqlite
```

After migration, update your agent configuration to use SQLite backend.

### Scenario 2: SQLite → ChromaDB (Vector Search)

Best for: Enabling semantic similarity search

```bash
# Migrate only semantic tier to ChromaDB
nanobot memory migrate \
  --from sqlite \
  --to chroma \
  --tier semantic
```

Note: ChromaDB backend requires an embedding provider configured.

### Scenario 3: Full Backup and Restore

```bash
# Create timestamped backup
nanobot memory backup --path ~/.nanobot/workspace/memory

# Backup appears as: ~/.nanobot/workspace/memory.backup.20240404_123045

# Restore by migrating back
nanobot memory migrate \
  --from filesystem \
  --from-path ~/.nanobot/workspace/memory.backup.20240404_123045 \
  --to filesystem \
  --to-path ~/.nanobot/workspace/memory
```

## Programmatic Migration

For custom migration workflows, use the Python API:

```python
import asyncio
from nanobot.agent.memory import (
    migrate_filesystem_to_sqlite,
    migrate_sqlite_to_chroma,
    MemoryMigrator,
    FileSystemBackend,
    SQLiteBackend,
)

async def custom_migration():
    # Convenience function
    result = await migrate_filesystem_to_sqlite(
        fs_path="~/.nanobot/workspace/memory",
        db_path="~/.nanobot/memory.db"
    )

    print(f"Migrated {sum(result.migrated.values())} entries")
    print(f"Duration: {result.duration_seconds:.2f}s")

    if result.errors:
        print(f"Errors: {len(result.errors)}")

    # Custom migrator with progress callback
    def on_progress(current, total, tier):
        print(f"{tier.value}: {current}/{total}")

    source = FileSystemBackend("~/.nanobot/workspace/memory")
    target = SQLiteBackend("~/.nanobot/memory.db")

    await source.initialize()
    await target.initialize()

    migrator = MemoryMigrator(source, target)
    migrator.set_progress_callback(on_progress)

    # Dry run
    report = await migrator.dry_run()
    print(f"Estimated time: {report.estimated_time_seconds:.1f}s")

    # Execute
    result = await migrator.migrate_all()

    # Validate
    is_valid = await migrator.validate()
    print(f"Validation: {'passed' if is_valid else 'failed'}")

asyncio.run(custom_migration())
```

## Migration Reference

### Supported Backend Combinations

| From | To | Notes |
|------|-----|-------|
| FileSystem | SQLite | Full support, recommended for performance |
| FileSystem | Chroma | Only semantic tier with embeddings |
| SQLite | Chroma | Only semantic tier with embeddings |
| SQLite | FileSystem | Full support |
| Chroma | SQLite | Limited, embeddings may not transfer |
| Chroma | FileSystem | Limited, embeddings may not transfer |

### Data Integrity

- **Duplicate Detection**: Entries with the same ID are skipped during migration
- **Validation**: Post-migration validation compares content checksums
- **Backup**: Automatic backup created before migration (unless disabled)
- **Rollback**: Restore from backup if migration fails

### Performance Considerations

- **Batch Size**: Default 100 entries per batch
- **Rate Limiting**: 1ms delay between batches to prevent overwhelming target
- **Progress Tracking**: Real-time progress callbacks for large migrations
- **Memory Usage**: Streams entries to minimize memory footprint

## Troubleshooting

### Migration Fails Midway

1. Check backup location: `~/.nanobot/workspace/memory.backup.*`
2. Restore from backup by reversing migration direction
3. Fix issues and retry

### Validation Failures

- Indicates data corruption or incomplete transfer
- Check logs for specific entry IDs
- Consider migrating in smaller batches with `--tier` option

### Large Dataset Migration

For very large memory stores:

1. Migrate one tier at a time
2. Use programmatic API for custom batch sizes
3. Monitor disk space during migration

```python
# Increase batch size for faster migration
migrator = MemoryMigrator(source, target, batch_size=500)
```

## Important Notes

- **No Auto-Migration**: System does NOT auto-detect backend changes. Manual migration required.
- **Embedding Dependency**: ChromaDB backend requires embedding provider setup
- **Concurrent Access**: Avoid using memory during migration
- **Version Compatibility**: Migrate between same nanobot versions when possible
