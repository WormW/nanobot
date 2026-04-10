"""CLI commands for memory management.

Provides commands for migrating, backing up, and managing memory data.
"""

import asyncio
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nanobot.agent.memory.migration import (
    MemoryMigrator,
    create_backup,
    migrate_filesystem_to_sqlite,
    migrate_sqlite_to_chroma,
)
from nanobot.agent.memory.storage import FileSystemBackend, SQLiteBackend
from nanobot.agent.memory.types import MemoryTier
from nanobot.config.paths import get_workspace_dir
from nanobot.memory_backfill.service import MemoryBackfillService

console = Console()
app = typer.Typer(help="Memory management commands")


@app.command("migrate")
def migrate(
    from_backend: str = typer.Option(
        "filesystem",
        "--from",
        help="Source backend: filesystem, sqlite, chroma"
    ),
    from_path: str = typer.Option(
        "~/.nanobot/workspace/memory",
        "--from-path",
        help="Path to source memory storage"
    ),
    to_backend: str = typer.Option(
        "sqlite",
        "--to",
        help="Target backend: sqlite, chroma"
    ),
    to_path: str = typer.Option(
        "~/.nanobot/memory.db",
        "--to-path",
        help="Path to target memory storage"
    ),
    tier: str = typer.Option(
        None,
        "--tier",
        help="Migrate only specific tier: working, episodic, semantic"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview migration without executing"
    ),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Create backup before migration"
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate after migration"
    ),
):
    """Migrate memory data between storage backends."""

    async def do_migrate():
        # Expand paths
        from_path_expanded = Path(from_path).expanduser()
        to_path_expanded = Path(to_path).expanduser()

        # Create source backend
        if from_backend == "filesystem":
            source = FileSystemBackend(str(from_path_expanded))
        elif from_backend == "sqlite":
            source = SQLiteBackend(str(from_path_expanded))
        else:
            console.print(f"[red]Unsupported source backend: {from_backend}[/red]")
            raise typer.Exit(1)

        # Create target backend
        if to_backend == "sqlite":
            target = SQLiteBackend(str(to_path_expanded))
        elif to_backend == "chroma":
            from nanobot.agent.memory.storage.chroma import ChromaBackend
            target = ChromaBackend(str(to_path_expanded))
        else:
            console.print(f"[red]Unsupported target backend: {to_backend}[/red]")
            raise typer.Exit(1)

        # Initialize backends
        await source.initialize()
        await target.initialize()

        # Dry run
        if dry_run:
            console.print("[yellow]Running dry-run (preview only)...[/yellow]")
            migrator = MemoryMigrator(source, target)
            report = await migrator.dry_run()

            table = Table(title="Migration Preview")
            table.add_column("Tier", style="cyan")
            table.add_column("Entries", style="green")

            for tier_enum, count in report.by_tier.items():
                table.add_row(tier_enum.value, str(count))

            table.add_row("Total", str(report.source_entries))
            console.print(table)
            console.print(f"\nEstimated time: {report.estimated_time_seconds:.1f}s")

            if report.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in report.warnings:
                    console.print(f"  - {warning}")
            return

        # Create backup
        if backup:
            console.print("[blue]Creating backup...[/blue]")
            try:
                backup_path = await create_backup(from_path_expanded)
                console.print(f"[green]Backup created: {backup_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Backup failed: {e}[/yellow]")
                if not typer.confirm("Continue without backup?"):
                    raise typer.Exit(1)

        # Perform migration
        console.print(f"\n[blue]Migrating from {from_backend} to {to_backend}...[/blue]")

        migrator = MemoryMigrator(source, target)

        # Progress callback
        def progress_callback(current, total, tier_enum):
            console.print(f"  {tier_enum.value}: {current}/{total}")

        migrator.set_progress_callback(progress_callback)

        # Execute migration
        if tier:
            tier_enum = MemoryTier(tier)
            console.print(f"[blue]Migrating only {tier} tier...[/blue]")
            count = await migrator.migrate_tier(tier_enum)
            result = type('Result', (), {
                'migrated': {tier_enum: count},
                'errors': [],
                'duration_seconds': 0
            })()
        else:
            result = await migrator.migrate_all()

        # Display results
        console.print("\n[green]Migration completed![/green]")

        table = Table(title="Migration Results")
        table.add_column("Tier", style="cyan")
        table.add_column("Migrated", style="green")

        for tier_enum, count in result.migrated.items():
            table.add_row(tier_enum.value, str(count))

        console.print(table)
        console.print(f"Duration: {result.duration_seconds:.2f}s")

        if result.errors:
            console.print(f"\n[yellow]{len(result.errors)} errors occurred:[/yellow]")
            for error in result.errors[:5]:
                console.print(f"  - {error}")
            if len(result.errors) > 5:
                console.print(f"  ... and {len(result.errors) - 5} more")

        # Validation
        if validate and not result.errors:
            console.print("\n[blue]Validating migration...[/blue]")
            is_valid = await migrator.validate()
            if is_valid:
                console.print("[green]Validation passed![/green]")
            else:
                console.print("[red]Validation failed - some entries may be missing[/red]")

    asyncio.run(do_migrate())


@app.command("stats")
def stats(
    backend: str = typer.Option(
        "filesystem",
        "--backend",
        help="Backend type: filesystem, sqlite, chroma"
    ),
    path: str = typer.Option(
        "~/.nanobot/workspace/memory",
        "--path",
        help="Path to memory storage"
    ),
):
    """Show memory statistics."""

    async def do_stats():
        path_expanded = Path(path).expanduser()

        # Create backend
        if backend == "filesystem":
            from_backend = FileSystemBackend(str(path_expanded))
        elif backend == "sqlite":
            from_backend = SQLiteBackend(str(path_expanded))
        else:
            console.print(f"[red]Unsupported backend: {backend}[/red]")
            raise typer.Exit(1)

        await from_backend.initialize()

        # Query each tier
        table = Table(title=f"Memory Statistics ({backend})")
        table.add_column("Tier", style="cyan")
        table.add_column("Count", style="green")

        total = 0
        for tier in MemoryTier:
            try:
                results = await from_backend.retrieve("", tier=tier, limit=100000)
                count = len(results)
                total += count
                table.add_row(tier.value, str(count))
            except Exception as e:
                table.add_row(tier.value, f"[red]Error: {e}[/red]")

        table.add_row("Total", str(total), style="bold")
        console.print(table)

    asyncio.run(do_stats())


@app.command("backup")
def backup(
    path: str = typer.Option(
        "~/.nanobot/workspace/memory",
        "--path",
        help="Path to memory storage to backup"
    ),
    output: str = typer.Option(
        None,
        "--output",
        help="Output path for backup (auto-generated if not specified)"
    ),
):
    """Create a backup of memory storage."""

    async def do_backup():
        path_expanded = Path(path).expanduser()

        if output:
            output_path = Path(output)
        else:
            output_path = None

        try:
            backup_path = await create_backup(path_expanded, suffix="")
            if output_path:
                # Rename to specified output
                backup_path.rename(output_path)
                backup_path = output_path

            console.print(f"[green]Backup created: {backup_path}[/green]")
        except Exception as e:
            console.print(f"[red]Backup failed: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(do_backup())


@app.command("backfill")
def backfill(
    session: str | None = typer.Option(
        None,
        "--session",
        "-s",
        help="Specific session to backfill (e.g., 'telegram:-123456')",
    ),
    workspace: str = typer.Option(
        "~/.nanobot/workspace",
        "--workspace",
        "-w",
        help="Path to workspace directory",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Preview what would be imported without actually importing",
    ),
    all_sessions: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Backfill all available sessions",
    ),
):
    """Backfill memory history.jsonl to external storage.

    Imports agent_turn entries from history.jsonl files to external memory
    systems like OpenViking. Supports incremental import with offset tracking.

    Examples:
        nb memory backfill --session telegram:-123456
        nb memory backfill --all --dry-run
    """

    async def do_backfill():
        ws_path = Path(workspace).expanduser()

        if not ws_path.exists():
            console.print(f"[red]Workspace not found: {ws_path}[/red]")
            raise typer.Exit(1)

        service = MemoryBackfillService(workspace=ws_path)

        if session:
            console.print(f"Backfilling session: {session}...")
            result = await service.backfill_session(session, dry_run=dry_run)

            if result.success:
                if dry_run:
                    console.print(
                        f"[green]Would import {result.entries_scanned} entries "
                        f"({result.entries_filtered} filtered)[/green]"
                    )
                else:
                    console.print(
                        f"[green]Imported {result.entries_imported} entries "
                        f"({result.entries_filtered} filtered)[/green]"
                    )
            else:
                console.print(f"[red]Failed: {result.errors}[/red]")

        elif all_sessions:
            sessions = service.list_sessions()
            if not sessions:
                console.print("[yellow]No sessions found.[/yellow]")
                return

            console.print(f"Found {len(sessions)} session(s)")
            total_imported = 0

            for s in sessions:
                result = await service.backfill_session(s, dry_run=dry_run)
                if result.success:
                    total_imported += result.entries_imported
                    console.print(f"  {s}: {result.entries_imported} imported")
                else:
                    console.print(f"  [red]{s}: failed[/red]")

            console.print(f"\n[green]Total: {total_imported} entries imported[/green]")

        else:
            console.print("[yellow]Use --session or --all[/yellow]")
            raise typer.Exit(1)

    asyncio.run(do_backfill())


@app.command("backfill-status")
def backfill_status(
    session: str | None = typer.Option(
        None,
        "--session",
        "-s",
        help="Specific session to check",
    ),
    workspace: str = typer.Option(
        "~/.nanobot/workspace",
        "--workspace",
        "-w",
        help="Path to workspace directory",
    ),
):
    """Show backfill status for sessions."""

    async def do_status():
        ws_path = Path(workspace).expanduser()
        service = MemoryBackfillService(workspace=ws_path)

        info = service.get_status(session)

        if session:
            if info.get("status") == "not_found":
                console.print(f"[yellow]Session '{session}' not found[/yellow]")
            else:
                table = Table(title=f"Backfill Status: {session}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")

                for key, value in info.items():
                    if key != "status":
                        table.add_row(key, str(value))

                console.print(table)
        else:
            sessions = info.get("sessions", {})
            if not sessions:
                console.print("[yellow]No backfill history found[/yellow]")
                return

            table = Table(title="Backfill Status")
            table.add_column("Session", style="cyan")
            table.add_column("Last Cursor", style="green")
            table.add_column("Imported", style="blue")
            table.add_column("Last Sync", style="dim")

            for s, data in sessions.items():
                table.add_row(
                    s,
                    str(data.get("last_cursor", 0)),
                    str(data.get("total_imported", 0)),
                    data.get("last_sync_at", "never")[:19],
                )

            console.print(table)

    asyncio.run(do_status())


if __name__ == "__main__":
    app()
