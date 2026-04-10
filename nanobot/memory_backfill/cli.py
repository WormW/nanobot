"""CLI commands for memory backfill operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nanobot.config.paths import get_workspace_dir
from nanobot.memory_backfill.service import MemoryBackfillService

console = Console()
app = typer.Typer(help="Memory backfill commands for importing history to external storage")


def _get_default_workspace() -> Path:
    """Get default workspace path."""
    return get_workspace_dir()


@app.command("backfill")
def backfill(
    session: str | None = typer.Option(
        None,
        "--session",
        "-s",
        help="Specific session to backfill (e.g., 'telegram:-123456' or 'workspace:myproject-coding')",
    ),
    workspace: Path = typer.Option(
        None,
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
):
    """Backfill memory history.jsonl to external storage (e.g., OpenViking).

    Examples:
        nb memory backfill --session telegram:-123456
        nb memory backfill --all --dry-run
        nb memory backfill --session workspace:myproject-coding --workspace ~/myworkspace
    """

    async def do_backfill():
        ws_path = workspace or _get_default_workspace()

        if not ws_path.exists():
            console.print(f"[red]Workspace not found: {ws_path}[/red]")
            raise typer.Exit(1)

        # Create service
        # Note: In production, you would configure the memory_sink here
        # to write to OpenViking or another external system
        service = MemoryBackfillService(workspace=ws_path)

        if session:
            # Backfill specific session
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"Backfilling {session}...", total=None)
                result = await service.backfill_session(session, dry_run=dry_run)

            _display_result(result, verbose)

        elif all_sessions:
            # Backfill all sessions
            sessions = service.list_sessions()
            if not sessions:
                console.print("[yellow]No sessions found with history files.[/yellow]")
                return

            console.print(f"Found {len(sessions)} session(s) to backfill:\n")
            for s in sessions:
                console.print(f"  - {s}")
            console.print()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Backfilling sessions...", total=len(sessions))
                results = []
                for s in sessions:
                    progress.update(task, description=f"Backfilling {s}...")
                    result = await service.backfill_session(s, dry_run=dry_run)
                    results.append(result)
                    progress.advance(task)

            _display_summary(results)

        else:
            console.print(
                "[yellow]Please specify --session or --all. Use --help for more information.[/yellow]"
            )
            raise typer.Exit(1)

    asyncio.run(do_backfill())


@app.command("status")
def status(
    session: str | None = typer.Option(
        None,
        "--session",
        "-s",
        help="Specific session to check",
    ),
    workspace: Path = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Path to workspace directory",
    ),
):
    """Show backfill status for sessions."""

    async def do_status():
        ws_path = workspace or _get_default_workspace()
        service = MemoryBackfillService(workspace=ws_path)

        if session:
            # Show status for specific session
            info = service.get_status(session)
            if info.get("status") == "not_found":
                console.print(f"[yellow]Session '{session}' not found.[/yellow]")
            else:
                _display_status_table([info])
        else:
            # Show status for all sessions
            info = service.get_status()
            sessions = info.get("sessions", {})
            if not sessions:
                console.print("[yellow]No backfill history found.[/yellow]")
                return

            session_list = [
                {"session": s, **o} for s, o in sessions.items()
            ]
            _display_status_table(session_list)

    asyncio.run(do_status())


def _display_result(result, verbose: bool = False):
    """Display a single backfill result."""
    if result.success:
        if result.entries_imported > 0:
            console.print(
                f"[green]✓[/green] Backfilled {result.session}: "
                f"{result.entries_imported} entries imported"
            )
        else:
            console.print(
                f"[dim]✓ {result.session}: No new entries to import[/dim]"
            )
    else:
        console.print(f"[red]✗[/red] Backfill failed for {result.session}")
        for error in result.errors:
            console.print(f"  [red]Error: {error}[/red]")

    if verbose:
        table = Table(title=f"Backfill Details: {result.session}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Entries Scanned", str(result.entries_scanned))
        table.add_row("Entries Imported", str(result.entries_imported))
        table.add_row("Entries Filtered", str(result.entries_filtered))
        table.add_row("Last Cursor", str(result.last_cursor))

        if result.errors:
            table.add_row("Errors", str(len(result.errors)))

        console.print(table)


def _display_summary(results: list):
    """Display summary of multiple backfill results."""
    total_imported = sum(r.entries_imported for r in results)
    total_scanned = sum(r.entries_scanned for r in results)
    total_errors = sum(len(r.errors) for r in results)

    console.print("\n" + "=" * 50)
    console.print(f"[bold]Backfill Summary[/bold]")
    console.print("=" * 50)
    console.print(f"Sessions processed: {len(results)}")
    console.print(f"Total entries scanned: {total_scanned}")
    console.print(f"Total entries imported: {total_imported}")
    if total_errors > 0:
        console.print(f"[red]Total errors: {total_errors}[/red]")


def _display_status_table(sessions: list[dict[str, Any]]):
    """Display status table for sessions."""
    table = Table(title="Backfill Status")
    table.add_column("Session", style="cyan")
    table.add_column("Last Cursor", style="green")
    table.add_column("Total Imported", style="blue")
    table.add_column("Last Sync", style="dim")

    for s in sessions:
        table.add_row(
            s.get("session", "unknown"),
            str(s.get("last_cursor", 0)),
            str(s.get("total_imported", 0)),
            s.get("last_sync_at", "never")[:19],  # Trim to date+time
        )

    console.print(table)


if __name__ == "__main__":
    app()
