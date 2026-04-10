"""Cron integration for automatic memory backfill.

Registers an hourly cron job to automatically backfill memory history
to external storage systems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.cron.types import CronJob, CronSchedule
from nanobot.memory_backfill.service import MemoryBackfillService, MemorySink


async def run_backfill_cron(
    workspace: Path,
    memory_sink: MemorySink | None = None,
) -> None:
    """Run backfill for all sessions.

    This function is designed to be called by the cron service.
    It will discover all sessions with history files and backfill
    any new entries since the last sync.

    Args:
        workspace: Path to the workspace directory
        memory_sink: Optional memory sink for writing to external storage
    """
    service = MemoryBackfillService(
        workspace=workspace,
        memory_sink=memory_sink,
    )

    sessions = service.list_sessions()
    if not sessions:
        logger.debug("No sessions found for backfill")
        return

    total_imported = 0
    total_errors = 0

    for session in sessions:
        try:
            result = await service.backfill_session(session, dry_run=False)
            total_imported += result.entries_imported
            if result.errors:
                total_errors += len(result.errors)
                logger.warning(
                    "Backfill errors for {}: {}",
                    session,
                    result.errors,
                )
        except Exception as e:
            logger.error("Failed to backfill session {}: {}", session, e)
            total_errors += 1

    logger.info(
        "Cron backfill completed: {} entries imported, {} errors",
        total_imported,
        total_errors,
    )


def create_backfill_cron_job(workspace: Path) -> CronJob:
    """Create a cron job for hourly memory backfill.

    Args:
        workspace: Path to the workspace directory

    Returns:
        CronJob configured to run backfill every hour
    """
    return CronJob(
        id="memory-backfill-hourly",
        name="Memory Backfill (Hourly)",
        enabled=True,
        schedule=CronSchedule(
            kind="cron",
            expr="0 * * * *",  # Every hour at minute 0
            tz="UTC",
        ),
        payload={
            "kind": "memory_backfill",
            "workspace": str(workspace),
        },
    )


# Compatibility alias for the service handler
async def handle_cron_backfill(payload: dict[str, Any]) -> str | None:
    """Handle a cron backfill job.

    This function is called by the cron service when the backfill job fires.
    """
    workspace_path = payload.get("workspace")
    if not workspace_path:
        return "No workspace specified in payload"

    try:
        await run_backfill_cron(Path(workspace_path))
        return None  # Success
    except Exception as e:
        return f"Backfill failed: {e}"
