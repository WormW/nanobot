"""OpenViking context database adapter (optional dependency).

Provides a VikingContextProvider that wraps AsyncOpenViking for use by
ContextBuilder and agent tools.  If ``openviking`` is not installed the
module still imports cleanly — callers check ``HAS_VIKING`` at runtime.

Install with:  pip install nanobot-ai[viking]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.context_engine import ContextEngine, TurnCapture

if TYPE_CHECKING:
    from nanobot.config.schema import VikingConfig

try:
    from openviking import AsyncOpenViking

    HAS_VIKING = True
except ImportError:
    HAS_VIKING = False
    AsyncOpenViking = None  # type: ignore[assignment,misc]


@dataclass(slots=True)
class RecalledItem:
    uri: str
    title: str
    score: float | None
    summary: str
    snippet: str


class VikingContextProvider(ContextEngine):
    """Thin adapter between nanobot and the OpenViking client.

    nanobot uses Viking as a context engine: recall relevant context before a
    turn, then archive the turn and commit the session afterward. This mirrors
    the OpenClaw plugin's recall/capture lifecycle more closely than injecting
    synchronous overviews into the system prompt.
    """

    def __init__(self, config: VikingConfig) -> None:
        self._config = config
        self._ov: Any = None  # AsyncOpenViking instance (set in initialize)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start the embedded OpenViking storage & index services."""
        if not HAS_VIKING:
            logger.warning("openviking package not installed — VikingContextProvider disabled")
            return

        # Point OpenViking at the user-specified config, if any.
        if self._config.config_path:
            os.environ.setdefault("OPENVIKING_CONFIG_FILE", self._config.config_path)

        try:
            self._ov = AsyncOpenViking()
            await self._ov.initialize()
            self._initialized = True
            logger.info("OpenViking initialized successfully")
        except Exception:
            logger.exception("Failed to initialize OpenViking — falling back to default memory")
            self._ov = None

    async def close(self) -> None:
        """Release OpenViking resources."""
        if self._ov is not None:
            try:
                await self._ov.close()
            except Exception:
                logger.exception("Error closing OpenViking")
            finally:
                self._ov = None
                self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._ov is not None

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    async def search_context(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 5,
        target_uri: str | None = None,
        score_threshold: float | None = None,
    ) -> str:
        """Run a semantic search across Viking resources and return formatted results."""
        if not self.is_ready:
            return "OpenViking is not available."
        try:
            results = await self._ov.search(
                query=query,
                session_id=session_id,
                limit=limit,
                target_uri=target_uri or self._config.target_uri,
                score_threshold=score_threshold,
            )
            if not results:
                return "No results found."
            items = self._normalize_search_results(results)
            if not items:
                return "No results found."
            return self._render_recalled_context(items)
        except Exception as exc:
            logger.exception("Viking search failed")
            return f"Search error: {exc}"

    def _normalize_search_results(self, raw: Any) -> list[RecalledItem]:
        """Normalize OpenViking search output into a stable prompt-facing shape."""
        if isinstance(raw, dict):
            candidates = raw.get("items") or raw.get("results") or raw.get("hits") or []
        elif hasattr(raw, "memories") or hasattr(raw, "resources") or hasattr(raw, "skills"):
            candidates = [
                *list(getattr(raw, "memories", []) or []),
                *list(getattr(raw, "resources", []) or []),
                *list(getattr(raw, "skills", []) or []),
            ]
            if not candidates and getattr(raw, "query_results", None):
                for qr in getattr(raw, "query_results", []) or []:
                    candidates.extend(list(getattr(qr, "matched_contexts", []) or []))
        elif isinstance(raw, list):
            candidates = raw
        else:
            candidates = []

        items: list[RecalledItem] = []
        for entry in candidates:
            if isinstance(entry, dict):
                data = entry
            else:
                data = {
                    "uri": getattr(entry, "uri", "") or getattr(entry, "id", ""),
                    "title": getattr(entry, "title", "") or getattr(entry, "name", ""),
                    "score": getattr(entry, "score", None),
                    "summary": getattr(entry, "summary", "") or getattr(entry, "abstract", "") or getattr(entry, "overview", ""),
                    "snippet": getattr(entry, "snippet", "") or getattr(entry, "content", "") or getattr(entry, "text", ""),
                }
            uri = str(data.get("uri") or data.get("id") or "")
            title = str(data.get("title") or data.get("name") or uri or "Context")
            score = data.get("score")
            if score is not None:
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = None
            summary = str(data.get("summary") or data.get("abstract") or data.get("overview") or "").strip()
            snippet = str(data.get("snippet") or data.get("content") or data.get("text") or "").strip()
            if summary and snippet.startswith(summary):
                snippet = snippet[len(summary):].strip()
            if not snippet and not summary and hasattr(entry, "__dict__"):
                snippet = str(entry)
            items.append(RecalledItem(
                uri=uri,
                title=title,
                score=score,
                summary=summary[:280],
                snippet=snippet[:420],
            ))
            if len(items) >= self._config.recall_limit:
                break
        return items

    @staticmethod
    def _render_recalled_context(items: list[RecalledItem]) -> str:
        """Render recalled items into a compact, prompt-safe context block."""
        lines = [
            "# Recalled Context",
            "",
            "Use this context only if it is relevant to the current request.",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append("")
            lines.append(f"{idx}. {item.title}")
            if item.uri:
                lines.append(f"URI: {item.uri}")
            if item.score is not None:
                lines.append(f"Score: {item.score:.3f}")
            if item.summary:
                lines.append(f"Summary: {item.summary}")
            if item.snippet:
                lines.append(f"Snippet: {item.snippet}")
        return "\n".join(lines)

    async def recall(self, *, session_id: str, query: str) -> str:
        """Recall prompt-ready context for an agent turn."""
        return await self.search_context(
            query=query,
            session_id=session_id,
            limit=self._config.recall_limit,
            target_uri=self._config.target_uri,
            score_threshold=self._config.recall_score_threshold,
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    async def add_resource(
        self,
        path: str,
        build_index: bool = True,
        summarize: bool = True,
    ) -> str:
        """Add a file or URL as a Viking resource."""
        if not self.is_ready:
            return "OpenViking is not available."
        try:
            await self._ov.add_resource(
                path,
                build_index=build_index,
                summarize=summarize,
                wait=True,
                timeout=120,
            )
            return f"Resource added: {path}"
        except Exception as exc:
            logger.exception("Viking add_resource failed")
            return f"Error adding resource: {exc}"

    # ------------------------------------------------------------------
    # Session commit (extract memories from conversation)
    # ------------------------------------------------------------------

    async def commit_session(self, session_id: str) -> str:
        """Commit a Viking session to extract long-term memories."""
        if not self.is_ready:
            return "OpenViking is not available."
        try:
            await self._ov.commit_session(session_id)
            return f"Session '{session_id}' committed."
        except Exception as exc:
            logger.exception("Viking commit_session failed")
            return f"Error committing session: {exc}"

    async def capture(self, turn: TurnCapture) -> None:
        """Append a turn into a Viking session and trigger memory extraction."""
        if not self.is_ready:
            return
        try:
            await self._ov.add_message(turn.session_id, role="user", content=turn.user_text)
            await self._ov.add_message(turn.session_id, role="assistant", content=turn.assistant_text)
            await self._ov.commit_session(turn.session_id)
        except Exception:
            logger.exception("Viking capture failed")
