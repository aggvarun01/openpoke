"""Persistence for bounded Execution Agent retrieval profiles."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...logging_config import logger


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PROFILE_DB_PATH = _DATA_DIR / "execution_agents" / "retrieval_profiles.sqlite"
_MAX_INSTRUCTION_LENGTH = 2000


@dataclass(frozen=True)
class AgentProfile:
    """Minimal profile data used to retrieve an existing Execution Agent."""

    agent_name: str
    instruction: str
    last_active_at: str
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None


class AgentProfileStore:
    """SQLite-backed storage for bounded, deterministic agent profiles."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=1, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    agent_name TEXT PRIMARY KEY,
                    instruction TEXT NOT NULL,
                    last_active_at TEXT NOT NULL,
                    embedding_json TEXT,
                    embedding_model TEXT,
                    embedding_dimensions INTEGER
                )
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(agent_profiles)")
            }
            for column, column_type in (
                ("embedding_json", "TEXT"),
                ("embedding_model", "TEXT"),
                ("embedding_dimensions", "INTEGER"),
            ):
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE agent_profiles ADD COLUMN {column} {column_type}"
                    )

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> AgentProfile:
        embedding_json = row["embedding_json"]
        embedding = json.loads(embedding_json) if embedding_json else None
        if embedding is not None and not isinstance(embedding, list):
            raise ValueError("stored profile embedding is not a vector")
        return AgentProfile(
            agent_name=row["agent_name"],
            instruction=row["instruction"],
            last_active_at=row["last_active_at"],
            embedding=embedding,
            embedding_model=row["embedding_model"],
            embedding_dimensions=row["embedding_dimensions"],
        )

    @staticmethod
    def _normalize_instruction(instruction: str) -> str:
        return " ".join(instruction.split())[:_MAX_INSTRUCTION_LENGTH]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get(self, agent_name: str) -> Optional[AgentProfile]:
        """Return a stored profile, if one exists."""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT agent_name, instruction, last_active_at, embedding_json, "
                "embedding_model, embedding_dimensions "
                "FROM agent_profiles WHERE agent_name = ?",
                (agent_name,),
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def upsert(self, agent_name: str, instruction: str) -> AgentProfile:
        """Create or refresh a profile using the latest delegated instruction."""
        if not agent_name.strip():
            raise ValueError("agent_name must not be empty")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")

        bounded_instruction = self._normalize_instruction(instruction)
        if not bounded_instruction:
            raise ValueError("instruction must not be empty")
        last_active_at = self._now()

        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO agent_profiles (agent_name, instruction, last_active_at)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_name) DO UPDATE SET
                    instruction = excluded.instruction,
                    last_active_at = excluded.last_active_at,
                    embedding_json = NULL,
                    embedding_model = NULL,
                    embedding_dimensions = NULL
                """,
                (agent_name, bounded_instruction, last_active_at),
            )

        return AgentProfile(agent_name, bounded_instruction, last_active_at)

    def get_or_backfill(
        self,
        agent_name: str,
        instruction: Optional[str],
    ) -> Optional[AgentProfile]:
        """Lazily create a missing profile from an existing recorded instruction."""
        profile = self.get(agent_name)
        if profile is not None or not instruction or not instruction.strip():
            return profile

        bounded_instruction = self._normalize_instruction(instruction)
        if not bounded_instruction:
            return None

        last_active_at = self._now()
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_profiles
                    (agent_name, instruction, last_active_at)
                VALUES (?, ?, ?)
                """,
                (agent_name, bounded_instruction, last_active_at),
            )
            row = conn.execute(
                "SELECT agent_name, instruction, last_active_at, embedding_json, "
                "embedding_model, embedding_dimensions "
                "FROM agent_profiles WHERE agent_name = ?",
                (agent_name,),
            ).fetchone()

        return self._row_to_profile(row) if row else None

    def update_embedding(
        self,
        agent_name: str,
        instruction: str,
        model: str,
        embedding: list[float],
    ) -> bool:
        """Store a vector only if the profile still has the embedded instruction."""
        if not model.strip():
            raise ValueError("model must not be empty")
        if not embedding or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in embedding
        ):
            raise ValueError("embedding must contain finite numeric values")

        with self._lock, closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                UPDATE agent_profiles
                SET embedding_json = ?, embedding_model = ?, embedding_dimensions = ?
                WHERE agent_name = ? AND instruction = ?
                """,
                (
                    json.dumps(embedding, separators=(",", ":")),
                    model,
                    len(embedding),
                    agent_name,
                    self._normalize_instruction(instruction),
                ),
            )
        return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Remove all derived agent profiles."""
        with self._lock, closing(self._connect()) as conn:
            conn.execute("DELETE FROM agent_profiles")


_agent_profile_store: Optional[AgentProfileStore] = None
_agent_profile_store_lock = threading.Lock()


def get_agent_profile_store() -> AgentProfileStore:
    """Return the shared agent profile store."""
    global _agent_profile_store
    if _agent_profile_store is None:
        with _agent_profile_store_lock:
            if _agent_profile_store is None:
                _agent_profile_store = AgentProfileStore(_PROFILE_DB_PATH)
    return _agent_profile_store


def get_or_backfill_agent_profile(agent_name: str) -> Optional[AgentProfile]:
    """Load a profile, creating it from an existing request log when necessary."""
    try:
        store = get_agent_profile_store()
        profile = store.get(agent_name)
        if profile is not None:
            return profile

        from .log_store import get_execution_agent_logs

        instruction = get_execution_agent_logs().load_latest_request(agent_name)
        return store.get_or_backfill(agent_name, instruction)
    except Exception as exc:
        logger.error(
            "agent profile backfill failed",
            extra={"agent_name": agent_name, "error": str(exc)},
        )
        return None


__all__ = [
    "AgentProfile",
    "AgentProfileStore",
    "get_agent_profile_store",
    "get_or_backfill_agent_profile",
]
