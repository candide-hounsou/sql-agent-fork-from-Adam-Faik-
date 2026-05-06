"""Persistent session history store for the SQL Agent chat app."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class SessionHistory:
    """Represents one named chat session."""

    session_id: str
    name: str
    created_at: str  # ISO 8601 string
    messages: List[dict] = field(default_factory=list)
    connector_info: Optional[str] = None  # Human-readable description of the DB

    @classmethod
    def new(cls, name: str = "New Session", connector_info: Optional[str] = None) -> "SessionHistory":
        return cls(
            session_id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            messages=[],
            connector_info=connector_info,
        )


class HistoryStore:
    """Reads and writes :class:`SessionHistory` objects to a JSON file.

    A file-level exclusive lock (``fcntl.LOCK_EX``) is acquired on every
    write to prevent corruption from concurrent Streamlit browser tabs.

    Parameters
    ----------
    store_path:
        Path to the JSON file.  The parent directory is created automatically.
    """

    def __init__(self, store_path: str = "data/history.json") -> None:
        self.store_path = store_path
        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> Dict[str, dict]:
        """Return the raw dict from disk (empty dict if file missing or corrupt)."""
        if not os.path.exists(self.store_path):
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_raw(self, data: Dict[str, dict]) -> None:
        """Atomically write *data* to disk with an exclusive file lock."""
        # Write to a temp file then replace to avoid partial writes
        tmp_path = self.store_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2, ensure_ascii=False)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp_path, self.store_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_session(self, session: SessionHistory) -> None:
        """Persist *session* (insert or update)."""
        data = self._load_raw()
        data[session.session_id] = asdict(session)
        self._write_raw(data)

    def load_session(self, session_id: str) -> Optional[SessionHistory]:
        """Return the :class:`SessionHistory` for *session_id*, or *None*."""
        data = self._load_raw()
        raw = data.get(session_id)
        if raw is None:
            return None
        return SessionHistory(**raw)

    def list_sessions(self) -> List[SessionHistory]:
        """Return all sessions sorted by creation time (newest first)."""
        data = self._load_raw()
        sessions = [SessionHistory(**v) for v in data.values()]
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete the session; return *True* if it existed."""
        data = self._load_raw()
        if session_id not in data:
            return False
        del data[session_id]
        self._write_raw(data)
        return True

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session; return *True* if it existed."""
        session = self.load_session(session_id)
        if session is None:
            return False
        session.name = new_name
        self.save_session(session)
        return True
