import sqlite3
from typing import Any, List, Optional, Tuple

from .base import DatabaseConnector


class SQLiteConnector(DatabaseConnector):
    """SQLite implementation of DatabaseConnector."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def execute(self, query: str) -> Tuple[List[str], List[Tuple]]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query)
        column_names = [d[0] for d in cursor.description]
        rows = cursor.fetchmany(100)
        return column_names, rows

    def get_schema(self) -> str:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        schema_parts = []
        for name, ddl in tables:
            if ddl:
                schema_parts.append(ddl)
        return "\n\n".join(schema_parts)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
