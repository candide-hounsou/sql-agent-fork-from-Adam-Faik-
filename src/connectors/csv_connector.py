"""CSV connector — loads CSV data into an in-memory SQLite database."""

import io
import sqlite3
from typing import IO, Any, List, Tuple, Union

from src.connectors.base import DatabaseConnector


class CSVConnector(DatabaseConnector):
    """Build an in-memory SQLite database from one or more CSV files.

    Parameters
    ----------
    csv_source:
        A file path (str), raw bytes, or a file-like object accepted by
        ``pandas.read_csv``.
    table_name:
        Name of the table created in the in-memory database.
    """

    def __init__(
        self,
        csv_source: Union[str, bytes, IO],
        table_name: str = "data",
    ) -> None:
        self._csv_source = csv_source
        self._table_name = table_name
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        import pandas as pd

        source = self._csv_source
        if isinstance(source, bytes):
            source = io.BytesIO(source)

        df = pd.read_csv(source)
        self._conn = sqlite3.connect(":memory:")
        df.to_sql(self._table_name, self._conn, index=False, if_exists="replace")

    def execute(self, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(100)
        cols = [desc[0] for desc in cursor.description]
        return cols, rows

    def get_schema(self) -> str:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(f"PRAGMA table_info({self._table_name});")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        return f"CREATE TABLE {self._table_name} ({col_defs});"

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
