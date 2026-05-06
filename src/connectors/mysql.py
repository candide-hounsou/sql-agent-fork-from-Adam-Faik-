"""MySQL database connector."""

from typing import Any, List, Tuple

from src.connectors.base import DatabaseConnector


class MySQLConnector(DatabaseConnector):
    """Connector for MySQL / MariaDB databases via mysql-connector-python."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "",
        user: str = "",
        password: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None
        # When set, execute() issues SET SESSION max_execution_time before each query.
        self.query_timeout_ms: int | None = None

    def connect(self) -> None:
        try:
            import mysql.connector
        except ImportError as exc:
            raise RuntimeError(
                "mysql-connector-python is required for MySQL connectivity. "
                "Install it with: pip install mysql-connector-python"
            ) from exc
        self._conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )

    def execute(self, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        """Execute *sql* and return (column_names, rows).

        If ``query_timeout_ms`` is set, a ``SET SESSION max_execution_time``
        command is issued first so MySQL aborts SELECT queries server-side if
        they exceed the deadline (milliseconds, MySQL 5.7.8+).
        """
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        if self.query_timeout_ms is not None:
            cursor.execute("SET SESSION max_execution_time = %s", (int(self.query_timeout_ms),))
        cursor.execute(sql)
        rows = cursor.fetchmany(100)
        cols = [desc[0] for desc in cursor.description]
        return cols, rows

    def get_schema(self) -> str:
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]
        lines = []
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`;")
            col_defs = ", ".join(f"{row[0]} {row[1]}" for row in cursor.fetchall())
            lines.append(f"CREATE TABLE {table} ({col_defs});")
        return "\n".join(lines)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
