"""PostgreSQL database connector."""

from typing import Any, List, Optional, Tuple

from src.connectors.base import DatabaseConnector


class PostgreSQLConnector(DatabaseConnector):
    """Connector for PostgreSQL databases via psycopg2."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "",
        user: str = "",
        password: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self._conn: Optional[Any] = None

    def connect(self) -> Any:
        try:
            import psycopg2
        except ImportError as exc:
            raise RuntimeError(
                "psycopg2 is required for PostgreSQL connectivity. "
                "Install it with: pip install psycopg2-binary"
            ) from exc
        self._conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
        return self._conn

    def execute(self, query: str) -> Tuple[List[str], List[Tuple]]:
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        cursor.execute(query)
        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(100)
        return column_names, rows

    def get_schema(self) -> str:
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
        )
        tables = [row[0] for row in cursor.fetchall()]
        lines = []
        for table in tables:
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table,),
            )
            col_defs = ", ".join(f"{col} {dtype}" for col, dtype in cursor.fetchall())
            lines.append(f"CREATE TABLE {table} ({col_defs});")
        return "\n".join(lines)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
