"""Unit tests for database connectors (Phase 5)."""

import io

import pytest


class TestSQLiteConnector:
    def test_execute_returns_cols_and_rows(self, sample_db_path):
        from src.connectors.sqlite import SQLiteConnector

        connector = SQLiteConnector(db_path=sample_db_path)
        cols, rows = connector.execute("SELECT * FROM customers")
        connector.close()

        assert "customer_id" in cols
        assert len(rows) > 0

    def test_get_schema_contains_table_names(self, sample_db_path):
        from src.connectors.sqlite import SQLiteConnector

        connector = SQLiteConnector(db_path=sample_db_path)
        schema = connector.get_schema()
        connector.close()

        assert "customers" in schema
        assert "orders" in schema

    def test_close_is_idempotent(self, sample_db_path):
        from src.connectors.sqlite import SQLiteConnector

        connector = SQLiteConnector(db_path=sample_db_path)
        connector.connect()
        connector.close()
        connector.close()  # should not raise


class TestCSVConnector:
    CSV_DATA = b"name,age,city\nAlice,30,Paris\nBob,25,Berlin\n"

    def test_execute_returns_correct_columns(self):
        from src.connectors.csv_connector import CSVConnector

        connector = CSVConnector(self.CSV_DATA, table_name="people")
        cols, rows = connector.execute("SELECT * FROM people")
        connector.close()

        assert cols == ["name", "age", "city"]
        assert len(rows) == 2

    def test_get_schema_includes_table(self):
        from src.connectors.csv_connector import CSVConnector

        connector = CSVConnector(self.CSV_DATA, table_name="people")
        schema = connector.get_schema()
        connector.close()

        assert "people" in schema

    def test_accepts_file_like_object(self):
        from src.connectors.csv_connector import CSVConnector

        buf = io.BytesIO(self.CSV_DATA)
        connector = CSVConnector(buf, table_name="t")
        cols, rows = connector.execute("SELECT name FROM t")
        connector.close()

        assert "name" in cols

    def test_aggregation_query(self):
        from src.connectors.csv_connector import CSVConnector

        connector = CSVConnector(self.CSV_DATA, table_name="people")
        cols, rows = connector.execute("SELECT COUNT(*) AS cnt FROM people")
        connector.close()

        assert rows[0][0] == 2


class TestConnectorFactory:
    def test_sqlite_returns_sqlite_connector(self, sample_db_path):
        from src.connectors.factory import get_connector
        from src.connectors.sqlite import SQLiteConnector

        connector = get_connector("sqlite", db_path=sample_db_path)
        assert isinstance(connector, SQLiteConnector)

    def test_csv_returns_csv_connector(self):
        from src.connectors.csv_connector import CSVConnector
        from src.connectors.factory import get_connector

        csv_bytes = b"col1,col2\n1,2\n"
        connector = get_connector("csv", csv_source=csv_bytes)
        assert isinstance(connector, CSVConnector)

    def test_postgresql_returns_postgresql_connector(self):
        from src.connectors.factory import get_connector
        from src.connectors.postgresql import PostgreSQLConnector

        connector = get_connector("postgresql", host="localhost", dbname="test", user="u", password="p")
        assert isinstance(connector, PostgreSQLConnector)

    def test_mysql_returns_mysql_connector(self):
        from src.connectors.factory import get_connector
        from src.connectors.mysql import MySQLConnector

        connector = get_connector("mysql", host="localhost", database="test", user="u", password="p")
        assert isinstance(connector, MySQLConnector)

    def test_unknown_type_raises_value_error(self):
        from src.connectors.factory import get_connector

        with pytest.raises(ValueError, match="Unsupported db_type"):
            get_connector("oracle")
