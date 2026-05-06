"""Unit tests for the execute_sql security guardrail and connector integration."""
from unittest.mock import MagicMock

from src.agent.nodes.execute import execute_sql
from src.connectors.base import DatabaseConnector


def _config(connector=None):
    return {"configurable": {"connector": connector} if connector else {}}


def _mock_connector(cols=None, rows=None):
    """Return a mock DatabaseConnector with preset execute() return values."""
    connector = MagicMock(spec=DatabaseConnector)
    connector.execute.return_value = (
        cols or ["customer_id", "customer_city", "customer_state"],
        rows or [("c1", "sao paulo", "SP")],
    )
    return connector


class TestSecurityGuardrail:
    """Tests that only SELECT statements pass through; DDL/DML are blocked."""

    def test_select_passes_through(self):
        """A valid SELECT query should reach the DB layer (mocked) without security error."""
        state = {"sql_query": "SELECT * FROM customers", "retry_count": 0}
        connector = _mock_connector()
        result = execute_sql(state, _config(connector))
        assert result.get("error") == ""
        assert result.get("raw_data") is not None

    def test_drop_table_is_blocked(self):
        state = {"sql_query": "DROP TABLE customers", "retry_count": 0}
        result = execute_sql(state, _config())
        assert "Security Violation" in result["error"]
        assert "DROP" in result["error"]
        assert result["retry_count"] == 1

    def test_insert_is_blocked(self):
        state = {"sql_query": "INSERT INTO customers VALUES ('x', 'city', 'ST')", "retry_count": 0}
        result = execute_sql(state, _config())
        assert "Security Violation" in result["error"]
        assert result["retry_count"] == 1

    def test_update_is_blocked(self):
        state = {"sql_query": "UPDATE customers SET customer_city = 'x'", "retry_count": 0}
        result = execute_sql(state, _config())
        assert "Security Violation" in result["error"]
        assert result["retry_count"] == 1

    def test_delete_is_blocked(self):
        state = {"sql_query": "DELETE FROM customers", "retry_count": 1}
        result = execute_sql(state, _config())
        assert "Security Violation" in result["error"]
        assert result["retry_count"] == 2

    def test_multiple_statements_first_select_second_drop_blocked(self):
        """When a second statement is non-SELECT, it should be blocked."""
        sql = "SELECT 1; DROP TABLE customers;"
        state = {"sql_query": sql, "retry_count": 0}
        result = execute_sql(state, _config())
        assert "Security Violation" in result["error"]

    def test_empty_query_attempts_db(self):
        """An empty query has no statements, so it skips the guardrail and hits the DB."""
        state = {"sql_query": "", "retry_count": 0}
        connector = _mock_connector(cols=[], rows=[])
        connector.execute.return_value = ([], [])
        result = execute_sql(state, _config(connector))
        assert "Security Violation" not in result.get("error", "")

    def test_connector_error_is_captured(self):
        """Errors raised by the connector are caught and returned as error state."""
        state = {"sql_query": "SELECT * FROM missing", "retry_count": 0}
        connector = MagicMock(spec=DatabaseConnector)
        connector.execute.side_effect = Exception("no such table: missing")
        result = execute_sql(state, _config(connector))
        assert "no such table: missing" in result["error"]
        assert result["retry_count"] == 1
