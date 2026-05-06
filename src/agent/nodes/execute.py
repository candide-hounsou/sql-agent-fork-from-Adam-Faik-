import sqlite3
import threading

import sqlparse
from langchain_core.runnables.config import RunnableConfig

from src.agent.state import AgentState
from src.connectors.base import DatabaseConnector

TIMEOUT_SECONDS = 5.0


def _execute_with_timeout(
    connector: DatabaseConnector, sql: str, timeout: float
) -> tuple:
    """Execute *sql* via *connector* with a hard *timeout* in seconds.

    **SQLite connectors** (``SQLiteConnector`` / ``CSVConnector``): a daemon
    timer thread fires after *timeout* seconds and calls
    ``sqlite3.Connection.interrupt()``, which causes the running C-level
    SQLite operation to raise ``sqlite3.OperationalError('interrupted')``.

    **PostgreSQL / MySQL connectors**: ``query_timeout_ms`` is set on the
    connector before the call so that their ``execute()`` implementations
    can issue a ``SET statement_timeout`` / ``SET SESSION max_execution_time``
    command before the query (see each connector's ``execute()`` docstring).

    A daemon worker thread and a ``worker.join(timeout + 1)`` universal
    wall-clock fallback ensure the caller is never blocked longer than
    ``timeout + 1`` seconds even for connectors that do not support any
    interrupt mechanism.
    """
    result_holder: list = [None]
    exc_holder: list = [None]

    def _target() -> None:
        try:
            result_holder[0] = connector.execute(sql)
        except Exception as exc:  # noqa: BLE001
            exc_holder[0] = exc

    # ------------------------------------------------------------------
    # SQLite interrupt setup
    # ------------------------------------------------------------------
    # Obtain the raw sqlite3.Connection if this is a SQLite-backed connector.
    # We need it open *before* starting the timer so that interrupt() is
    # callable the moment the deadline is reached.
    sqlite_conn: sqlite3.Connection | None = None
    raw = getattr(connector, "_conn", None)
    if isinstance(raw, sqlite3.Connection):
        sqlite_conn = raw
    elif raw is None and hasattr(connector, "connect"):
        try:
            connector.connect()
            raw = getattr(connector, "_conn", None)
            if isinstance(raw, sqlite3.Connection):
                sqlite_conn = raw
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # PostgreSQL / MySQL statement-level timeout
    # ------------------------------------------------------------------
    # Connectors that support server-side timeouts expose a query_timeout_ms
    # attribute.  Setting it here propagates the deadline into their execute().
    if hasattr(connector, "query_timeout_ms"):
        connector.query_timeout_ms = int(timeout * 1000)

    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        if sqlite_conn is not None:
            try:
                sqlite_conn.interrupt()
            except Exception:  # noqa: BLE001
                pass

    timer = threading.Timer(timeout, _on_timeout)
    timer.daemon = True
    worker = threading.Thread(target=_target, daemon=True)

    timer.start()
    worker.start()
    worker.join(timeout + 1.0)  # extra second for the interrupt to propagate
    timer.cancel()

    if worker.is_alive():
        raise TimeoutError(f"Query timed out after {timeout:.0f} seconds.")

    if exc_holder[0] is not None:
        if timed_out.is_set() and "interrupted" in str(exc_holder[0]).lower():
            raise TimeoutError(f"Query timed out after {timeout:.0f} seconds.")
        raise exc_holder[0]

    return result_holder[0]  # type: ignore[return-value]


def execute_sql(state: AgentState, config: RunnableConfig) -> dict:
    print("--- NODE: EXECUTING SQL ---")
    sql_query = state.get("sql_query", "")

    # --- Security guardrail: only allow SELECT statements ---
    parsed_statements = sqlparse.parse(sql_query)
    for statement in parsed_statements:
        if not statement.is_whitespace:
            raw_type = statement.get_type()
            if raw_type:
                stmt_type = raw_type.upper()
                if stmt_type != "SELECT":
                    security_error = (
                        f"Security Violation: '{stmt_type}' operation detected. "
                        "Only SELECT queries are strictly allowed."
                    )
                    print(f"🚨 ALERT: {security_error}")
                    current_retries = state.get("retry_count", 0)
                    return {"error": security_error, "retry_count": current_retries + 1}

    # --- Resolve connector from config or fall back to default SQLite ---
    configurable = config.get("configurable", {})
    connector: DatabaseConnector | None = configurable.get("connector")
    if connector is None:
        from src.connectors.sqlite import SQLiteConnector

        connector = SQLiteConnector("data/olist.db")

    try:
        cols, rows = _execute_with_timeout(connector, sql_query, TIMEOUT_SECONDS)

        if not rows:
            formatted_results = "No results found."
            raw_data_list = []
        else:
            raw_data_list = [dict(zip(cols, row)) for row in rows]
            formatted_results = f"Columns: {', '.join(cols)}\nData:\n"
            llm_row_limit = 10
            for row in rows[:llm_row_limit]:
                formatted_results += str(dict(zip(cols, row))) + "\n"
            if len(rows) > llm_row_limit:
                hidden_rows = len(rows) - llm_row_limit
                formatted_results += (
                    f"\n... (WARNING: {hidden_rows} additional rows were retrieved but "
                    "hidden from you to save tokens. You MUST mention to the user that "
                    "this is a partial view and they should look at the table/CSV for "
                    "the full data.)\n"
                )
            max_chars = 3000
            if len(formatted_results) > max_chars:
                formatted_results = (
                    formatted_results[:max_chars]
                    + "\n... [FATAL: DATA TRUNCATED DUE TO EXTREME LENGTH]"
                )

        print("✓ Execution successful. Data fetched.\n")
        current_retries = state.get("retry_count", 0)
        return {
            "db_results": formatted_results,
            "raw_data": raw_data_list,
            "error": "",
            "retry_count": current_retries,
        }

    except TimeoutError:
        error_msg = (
            f"Timeout Error: Execution timed out after {TIMEOUT_SECONDS:.0f} seconds. "
            "Your query is too slow. Check for missing ON clauses in your JOINs "
            "(Cartesian products)."
        )
        print(f"✗ Execution failed! {error_msg}\n")
        current_retries = state.get("retry_count", 0)
        return {"error": error_msg, "retry_count": current_retries + 1}

    except sqlite3.OperationalError as e:
        error_msg = f"SQLite Error: {str(e)}"
        print(f"✗ Execution failed! {error_msg}\n")
        current_retries = state.get("retry_count", 0)
        return {"error": error_msg, "retry_count": current_retries + 1}

    except Exception as e:
        error_msg = f"System Error: {str(e)}"
        print(f"✗ Execution failed! {error_msg}\n")
        current_retries = state.get("retry_count", 0)
        return {"error": error_msg, "retry_count": current_retries + 1}
