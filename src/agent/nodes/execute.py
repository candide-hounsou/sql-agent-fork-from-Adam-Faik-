import sqlite3

import sqlparse
from langchain_core.runnables.config import RunnableConfig

from src.agent.state import AgentState
from src.connectors.base import DatabaseConnector


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

    TIMEOUT_SECONDS = 5.0

    try:
        cols, rows = connector.execute(sql_query)

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

    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            error_msg = (
                f"SQLite Error: Execution timed out after {TIMEOUT_SECONDS} seconds. "
                "Your query is too slow. Check for missing ON clauses in your JOINs "
                "(Cartesian products)."
            )
        else:
            error_msg = f"SQLite Error: {str(e)}"
        print(f"✗ Execution failed! {error_msg}\n")
        current_retries = state.get("retry_count", 0)
        return {"error": error_msg, "retry_count": current_retries + 1}

    except Exception as e:
        error_msg = f"System Error: {str(e)}"
        print(f"✗ Execution failed! {error_msg}\n")
        current_retries = state.get("retry_count", 0)
        return {"error": error_msg, "retry_count": current_retries + 1}
