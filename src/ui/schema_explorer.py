"""Schema explorer sidebar panel for Streamlit."""

from __future__ import annotations


def render_schema_panel(connector=None) -> None:
    """Render an expandable sidebar panel showing tables, columns, and sample values.

    Parameters
    ----------
    connector:
        An active :class:`src.connectors.base.DatabaseConnector` instance.
        If *None*, falls back to reading ``data/schema.txt``.
    """
    import streamlit as st

    st.subheader("🗂️ Schema Explorer")

    if connector is None:
        # Fallback: read the static schema.txt
        try:
            with open("data/schema.txt", "r", encoding="utf-8") as f:
                schema_text = f.read()
            with st.expander("📄 Full Schema (text)", expanded=False):
                st.code(schema_text, language="sql")
        except FileNotFoundError:
            st.info("No database connected and no schema.txt found.")
        return

    # Build table list from the connector
    try:
        schema_text = connector.get_schema()
    except Exception as e:
        st.warning(f"Could not load schema: {e}")
        return

    # Parse table names from schema text
    import re
    table_names = re.findall(r"CREATE TABLE\s+(\w+)", schema_text, re.IGNORECASE)

    if not table_names:
        with st.expander("📄 Schema", expanded=False):
            st.code(schema_text, language="sql")
        return

    for table in table_names:
        with st.expander(f"📋 {table}", expanded=False):
            try:
                # Get column info
                cols, rows = connector.execute(f"SELECT * FROM {table} LIMIT 3")
                # Show column names
                st.markdown(f"**Columns:** {', '.join(cols)}")
                # Show sample rows
                if rows:
                    import pandas as pd
                    sample_df = pd.DataFrame(rows, columns=cols)
                    st.dataframe(sample_df, use_container_width=True)
                # Show row count
                count_cols, count_rows = connector.execute(
                    f"SELECT COUNT(*) AS row_count FROM {table}"
                )
                if count_rows:
                    st.caption(f"Total rows: {count_rows[0][0]:,}")
            except Exception as e:
                st.caption(f"Preview unavailable: {e}")
