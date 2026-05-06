"""Generate context-aware question suggestions for onboarding new users."""

from __future__ import annotations

from typing import List


def generate_suggestions(schema_summary: str, llm) -> List[str]:
    """Generate 5–8 example questions tailored to the connected schema.

    Results are meant to be cached in ``st.session_state`` so this function
    is only called once per database connection.

    Parameters
    ----------
    schema_summary:
        A short text describing the database schema (table names + key columns).
    llm:
        Any LangChain chat model with an ``invoke`` method.

    Returns
    -------
    List[str]
        A list of 5–8 natural-language questions the user could ask.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You are a helpful data analyst onboarding a new user to a SQL database. "
        "Based on the schema summary provided, generate exactly 6 interesting and "
        "diverse example questions that showcase the database's analytical potential. "
        "Return ONLY the questions as a numbered list (1. ... 2. ... etc.), "
        "with no additional commentary."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Schema summary:\n{schema_summary}"),
    ])

    raw = response.content.strip()
    suggestions: List[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. " or "- "
        import re
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
        if cleaned:
            suggestions.append(cleaned)

    # Guarantee at least 5 suggestions (repeat last if needed for edge-cases)
    while len(suggestions) < 5 and suggestions:
        suggestions.append(suggestions[-1])

    return suggestions[:8]
