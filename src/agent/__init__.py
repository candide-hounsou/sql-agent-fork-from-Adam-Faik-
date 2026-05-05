import os

from src.agent.graph import create_graph as _create_graph  # noqa: F401


def _require_openai_credentials() -> None:
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY"):
        return
    raise RuntimeError(
        "Missing OpenAI credentials. Set OPENAI_API_KEY in the project .env file "
        "or export OPENAI_API_KEY before starting the app."
    )


def create_graph():
    """Create the compiled LangGraph agent. Validates credentials at call time."""
    _require_openai_credentials()
    return _create_graph()


__all__ = ["create_graph"]
