"""LLM response caching via LangChain's SQLiteCache."""

import os


def setup_cache(cache_path: str = "data/llm_cache.db") -> None:
    """Enable LangChain's global SQLite LLM cache.

    Call once at app start-up (before any LLM invocation).  Subsequent calls
    with the same inputs will be served from the local SQLite file instead of
    hitting the remote API.

    Parameters
    ----------
    cache_path:
        Path to the SQLite file used as the cache store.
    """
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache
    except ImportError as exc:
        raise RuntimeError(
            "langchain-community is required for LLM caching. "
            "Install it with: pip install langchain-community"
        ) from exc

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=cache_path))
    print(f"✅ LLM cache enabled → {cache_path}")


def disable_cache() -> None:
    """Disable the global LLM cache (useful for testing fresh responses)."""
    try:
        from langchain_core.globals import set_llm_cache
        set_llm_cache(None)
        print("ℹ️  LLM cache disabled.")
    except ImportError:
        pass
