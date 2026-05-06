"""Token usage tracking and cost estimation for LLM calls."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# ---------------------------------------------------------------------------
# Per-model pricing table (USD per 1 000 tokens)
# Update these values as provider pricing changes.
# ---------------------------------------------------------------------------
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    # Anthropic
    "claude-3-5-haiku-20241022": {"prompt": 0.001, "completion": 0.005},
    "claude-sonnet-4-5": {"prompt": 0.003, "completion": 0.015},
    # Google
    "gemini-2.0-flash": {"prompt": 0.000075, "completion": 0.0003},
    "gemini-2.5-pro": {"prompt": 0.00125, "completion": 0.01},
}

# Fallback pricing for unknown models (conservative estimate)
_DEFAULT_PRICING: Dict[str, float] = {"prompt": 0.001, "completion": 0.002}


def get_session_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    """Estimate the USD cost for *prompt_tokens* + *completion_tokens*.

    Parameters
    ----------
    prompt_tokens:
        Number of tokens in the LLM input.
    completion_tokens:
        Number of tokens in the LLM output.
    model:
        Model name string (e.g. ``"gpt-4o-mini"``).

    Returns
    -------
    float
        Estimated cost in USD.
    """
    pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    cost = (prompt_tokens / 1_000) * pricing["prompt"] + (
        completion_tokens / 1_000
    ) * pricing["completion"]
    return round(cost, 6)


class TokenTracker(BaseCallbackHandler):
    """LangChain callback handler that counts tokens and accumulates cost.

    Usage::

        tracker = TokenTracker(model="gpt-4o-mini")
        llm.invoke(messages, config={"callbacks": [tracker]})
        print(tracker.total_tokens, tracker.total_cost_usd)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__()
        self.model = model
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called after every LLM invocation."""
        usage = (
            getattr(response, "llm_output", None) or {}
        ).get("token_usage", {})

        prompt_toks: int = usage.get("prompt_tokens", 0)
        completion_toks: int = usage.get("completion_tokens", 0)

        self._accumulate(prompt_toks, completion_toks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accumulate(self, prompt_toks: int, completion_toks: int) -> None:
        self.prompt_tokens += prompt_toks
        self.completion_tokens += completion_toks
        self.total_tokens += prompt_toks + completion_toks
        self.total_cost_usd += get_session_cost(prompt_toks, completion_toks, self.model)
        self.total_cost_usd = round(self.total_cost_usd, 6)

    def reset(self) -> None:
        """Reset all counters (call between sessions)."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
