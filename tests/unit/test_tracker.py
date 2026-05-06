"""Unit tests for src/llm/tracker.py — cost calculation and token tracking."""

import pytest
from unittest.mock import MagicMock
from langchain_core.outputs import LLMResult


class TestGetSessionCost:
    def test_known_model_cost(self):
        from src.llm.tracker import get_session_cost

        cost = get_session_cost(1000, 500, "gpt-4o-mini")
        # 1000 prompt * 0.00015/1000 + 500 completion * 0.0006/1000
        expected = 1.0 * 0.00015 + 0.5 * 0.0006
        assert abs(cost - expected) < 1e-9

    def test_unknown_model_uses_default_pricing(self):
        from src.llm.tracker import get_session_cost

        cost = get_session_cost(1000, 1000, "some-unknown-model-xyz")
        assert cost > 0

    def test_zero_tokens_returns_zero(self):
        from src.llm.tracker import get_session_cost

        assert get_session_cost(0, 0, "gpt-4o") == 0.0

    def test_expensive_model_costs_more(self):
        from src.llm.tracker import get_session_cost

        cheap = get_session_cost(1000, 1000, "gpt-4o-mini")
        expensive = get_session_cost(1000, 1000, "gpt-4o")
        assert expensive > cheap

    def test_all_listed_models_have_positive_cost(self):
        from src.llm.tracker import get_session_cost, MODEL_PRICING

        for model in MODEL_PRICING:
            cost = get_session_cost(100, 100, model)
            assert cost > 0, f"Expected positive cost for {model}"


class TestTokenTracker:
    def _make_llm_result(self, prompt_tokens: int, completion_tokens: int) -> LLMResult:
        result = MagicMock(spec=LLMResult)
        result.llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }
        return result

    def test_initial_state_is_zero(self):
        from src.llm.tracker import TokenTracker

        tracker = TokenTracker(model="gpt-4o-mini")
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_on_llm_end_accumulates_tokens(self):
        from src.llm.tracker import TokenTracker

        tracker = TokenTracker(model="gpt-4o-mini")
        result = self._make_llm_result(200, 100)
        tracker.on_llm_end(result)

        assert tracker.prompt_tokens == 200
        assert tracker.completion_tokens == 100
        assert tracker.total_tokens == 300

    def test_on_llm_end_accumulates_across_calls(self):
        from src.llm.tracker import TokenTracker

        tracker = TokenTracker(model="gpt-4o-mini")
        tracker.on_llm_end(self._make_llm_result(100, 50))
        tracker.on_llm_end(self._make_llm_result(100, 50))

        assert tracker.total_tokens == 300

    def test_cost_is_positive_after_invocation(self):
        from src.llm.tracker import TokenTracker

        tracker = TokenTracker(model="gpt-4o-mini")
        tracker.on_llm_end(self._make_llm_result(1000, 500))
        assert tracker.total_cost_usd > 0

    def test_reset_clears_all_counters(self):
        from src.llm.tracker import TokenTracker

        tracker = TokenTracker(model="gpt-4o-mini")
        tracker.on_llm_end(self._make_llm_result(500, 200))
        tracker.reset()

        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_missing_token_usage_does_not_crash(self):
        from src.llm.tracker import TokenTracker

        tracker = TokenTracker(model="gpt-4o-mini")
        result = MagicMock(spec=LLMResult)
        result.llm_output = {}
        tracker.on_llm_end(result)  # should not raise
        assert tracker.total_tokens == 0
