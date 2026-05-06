"""Unit tests for src/ui/prompt_suggestions.py (Phase 7 — onboarding UX)."""

from unittest.mock import MagicMock


class TestGenerateSuggestions:
    SCHEMA_SUMMARY = (
        "Tables: customers (customer_id, city, state), "
        "orders (order_id, customer_id, status), "
        "products (product_id, category, price)"
    )

    def _make_llm(self, response_text: str):
        llm = MagicMock()
        response = MagicMock()
        response.content = response_text
        llm.invoke.return_value = response
        return llm

    def test_returns_at_least_five_suggestions(self):
        from src.ui.prompt_suggestions import generate_suggestions

        mock_response = (
            "1. How many customers are there?\n"
            "2. What are the top 5 cities by number of orders?\n"
            "3. Which product category has the highest revenue?\n"
            "4. How many orders were delivered last year?\n"
            "5. What is the average order value?\n"
            "6. Which customers placed more than 3 orders?\n"
        )
        llm = self._make_llm(mock_response)
        result = generate_suggestions(self.SCHEMA_SUMMARY, llm)

        assert len(result) >= 5

    def test_returns_strings(self):
        from src.ui.prompt_suggestions import generate_suggestions

        mock_response = "1. Count customers\n2. List orders\n3. Top products\n4. Revenue\n5. Avg price\n"
        llm = self._make_llm(mock_response)
        result = generate_suggestions(self.SCHEMA_SUMMARY, llm)

        assert all(isinstance(s, str) for s in result)

    def test_strips_numbering(self):
        from src.ui.prompt_suggestions import generate_suggestions

        mock_response = "1. First question?\n2. Second question?\n3. Third?\n4. Fourth?\n5. Fifth?\n"
        llm = self._make_llm(mock_response)
        result = generate_suggestions(self.SCHEMA_SUMMARY, llm)

        for item in result:
            assert not item[0].isdigit(), f"Numbering not stripped: {item!r}"

    def test_at_most_eight_suggestions(self):
        from src.ui.prompt_suggestions import generate_suggestions

        # Ten items in the response
        lines = "\n".join(f"{i}. Question {i}?" for i in range(1, 11))
        llm = self._make_llm(lines)
        result = generate_suggestions(self.SCHEMA_SUMMARY, llm)

        assert len(result) <= 8

    def test_pads_short_responses_to_five(self):
        from src.ui.prompt_suggestions import generate_suggestions

        # Only 2 lines returned
        mock_response = "1. Only question A?\n2. Only question B?\n"
        llm = self._make_llm(mock_response)
        result = generate_suggestions(self.SCHEMA_SUMMARY, llm)

        assert len(result) >= 5

    def test_llm_invoke_called_once(self):
        from src.ui.prompt_suggestions import generate_suggestions

        mock_response = "1. Q1\n2. Q2\n3. Q3\n4. Q4\n5. Q5\n"
        llm = self._make_llm(mock_response)
        generate_suggestions(self.SCHEMA_SUMMARY, llm)

        llm.invoke.assert_called_once()
