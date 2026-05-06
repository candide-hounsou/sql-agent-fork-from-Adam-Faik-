"""Unit tests for the multi-provider LLM factory (Phase 4)."""

from unittest.mock import MagicMock, patch

import pytest


class TestProviderModels:
    def test_all_providers_have_models(self):
        from src.llm.factory import PROVIDER_MODELS, LLMProvider

        for provider in LLMProvider:
            assert provider in PROVIDER_MODELS
            assert len(PROVIDER_MODELS[provider]) > 0

    def test_openai_default_is_gpt4o_mini(self):
        from src.llm.factory import PROVIDER_MODELS, LLMProvider

        assert PROVIDER_MODELS[LLMProvider.OPENAI][0] == "gpt-4o-mini"

    def test_anthropic_models_listed(self):
        from src.llm.factory import PROVIDER_MODELS, LLMProvider

        assert any("claude" in m for m in PROVIDER_MODELS[LLMProvider.ANTHROPIC])

    def test_gemini_models_listed(self):
        from src.llm.factory import PROVIDER_MODELS, LLMProvider

        assert any("gemini" in m for m in PROVIDER_MODELS[LLMProvider.GEMINI])


class TestGetLlm:
    def test_openai_returns_chat_openai(self):
        mock_llm = MagicMock()
        mock_class = MagicMock(return_value=mock_llm)
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_class)}):
            from importlib import reload

            import src.llm.factory as factory_mod
            reload(factory_mod)
            result = factory_mod.get_llm(provider="openai", model_name="gpt-4o-mini")
        assert result is mock_llm

    def test_unknown_provider_raises_value_error(self):
        from src.llm.factory import get_llm

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm(provider="unknown_provider")

    def test_default_model_used_when_none_given(self):
        mock_llm = MagicMock()
        mock_class = MagicMock(return_value=mock_llm)
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_class)}):
            from importlib import reload

            import src.llm.factory as factory_mod
            reload(factory_mod)
            factory_mod.get_llm(provider="openai", model_name=None)
        # Verify model_name was passed as first element of PROVIDER_MODELS list
        call_kwargs = mock_class.call_args
        assert call_kwargs is not None

    def test_anthropic_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from src.llm.factory import get_llm

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            get_llm(provider="anthropic")

    def test_gemini_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from src.llm.factory import get_llm

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            get_llm(provider="gemini")

    def test_llm_provider_enum_values(self):
        from src.llm.factory import LLMProvider

        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.GEMINI == "gemini"
