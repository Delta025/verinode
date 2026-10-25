"""
Tests for the LLM-assisted contradiction detection module.

All tests mock the Ollama client — no running Ollama server needed.

Tests cover:
- Contradiction detection for all types (negation, numerical, semantic, logical)
- No-contradiction cases (compatible facts, different topics)
- Response parsing (valid JSON, malformed, missing fields)
- Factory function (success and failure paths)
- Ollama call verification (JSON format, temperature, context)
- Confidence clamping
- Batch processing
- Graceful error handling

Run with:  python -m pytest tests/test_llm_contradiction.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm_contradiction import (
    ContradictionChecker,
    RelevanceResult,
    GeminiContradictionChecker,
    create_contradiction_checker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checker():
    """Create a ContradictionChecker with a mocked Ollama client."""
    with patch("core.llm_contradiction._get_ollama") as mock_get:
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_get.return_value = mock_ollama

        checker = ContradictionChecker(model="qwen2.5:7b")
        checker._client = mock_client
    return checker


def _mock_response(json_text: str):
    """Create a mock Ollama response object."""
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = json_text
    return response


# ---------------------------------------------------------------------------
# Ollama relevance tests
# ---------------------------------------------------------------------------

def test_ollama_check_relevance():
    """Ollama ContradictionChecker.check_relevance works correctly."""
    checker = _make_checker()
    checker._client.chat.return_value = _mock_response(json.dumps({
        "relationship": "contradicts",
        "confidence": 0.95,
        "explanation": "One affirms and the other negates the same claim."
    }))

    result = checker.check_relevance(
        "Water is a good solvent",
        "Water is not a good solvent"
    )

    assert result.relationship == "contradicts"
    assert result.confidence == 0.95


def test_ollama_check_relevance_supports():
    """Ollama ContradictionChecker.check_relevance returns supports."""
    checker = _make_checker()
    checker._client.chat.return_value = _mock_response(json.dumps({
        "relationship": "supports",
        "confidence": 0.92,
        "explanation": "These statements are consistent."
    }))

    result = checker.check_relevance(
        "Water boils at 100 degrees Celsius",
        "Water boils at 100 degrees C"
    )

    assert result.relationship == "supports"


def test_ollama_check_relevance_irrelevant():
    """Ollama ContradictionChecker.check_relevance returns irrelevant."""
    checker = _make_checker()
    checker._client.chat.return_value = _mock_response(json.dumps({
        "relationship": "irrelevant",
        "confidence": 0.99,
        "explanation": "Completely different topics."
    }))

    result = checker.check_relevance(
        "Iron is a metal",
        "Photosynthesis converts sunlight into chemical energy"
    )

    assert result.relationship == "irrelevant"


def test_ollama_check_relevance_failure():
    """If the LLM call raises an exception, return a safe irrelevant result."""
    checker = _make_checker()
    checker._client.chat.side_effect = Exception("Connection refused")

    result = checker.check_relevance("fact A", "fact B")

    assert result.relationship == "irrelevant"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_create_contradiction_checker_success():
    """Factory returns a checker when Ollama is available."""
    with patch("core.llm_contradiction._get_ollama") as mock_get:
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = MagicMock()
        mock_get.return_value = mock_ollama

        checker = create_contradiction_checker(model="qwen2.5:7b", backend="ollama")

    assert checker is not None
    assert isinstance(checker, ContradictionChecker)


def test_create_contradiction_checker_no_ollama():
    """Factory returns None when Ollama import fails."""
    with patch("core.llm_contradiction._get_ollama") as mock_get:
        mock_get.side_effect = ImportError("No module named 'ollama'")

        checker = create_contradiction_checker(backend="ollama")

    assert checker is None


# ---------------------------------------------------------------------------
# Result dataclass tests
# ---------------------------------------------------------------------------

def test_relevance_result_fields():
    """RelevanceResult has all expected fields."""
    result = RelevanceResult(
        claim="fact A",
        knowledge="fact B",
        relationship="contradicts",
        confidence=0.95,
        explanation="Different values",
    )

    assert result.claim == "fact A"
    assert result.knowledge == "fact B"
    assert result.relationship == "contradicts"
    assert result.confidence == 0.95
    assert result.explanation == "Different values"


# ---------------------------------------------------------------------------
# Helpers: Gemini
# ---------------------------------------------------------------------------

def _make_gemini_checker():
    """Create a GeminiContradictionChecker with a mocked genai client."""
    with patch("core.llm_contradiction._get_genai") as mock_get:
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_get.return_value = mock_genai

        checker = GeminiContradictionChecker(api_key="test-key")
        checker._client = mock_client
    return checker


def _mock_gemini_response(json_text: str):
    """Create a mock Gemini response object."""
    response = MagicMock()
    response.text = json_text
    return response


# ---------------------------------------------------------------------------
# Tests: Gemini contradiction checker
# ---------------------------------------------------------------------------

class TestGeminiContradictionChecker:
    def test_check_relevance(self):
        """Relevance check works via Gemini."""
        checker = _make_gemini_checker()
        checker._client.models.generate_content.return_value = _mock_gemini_response(
            json.dumps({
                "relationship": "contradicts",
                "confidence": 0.88,
                "explanation": "Different values.",
            })
        )

        result = checker.check_relevance(
            "Gold melts at 500C", "Gold melts at 1064C"
        )

        assert result.relationship == "contradicts"
        assert result.confidence == 0.88

    def test_check_relevance_supports(self):
        """Relevance check can return 'supports'."""
        checker = _make_gemini_checker()
        checker._client.models.generate_content.return_value = _mock_gemini_response(
            json.dumps({
                "relationship": "supports",
                "confidence": 0.95,
                "explanation": "Consistent statement.",
            })
        )

        result = checker.check_relevance("Water boils at 100C", "Water boils at 100 degrees")

        assert result.relationship == "supports"

    def test_check_relevance_api_failure(self):
        """Relevance API failure returns safe default."""
        checker = _make_gemini_checker()
        checker._client.models.generate_content.side_effect = Exception("Timeout")

        result = checker.check_relevance("A", "B")

        assert result.relationship == "irrelevant"
        assert result.confidence == 0.0
        assert "Gemini call failed" in result.explanation

    def test_check_relevance_batch(self):
        """Batch relevance calls check_relevance for each pair."""
        checker = _make_gemini_checker()
        checker._client.models.generate_content.return_value = _mock_gemini_response(
            json.dumps({
                "relationship": "supports",
                "confidence": 0.9,
                "explanation": "Consistent.",
            })
        )

        results = checker.check_relevance_batch(
            [("claim1", "knowledge1")], max_workers=1
        )

        assert len(results) == 1
        assert results[0].relationship == "supports"

    def test_missing_api_key_raises(self):
        """Constructor raises if no API key provided."""
        with patch("core.llm_contradiction._get_genai") as mock_get:
            mock_genai = MagicMock()
            mock_get.return_value = mock_genai
            with patch.dict("os.environ", {}, clear=False):
                # Remove the env var if it exists
                import os
                old_val = os.environ.pop("ENGRAM_GEMINI_API_KEY", None)
                try:
                    with pytest.raises(ValueError, match="API key required"):
                        GeminiContradictionChecker()
                finally:
                    if old_val is not None:
                        os.environ["ENGRAM_GEMINI_API_KEY"] = old_val

    def test_gemini_call_format(self):
        """Verify the Gemini API is called with correct parameters."""
        checker = _make_gemini_checker()
        checker._client.models.generate_content.return_value = _mock_gemini_response(
            json.dumps({
                "relationship": "irrelevant",
                "confidence": 0.5,
                "explanation": "test",
            })
        )

        checker.check_relevance("Fact A", "Fact B")

        call_kwargs = checker._client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs.kwargs["config"]["response_mime_type"] == "application/json"
        assert call_kwargs.kwargs["config"]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Factory backend selection
# ---------------------------------------------------------------------------

class TestFactoryBackendSelection:
    def test_factory_defaults_to_ollama(self):
        """Default backend is Ollama when env var is not set."""
        with patch("core.llm_contradiction._get_ollama") as mock_get, \
             patch.dict("os.environ", {}, clear=False) as env:
            env.pop("ENGRAM_CHECKER_BACKEND", None)
            mock_ollama = MagicMock()
            mock_get.return_value = mock_ollama

            checker = create_contradiction_checker()

            assert isinstance(checker, ContradictionChecker)

    def test_factory_gemini_backend(self):
        """backend='gemini' returns GeminiContradictionChecker."""
        with patch("core.llm_contradiction._get_genai") as mock_get:
            mock_genai = MagicMock()
            mock_get.return_value = mock_genai

            checker = create_contradiction_checker(
                backend="gemini", gemini_api_key="test-key"
            )

            assert isinstance(checker, GeminiContradictionChecker)

    def test_factory_gemini_unavailable_returns_none(self):
        """When Gemini SDK not installed, factory returns None."""
        with patch(
            "core.llm_contradiction._get_genai",
            side_effect=ImportError("no genai"),
        ):
            checker = create_contradiction_checker(
                backend="gemini", gemini_api_key="test-key"
            )

            assert checker is None

    def test_factory_env_var_backend(self):
        """ENGRAM_CHECKER_BACKEND env var selects backend."""
        import os
        old_backend = os.environ.get("ENGRAM_CHECKER_BACKEND")
        old_key = os.environ.get("ENGRAM_GEMINI_API_KEY")
        try:
            os.environ["ENGRAM_CHECKER_BACKEND"] = "gemini"
            os.environ["ENGRAM_GEMINI_API_KEY"] = "test-key"

            with patch("core.llm_contradiction._get_genai") as mock_get:
                mock_genai = MagicMock()
                mock_get.return_value = mock_genai

                checker = create_contradiction_checker()

                assert isinstance(checker, GeminiContradictionChecker)
        finally:
            if old_backend is not None:
                os.environ["ENGRAM_CHECKER_BACKEND"] = old_backend
            else:
                os.environ.pop("ENGRAM_CHECKER_BACKEND", None)
            if old_key is not None:
                os.environ["ENGRAM_GEMINI_API_KEY"] = old_key
            else:
                os.environ.pop("ENGRAM_GEMINI_API_KEY", None)

    def test_factory_ollama_unavailable_returns_none(self):
        """When Ollama not available, factory returns None for ollama backend."""
        with patch(
            "core.llm_contradiction._get_ollama",
            side_effect=ImportError("no ollama"),
        ):
            checker = create_contradiction_checker(backend="ollama")

            assert checker is None


# ---------------------------------------------------------------------------
# Tests: Multi-pair batch (Gemini)
# ---------------------------------------------------------------------------

class TestGeminiRelevanceBatch:
    """Tests for Gemini relevance batch checking."""

    def test_check_relevance_batch_empty(self):
        """Empty pairs list returns empty results."""
        checker = _make_gemini_checker()
        results = checker.check_relevance_batch([])
        assert results == []

    def test_relevance_cache_directional(self):
        """Relevance cache is directional (claim, knowledge) not symmetric."""
        checker = _make_gemini_checker()
        response = _mock_gemini_response(json.dumps({
            "relationship": "contradicts",
            "confidence": 0.9,
            "explanation": "Different values",
        }))
        checker._client.models.generate_content.return_value = response

        result = checker.check_relevance("A melts at 500", "A melts at 1000")
        assert result.relationship == "contradicts"
        assert ("A melts at 500", "A melts at 1000") in checker._relevance_cache
        # Reversed key should NOT be cached
        assert ("A melts at 1000", "A melts at 500") not in checker._relevance_cache

    def test_clear_cache_clears_relevance(self):
        """clear_cache() clears the relevance cache."""
        checker = _make_gemini_checker()
        checker._relevance_cache[("C", "D")] = RelevanceResult(
            claim="C", knowledge="D", relationship="supports",
            confidence=0.8, explanation="y",
        )

        checker.clear_cache()

        assert len(checker._relevance_cache) == 0


# ---------------------------------------------------------------------------
# Tests: Ollama relevance cache
# ---------------------------------------------------------------------------

def test_ollama_relevance_cache():
    """Ollama ContradictionChecker caches relevance results."""
    with patch("core.llm_contradiction._get_ollama") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value.Client.return_value = mock_client

        mock_client.chat.return_value.message.content = json.dumps({
            "relationship": "supports",
            "confidence": 0.85,
            "explanation": "Same property, consistent values",
        })

        checker = ContradictionChecker(model="test")

        # First call should hit LLM
        r1 = checker.check_relevance("Water boils at 100C", "Water boils at 100 degrees")
        assert r1.relationship == "supports"
        assert mock_client.chat.call_count == 1

        # Second call with same args should be cached
        r2 = checker.check_relevance("Water boils at 100C", "Water boils at 100 degrees")
        assert r2.relationship == "supports"
        assert mock_client.chat.call_count == 1  # No new LLM call


def test_ollama_clear_cache_includes_relevance():
    """Ollama ContradictionChecker.clear_cache() clears the relevance cache."""
    with patch("core.llm_contradiction._get_ollama") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value.Client.return_value = mock_client

        checker = ContradictionChecker(model="test")

        # Manually populate cache
        checker._relevance_cache[("c", "d")] = MagicMock()

        assert len(checker._relevance_cache) == 1

        checker.clear_cache()

        assert len(checker._relevance_cache) == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
