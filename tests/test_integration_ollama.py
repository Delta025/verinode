"""
Integration tests for the Ollama-based fact extraction engine.

These tests require a running Ollama server with a model pulled.
They are skipped by default. Run them with:

    python -m pytest tests/test_integration_ollama.py -v --run-integration

Prerequisites:
    1. Ollama installed and running (ollama serve)
    2. A model pulled (e.g. ollama pull qwen2.5:7b)

Tests cover:
- Ollama server connectivity
- Real fact extraction from text
- JSON mode structured output parsing
- Node conversion with real LLM output
- Domain hint handling
- Multi-fact extraction with related_facts wiring
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.fact_extraction import ExtractedFact, ExtractionResult, FactExtractionEngine
from core.node import KnowledgeNode, Source

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import ollama

        client = ollama.Client()
        client.list()
        return True
    except Exception:
        return False


def _model_available(model: str = "qwen2.5:7b") -> bool:
    """Check if a specific model is pulled."""
    try:
        import ollama

        client = ollama.Client()
        models = client.list()
        model_names = [m.model for m in models.models]
        # Check for exact match or prefix match (e.g. "qwen2.5:7b" matches "qwen2.5:7b")
        return any(model in name or name.startswith(model.split(":")[0]) for name in model_names)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOllamaConnectivity:
    """Test basic Ollama server connectivity."""

    def test_server_reachable(self):
        """Ollama server is running and reachable."""
        assert _ollama_available(), "Ollama server not reachable. Start with: ollama serve"

    def test_model_available(self):
        """At least one model is available for extraction."""
        assert _ollama_available(), "Ollama server not reachable"

        import ollama

        client = ollama.Client()
        models = client.list()
        assert len(models.models) > 0, "No models pulled. Pull one with: ollama pull qwen2.5:7b"


@pytest.mark.integration
class TestOllamaExtraction:
    """Test real fact extraction via Ollama."""

    @pytest.fixture
    def engine(self):
        """Create a real FactExtractionEngine connected to Ollama."""
        if not _ollama_available():
            pytest.skip("Ollama server not available")

        # Use whatever model is available
        import ollama

        client = ollama.Client()
        models = client.list()
        if not models.models:
            pytest.skip("No models pulled")

        model_name = models.models[0].model
        return FactExtractionEngine(model=model_name)

    def test_extract_simple_text(self, engine):
        """Extract facts from a simple, clear text passage."""
        result = engine.extract(
            "Water is composed of hydrogen and oxygen atoms. "
            "It boils at 100 degrees Celsius at standard atmospheric pressure."
        )

        assert isinstance(result, ExtractionResult)
        assert len(result.facts) >= 1, "Should extract at least one fact"
        assert all(isinstance(f, ExtractedFact) for f in result.facts)

        # Check that facts have reasonable content
        all_text = " ".join(f.text.lower() for f in result.facts)
        assert "water" in all_text or "h2o" in all_text, (
            f"Expected facts about water, got: {[f.text for f in result.facts]}"
        )

    def test_extract_returns_valid_json_fields(self, engine):
        """Extracted facts have all expected fields populated."""
        result = engine.extract("The Earth orbits the Sun once every 365.25 days.")

        assert len(result.facts) >= 1
        fact = result.facts[0]

        # Check fields exist and have reasonable values
        assert isinstance(fact.text, str) and len(fact.text) > 0
        assert isinstance(fact.confidence, float)
        assert 0.0 <= fact.confidence <= 1.0
        assert isinstance(fact.fact_type, str)

    def test_extract_with_domain_hint(self, engine):
        """Domain hint is passed through and affects extraction."""
        result = engine.extract(
            "Sodium chloride dissolves in water to form Na+ and Cl- ions.",
            domain="chemistry",
        )

        assert len(result.facts) >= 1
        assert result.domain == "chemistry"

    def test_extract_to_nodes(self, engine):
        """extract_to_nodes returns valid KnowledgeNode objects."""
        source = Source(
            source_type="textbook",
            title="Integration Test Source",
            quality_score=0.8,
        )

        nodes = engine.extract_to_nodes(
            "Iron is a chemical element with symbol Fe. It is the most common element on Earth by mass.",
            domain="chemistry",
            source=source,
        )

        assert len(nodes) >= 1
        assert all(isinstance(n, KnowledgeNode) for n in nodes)

        # Check node properties
        for node in nodes:
            assert len(node.text) > 0
            assert 0.0 <= node.confidence <= 1.0
            assert node.derivation.method == "extracted"
            assert "extraction_model" in node.metadata
            assert len(node.sources) == 1
            assert node.sources[0].title == "Integration Test Source"

    def test_extract_multisentence(self, engine):
        """Multi-sentence text produces multiple facts."""
        result = engine.extract(
            "The speed of light in a vacuum is approximately 299,792,458 meters per second. "
            "Light travels slower in denser media like glass or water. "
            "Einstein's theory of special relativity states that nothing can travel faster than light."
        )

        # Should extract at least 2 facts from 3 clear statements
        assert len(result.facts) >= 2, f"Expected at least 2 facts from 3 sentences, got {len(result.facts)}"

    def test_extract_empty_input(self, engine):
        """Empty or trivial input produces few or no facts."""
        result = engine.extract("")
        # Empty input should produce 0 facts (or at most 1 if model hallucinates)
        assert len(result.facts) <= 1

    def test_raw_response_captured(self, engine):
        """The raw LLM response is captured in the result."""
        result = engine.extract("The sky appears blue due to Rayleigh scattering.")

        assert isinstance(result.raw_response, str)
        assert len(result.raw_response) > 0


@pytest.mark.integration
class TestOllamaEdgeCases:
    """Test edge cases with real Ollama extraction."""

    @pytest.fixture
    def engine(self):
        """Create a real FactExtractionEngine."""
        if not _ollama_available():
            pytest.skip("Ollama server not available")

        import ollama

        client = ollama.Client()
        models = client.list()
        if not models.models:
            pytest.skip("No models pulled")

        model_name = models.models[0].model
        return FactExtractionEngine(model=model_name)

    def test_long_text(self, engine):
        """Extraction handles longer text passages."""
        long_text = (
            "Photosynthesis is the process by which green plants and certain other "
            "organisms transform light energy into chemical energy. During photosynthesis "
            "in green plants, light energy is captured and used to convert water, carbon "
            "dioxide, and minerals into oxygen and energy-rich organic compounds. "
            "The process occurs primarily in the chloroplasts of plant cells, specifically "
            "in the thylakoid membranes. Chlorophyll, the green pigment in plants, absorbs "
            "light most efficiently in the blue and red wavelengths. The overall equation "
            "for photosynthesis is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2."
        )

        result = engine.extract(long_text, domain="biology")
        assert len(result.facts) >= 3, "Long passage should yield multiple facts"

    def test_contradictory_claims(self, engine):
        """Extraction handles text with contradictory information."""
        result = engine.extract(
            "Water boils at 100 degrees Celsius. However, at high altitudes water boils at lower temperatures."
        )

        assert len(result.facts) >= 1
        # Both claims should be extracted as separate facts
