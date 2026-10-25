"""
Tests for the fact extraction engine.

All tests mock the Ollama client so they run without a running Ollama
server or any downloaded models.

Tests cover:
- JSON response parsing (valid, malformed, edge cases)
- Ollama JSON-mode dict-wrapper handling ({"facts": [...]})
- ExtractedFact -> KnowledgeNode conversion
- Related-fact wiring (predecessor relationships)
- Domain hints
- Confidence clamping
- Source attachment
- End-to-end extract + node conversion flow

Run with:  python -m pytest tests/test_fact_extraction.py -v
Or:        python tests/test_fact_extraction.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.node import KnowledgeNode, NodeType, Source
from core.fact_extraction import (
    FactExtractionEngine,
    ExtractedFact,
    ExtractionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_text: str):
    """Create a mock Ollama chat response."""
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = json_text
    return response


SAMPLE_RESPONSE = json.dumps([
    {
        "text": "Water is composed of hydrogen and oxygen",
        "fact_type": "fact",
        "confidence": 0.95,
        "domain": "chemistry",
        "related_facts": [],
        "source_quote": "Water is H2O"
    },
    {
        "text": "Water boils at 100 degrees Celsius at sea level",
        "fact_type": "observation",
        "confidence": 0.9,
        "domain": "chemistry",
        "related_facts": [0],
        "source_quote": "boils at 100C"
    },
    {
        "text": "The boiling point of water decreases at higher altitudes",
        "fact_type": "inference",
        "confidence": 0.85,
        "domain": "chemistry",
        "related_facts": [1],
        "source_quote": "altitude affects boiling"
    }
])


def _make_engine():
    """Create a FactExtractionEngine with a mocked Ollama client."""
    with patch("core.fact_extraction._get_ollama") as mock_get:
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_get.return_value = mock_ollama

        engine = FactExtractionEngine(model="qwen2.5:7b")
        engine._client = mock_client
    return engine


# ---------------------------------------------------------------------------
# Tests: parsing
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    """Valid JSON array is parsed into ExtractedFact objects."""
    engine = _make_engine()
    facts = engine._parse_response(SAMPLE_RESPONSE)
    assert len(facts) == 3
    assert facts[0].text == "Water is composed of hydrogen and oxygen"
    assert facts[0].fact_type == "fact"
    assert facts[0].confidence == 0.95
    assert facts[0].domain == "chemistry"
    print("  parse_valid_json passed")


def test_parse_code_fenced_json():
    """JSON wrapped in markdown code fences is handled."""
    engine = _make_engine()
    wrapped = f"```json\n{SAMPLE_RESPONSE}\n```"
    facts = engine._parse_response(wrapped)
    assert len(facts) == 3
    print("  parse_code_fenced_json passed")


def test_parse_single_object():
    """A single JSON object (not array) is wrapped automatically."""
    engine = _make_engine()
    single = json.dumps({
        "text": "The sky is blue",
        "fact_type": "observation",
        "confidence": 0.8,
        "domain": "general",
        "related_facts": [],
        "source_quote": "sky is blue"
    })
    facts = engine._parse_response(single)
    assert len(facts) == 1
    assert facts[0].text == "The sky is blue"
    print("  parse_single_object passed")


def test_parse_malformed_json():
    """Malformed JSON returns empty list instead of crashing."""
    engine = _make_engine()
    facts = engine._parse_response("this is not json {{{")
    assert facts == []
    print("  parse_malformed_json passed")


def test_parse_empty_text_skipped():
    """Items with empty text are skipped."""
    engine = _make_engine()
    data = json.dumps([
        {"text": "", "confidence": 0.9},
        {"text": "  ", "confidence": 0.9},
        {"text": "Valid fact", "confidence": 0.8}
    ])
    facts = engine._parse_response(data)
    assert len(facts) == 1
    assert facts[0].text == "Valid fact"
    print("  parse_empty_text_skipped passed")


def test_confidence_clamping():
    """Confidence values outside [0, 1] are clamped."""
    engine = _make_engine()
    data = json.dumps([
        {"text": "Over-confident", "confidence": 1.5},
        {"text": "Under-confident", "confidence": -0.3},
    ])
    facts = engine._parse_response(data)
    assert facts[0].confidence == 1.0
    assert facts[1].confidence == 0.0
    print("  confidence_clamping passed")


def test_default_domain_from_hint():
    """When facts have no domain, the default_domain is used."""
    engine = _make_engine()
    data = json.dumps([{"text": "Fact without domain"}])
    facts = engine._parse_response(data, default_domain="physics")
    assert facts[0].domain == "physics"
    print("  default_domain_from_hint passed")


def test_parse_dict_wrapper_facts_key():
    """Ollama JSON mode may return {"facts": [...]} — unwrap correctly."""
    engine = _make_engine()
    wrapped = json.dumps({
        "facts": [
            {"text": "Iron is a metal", "fact_type": "fact", "confidence": 0.95},
            {"text": "Iron rusts when exposed to moisture", "fact_type": "observation", "confidence": 0.9}
        ]
    })
    facts = engine._parse_response(wrapped)
    assert len(facts) == 2
    assert facts[0].text == "Iron is a metal"
    assert facts[1].text == "Iron rusts when exposed to moisture"
    print("  parse_dict_wrapper_facts_key passed")


def test_parse_dict_wrapper_results_key():
    """Ollama may also return {"results": [...]} — unwrap correctly."""
    engine = _make_engine()
    wrapped = json.dumps({
        "results": [
            {"text": "The sun is a star", "confidence": 0.99}
        ]
    })
    facts = engine._parse_response(wrapped)
    assert len(facts) == 1
    assert facts[0].text == "The sun is a star"
    print("  parse_dict_wrapper_results_key passed")


def test_parse_dict_unknown_key_treated_as_single_fact():
    """A dict without known wrapper keys is treated as a single fact."""
    engine = _make_engine()
    single = json.dumps({
        "text": "Gravity pulls objects toward Earth",
        "fact_type": "fact",
        "confidence": 0.95
    })
    facts = engine._parse_response(single)
    assert len(facts) == 1
    assert facts[0].text == "Gravity pulls objects toward Earth"
    print("  parse_dict_unknown_key_treated_as_single_fact passed")


# ---------------------------------------------------------------------------
# Tests: node conversion
# ---------------------------------------------------------------------------

def test_facts_to_nodes_basic():
    """facts_to_nodes creates KnowledgeNode objects with correct attributes."""
    engine = _make_engine()
    result = ExtractionResult(
        facts=[
            ExtractedFact(
                text="Water is H2O",
                fact_type="fact",
                confidence=0.95,
                domain="chemistry",
                source_quote="Water is H2O",
            ),
            ExtractedFact(
                text="Water boils at 100C",
                fact_type="observation",
                confidence=0.9,
                domain="chemistry",
                source_quote="boils at 100C",
                related_facts=[0],
            ),
        ],
        source_text="Water is H2O and boils at 100C",
        model="qwen2.5:7b",
    )

    nodes = engine.facts_to_nodes(result)
    assert len(nodes) == 2
    assert isinstance(nodes[0], KnowledgeNode)
    assert nodes[0].text == "Water is H2O"
    assert nodes[0].node_type == NodeType.FACT
    assert nodes[0].confidence == 0.95
    assert nodes[0].domain == "chemistry"
    assert nodes[0].derivation.method == "extracted"

    # Second node should have first as predecessor
    assert nodes[1].node_type == NodeType.OBSERVATION
    assert nodes[0].id in nodes[1].predecessor_nodes
    print("  facts_to_nodes_basic passed")


def test_facts_to_nodes_with_source():
    """Source metadata is attached to every node."""
    engine = _make_engine()
    source = Source(
        source_type="textbook",
        title="Chemistry 101",
        author="Dr. Smith",
        quality_score=0.9,
    )
    result = ExtractionResult(
        facts=[ExtractedFact(text="NaCl is table salt", confidence=0.95, domain="chemistry")],
        source_text="NaCl is table salt",
        model="qwen2.5:7b",
    )

    nodes = engine.facts_to_nodes(result, source=source)
    assert len(nodes[0].sources) == 1
    assert nodes[0].sources[0].title == "Chemistry 101"
    print("  facts_to_nodes_with_source passed")


def test_related_facts_wiring():
    """related_facts indices produce correct predecessor edges."""
    engine = _make_engine()
    result = ExtractionResult(
        facts=[
            ExtractedFact(text="A", confidence=0.9),
            ExtractedFact(text="B", confidence=0.8, related_facts=[0]),
            ExtractedFact(text="C", confidence=0.7, related_facts=[0, 1]),
        ],
        source_text="A, B, C",
        model="test",
    )

    nodes = engine.facts_to_nodes(result)

    # B depends on A
    assert nodes[0].id in nodes[1].predecessor_nodes
    # C depends on both A and B
    assert nodes[0].id in nodes[2].predecessor_nodes
    assert nodes[1].id in nodes[2].predecessor_nodes
    # A has no predecessors
    assert len(nodes[0].predecessor_nodes) == 0
    print("  related_facts_wiring passed")


def test_self_reference_ignored():
    """A fact referencing its own index is silently ignored."""
    engine = _make_engine()
    result = ExtractionResult(
        facts=[ExtractedFact(text="Self-referencing", confidence=0.5, related_facts=[0])],
        source_text="text",
        model="test",
    )
    nodes = engine.facts_to_nodes(result)
    assert len(nodes[0].predecessor_nodes) == 0
    print("  self_reference_ignored passed")


def test_out_of_range_reference_ignored():
    """References to indices outside the facts list are ignored."""
    engine = _make_engine()
    result = ExtractionResult(
        facts=[ExtractedFact(text="Lonely fact", confidence=0.5, related_facts=[5, 99])],
        source_text="text",
        model="test",
    )
    nodes = engine.facts_to_nodes(result)
    assert len(nodes[0].predecessor_nodes) == 0
    print("  out_of_range_reference_ignored passed")


# ---------------------------------------------------------------------------
# Tests: end-to-end extract (mocked Ollama)
# ---------------------------------------------------------------------------

def test_extract_end_to_end():
    """Full extract() call with mocked Ollama returns ExtractionResult."""
    engine = _make_engine()
    engine._client.chat.return_value = _mock_response(SAMPLE_RESPONSE)

    result = engine.extract(
        "Water is H2O. It boils at 100C. At altitude the boiling point drops.",
        domain="chemistry",
    )

    assert isinstance(result, ExtractionResult)
    assert len(result.facts) == 3
    assert result.model == engine.model
    assert result.domain == "chemistry"
    assert result.source_text.startswith("Water")
    print("  extract_end_to_end passed")


def test_extract_to_nodes_end_to_end():
    """extract_to_nodes returns KnowledgeNode list directly."""
    engine = _make_engine()
    engine._client.chat.return_value = _mock_response(SAMPLE_RESPONSE)

    source = Source(source_type="lecture", title="Chem 101", quality_score=0.85)
    nodes = engine.extract_to_nodes(
        "Water is H2O. It boils at 100C.",
        domain="chemistry",
        source=source,
    )

    assert len(nodes) == 3
    assert all(isinstance(n, KnowledgeNode) for n in nodes)
    assert all(len(n.sources) == 1 for n in nodes)
    # Node 1 should depend on node 0 per SAMPLE_RESPONSE
    assert nodes[0].id in nodes[1].predecessor_nodes
    print("  extract_to_nodes_end_to_end passed")


def test_extract_with_empty_response():
    """If the LLM returns empty JSON array, result has no facts."""
    engine = _make_engine()
    engine._client.chat.return_value = _mock_response("[]")

    result = engine.extract("Nothing interesting.")
    assert result.facts == []
    print("  extract_with_empty_response passed")


def test_metadata_on_nodes():
    """Extracted nodes carry extraction metadata."""
    engine = _make_engine()
    engine._client.chat.return_value = _mock_response(SAMPLE_RESPONSE)

    nodes = engine.extract_to_nodes("Water is H2O.")
    assert "source_quote" in nodes[0].metadata
    assert "extraction_model" in nodes[0].metadata
    print("  metadata_on_nodes passed")


def test_extract_passes_json_format():
    """Verify that extract() passes format='json' to Ollama."""
    engine = _make_engine()
    engine._client.chat.return_value = _mock_response(SAMPLE_RESPONSE)

    engine.extract("Test text")

    call_kwargs = engine._client.chat.call_args
    assert call_kwargs.kwargs.get("format") == "json" or \
           (len(call_kwargs.args) == 0 and call_kwargs[1].get("format") == "json")
    print("  extract_passes_json_format passed")


def test_extract_passes_temperature_and_ctx():
    """Verify that temperature and num_ctx are passed via options."""
    engine = _make_engine()
    engine.temperature = 0.1
    engine.num_ctx = 8192
    engine._client.chat.return_value = _mock_response("[]")

    engine.extract("Test text")

    call_kwargs = engine._client.chat.call_args
    options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
    assert options["temperature"] == 0.1
    assert options["num_ctx"] == 8192
    print("  extract_passes_temperature_and_ctx passed")


def test_extract_dict_wrapper_end_to_end():
    """Full extract with Ollama returning {"facts": [...]} dict wrapper."""
    engine = _make_engine()
    wrapped_response = json.dumps({
        "facts": [
            {
                "text": "The Earth orbits the Sun",
                "fact_type": "fact",
                "confidence": 0.99,
                "domain": "astronomy",
                "related_facts": [],
                "source_quote": "Earth orbits Sun"
            }
        ]
    })
    engine._client.chat.return_value = _mock_response(wrapped_response)

    result = engine.extract("The Earth orbits the Sun.")
    assert len(result.facts) == 1
    assert result.facts[0].text == "The Earth orbits the Sun"
    print("  extract_dict_wrapper_end_to_end passed")


# ---------------------------------------------------------------------------
# Tests: _safe_float helper
# ---------------------------------------------------------------------------

def test_safe_float_valid():
    """_safe_float converts valid numeric values."""
    from core.fact_extraction import _safe_float
    assert _safe_float(0.85) == 0.85
    assert _safe_float("0.7") == 0.7
    assert _safe_float(1) == 1.0


def test_safe_float_invalid():
    """_safe_float returns default for invalid values."""
    from core.fact_extraction import _safe_float
    assert _safe_float("not-a-number") == 0.5
    assert _safe_float(None) == 0.5
    assert _safe_float("high", 0.3) == 0.3


def test_parse_response_with_string_confidence():
    """Confidence as string in LLM response doesn't crash."""
    engine = _make_engine()
    response_json = json.dumps([
        {
            "text": "Test fact with string confidence",
            "fact_type": "fact",
            "confidence": "not_a_number",
            "domain": "general",
            "related_facts": [],
            "source_quote": "test"
        }
    ])
    engine._client.chat.return_value = _mock_response(response_json)
    result = engine.extract("Test text")
    assert len(result.facts) == 1
    # Should fall back to default 0.5 and be clamped to [0,1]
    assert 0.0 <= result.facts[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("\nRunning fact extraction tests...\n")
    test_parse_valid_json()
    test_parse_code_fenced_json()
    test_parse_single_object()
    test_parse_malformed_json()
    test_parse_empty_text_skipped()
    test_confidence_clamping()
    test_default_domain_from_hint()
    test_parse_dict_wrapper_facts_key()
    test_parse_dict_wrapper_results_key()
    test_parse_dict_unknown_key_treated_as_single_fact()
    test_facts_to_nodes_basic()
    test_facts_to_nodes_with_source()
    test_related_facts_wiring()
    test_self_reference_ignored()
    test_out_of_range_reference_ignored()
    test_extract_end_to_end()
    test_extract_to_nodes_end_to_end()
    test_extract_with_empty_response()
    test_metadata_on_nodes()
    test_extract_passes_json_format()
    test_extract_passes_temperature_and_ctx()
    test_extract_dict_wrapper_end_to_end()
    print("\nAll fact extraction tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
