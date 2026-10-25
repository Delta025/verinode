"""
Integration tests for LLM-assisted contradiction detection.

These tests require a running Ollama server with a pulled model
(default: qwen2.5:7b). Skip unless --run-integration is passed.

Tests cover:
- Numerical contradiction detection
- Semantic contradiction detection
- Logical contradiction detection
- False positive guards (different properties, compatible facts)
- Full pipeline integration with conflicting facts
- ContradictionChecker graceful fallback
- DependencyDetector with live LLM

Run with:
    pytest tests/test_integration_contradiction.py -v --run-integration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from core.llm_contradiction import create_contradiction_checker

# All tests require a running Ollama server
pytestmark = pytest.mark.integration


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def checker():
    """Create a shared ContradictionChecker (module-scoped to reuse connection)."""
    c = create_contradiction_checker()
    if c is None:
        pytest.skip("Ollama not available")
    return c


# ------------------------------------------------------------------
# Tests: contradiction types
# ------------------------------------------------------------------


class TestContradictionTypes:
    """Test that the LLM correctly identifies different contradiction types."""

    def test_numerical_contradiction(self, checker):
        """LLM detects incompatible numerical values for the same property."""
        result = checker.check_relevance(
            "Water freezes at 0 degrees Celsius",
            "Water freezes at -100 degrees Celsius",
        )
        assert result.relationship == "contradicts"
        assert result.confidence >= 0.6

    def test_semantic_contradiction(self, checker):
        """LLM detects semantically incompatible claims without negation."""
        result = checker.check_relevance(
            "Water is safe to drink",
            "Water is toxic and dangerous to consume",
        )
        assert result.relationship == "contradicts"
        assert result.confidence >= 0.6

    def test_logical_contradiction(self, checker):
        """LLM detects logically invalid inferences."""
        result = checker.check_relevance(
            "Water boils at 100 degrees Celsius at sea level",
            "Water cannot freeze unless cooled below -100 degrees Celsius",
        )
        # The LLM should recognise these are about different properties
        # and the second makes an invalid inference
        assert result.relationship == "contradicts"
        assert result.confidence >= 0.5


# ------------------------------------------------------------------
# Tests: false positive guards
# ------------------------------------------------------------------


class TestFalsePositiveGuards:
    """Test that the LLM does NOT flag non-contradictions."""

    def test_different_properties_not_contradiction(self, checker):
        """Statements about different properties of the same subject are NOT contradictions."""
        result = checker.check_relevance(
            "Water boils at 100 degrees Celsius",
            "Water freezes at 0 degrees Celsius",
        )
        assert result.relationship != "contradicts"

    def test_compatible_facts_not_contradiction(self, checker):
        """Two true, compatible facts should not be flagged."""
        result = checker.check_relevance(
            "Iron is a metal found in the Earth's crust",
            "Iron is used in the production of steel",
        )
        assert result.relationship != "contradicts"

    def test_specific_vs_general_not_contradiction(self, checker):
        """A more specific statement does not contradict a more general one."""
        result = checker.check_relevance(
            "Water boils at 100 degrees Celsius",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure",
        )
        assert result.relationship != "contradicts"


# ------------------------------------------------------------------
# Tests: pipeline integration
# ------------------------------------------------------------------


class TestPipelineIntegration:
    """Test LLM contradiction detection integrated with the ingestion pipeline."""

    def test_pipeline_flags_numerical_contradiction(self, checker):
        """Full pipeline catches a numerical contradiction via LLM."""
        from core.graph import KnowledgeGraph
        from core.node import KnowledgeNode, Source
        from core.pipeline import Pipeline

        graph = KnowledgeGraph()
        existing = KnowledgeNode(
            text="Water freezes at 0 degrees Celsius",
            confidence=0.95,
            domain="chemistry",
        )
        graph.add_node(existing)

        pipeline = Pipeline(
            graph,
            contradiction_checker=checker,
            contradiction_threshold=0.7,
        )
        pipeline.build_index()

        result = pipeline.ingest(
            "Water freezes at -100 degrees Celsius.",
            source=Source(source_type="textbook", title="Test", quality_score=0.9),
            domain="chemistry",
        )

        # Should be flagged (numerical contradiction)
        flagged = [v for v in result.verdicts if v.action == "flagged"]
        assert len(flagged) >= 1, f"Expected flagged verdict, got: {[(v.action, v.reason) for v in result.verdicts]}"


# ------------------------------------------------------------------
# Tests: fallback behaviour
# ------------------------------------------------------------------


class TestFallback:
    """Test graceful degradation when Ollama is unavailable."""

    def test_create_checker_returns_none_on_bad_host(self):
        """Factory returns None when Ollama is unreachable."""
        result = create_contradiction_checker(host="http://localhost:99999")
        assert result is None

    def test_dependency_detector_with_live_llm(self, checker):
        """DependencyDetector correctly uses live LLM for contradiction classification."""
        from core.dependency_detection import DependencyDetector, RelationType
        from core.graph import KnowledgeGraph
        from core.node import KnowledgeNode, NodeType

        graph = KnowledgeGraph()
        node_a = KnowledgeNode(
            text="Water freezes at 0 degrees Celsius",
            node_type=NodeType.FACT,
            confidence=0.9,
            domain="chemistry",
        )
        node_b = KnowledgeNode(
            text="Water freezes at -100 degrees Celsius",
            node_type=NodeType.FACT,
            confidence=0.9,
            domain="chemistry",
        )
        graph.add_node(node_a)
        graph.add_node(node_b)

        detector = DependencyDetector(
            similarity_threshold=0.01,
            dependency_confidence_threshold=0.1,
        )
        deps = detector.detect_all(
            graph,
            contradiction_checker=checker,
        )

        contradictions = [d for d in deps if d.relation == RelationType.CONTRADICTS]
        assert len(contradictions) >= 1, (
            f"Expected contradiction, got relations: {[(d.relation, d.evidence) for d in deps]}"
        )
        # Evidence should mention LLM
        assert any("LLM" in d.evidence for d in contradictions)
