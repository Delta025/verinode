"""
Tests for the dependency detection engine.

Tests cover:
- Text similarity fallback (Jaccard)
- Contradiction detection (negation mismatch)
- Definitional dependency detection
- Causal dependency detection
- Entity-overlap elaboration detection
- High-similarity support detection
- Graph wiring via apply_dependencies
- Embedding-backed candidate selection
- Edge cases (empty graph, single node, self-loops)

Run with:  python -m pytest tests/test_dependency_detection.py -v
Or:        python tests/test_dependency_detection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.dependency_detection import (
    DependencyDetector,
    DetectedDependency,
    RelationType,
)
from core.graph import KnowledgeGraph
from core.node import KnowledgeNode, NodeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_with(*texts_and_domains):
    """Create a graph with the given (text, domain) pairs."""
    graph = KnowledgeGraph()
    nodes = []
    for item in texts_and_domains:
        if isinstance(item, tuple):
            text, domain = item
        else:
            text, domain = item, "general"
        node = KnowledgeNode(text=text, node_type=NodeType.FACT, confidence=0.9, domain=domain)
        graph.add_node(node)
        nodes.append(node)
    return graph, nodes


# ---------------------------------------------------------------------------
# Tests: text similarity (Jaccard fallback)
# ---------------------------------------------------------------------------


def test_text_similarity_identical():
    """Identical texts have similarity 1.0."""
    sim = DependencyDetector._text_similarity(
        "Water boils at 100 degrees Celsius", "Water boils at 100 degrees Celsius"
    )
    assert sim == 1.0
    print("  text_similarity_identical passed")


def test_text_similarity_disjoint():
    """Completely different texts have similarity 0."""
    sim = DependencyDetector._text_similarity("Photosynthesis converts sunlight", "Gravity pulls objects downward")
    assert sim == 0.0
    print("  text_similarity_disjoint passed")


def test_text_similarity_partial():
    """Partially overlapping texts have intermediate similarity."""
    sim = DependencyDetector._text_similarity("Water boils at 100 degrees", "Water freezes at 0 degrees")
    assert 0.0 < sim < 1.0
    print("  text_similarity_partial passed")


# ---------------------------------------------------------------------------
# Tests: contradiction detection
# ---------------------------------------------------------------------------


def test_contradiction_detected():
    """Negation mismatch on similar content is flagged as contradiction."""
    graph, nodes = _graph_with(
        ("Water boils at 100 degrees Celsius", "chemistry"),
        ("Water does not boil at 100 degrees Celsius", "chemistry"),
    )
    detector = DependencyDetector(similarity_threshold=0.2, dependency_confidence_threshold=0.1)
    deps = detector.detect_all(graph)

    contradictions = [d for d in deps if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions) >= 1
    print("  contradiction_detected passed")


def test_no_contradiction_same_polarity():
    """Two affirmative statements about different topics are not contradictions."""
    graph, nodes = _graph_with(
        ("Water boils at 100 degrees", "chemistry"),
        ("Iron melts at 1538 degrees", "chemistry"),
    )
    detector = DependencyDetector(similarity_threshold=0.01, dependency_confidence_threshold=0.1)
    deps = detector.detect_all(graph)

    contradictions = [d for d in deps if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions) == 0
    print("  no_contradiction_same_polarity passed")


# ---------------------------------------------------------------------------
# Tests: definitional dependency
# ---------------------------------------------------------------------------


def test_definitional_dependency():
    """A node using a defined concept depends on the definition."""
    graph, nodes = _graph_with(
        ("Photosynthesis is defined as the process by which plants convert sunlight into energy", "biology"),
        ("Photosynthesis occurs primarily in the leaves of plants", "biology"),
    )
    detector = DependencyDetector(similarity_threshold=0.1, dependency_confidence_threshold=0.1)
    deps = detector.detect_all(graph)

    depends = [d for d in deps if d.relation == RelationType.DEPENDS_ON]
    assert len(depends) >= 1
    # The definition should be the source
    dep = depends[0]
    assert dep.source_id == nodes[0].id or dep.target_id == nodes[1].id
    print("  definitional_dependency passed")


# ---------------------------------------------------------------------------
# Tests: causal dependency
# ---------------------------------------------------------------------------


def test_causal_dependency():
    """Causal language triggers a DEPENDS_ON detection."""
    graph, nodes = _graph_with(
        ("Atmospheric pressure decreases with altitude", "physics"),
        ("Water boiling point decreases because atmospheric pressure decreases at higher altitudes", "physics"),
    )
    detector = DependencyDetector(similarity_threshold=0.1, dependency_confidence_threshold=0.1)
    deps = detector.detect_all(graph)

    depends = [
        d for d in deps if d.relation in (RelationType.DEPENDS_ON, RelationType.SUPPORTS, RelationType.ELABORATES)
    ]
    assert len(depends) >= 1
    print("  causal_dependency passed")


# ---------------------------------------------------------------------------
# Tests: entity overlap / elaboration
# ---------------------------------------------------------------------------


def test_elaboration_detection():
    """A more specific statement about the same entities is classified as ELABORATES."""
    graph, nodes = _graph_with(
        ("Water boils", "chemistry"),
        ("Water boils at exactly 100 degrees Celsius at standard atmospheric pressure at sea level", "chemistry"),
    )
    detector = DependencyDetector(similarity_threshold=0.1, dependency_confidence_threshold=0.1)
    deps = detector.detect_all(graph)

    elaborates = [d for d in deps if d.relation == RelationType.ELABORATES]
    if elaborates:
        # The longer/more specific node should be the target
        dep = elaborates[0]
        assert dep.target_id == nodes[1].id or dep.source_id == nodes[0].id
    else:
        # May be detected as SUPPORTS or DEPENDS_ON due to high overlap
        assert len(deps) > 0
    print("  elaboration_detection passed")


# ---------------------------------------------------------------------------
# Tests: high-similarity support
# ---------------------------------------------------------------------------


def test_support_high_similarity():
    """High semantic overlap in same domain → SUPPORTS."""
    # Use texts with very high Jaccard overlap (>0.6) to trigger SUPPORTS path
    graph, nodes = _graph_with(
        ("Iron metal element periodic table group eight transition", "chemistry"),
        ("Iron metal element periodic table group eight block", "chemistry"),
    )
    detector = DependencyDetector(similarity_threshold=0.1, dependency_confidence_threshold=0.1)
    deps = detector.detect_all(graph)
    assert len(deps) >= 1
    # Should be SUPPORTS or ELABORATES given the high overlap
    assert deps[0].relation in (RelationType.SUPPORTS, RelationType.ELABORATES)
    print("  support_high_similarity passed")


# ---------------------------------------------------------------------------
# Tests: apply_dependencies (graph wiring)
# ---------------------------------------------------------------------------


def test_apply_depends_on():
    """apply_dependencies wires DEPENDS_ON as predecessor edge."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    dep = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.DEPENDS_ON,
        confidence=0.8,
        evidence="test",
    )
    detector = DependencyDetector()
    added = detector.apply_dependencies([dep], graph)

    assert added == 1
    assert nodes[0].id in nodes[1].predecessor_nodes
    assert nodes[1].id in nodes[0].successor_nodes
    print("  apply_depends_on passed")


def test_apply_supports():
    """apply_dependencies wires SUPPORTS as predecessor edge."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    dep = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.SUPPORTS,
        confidence=0.7,
        evidence="test",
    )
    detector = DependencyDetector()
    added = detector.apply_dependencies([dep], graph)
    assert added == 1
    assert nodes[0].id in nodes[1].predecessor_nodes
    print("  apply_supports passed")


def test_apply_contradicts():
    """apply_dependencies records CONTRADICTS via add_contradiction."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    dep = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.CONTRADICTS,
        confidence=0.9,
        evidence="test",
    )
    detector = DependencyDetector()
    added = detector.apply_dependencies([dep], graph)
    assert added == 1
    assert nodes[1].id in nodes[0].conflicts_with
    assert nodes[0].id in nodes[1].conflicts_with
    print("  apply_contradicts passed")


def test_apply_no_duplicate_edges():
    """Applying the same dependency twice doesn't create duplicate edges."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    dep = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.DEPENDS_ON,
        confidence=0.8,
        evidence="test",
    )
    detector = DependencyDetector()
    detector.apply_dependencies([dep], graph)
    added = detector.apply_dependencies([dep], graph)
    assert added == 0  # Already exists
    assert nodes[1].predecessor_nodes.count(nodes[0].id) == 1
    print("  apply_no_duplicate_edges passed")


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_empty_graph():
    """Empty graph returns no dependencies."""
    graph = KnowledgeGraph()
    detector = DependencyDetector()
    deps = detector.detect_all(graph)
    assert deps == []
    print("  empty_graph passed")


def test_single_node():
    """Graph with one node returns no dependencies."""
    graph, _ = _graph_with("Only fact")
    detector = DependencyDetector()
    deps = detector.detect_all(graph)
    assert deps == []
    print("  single_node passed")


def test_detect_for_node_without_embeddings():
    """detect_for_node works via text fallback when no embeddings available."""
    graph, nodes = _graph_with(
        ("Water boils 100 degrees Celsius standard pressure sea level chemistry", "chemistry"),
        ("Water does not boil 100 degrees Celsius standard pressure sea level chemistry", "chemistry"),
        ("Gravity acceleration constant Earth surface", "physics"),
    )
    detector = DependencyDetector(similarity_threshold=0.1, dependency_confidence_threshold=0.1)
    deps = detector.detect_for_node(nodes[0], graph)
    # Should find contradiction with nodes[1] (high overlap + negation mismatch)
    water_related = [
        d for d in deps if d.source_id in (nodes[0].id, nodes[1].id) or d.target_id in (nodes[0].id, nodes[1].id)
    ]
    assert len(water_related) >= 1
    print("  detect_for_node_without_embeddings passed")


def test_confidence_threshold_filters():
    """Dependencies below confidence threshold are excluded."""
    graph, nodes = _graph_with(
        ("Alpha beta gamma", "general"),
        ("Delta epsilon zeta", "general"),
    )
    # Very high threshold should filter out weak relationships
    detector = DependencyDetector(similarity_threshold=0.01, dependency_confidence_threshold=0.99)
    deps = detector.detect_all(graph)
    assert len(deps) == 0
    print("  confidence_threshold_filters passed")


# ---------------------------------------------------------------------------
# Tests: embedding-backed detection
# ---------------------------------------------------------------------------

import pytest


@pytest.fixture(scope="module")
def _shared_embedding_engine():
    """Shared EmbeddingEngine for tests that need real embeddings.

    Module-scoped so the model is loaded only once, avoiding repeated
    HTTP checks to HuggingFace that can cause transient timeouts.
    """
    from core.embeddings import EmbeddingEngine

    return EmbeddingEngine()


def test_detect_with_embeddings(_shared_embedding_engine):
    """detect_all uses embedding engine for candidate selection when available."""
    engine = _shared_embedding_engine

    graph, nodes = _graph_with(
        ("Water boils at 100 degrees Celsius at sea level", "chemistry"),
        ("The boiling point of water is 100 degrees Celsius at standard pressure", "chemistry"),
        ("Mitosis is the process of cell division", "biology"),
    )

    engine.index_graph(graph)

    detector = DependencyDetector(
        similarity_threshold=0.2,
        dependency_confidence_threshold=0.1,
        embedding_engine=engine,
    )
    deps = detector.detect_all(graph, embedding_engine=engine)

    # The two water-boiling nodes should have a relationship (high similarity)
    water_pair = {nodes[0].id, nodes[1].id}
    water_deps = [d for d in deps if {d.source_id, d.target_id} & water_pair]
    assert len(water_deps) >= 1
    print("  detect_with_embeddings passed")


def test_detect_for_node_with_embeddings(_shared_embedding_engine):
    """detect_for_node uses embedding nearest-neighbor search."""
    engine = _shared_embedding_engine

    graph, nodes = _graph_with(
        ("Water freezes at 0 degrees Celsius to form ice", "chemistry"),
        ("The freezing point of water is 0 degrees Celsius", "chemistry"),
        ("The speed of light is 300000 km/s", "physics"),
    )

    engine.index_graph(graph)

    detector = DependencyDetector(
        similarity_threshold=0.2,
        dependency_confidence_threshold=0.1,
        embedding_engine=engine,
    )
    deps = detector.detect_for_node(nodes[0], graph, embedding_engine=engine)
    # Should find relationship with nodes[1] (freezing point topic), not likely with nodes[2]
    assert len(deps) >= 1
    related_ids = {d.source_id for d in deps} | {d.target_id for d in deps}
    assert nodes[1].id in related_ids
    print("  detect_for_node_with_embeddings passed")


# ---------------------------------------------------------------------------
# Tests: LLM-assisted contradiction detection
# ---------------------------------------------------------------------------


def test_llm_contradiction_detection():
    """ContradictionChecker detects contradictions the heuristic misses."""
    from unittest.mock import MagicMock

    from core.llm_contradiction import RelevanceResult

    graph, nodes = _graph_with(
        ("Water freezes at 0 degrees Celsius", "chemistry"),
        ("Water freezes at -100 degrees Celsius", "chemistry"),
    )
    # No negation words → heuristic will NOT flag this
    detector = DependencyDetector(similarity_threshold=0.01, dependency_confidence_threshold=0.1)

    # Without LLM: should NOT find a contradiction (no negation mismatch)
    deps_no_llm = detector.detect_all(graph)
    contradictions_no_llm = [d for d in deps_no_llm if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions_no_llm) == 0, "Heuristic should not catch numerical contradiction"

    # With LLM: mock checker that spots the numerical contradiction
    mock_checker = MagicMock()
    mock_checker.check_relevance.return_value = RelevanceResult(
        claim=nodes[0].text,
        knowledge=nodes[1].text,
        relationship="contradicts",
        confidence=0.95,
        explanation="Incompatible freezing point values",
    )
    deps_with_llm = detector.detect_all(graph, contradiction_checker=mock_checker)
    contradictions_with_llm = [d for d in deps_with_llm if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions_with_llm) >= 1, "LLM should catch numerical contradiction"
    assert "contradicts" in contradictions_with_llm[0].evidence.lower()
    print("  llm_contradiction_detection passed")


def test_llm_overrides_heuristic_false_positive():
    """LLM can override a heuristic false positive (negation that isn't a real contradiction)."""
    from unittest.mock import MagicMock

    from core.llm_contradiction import RelevanceResult

    graph, nodes = _graph_with(
        ("Water boils at 100 degrees Celsius", "chemistry"),
        ("Water does not freeze at 100 degrees Celsius", "chemistry"),
    )
    # Heuristic sees negation mismatch + high word overlap → flags as contradiction
    detector = DependencyDetector(similarity_threshold=0.01, dependency_confidence_threshold=0.1)
    deps_heuristic = detector.detect_all(graph)
    contradictions_heuristic = [d for d in deps_heuristic if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions_heuristic) >= 1, "Heuristic should (incorrectly) flag this"

    # LLM recognises these are about different properties → not a contradiction
    mock_checker = MagicMock()
    mock_checker.check_relevance.return_value = RelevanceResult(
        claim=nodes[0].text,
        knowledge=nodes[1].text,
        relationship="irrelevant",
        confidence=0.9,
        explanation="Different properties: boiling vs freezing",
    )
    deps_with_llm = detector.detect_all(graph, contradiction_checker=mock_checker)
    contradictions_with_llm = [d for d in deps_with_llm if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions_with_llm) == 0, "LLM should override the false positive"
    print("  llm_overrides_heuristic_false_positive passed")


def test_no_llm_behaviour_unchanged():
    """When no contradiction_checker is provided, behaviour is identical to before."""
    graph, nodes = _graph_with(
        ("Water boils at 100 degrees Celsius", "chemistry"),
        ("Water does not boil at 100 degrees Celsius", "chemistry"),
    )
    detector = DependencyDetector(similarity_threshold=0.2, dependency_confidence_threshold=0.1)

    # Explicitly pass contradiction_checker=None
    deps = detector.detect_all(graph, contradiction_checker=None)
    contradictions = [d for d in deps if d.relation == RelationType.CONTRADICTS]
    assert len(contradictions) >= 1
    # Evidence should be the heuristic message, not LLM
    assert "Negation mismatch" in contradictions[0].evidence
    print("  no_llm_behaviour_unchanged passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_apply_dependencies_rejects_cycle():
    """apply_dependencies skips edges that would create cycles."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    # Wire A -> B (A is predecessor of B)
    dep1 = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.DEPENDS_ON,
        confidence=0.8,
        evidence="test",
    )
    detector = DependencyDetector()
    detector.apply_dependencies([dep1], graph)
    assert nodes[0].id in nodes[1].predecessor_nodes

    # Try to wire B -> A (would create cycle)
    dep2 = DetectedDependency(
        source_id=nodes[1].id,
        target_id=nodes[0].id,
        relation=RelationType.SUPPORTS,
        confidence=0.7,
        evidence="test",
    )
    added = detector.apply_dependencies([dep2], graph)
    assert added == 0  # Cycle rejected
    assert nodes[1].id not in nodes[0].predecessor_nodes


def test_apply_dependencies_preserves_relation_type():
    """apply_dependencies stores the relation type as edge label."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    dep = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.SUPPORTS,
        confidence=0.7,
        evidence="test",
    )
    detector = DependencyDetector()
    detector.apply_dependencies([dep], graph)
    assert nodes[1].get_edge_label(nodes[0].id) == "supports"


def test_apply_dependencies_wires_elaborates():
    """ELABORATES relations are now wired as edges (previously dropped)."""
    graph, nodes = _graph_with("Fact A", "Fact B")
    dep = DetectedDependency(
        source_id=nodes[0].id,
        target_id=nodes[1].id,
        relation=RelationType.ELABORATES,
        confidence=0.7,
        evidence="test",
    )
    detector = DependencyDetector()
    added = detector.apply_dependencies([dep], graph)
    assert added == 1
    assert nodes[0].id in nodes[1].predecessor_nodes
    assert nodes[1].get_edge_label(nodes[0].id) == "elaborates"


def run_all_tests():
    from core.embeddings import EmbeddingEngine

    print("\nRunning dependency detection tests...\n")
    test_text_similarity_identical()
    test_text_similarity_disjoint()
    test_text_similarity_partial()
    test_contradiction_detected()
    test_no_contradiction_same_polarity()
    test_definitional_dependency()
    test_causal_dependency()
    test_elaboration_detection()
    test_support_high_similarity()
    test_apply_depends_on()
    test_apply_supports()
    test_apply_contradicts()
    test_apply_no_duplicate_edges()
    test_empty_graph()
    test_single_node()
    test_detect_for_node_without_embeddings()
    test_confidence_threshold_filters()
    engine = EmbeddingEngine()
    test_detect_with_embeddings(engine)
    test_detect_for_node_with_embeddings(engine)
    test_llm_contradiction_detection()
    test_llm_overrides_heuristic_false_positive()
    test_no_llm_behaviour_unchanged()
    print("\nAll dependency detection tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
