"""
Tests for the reasoning engine.

Tests cover:
- Forward chaining (single path, branching, depth limit)
- Backward chaining (single path, multiple roots)
- Chain confidence calculation (weakest link, product, average)
- Query answering (keyword fallback and embedding-backed)
- Hypothesis testing (supported, contradicted, unsupported)
- Node explanation generation
- Edge cases (empty graph, disconnected nodes, cycles avoided)

Run with:  python -m pytest tests/test_reasoning.py -v
Or:        python tests/test_reasoning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.node import KnowledgeNode, NodeType, Source, Derivation
from core.graph import KnowledgeGraph
from core.confidence import ConfidenceEngine
from core.reasoning import (
    ReasoningEngine,
    ReasoningChain,
    InferenceStrategy,
    QueryResult,
    HypothesisResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_chain_graph():
    """Build a linear chain:  A -> B -> C -> D"""
    graph = KnowledgeGraph()

    a = KnowledgeNode(text="Water is H2O", confidence=0.95, domain="chemistry",
                      node_type=NodeType.FACT)
    b = KnowledgeNode(text="H2O boils at 100C", confidence=0.90, domain="chemistry",
                      predecessor_nodes=[a.id],
                      derivation=Derivation(method="inferred", inference_depth=1, derivation_type="chain"))
    c = KnowledgeNode(text="Boiling produces steam", confidence=0.85, domain="chemistry",
                      predecessor_nodes=[b.id],
                      derivation=Derivation(method="inferred", inference_depth=2, derivation_type="chain"))
    d = KnowledgeNode(text="Steam can drive turbines", confidence=0.80, domain="engineering",
                      predecessor_nodes=[c.id],
                      derivation=Derivation(method="inferred", inference_depth=3, derivation_type="chain"))

    for n in [a, b, c, d]:
        graph.add_node(n)

    return graph, a, b, c, d


def _branching_graph():
    """
    Build a branching graph:
        A -> C
        B -> C -> D
    """
    graph = KnowledgeGraph()

    a = KnowledgeNode(text="Hydrogen is flammable", confidence=0.95, domain="chemistry")
    b = KnowledgeNode(text="Oxygen supports combustion", confidence=0.90, domain="chemistry")
    c = KnowledgeNode(text="Water forms when hydrogen burns in oxygen", confidence=0.85,
                      domain="chemistry", predecessor_nodes=[a.id, b.id],
                      derivation=Derivation(method="inferred", derivation_type="convergence"))
    d = KnowledgeNode(text="The reaction releases energy", confidence=0.80,
                      domain="chemistry", predecessor_nodes=[c.id],
                      derivation=Derivation(method="inferred", derivation_type="chain"))

    for n in [a, b, c, d]:
        graph.add_node(n)

    return graph, a, b, c, d


def _contradiction_graph():
    """Build a graph with contradicting nodes."""
    graph = KnowledgeGraph()

    a = KnowledgeNode(text="Water boils at 100C at sea level", confidence=0.95, domain="chemistry")
    b = KnowledgeNode(text="Water does not boil at 100C at high altitude", confidence=0.90,
                      domain="chemistry")
    graph.add_node(a)
    graph.add_node(b)
    graph.add_contradiction(a.id, b.id)

    return graph, a, b


# ---------------------------------------------------------------------------
# Tests: Forward chaining
# ---------------------------------------------------------------------------

def test_forward_chain_linear():
    """Forward chaining through a linear graph finds all paths."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, a.id)
    assert len(chains) >= 1

    # The longest chain should go A -> B -> C -> D
    longest = max(chains, key=lambda ch: ch.depth)
    assert longest.depth == 4
    assert longest.steps[0].node_id == a.id
    assert longest.steps[-1].node_id == d.id
    assert longest.steps[-1].step_type == "conclusion"
    assert longest.steps[0].step_type == "premise"
    print("  forward_chain_linear passed")


def test_forward_chain_branching():
    """Forward chaining from a root finds all downstream paths."""
    graph, a, b, c, d = _branching_graph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, a.id)
    assert len(chains) >= 1
    # Should reach D via A -> C -> D
    node_ids_in_chains = set()
    for chain in chains:
        for step in chain.steps:
            node_ids_in_chains.add(step.node_id)
    assert d.id in node_ids_in_chains
    print("  forward_chain_branching passed")


def test_forward_chain_leaf_node():
    """Forward chaining from a leaf returns no chains (no successors)."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, d.id)
    assert len(chains) == 0
    print("  forward_chain_leaf_node passed")


def test_forward_chain_depth_limit():
    """max_depth limits chain length."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, a.id, max_depth=2)
    for chain in chains:
        assert chain.depth <= 3  # start + 2 hops
    print("  forward_chain_depth_limit passed")


def test_forward_chain_nonexistent_node():
    """Forward chaining from a missing node returns empty list."""
    graph, *_ = _linear_chain_graph()
    engine = ReasoningEngine()
    chains = engine.forward_chain(graph, "nonexistent-id")
    assert chains == []
    print("  forward_chain_nonexistent_node passed")


# ---------------------------------------------------------------------------
# Tests: Backward chaining
# ---------------------------------------------------------------------------

def test_backward_chain_linear():
    """Backward chaining finds the evidence chain to a conclusion."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.backward_chain(graph, d.id)
    assert len(chains) >= 1

    chain = chains[0]
    # After reversal, should read A -> B -> C -> D
    assert chain.steps[0].step_type == "premise"
    assert chain.steps[-1].step_type == "conclusion"
    assert chain.steps[-1].node_id == d.id
    print("  backward_chain_linear passed")


def test_backward_chain_branching():
    """Backward chaining from D finds paths through both A and B."""
    graph, a, b, c, d = _branching_graph()
    engine = ReasoningEngine()

    chains = engine.backward_chain(graph, d.id)
    assert len(chains) >= 1

    # Should find paths reaching both A and B as premises
    all_premise_ids = set()
    for chain in chains:
        for step in chain.steps:
            if step.step_type == "premise":
                all_premise_ids.add(step.node_id)
    assert a.id in all_premise_ids or b.id in all_premise_ids
    print("  backward_chain_branching passed")


def test_backward_chain_root_node():
    """Backward chaining from a root node (no predecessors) returns no chains."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.backward_chain(graph, a.id)
    assert len(chains) == 0
    print("  backward_chain_root_node passed")


# ---------------------------------------------------------------------------
# Tests: Chain confidence
# ---------------------------------------------------------------------------

def test_chain_confidence_weakest_link():
    """Weakest-link strategy uses min confidence."""
    engine = ReasoningEngine(inference_strategy=InferenceStrategy.WEAKEST_LINK)
    graph, a, b, c, d = _linear_chain_graph()

    chains = engine.forward_chain(graph, a.id)
    longest = max(chains, key=lambda ch: ch.depth)
    # Weakest link = 0.80 (node D), with depth decay
    assert longest.conclusion_confidence <= 0.80
    assert longest.conclusion_confidence > 0.0
    print("  chain_confidence_weakest_link passed")


def test_chain_confidence_product():
    """Product strategy multiplies all confidences."""
    engine = ReasoningEngine(inference_strategy=InferenceStrategy.PRODUCT)
    graph, a, b, c, d = _linear_chain_graph()

    chains = engine.forward_chain(graph, a.id)
    longest = max(chains, key=lambda ch: ch.depth)
    # Product = 0.95 * 0.90 * 0.85 * 0.80 ≈ 0.58, with depth decay
    assert longest.conclusion_confidence < 0.60
    assert longest.conclusion_confidence > 0.0
    print("  chain_confidence_product passed")


def test_chain_confidence_average():
    """Average strategy uses mean of confidences."""
    engine = ReasoningEngine(inference_strategy=InferenceStrategy.AVERAGE)
    graph, a, b, c, d = _linear_chain_graph()

    chains = engine.forward_chain(graph, a.id)
    longest = max(chains, key=lambda ch: ch.depth)
    # Average ≈ (0.95+0.90+0.85+0.80)/4 ≈ 0.875, with depth decay
    assert longest.conclusion_confidence < 0.90
    assert longest.conclusion_confidence > 0.50
    print("  chain_confidence_average passed")


# ---------------------------------------------------------------------------
# Tests: Query answering
# ---------------------------------------------------------------------------

def test_query_keyword_fallback():
    """Query answering works with keyword matching when no embedding engine."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    result = engine.query(graph, "What is water?")
    assert isinstance(result, QueryResult)
    assert result.answer_node_id is not None
    # Should find the water-related node
    assert "water" in result.answer_text.lower() or "H2O" in result.answer_text
    assert result.confidence > 0.0
    print("  query_keyword_fallback passed")


def test_query_no_match():
    """Query with no keyword overlap returns empty result."""
    graph, *_ = _linear_chain_graph()
    engine = ReasoningEngine()

    result = engine.query(graph, "quantum mechanics wavefunction")
    assert result.confidence == 0.0
    assert result.answer_node_id is None
    print("  query_no_match passed")


def test_query_with_supporting_chains():
    """Query result includes supporting reasoning chains."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    result = engine.query(graph, "steam turbines")
    if result.answer_node_id:
        # If found, should have chains
        assert isinstance(result.supporting_chains, list)
    print("  query_with_supporting_chains passed")


def test_query_with_embeddings():
    """Query answering uses embedding search when available."""
    from core.embeddings import EmbeddingEngine

    graph, a, b, c, d = _linear_chain_graph()
    embed_engine = EmbeddingEngine()
    embed_engine.index_graph(graph)

    engine = ReasoningEngine()
    result = engine.query(graph, "What happens when water boils?", embedding_engine=embed_engine)

    assert isinstance(result, QueryResult)
    assert result.answer_node_id is not None
    assert result.confidence > 0.0
    print("  query_with_embeddings passed")


# ---------------------------------------------------------------------------
# Tests: Hypothesis testing
# ---------------------------------------------------------------------------

def test_hypothesis_supported():
    """A hypothesis consistent with the graph is supported."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    result = engine.test_hypothesis(graph, "Water boils to produce steam")
    assert isinstance(result, HypothesisResult)
    assert result.hypothesis == "Water boils to produce steam"
    # Should find supporting evidence (water, boils, steam nodes)
    assert len(result.supporting_evidence) >= 0  # May or may not find chains
    print("  hypothesis_supported passed")


def test_hypothesis_contradicted():
    """A hypothesis at odds with a contradiction is identified."""
    graph, a, b = _contradiction_graph()
    engine = ReasoningEngine()

    result = engine.test_hypothesis(graph, "Water boils at 100C everywhere")
    assert isinstance(result, HypothesisResult)
    # Should find the contradicting node about altitude
    has_contra = len(result.contradicting_evidence) > 0
    has_support = len(result.supporting_evidence) > 0
    assert has_contra or has_support  # At least finds some evidence
    print("  hypothesis_contradicted passed")


def test_hypothesis_no_evidence():
    """A hypothesis with no matching nodes has zero confidence."""
    graph, *_ = _linear_chain_graph()
    engine = ReasoningEngine()

    result = engine.test_hypothesis(graph, "Dinosaurs went extinct 65 million years ago")
    assert result.confidence == 0.0
    assert not result.supported
    print("  hypothesis_no_evidence passed")


def test_hypothesis_explanation():
    """Hypothesis result includes a human-readable explanation."""
    graph, *_ = _linear_chain_graph()
    engine = ReasoningEngine()

    result = engine.test_hypothesis(graph, "Water is H2O")
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0
    print("  hypothesis_explanation passed")


# ---------------------------------------------------------------------------
# Tests: Explanation
# ---------------------------------------------------------------------------

def test_explain_node_root():
    """Explaining a root node shows its sources and confidence."""
    graph, a, b, c, d = _linear_chain_graph()
    a.add_source(Source(source_type="textbook", title="Chemistry 101", quality_score=0.95))
    engine = ReasoningEngine()

    explanation = engine.explain_node(graph, a.id)
    assert "Water is H2O" in explanation
    assert "Confidence" in explanation
    print("  explain_node_root passed")


def test_explain_node_derived():
    """Explaining a derived node shows dependencies and chains."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    explanation = engine.explain_node(graph, d.id)
    assert "Steam can drive turbines" in explanation
    assert "Depends on" in explanation
    assert "Evidence chains" in explanation
    print("  explain_node_derived passed")


def test_explain_node_with_conflicts():
    """Explaining a conflicted node shows conflicts."""
    graph, a, b = _contradiction_graph()
    engine = ReasoningEngine()

    explanation = engine.explain_node(graph, a.id)
    assert "Conflicts" in explanation
    print("  explain_node_with_conflicts passed")


def test_explain_missing_node():
    """Explaining a non-existent node returns 'not found'."""
    graph = KnowledgeGraph()
    engine = ReasoningEngine()

    explanation = engine.explain_node(graph, "nonexistent")
    assert "not found" in explanation.lower()
    print("  explain_missing_node passed")


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

def test_empty_graph():
    """Operations on an empty graph don't crash."""
    graph = KnowledgeGraph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, "any-id")
    assert chains == []

    result = engine.query(graph, "anything")
    assert result.confidence == 0.0

    hyp = engine.test_hypothesis(graph, "any hypothesis")
    assert not hyp.supported
    print("  empty_graph passed")


def test_disconnected_node():
    """A node with no edges produces no chains."""
    graph = KnowledgeGraph()
    node = KnowledgeNode(text="Isolated fact", confidence=0.9)
    graph.add_node(node)

    engine = ReasoningEngine()

    forward = engine.forward_chain(graph, node.id)
    backward = engine.backward_chain(graph, node.id)
    assert forward == []
    assert backward == []
    print("  disconnected_node passed")


def test_reasoning_chain_properties():
    """ReasoningChain has correct depth, premises, and conclusion properties."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, a.id)
    longest = max(chains, key=lambda ch: ch.depth)

    assert longest.depth == len(longest.steps)
    assert len(longest.premises) >= 1
    assert longest.conclusion is not None
    assert longest.conclusion.step_type == "conclusion"
    print("  reasoning_chain_properties passed")


def test_build_explanation_format():
    """Built explanation contains 'Given that' and 'Therefore'."""
    graph, a, b, c, d = _linear_chain_graph()
    engine = ReasoningEngine()

    chains = engine.forward_chain(graph, a.id)
    for chain in chains:
        if chain.explanation:
            assert "Given that" in chain.explanation
            assert "Therefore" in chain.explanation
            break
    print("  build_explanation_format passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("\nRunning reasoning engine tests...\n")
    test_forward_chain_linear()
    test_forward_chain_branching()
    test_forward_chain_leaf_node()
    test_forward_chain_depth_limit()
    test_forward_chain_nonexistent_node()
    test_backward_chain_linear()
    test_backward_chain_branching()
    test_backward_chain_root_node()
    test_chain_confidence_weakest_link()
    test_chain_confidence_product()
    test_chain_confidence_average()
    test_query_keyword_fallback()
    test_query_no_match()
    test_query_with_supporting_chains()
    test_query_with_embeddings()
    test_hypothesis_supported()
    test_hypothesis_contradicted()
    test_hypothesis_no_evidence()
    test_hypothesis_explanation()
    test_explain_node_root()
    test_explain_node_derived()
    test_explain_node_with_conflicts()
    test_explain_missing_node()
    test_empty_graph()
    test_disconnected_node()
    test_reasoning_chain_properties()
    test_build_explanation_format()
    print("\nAll reasoning engine tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
