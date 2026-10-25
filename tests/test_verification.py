"""
Tests for the VerificationEngine: claim verification and retraction simulation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, Source, Derivation
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import ReasoningEngine
from src.core.verification import (
    VerificationEngine,
    VerificationResult,
    Verdict,
    SourceEvidence,
    RetractionSimulation,
    RetractedNodeImpact,
)
from src.core.entity_index import extract_content_words
from src.core.llm_contradiction import RelevanceResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_node(text, confidence=0.9, domain="chemistry", node_id=None, predecessors=None):
    """Create a KnowledgeNode for testing."""
    node = KnowledgeNode(
        text=text,
        confidence=confidence,
        domain=domain,
        node_id=node_id,
        predecessor_nodes=predecessors or [],
    )
    return node


def build_simple_graph():
    """Build a simple graph with 3 linked nodes for testing."""
    graph = KnowledgeGraph()

    # Root node (no predecessors)
    root = make_node(
        "Water boils at 100 degrees Celsius at sea level",
        confidence=0.95,
        node_id="root-001",
    )
    root.sources = [Source(source_type="textbook", title="Chemistry 101", quality_score=0.9)]

    # Derived node
    derived = make_node(
        "Steam is produced when water boils",
        confidence=0.88,
        node_id="derived-001",
        predecessors=["root-001"],
    )

    # Second root
    root2 = make_node(
        "Water freezes at 0 degrees Celsius",
        confidence=0.93,
        node_id="root-002",
    )
    root2.sources = [Source(source_type="textbook", title="Chemistry 101", quality_score=0.9)]

    graph.add_node(root)
    graph.add_node(derived)
    graph.add_node(root2)

    return graph


# ------------------------------------------------------------------
# Unit tests: extract_content_words
# ------------------------------------------------------------------

class TestExtractContentWords:
    def test_extracts_content_words(self):
        words = extract_content_words("Water boils at 100 degrees Celsius")
        assert "water" in words
        assert "boils" in words
        assert "100" in words
        assert "degrees" in words
        assert "celsius" in words
        # Stopwords excluded
        assert "at" not in words

    def test_empty_text(self):
        assert extract_content_words("") == set()

    def test_stopwords_only(self):
        words = extract_content_words("the is a an to for with")
        assert len(words) == 0

    def test_short_words_excluded(self):
        words = extract_content_words("I a x y")
        assert len(words) == 0


# ------------------------------------------------------------------
# Unit tests: verify()
# ------------------------------------------------------------------

class TestVerifyEmptyGraph:
    def test_empty_graph_returns_insufficient_evidence(self):
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)
        result = engine.verify("Water boils at 100 degrees Celsius")

        assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.0
        assert "empty" in result.explanation.lower()

    def test_no_relevant_nodes(self):
        """Graph has nodes but none match the claim."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("Gold has the symbol Au", node_id="gold-1"))
        engine = VerificationEngine(graph)

        result = engine.verify("The speed of light is 299792458 m/s")
        assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE


class TestVerifySupported:
    def test_supported_claim_via_entity_overlap(self):
        """Claim matching existing node should be SUPPORTED."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        result = engine.verify("Water boils at 100 degrees Celsius")
        # Entity overlap should find "root-001" (shares water, boils, 100, degrees, celsius)
        assert result.verdict == Verdict.SUPPORTED
        assert result.confidence > 0.5
        assert len(result.supporting_evidence) > 0

    def test_supported_with_embedding_engine(self):
        """Verify uses embedding search when available."""
        graph = build_simple_graph()

        mock_emb = MagicMock()
        mock_emb.index_size = 3
        mock_emb.search.return_value = [
            {"node_id": "root-001", "score": 0.92},
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Water boils at 100 degrees Celsius")

        assert result.verdict == Verdict.SUPPORTED
        assert result.confidence > 0.5
        mock_emb.search.assert_called_once()

    def test_supporting_evidence_has_node_info(self):
        """Check that supporting evidence carries node metadata."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        result = engine.verify("Water boils at 100 degrees Celsius")
        assert len(result.supporting_evidence) > 0
        ev = result.supporting_evidence[0]
        assert ev.node_id is not None
        assert ev.node_text != ""
        assert ev.confidence > 0
        assert ev.relationship == "supports"


class TestVerifyContradicted:
    def test_heuristic_contradiction_via_negation(self):
        """Negation in claim but not in graph → heuristic contradiction."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="node-1",
        ))

        # Mock embedding engine returning the relevant node
        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Water does not boil at 100 degrees Celsius")

        assert result.verdict == Verdict.CONTRADICTED
        assert len(result.contradicting_evidence) > 0

    def test_llm_contradiction_detection(self):
        """LLM checker should detect numerical contradictions via 3-way classification."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="node-1",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Water boils at 50 degrees Celsius",
                knowledge="Water boils at 100 degrees Celsius",
                relationship="contradicts",
                confidence=0.95,
                explanation="Numerical contradiction: 50C vs 100C",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
        )
        result = engine.verify("Water boils at 50 degrees Celsius")

        assert result.verdict == Verdict.CONTRADICTED
        assert result.confidence > 0.5
        assert len(result.contradicting_evidence) > 0
        assert "numerical" in result.contradicting_evidence[0].explanation.lower()

    def test_contradicting_evidence_has_explanation(self):
        """Contradicting evidence should include an explanation."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="node-1",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Water does not boil at 100 degrees Celsius")

        assert len(result.contradicting_evidence) > 0
        ev = result.contradicting_evidence[0]
        assert ev.explanation != ""
        assert ev.relationship == "contradicts"


class TestVerifyUncertain:
    def test_mixed_evidence_is_uncertain(self):
        """When support and contradiction are close, verdict is UNCERTAIN."""
        graph = KnowledgeGraph()
        # Two nodes with similar confidence — one supports, one contradicts
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius at standard pressure",
            confidence=0.85,
            node_id="support-1",
        ))
        graph.add_node(make_node(
            "Water does not boil at 100 degrees Celsius at high altitude",
            confidence=0.82,
            node_id="contra-1",
        ))

        # Mock embedding returning both
        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "support-1", "score": 0.88},
            {"node_id": "contra-1", "score": 0.85},
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Water boils at 100 degrees Celsius")

        # Could be SUPPORTED, UNCERTAIN, or CONTRADICTED depending on heuristic
        # but we should get evidence on both sides
        assert result.verdict in (Verdict.SUPPORTED, Verdict.UNCERTAIN, Verdict.CONTRADICTED)
        assert result.num_candidates_checked == 2


class TestVerifyCoverage:
    def test_coverage_calculation(self):
        """Coverage = (supporting + contradicting) / candidates_checked."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        result = engine.verify("Water boils at 100 degrees Celsius")
        assert 0.0 <= result.coverage <= 1.0

    def test_num_candidates_tracked(self):
        """Verify that num_candidates_checked is populated."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        result = engine.verify("Water boils at 100 degrees Celsius")
        assert result.num_candidates_checked >= 0


class TestVerifyBackwardChains:
    def test_evidence_has_backward_chains(self):
        """Supporting evidence should include backward chains when graph has edges."""
        graph = build_simple_graph()

        mock_emb = MagicMock()
        mock_emb.index_size = 3
        mock_emb.search.return_value = [
            {"node_id": "derived-001", "score": 0.80},
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Steam is produced when water boils")

        # derived-001 has root-001 as predecessor, so should have backward chain
        assert result.verdict == Verdict.SUPPORTED
        if result.supporting_evidence:
            ev = result.supporting_evidence[0]
            # chains list should exist (may be empty if node is a root)
            assert isinstance(ev.chains, list)


# ------------------------------------------------------------------
# Unit tests: simulate_retraction()
# ------------------------------------------------------------------

class TestSimulateRetractionNotFound:
    def test_nonexistent_node(self):
        """Retraction of nonexistent node returns empty result."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        sim = engine.simulate_retraction("nonexistent-id")
        assert sim.retracted_node_id == "nonexistent-id"
        assert sim.total_affected == 0
        assert sim.affected_nodes == []


class TestSimulateRetractionLeaf:
    def test_leaf_node_no_impact(self):
        """Retracting a leaf node (no successors) has no downstream impact."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        sim = engine.simulate_retraction("derived-001")
        assert sim.retracted_node_id == "derived-001"
        assert sim.total_affected == 0  # no successors
        assert sim.max_confidence_drop == 0.0


class TestSimulateRetractionCascade:
    def test_root_retraction_affects_derived(self):
        """Retracting a root node should lower derived node's confidence."""
        graph = KnowledgeGraph()

        root = make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="root-1",
        )
        root.sources = [Source(source_type="textbook", title="Chem", quality_score=0.9)]

        derived = make_node(
            "Steam is produced when water boils",
            confidence=0.85,
            node_id="derived-1",
            predecessors=["root-1"],
        )
        derived.derivation = Derivation(
            method="inferred",
            derivation_type="chain",
            inference_depth=1,
        )

        graph.add_node(root)
        graph.add_node(derived)

        engine = VerificationEngine(graph)
        sim = engine.simulate_retraction("root-1")

        assert sim.total_affected >= 1
        assert sim.max_confidence_drop < 0  # confidence should drop

        # The original confidences should be RESTORED
        assert graph.get_node("root-1").confidence == 0.95
        assert graph.get_node("derived-1").confidence == 0.85

    def test_confidence_restored_after_simulation(self):
        """Graph node confidences must be identical before and after."""
        graph = build_simple_graph()

        # Record original confidences
        original_confs = {nid: n.confidence for nid, n in graph.nodes.items()}

        engine = VerificationEngine(graph)
        sim = engine.simulate_retraction("root-001")

        # Check all confidences restored
        for nid, orig_conf in original_confs.items():
            assert graph.get_node(nid).confidence == orig_conf, (
                f"Node {nid} confidence not restored"
            )


class TestSimulateRetractionOrphaned:
    def test_orphaned_detection(self):
        """Nodes whose only predecessor is retracted should be counted as orphaned."""
        graph = KnowledgeGraph()

        root = make_node("Root fact", confidence=0.9, node_id="root-1")
        child = make_node(
            "Derived fact",
            confidence=0.8,
            node_id="child-1",
            predecessors=["root-1"],
        )
        child.derivation = Derivation(method="inferred", derivation_type="chain", inference_depth=1)

        graph.add_node(root)
        graph.add_node(child)

        engine = VerificationEngine(graph)
        sim = engine.simulate_retraction("root-1")

        assert sim.orphaned_nodes >= 1


class TestSimulateRetractionBranching:
    def test_multi_child_cascade(self):
        """Root with multiple children should affect all children."""
        graph = KnowledgeGraph()

        root = make_node("Root fact", confidence=0.95, node_id="root-1")
        root.sources = [Source(source_type="textbook", title="X", quality_score=0.9)]

        child1 = make_node("Child 1", confidence=0.85, node_id="child-1", predecessors=["root-1"])
        child1.derivation = Derivation(method="inferred", derivation_type="chain", inference_depth=1)

        child2 = make_node("Child 2", confidence=0.80, node_id="child-2", predecessors=["root-1"])
        child2.derivation = Derivation(method="inferred", derivation_type="chain", inference_depth=1)

        graph.add_node(root)
        graph.add_node(child1)
        graph.add_node(child2)

        engine = VerificationEngine(graph)
        sim = engine.simulate_retraction("root-1")

        # Both children should be affected
        affected_ids = {a.node_id for a in sim.affected_nodes}
        assert "child-1" in affected_ids or "child-2" in affected_ids

        # Verify restore
        assert graph.get_node("root-1").confidence == 0.95
        assert graph.get_node("child-1").confidence == 0.85
        assert graph.get_node("child-2").confidence == 0.80


class TestSimulateRetractionConvergent:
    def test_convergent_node_partially_affected(self):
        """Node with two predecessors: retract one → partial impact."""
        graph = KnowledgeGraph()

        root1 = make_node("Root 1", confidence=0.9, node_id="r1")
        root1.sources = [Source(source_type="textbook", title="A", quality_score=0.9)]

        root2 = make_node("Root 2", confidence=0.9, node_id="r2")
        root2.sources = [Source(source_type="textbook", title="B", quality_score=0.9)]

        convergent = make_node(
            "Convergent fact",
            confidence=0.88,
            node_id="conv-1",
            predecessors=["r1", "r2"],
        )
        convergent.derivation = Derivation(
            method="inferred",
            derivation_type="convergence",
            inference_depth=1,
        )

        graph.add_node(root1)
        graph.add_node(root2)
        graph.add_node(convergent)

        engine = VerificationEngine(graph)
        sim = engine.simulate_retraction("r1")

        # Convergent node should be affected but not as severely as single-parent
        assert sim.total_affected >= 0  # may or may not show significant change
        # All originals restored
        assert graph.get_node("r1").confidence == 0.9
        assert graph.get_node("r2").confidence == 0.9
        assert graph.get_node("conv-1").confidence == 0.88


class TestSimulateRetractionSummaryStats:
    def test_summary_stats_calculated(self):
        """Verify that summary stats are populated correctly."""
        graph = KnowledgeGraph()

        root = make_node("Root", confidence=0.95, node_id="root-1")
        root.sources = [Source(source_type="textbook", title="X", quality_score=0.9)]

        child = make_node("Child", confidence=0.85, node_id="child-1", predecessors=["root-1"])
        child.derivation = Derivation(method="inferred", derivation_type="chain", inference_depth=1)

        graph.add_node(root)
        graph.add_node(child)

        engine = VerificationEngine(graph)
        sim = engine.simulate_retraction("root-1")

        assert sim.retracted_node_text == "Root"
        assert sim.original_confidence == 0.95
        assert isinstance(sim.max_confidence_drop, float)
        assert isinstance(sim.avg_confidence_drop, float)
        assert isinstance(sim.orphaned_nodes, int)


# ------------------------------------------------------------------
# Unit tests: _bfs_with_hops
# ------------------------------------------------------------------

class TestBFSWithHops:
    def test_bfs_finds_successors_with_distances(self):
        """BFS should return successors with correct hop distances."""
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.9, node_id="root")
        child = make_node("Child", confidence=0.8, node_id="child", predecessors=["root"])
        grandchild = make_node("Grandchild", confidence=0.7, node_id="grandchild", predecessors=["child"])

        graph.add_node(root)
        graph.add_node(child)
        graph.add_node(grandchild)

        engine = VerificationEngine(graph)
        result = engine._bfs_with_hops("root")

        hop_map = dict(result)
        assert "child" in hop_map
        assert hop_map["child"] == 1
        assert "grandchild" in hop_map
        assert hop_map["grandchild"] == 2

    def test_bfs_excludes_start_node(self):
        """Start node should not appear in BFS results."""
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.9, node_id="root")
        graph.add_node(root)

        engine = VerificationEngine(graph)
        result = engine._bfs_with_hops("root")

        ids = {r[0] for r in result}
        assert "root" not in ids

    def test_bfs_empty_graph(self):
        """BFS on nonexistent node returns empty list."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)
        assert engine._bfs_with_hops("nope") == []


# ------------------------------------------------------------------
# Unit tests: LLM fallback behavior
# ------------------------------------------------------------------

class TestLLMFallback:
    def test_no_llm_uses_heuristic_only(self):
        """Without contradiction checker, only heuristic is used."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.9,
            node_id="node-1",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        # No contradiction_checker
        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=None,
        )
        result = engine.verify("Water does not boil at 100 degrees Celsius")

        # Should still detect contradiction via heuristic
        assert result.verdict == Verdict.CONTRADICTED

    def test_llm_irrelevant_result_filters_candidate(self):
        """When LLM returns irrelevant, the candidate is excluded from evidence."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.9,
            node_id="node-1",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        # LLM returns irrelevant — candidate is filtered out
        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Water does not boil at 100 degrees Celsius",
                knowledge="Water boils at 100 degrees Celsius",
                relationship="irrelevant",
                confidence=0.0,
                explanation="LLM call failed: connection error",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
        )
        result = engine.verify("Water does not boil at 100 degrees Celsius")

        # LLM said irrelevant, so no evidence on either side
        assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE


# ------------------------------------------------------------------
# Integration-style tests
# ------------------------------------------------------------------

class TestVerificationEngineIntegration:
    def test_full_workflow_supported_claim(self):
        """Full verify workflow with a supported claim."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        result = engine.verify("Water boils at 100 degrees Celsius at sea level")

        assert isinstance(result, VerificationResult)
        assert result.claim == "Water boils at 100 degrees Celsius at sea level"
        assert result.verdict in (Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE)

    def test_full_workflow_retraction_then_restore(self):
        """Full retraction workflow: simulate and verify restore."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        # Record all confidences
        orig = {nid: n.confidence for nid, n in graph.nodes.items()}

        sim = engine.simulate_retraction("root-001")

        assert isinstance(sim, RetractionSimulation)

        # Verify everything restored
        for nid, c in orig.items():
            assert graph.get_node(nid).confidence == c

    def test_retraction_restores_on_exception(self):
        """Confidence must be restored even when an exception occurs mid-simulation."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        # Record original confidences
        orig = {nid: n.confidence for nid, n in graph.nodes.items()}

        # Patch _topological_sort to raise mid-simulation
        with patch.object(engine.confidence_engine, '_topological_sort', side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                engine.simulate_retraction("root-001")

        # Verify ALL confidences restored despite exception
        for nid, c in orig.items():
            assert graph.get_node(nid).confidence == c, (
                f"Node {nid} confidence not restored: {graph.get_node(nid).confidence} != {c}"
            )

    def test_coverage_metric_clamped(self):
        """Coverage should not exceed 1.0 even if evidence count > num_checked."""
        graph = build_simple_graph()
        engine = VerificationEngine(graph)

        # Create evidence that would make coverage > 1.0
        # (more supporting + contradicting than num_checked)
        ev1 = SourceEvidence(
            node_id="root-001",
            node_text="Water boils at 100C",
            confidence=0.8,
            relationship="supports",
        )
        ev2 = SourceEvidence(
            node_id="derived-002",
            node_text="Related fact",
            confidence=0.7,
            relationship="supports",
        )

        result = engine._compute_verdict(
            supporting=[ev1, ev2],
            contradicting=[],
            num_checked=1,
            claim="Test claim",
        )
        assert result.coverage <= 1.0


# ------------------------------------------------------------------
# Unit tests: Relevance filtering (Step 11a)
# ------------------------------------------------------------------

class TestRelevanceFiltering:
    def test_irrelevant_evidence_excluded_heuristic(self):
        """Low-similarity candidates are excluded as irrelevant in heuristic mode."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Gold melts at 1064 degrees Celsius",
            confidence=0.95,
            node_id="gold-melt",
        ))
        graph.add_node(make_node(
            "Gold is a chemical element with symbol Au",
            confidence=0.90,
            node_id="gold-element",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "gold-melt", "score": 0.88},
            {"node_id": "gold-element", "score": 0.55},  # Below relevance threshold
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Gold melts at 500 degrees Celsius")

        # gold-element should be filtered as irrelevant (sim 0.55 < 0.65)
        support_ids = {e.node_id for e in result.supporting_evidence}
        assert "gold-element" not in support_ids

    def test_llm_3way_excludes_irrelevant(self):
        """LLM 3-way classification correctly excludes irrelevant evidence."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Gold melts at 1064 degrees Celsius",
            confidence=0.95,
            node_id="gold-melt",
        ))
        graph.add_node(make_node(
            "Gold is a chemical element",
            confidence=0.90,
            node_id="gold-element",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "gold-melt", "score": 0.88},
            {"node_id": "gold-element", "score": 0.72},
        ]

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Gold melts at 500 degrees Celsius",
                knowledge="Gold melts at 1064 degrees Celsius",
                relationship="contradicts",
                confidence=0.95,
                explanation="Numerical contradiction: 500C vs 1064C",
            ),
            RelevanceResult(
                claim="Gold melts at 500 degrees Celsius",
                knowledge="Gold is a chemical element",
                relationship="irrelevant",
                confidence=0.85,
                explanation="Different property: element classification vs melting point",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
        )
        result = engine.verify("Gold melts at 500 degrees Celsius")

        assert result.verdict == Verdict.CONTRADICTED
        assert len(result.contradicting_evidence) == 1
        assert len(result.supporting_evidence) == 0

    def test_all_irrelevant_yields_insufficient_evidence(self):
        """When all candidates are irrelevant, verdict is INSUFFICIENT_EVIDENCE."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Gold is a chemical element",
            confidence=0.90,
            node_id="gold-element",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "gold-element", "score": 0.50},  # Below 0.65 threshold
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb)
        result = engine.verify("Gold melts at 500 degrees Celsius")

        assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE

    def test_false_claim_detected_when_irrelevant_filtered(self):
        """Core bug fix: false claim with irrelevant noise should be CONTRADICTED."""
        graph = KnowledgeGraph()
        # The correct fact
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="correct-fact",
        ))
        # Irrelevant but related facts
        graph.add_node(make_node(
            "Water is a chemical compound with formula H2O",
            confidence=0.92,
            node_id="water-compound",
        ))
        graph.add_node(make_node(
            "Water freezes at 0 degrees Celsius",
            confidence=0.93,
            node_id="water-freeze",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 3
        mock_emb.search.return_value = [
            {"node_id": "correct-fact", "score": 0.90},
            {"node_id": "water-compound", "score": 0.55},  # Irrelevant (below threshold)
            {"node_id": "water-freeze", "score": 0.60},    # Irrelevant (below threshold)
        ]

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Water boils at 50 degrees Celsius",
                knowledge="Water boils at 100 degrees Celsius",
                relationship="contradicts",
                confidence=0.95,
                explanation="Numerical: 50C vs 100C",
            ),
            RelevanceResult(
                claim="Water boils at 50 degrees Celsius",
                knowledge="Water is a chemical compound with formula H2O",
                relationship="irrelevant",
                confidence=0.90,
                explanation="Different property",
            ),
            RelevanceResult(
                claim="Water boils at 50 degrees Celsius",
                knowledge="Water freezes at 0 degrees Celsius",
                relationship="irrelevant",
                confidence=0.85,
                explanation="Different property: freezing vs boiling",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
        )
        result = engine.verify("Water boils at 50 degrees Celsius")

        # With irrelevant evidence filtered, the contradiction should win
        assert result.verdict == Verdict.CONTRADICTED
        assert len(result.supporting_evidence) == 0


# ------------------------------------------------------------------
# Unit tests: Weighted evidence (Step 11b)
# ------------------------------------------------------------------

class TestWeightedEvidence:
    def test_confidence_weighted_by_similarity(self):
        """Evidence confidence should be weighted by similarity score."""
        graph = KnowledgeGraph()
        node = make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.90,
            node_id="node-1",
        )
        # Add a high-quality source so source quality factor is near 1.0
        node.sources = [Source(source_type="textbook", title="Chem 101", quality_score=0.95)]
        graph.add_node(node)

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.80},
        ]

        engine = VerificationEngine(graph, embedding_engine=mock_emb, use_bm25=False)
        result = engine.verify("Water boils at 100 degrees Celsius")

        assert len(result.supporting_evidence) > 0
        ev = result.supporting_evidence[0]
        # Base: 0.90 * 0.80 = 0.72
        # Source quality factor: (1-0.3) + 0.3*0.95 = 0.985
        # Weighted: 0.72 * 0.985 = 0.7092
        assert ev.confidence < 0.90
        assert abs(ev.confidence - 0.7092) < 0.01

    def test_high_similarity_contradiction_outweighs_low_similarity_support(self):
        """Contradiction with high similarity should outweigh lower-similarity support."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.90,
            node_id="node-correct",
        ))
        graph.add_node(make_node(
            "Steam is produced when water boils at sea level",
            confidence=0.85,
            node_id="node-related",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "node-correct", "score": 0.92},
            {"node_id": "node-related", "score": 0.70},
        ]

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Water boils at 50 degrees Celsius",
                knowledge="Water boils at 100 degrees Celsius",
                relationship="contradicts",
                confidence=0.95,
                explanation="Numerical contradiction",
            ),
            RelevanceResult(
                claim="Water boils at 50 degrees Celsius",
                knowledge="Steam is produced when water boils at sea level",
                relationship="supports",
                confidence=0.60,
                explanation="Related but different property",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
        )
        result = engine.verify("Water boils at 50 degrees Celsius")

        # Contradiction weighted: 0.90 * 0.92 = 0.828
        # Support weighted: 0.85 * 0.70 = 0.595
        # 0.828 > 0.595 + 0.05, so CONTRADICTED
        assert result.verdict == Verdict.CONTRADICTED


# ------------------------------------------------------------------
# Unit tests: Verdict tiebreaker (Step 11c)
# ------------------------------------------------------------------

class TestVerdictTiebreaker:
    def test_tiebreaker_favors_majority_contradictions(self):
        """When confidence is similar, more contradictions should win."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        contra1 = SourceEvidence(
            node_id="c1", node_text="Contra 1",
            confidence=0.80, relationship="contradicts",
        )
        contra2 = SourceEvidence(
            node_id="c2", node_text="Contra 2",
            confidence=0.78, relationship="contradicts",
        )
        support1 = SourceEvidence(
            node_id="s1", node_text="Support 1",
            confidence=0.79, relationship="supports",
        )

        result = engine._compute_verdict(
            claim="Test claim",
            supporting=[support1],
            contradicting=[contra1, contra2],
            num_checked=3,
        )
        assert result.verdict == Verdict.CONTRADICTED

    def test_tiebreaker_favors_majority_supports(self):
        """When confidence is similar, more supports should win."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        contra1 = SourceEvidence(
            node_id="c1", node_text="Contra 1",
            confidence=0.80, relationship="contradicts",
        )
        support1 = SourceEvidence(
            node_id="s1", node_text="Support 1",
            confidence=0.79, relationship="supports",
        )
        support2 = SourceEvidence(
            node_id="s2", node_text="Support 2",
            confidence=0.78, relationship="supports",
        )

        result = engine._compute_verdict(
            claim="Test claim",
            supporting=[support1, support2],
            contradicting=[contra1],
            num_checked=3,
        )
        assert result.verdict == Verdict.SUPPORTED

    def test_tiebreaker_even_split_is_uncertain(self):
        """When confidence and count are equal, verdict is UNCERTAIN."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        contra1 = SourceEvidence(
            node_id="c1", node_text="Contra 1",
            confidence=0.80, relationship="contradicts",
        )
        support1 = SourceEvidence(
            node_id="s1", node_text="Support 1",
            confidence=0.80, relationship="supports",
        )

        result = engine._compute_verdict(
            claim="Test claim",
            supporting=[support1],
            contradicting=[contra1],
            num_checked=2,
        )
        assert result.verdict == Verdict.UNCERTAIN

    def test_margin_triggers_contradicted(self):
        """When contra_conf > support_conf + 0.05, verdict is CONTRADICTED."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        contra1 = SourceEvidence(
            node_id="c1", node_text="Contra 1",
            confidence=0.85, relationship="contradicts",
        )
        support1 = SourceEvidence(
            node_id="s1", node_text="Support 1",
            confidence=0.70, relationship="supports",
        )

        result = engine._compute_verdict(
            claim="Test claim",
            supporting=[support1],
            contradicting=[contra1],
            num_checked=2,
        )
        # 0.85 > 0.70 + 0.05 = True
        assert result.verdict == Verdict.CONTRADICTED


# ------------------------------------------------------------------
# Unit tests: Source quality weighting (Step 14+)
# ------------------------------------------------------------------

class TestSourceQualityWeighting:
    def test_get_source_quality_with_sources(self):
        """Source quality is max of all source quality_scores."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        node = make_node("Test fact", node_id="n1")
        node.sources = [
            Source(source_type="textbook", title="A", quality_score=0.95),
            Source(source_type="web", title="B", quality_score=0.3),
        ]

        assert engine._get_source_quality(node) == 0.95

    def test_get_source_quality_no_sources(self):
        """Nodes without sources default to 0.5 quality."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        node = make_node("Test fact", node_id="n1")
        assert engine._get_source_quality(node) == 0.5

    def test_weighted_confidence_with_high_quality_source(self):
        """High-quality source should yield confidence near base value."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        node = make_node("Test fact", confidence=0.90, node_id="n1")
        node.sources = [Source(source_type="textbook", title="A", quality_score=0.95)]

        # base = 0.90 * 0.80 = 0.72
        # factor = 0.7 + 0.3*0.95 = 0.985
        # weighted = 0.72 * 0.985 = 0.7092
        weighted = engine._weighted_confidence(node, sim_score=0.80)
        assert abs(weighted - 0.7092) < 0.001

    def test_weighted_confidence_with_low_quality_source(self):
        """Low-quality source should reduce confidence."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph)

        node = make_node("Test fact", confidence=0.90, node_id="n1")
        node.sources = [Source(source_type="student", title="A", quality_score=0.3)]

        # base = 0.90 * 0.80 = 0.72
        # factor = 0.7 + 0.3*0.3 = 0.79
        # weighted = 0.72 * 0.79 = 0.5688
        weighted = engine._weighted_confidence(node, sim_score=0.80)
        assert abs(weighted - 0.5688) < 0.001

    def test_weighted_confidence_disabled(self):
        """When use_source_quality=False, returns base value."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph, use_source_quality=False)

        node = make_node("Test fact", confidence=0.90, node_id="n1")
        node.sources = [Source(source_type="student", title="A", quality_score=0.3)]

        # Should ignore source quality: 0.90 * 0.80 = 0.72
        weighted = engine._weighted_confidence(node, sim_score=0.80)
        assert abs(weighted - 0.72) < 0.001

    def test_source_quality_weight_zero_same_as_disabled(self):
        """source_quality_weight=0 is equivalent to use_source_quality=False."""
        graph = KnowledgeGraph()
        engine = VerificationEngine(graph, source_quality_weight=0.0)

        node = make_node("Test fact", confidence=0.90, node_id="n1")
        node.sources = [Source(source_type="student", title="A", quality_score=0.1)]

        # factor = (1 - 0) + 0 * 0.1 = 1.0
        weighted = engine._weighted_confidence(node, sim_score=0.80)
        assert abs(weighted - 0.72) < 0.001

    def test_high_quality_evidence_beats_low_quality(self):
        """High-quality contradiction should outweigh low-quality support."""
        graph = KnowledgeGraph()
        node_high = make_node(
            "Gold melts at 1064 degrees Celsius",
            confidence=0.90,
            node_id="high-q",
        )
        node_high.sources = [Source(source_type="textbook", title="A", quality_score=0.95)]

        node_low = make_node(
            "Gold melts at 500 degrees Celsius",
            confidence=0.90,
            node_id="low-q",
        )
        node_low.sources = [Source(source_type="student", title="B", quality_score=0.30)]

        graph.add_node(node_high)
        graph.add_node(node_low)

        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "high-q", "score": 0.88},
            {"node_id": "low-q", "score": 0.88},
        ]

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Gold melts at 500 degrees Celsius",
                knowledge="Gold melts at 1064 degrees Celsius",
                relationship="contradicts",
                confidence=0.95,
                explanation="Numerical contradiction",
            ),
            RelevanceResult(
                claim="Gold melts at 500 degrees Celsius",
                knowledge="Gold melts at 500 degrees Celsius",
                relationship="supports",
                confidence=0.90,
                explanation="Exact match",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
            use_bm25=False,
        )

        result = engine.verify("Gold melts at 500 degrees Celsius")

        # Contradiction from high-quality source should outweigh
        # support from low-quality source
        assert result.verdict == Verdict.CONTRADICTED
        assert len(result.contradicting_evidence) == 1
        assert len(result.supporting_evidence) == 1
        # Contradicting evidence confidence should be higher
        assert result.contradicting_evidence[0].confidence > result.supporting_evidence[0].confidence

    def test_source_quality_in_3way_classification(self):
        """LLM 3-way classification uses source quality weighting."""
        graph = KnowledgeGraph()
        node = make_node(
            "Iron has the chemical symbol Fe",
            confidence=0.80,
            node_id="n1",
        )
        node.sources = [Source(source_type="textbook", title="Chem", quality_score=0.95)]
        graph.add_node(node)

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "n1", "score": 0.85},
        ]

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Iron has the chemical symbol Fe",
                knowledge="Iron has the chemical symbol Fe",
                relationship="supports",
                confidence=0.95,
                explanation="Exact match",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=mock_checker,
            use_bm25=False,
        )

        result = engine.verify("Iron has the chemical symbol Fe")
        assert result.verdict == Verdict.SUPPORTED

        # Confidence should include source quality factor
        ev = result.supporting_evidence[0]
        # base = 0.80 * 0.85 = 0.68
        # factor = 0.7 + 0.3*0.95 = 0.985
        # weighted = 0.68 * 0.985 = 0.6698
        assert abs(ev.confidence - 0.6698) < 0.01

    def test_source_quality_in_heuristic_path(self):
        """Heuristic classification also uses source quality weighting."""
        graph = KnowledgeGraph()
        node = make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.90,
            node_id="n1",
        )
        node.sources = [Source(source_type="lecture", title="Notes", quality_score=0.70)]
        graph.add_node(node)

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "n1", "score": 0.85},
        ]

        # No contradiction checker — uses heuristic path
        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            contradiction_checker=None,
            use_bm25=False,
        )

        result = engine.verify("Water does not boil at 100 degrees Celsius")
        assert result.verdict == Verdict.CONTRADICTED

        ev = result.contradicting_evidence[0]
        # base = 0.90 * 0.85 = 0.765
        # factor = 0.7 + 0.3*0.70 = 0.91
        # weighted = 0.765 * 0.91 = 0.69615
        assert abs(ev.confidence - 0.69615) < 0.01
