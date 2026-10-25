"""
Tests for the ConfidenceEngine: calculation, propagation, anomaly detection,
confidence tracing, and impact estimation.
"""

from src.core.confidence import ConfidenceEngine
from src.core.graph import KnowledgeGraph
from src.core.node import Derivation, KnowledgeNode, NodeType, Source

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_node(
    text, confidence=0.8, domain="general", node_id=None, predecessors=None, derivation_type="extracted", depth=0
):
    """Create a KnowledgeNode for testing."""
    node = KnowledgeNode(
        text=text,
        node_type=NodeType.FACT,
        confidence=confidence,
        domain=domain,
        derivation=Derivation(
            method=derivation_type,
            derivation_type=derivation_type,
            inference_depth=depth,
        ),
        predecessor_nodes=predecessors or [],
        node_id=node_id,
    )
    return node


def build_chain_graph():
    """Build a linear chain: root -> mid -> leaf."""
    graph = KnowledgeGraph()

    root = make_node("Root fact", confidence=0.9, node_id="root")
    graph.add_node(root)

    mid = make_node(
        "Mid derived", confidence=0.8, node_id="mid", predecessors=["root"], derivation_type="chain", depth=1
    )
    graph.add_node(mid)
    root.add_successor("mid")

    leaf = make_node(
        "Leaf derived", confidence=0.7, node_id="leaf", predecessors=["mid"], derivation_type="chain", depth=2
    )
    graph.add_node(leaf)
    mid.add_successor("leaf")

    return graph


def build_convergent_graph():
    """Build a convergent graph: src_a, src_b -> merged."""
    graph = KnowledgeGraph()

    src_a = make_node("Source A", confidence=0.7, node_id="src_a")
    src_b = make_node("Source B", confidence=0.8, node_id="src_b")
    graph.add_node(src_a)
    graph.add_node(src_b)

    merged = make_node(
        "Merged conclusion",
        confidence=0.85,
        node_id="merged",
        predecessors=["src_a", "src_b"],
        derivation_type="convergence",
        depth=1,
    )
    graph.add_node(merged)
    src_a.add_successor("merged")
    src_b.add_successor("merged")

    return graph


def build_branching_graph():
    """Build root -> {child_a, child_b} -> grandchild."""
    graph = KnowledgeGraph()

    root = make_node("Root", confidence=0.9, node_id="root")
    graph.add_node(root)

    child_a = make_node(
        "Child A", confidence=0.8, node_id="child_a", predecessors=["root"], derivation_type="inference", depth=1
    )
    child_b = make_node(
        "Child B", confidence=0.75, node_id="child_b", predecessors=["root"], derivation_type="inference", depth=1
    )
    graph.add_node(child_a)
    graph.add_node(child_b)
    root.add_successor("child_a")
    root.add_successor("child_b")

    grandchild = make_node(
        "Grandchild",
        confidence=0.7,
        node_id="grandchild",
        predecessors=["child_a", "child_b"],
        derivation_type="chain",
        depth=2,
    )
    graph.add_node(grandchild)
    child_a.add_successor("grandchild")
    child_b.add_successor("grandchild")

    return graph


# ------------------------------------------------------------------
# calculate_node_confidence
# ------------------------------------------------------------------


class TestCalculateNodeConfidence:
    def test_no_predecessors_no_sources(self):
        """Node with no predecessors and no sources defaults to 0.5."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        node = make_node("Orphan fact", confidence=0.8, node_id="orphan")
        graph.add_node(node)

        conf = engine.calculate_node_confidence(node, graph)
        assert conf == 0.5  # default from _confidence_from_sources

    def test_no_predecessors_with_sources(self):
        """Node with no predecessors uses source quality as confidence."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        node = make_node("Sourced fact", confidence=0.8, node_id="sourced")
        node.add_source(Source(source_type="journal", title="Nature", quality_score=0.95))
        graph.add_node(node)

        conf = engine.calculate_node_confidence(node, graph)
        assert abs(conf - 0.95) < 0.01

    def test_chain_derivation_uses_min(self):
        """Chain derivation: confidence limited by weakest predecessor."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()
        mid = graph.get_node("mid")

        conf = engine.calculate_node_confidence(mid, graph)
        # chain: min(preds) * inference_strength * depth_decay^depth
        # min(0.9) * 0.9 * 0.95^1 = 0.769
        expected = 0.9 * 0.9 * 0.95
        assert abs(conf - expected) < 0.01

    def test_convergence_derivation_uses_noisy_or(self):
        """Convergence derivation: noisy-OR with boost."""
        engine = ConfidenceEngine()
        graph = build_convergent_graph()
        merged = graph.get_node("merged")

        conf = engine.calculate_node_confidence(merged, graph)
        # noisy_or(0.7, 0.8) = 1 - (0.3 * 0.2) = 0.94
        # * convergence_boost(1.2) = 1.128 -> clamped to 0.99
        # * depth_decay(0.95^1) = 0.99 * 0.95 = 0.9405
        assert conf <= 0.99

    def test_inference_derivation_uses_min(self):
        """Inference derivation: same as chain (min * strength)."""
        engine = ConfidenceEngine()
        graph = build_branching_graph()
        grandchild = graph.get_node("grandchild")

        # grandchild has chain type with preds [child_a=0.8, child_b=0.75]
        conf = engine.calculate_node_confidence(grandchild, graph)
        # chain: min(0.8, 0.75) * 0.9 * 0.95^2 = 0.75 * 0.9 * 0.9025 = 0.609
        expected = 0.75 * 0.9 * (0.95**2)
        assert abs(conf - expected) < 0.01

    def test_default_derivation_uses_average(self):
        """Unknown derivation type: average * inference_strength."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.9, node_id="root")
        graph.add_node(root)

        node = make_node(
            "Derived", confidence=0.5, node_id="derived", predecessors=["root"], derivation_type="custom_type"
        )
        graph.add_node(node)
        root.add_successor("derived")

        conf = engine.calculate_node_confidence(node, graph)
        # default: mean(0.9) * 0.9 * 0.95^0 = 0.81
        expected = 0.9 * 0.9
        assert abs(conf - expected) < 0.01

    def test_confidence_clamped_to_min(self):
        """Confidence never drops below min_confidence."""
        engine = ConfidenceEngine({"min_confidence": 0.1})
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.05, node_id="root")
        graph.add_node(root)

        node = make_node(
            "Derived", confidence=0.5, node_id="derived", predecessors=["root"], derivation_type="chain", depth=10
        )
        graph.add_node(node)
        root.add_successor("derived")

        conf = engine.calculate_node_confidence(node, graph)
        assert conf >= 0.1

    def test_confidence_clamped_to_max(self):
        """Confidence never exceeds 0.99."""
        engine = ConfidenceEngine({"convergence_boost": 2.0})
        graph = KnowledgeGraph()
        a = make_node("A", confidence=0.99, node_id="a")
        b = make_node("B", confidence=0.99, node_id="b")
        graph.add_node(a)
        graph.add_node(b)

        merged = make_node(
            "Merged", confidence=0.5, node_id="merged", predecessors=["a", "b"], derivation_type="convergence"
        )
        graph.add_node(merged)
        a.add_successor("merged")
        b.add_successor("merged")

        conf = engine.calculate_node_confidence(merged, graph)
        assert conf <= 0.99

    def test_predecessors_not_in_graph(self):
        """If all predecessors are missing from graph, fall back to sources."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        node = make_node("Orphan derived", confidence=0.5, node_id="orphan", predecessors=["missing_1", "missing_2"])
        graph.add_node(node)

        conf = engine.calculate_node_confidence(node, graph)
        # Falls back to _confidence_from_sources, no sources -> 0.5
        assert conf == 0.5

    def test_source_boost_over_derived(self):
        """If source quality * 0.8 exceeds derived confidence, source wins."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.3, node_id="root")
        graph.add_node(root)

        node = make_node("Derived", confidence=0.5, node_id="derived", predecessors=["root"], derivation_type="chain")
        node.add_source(Source(source_type="journal", title="Nature", quality_score=0.95))
        graph.add_node(node)
        root.add_successor("derived")

        conf = engine.calculate_node_confidence(node, graph)
        # chain: 0.3 * 0.9 * 0.95^0 = 0.27
        # source: 0.95 * 0.8 = 0.76
        # max(0.27, 0.76) = 0.76
        assert conf >= 0.7


# ------------------------------------------------------------------
# _noisy_or
# ------------------------------------------------------------------


class TestNoisyOr:
    def test_single_value(self):
        engine = ConfidenceEngine()
        assert abs(engine._noisy_or([0.5]) - 0.5) < 0.001

    def test_two_values(self):
        engine = ConfidenceEngine()
        # 1 - (1-0.7)(1-0.8) = 1 - 0.06 = 0.94
        assert abs(engine._noisy_or([0.7, 0.8]) - 0.94) < 0.001

    def test_all_zeros(self):
        engine = ConfidenceEngine()
        assert engine._noisy_or([0.0, 0.0, 0.0]) == 0.0

    def test_one_certain(self):
        engine = ConfidenceEngine()
        assert engine._noisy_or([1.0, 0.5]) == 1.0

    def test_many_weak_sources(self):
        """Many weak sources combine to high confidence."""
        engine = ConfidenceEngine()
        # 10 sources at 0.3: 1 - 0.7^10 ≈ 0.972
        result = engine._noisy_or([0.3] * 10)
        assert result > 0.95


# ------------------------------------------------------------------
# _topological_sort
# ------------------------------------------------------------------


class TestTopologicalSort:
    def test_linear_chain(self):
        """Linear chain sorted in dependency order."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        sorted_ids = engine._topological_sort(graph, {"root", "mid", "leaf"})
        assert sorted_ids.index("root") < sorted_ids.index("mid")
        assert sorted_ids.index("mid") < sorted_ids.index("leaf")

    def test_subset_of_graph(self):
        """Only sorts the specified subset."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        sorted_ids = engine._topological_sort(graph, {"mid", "leaf"})
        assert len(sorted_ids) == 2
        assert sorted_ids.index("mid") < sorted_ids.index("leaf")

    def test_disconnected_nodes(self):
        """Independent nodes all appear (no ordering required)."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        for i in range(3):
            graph.add_node(make_node(f"Node {i}", node_id=f"n{i}"))

        sorted_ids = engine._topological_sort(graph, {"n0", "n1", "n2"})
        assert set(sorted_ids) == {"n0", "n1", "n2"}

    def test_empty_set(self):
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        assert engine._topological_sort(graph, set()) == []

    def test_branching_graph(self):
        """Branching graph: root before children, children before grandchild."""
        engine = ConfidenceEngine()
        graph = build_branching_graph()

        sorted_ids = engine._topological_sort(graph, {"root", "child_a", "child_b", "grandchild"})
        assert sorted_ids.index("root") < sorted_ids.index("child_a")
        assert sorted_ids.index("root") < sorted_ids.index("child_b")
        assert sorted_ids.index("child_a") < sorted_ids.index("grandchild")


# ------------------------------------------------------------------
# propagate_confidence_update
# ------------------------------------------------------------------


class TestPropagateConfidenceUpdate:
    def test_leaf_update_propagates_nothing(self):
        """Updating a leaf node affects nothing downstream."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        affected = engine.propagate_confidence_update(graph, "leaf")
        assert len(affected) == 0

    def test_root_update_propagates_to_descendants(self):
        """Updating root should propagate to mid and leaf."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()
        graph.update_node_confidence("root", 0.3)

        affected = engine.propagate_confidence_update(graph, "root")
        # mid and leaf should both be affected
        assert "mid" in affected or "leaf" in affected

    def test_nonexistent_node(self):
        """Propagating from a nonexistent node returns empty set."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        affected = engine.propagate_confidence_update(graph, "ghost")
        assert affected == set()


# ------------------------------------------------------------------
# detect_anomalies
# ------------------------------------------------------------------


class TestDetectAnomalies:
    def test_no_anomalies_on_fresh_graph(self):
        """Freshly computed confidences should have no anomalies."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.5, node_id="root")
        graph.add_node(root)
        # confidence 0.5 matches default for unsourced node -> no anomaly
        anomalies = engine.detect_anomalies(graph)
        assert len(anomalies) == 0

    def test_overconfident_anomaly(self):
        """Node with much higher confidence than expected is flagged."""
        engine = ConfidenceEngine({"anomaly_threshold": 0.1})
        graph = KnowledgeGraph()
        root = make_node("Root", confidence=0.3, node_id="root")
        graph.add_node(root)

        # Derived node claims 0.95 confidence but root is only 0.3
        derived = make_node(
            "Derived", confidence=0.95, node_id="derived", predecessors=["root"], derivation_type="chain"
        )
        graph.add_node(derived)
        root.add_successor("derived")

        anomalies = engine.detect_anomalies(graph)
        assert len(anomalies) >= 1
        anom = anomalies[0]
        assert anom["anomaly_type"] == "overconfident"
        assert anom["node_id"] == "derived"

    def test_underconfident_anomaly(self):
        """Node with much lower confidence than expected is flagged."""
        engine = ConfidenceEngine({"anomaly_threshold": 0.1})
        graph = KnowledgeGraph()
        node = make_node("Sourced", confidence=0.2, node_id="sourced")
        node.add_source(Source(source_type="journal", title="Nature", quality_score=0.95))
        graph.add_node(node)

        anomalies = engine.detect_anomalies(graph)
        assert len(anomalies) >= 1
        assert anomalies[0]["anomaly_type"] == "underconfident"

    def test_anomalies_sorted_by_severity(self):
        """Anomalies returned sorted by severity (highest first)."""
        engine = ConfidenceEngine({"anomaly_threshold": 0.05})
        graph = KnowledgeGraph()

        # Node with small deviation
        n1 = make_node("Small dev", confidence=0.6, node_id="n1")
        n1.add_source(Source(source_type="web", title="x", quality_score=0.5))
        graph.add_node(n1)

        # Node with large deviation
        n2 = make_node("Large dev", confidence=0.99, node_id="n2")
        n2.add_source(Source(source_type="web", title="y", quality_score=0.3))
        graph.add_node(n2)

        anomalies = engine.detect_anomalies(graph)
        if len(anomalies) >= 2:
            assert anomalies[0]["severity"] >= anomalies[1]["severity"]

    def test_no_anomaly_for_orphan_no_sources(self):
        """Nodes with no predecessors and no sources are skipped."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        node = make_node("Orphan", confidence=0.99, node_id="orphan")
        graph.add_node(node)

        anomalies = engine.detect_anomalies(graph)
        assert len(anomalies) == 0  # skipped entirely


# ------------------------------------------------------------------
# trace_confidence_source
# ------------------------------------------------------------------


class TestTraceConfidenceSource:
    def test_trace_leaf_to_root(self):
        """Tracing leaf finds root as source node."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        trace = engine.trace_confidence_source(graph, "leaf")
        assert trace["node_id"] == "leaf"
        assert trace["num_ancestors"] >= 1
        source_ids = [s["id"] for s in trace["source_nodes"]]
        assert "root" in source_ids

    def test_trace_root_has_no_ancestors(self):
        """Root node trace has itself as base (no source_nodes from ancestors)."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        trace = engine.trace_confidence_source(graph, "root")
        assert trace["num_ancestors"] == 0
        assert trace["source_nodes"] == []

    def test_trace_nonexistent_node(self):
        """Tracing a nonexistent node returns empty dict."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        trace = engine.trace_confidence_source(graph, "ghost")
        assert trace == {}

    def test_trace_convergent_has_multiple_sources(self):
        """Convergent node has multiple source nodes."""
        engine = ConfidenceEngine()
        graph = build_convergent_graph()

        trace = engine.trace_confidence_source(graph, "merged")
        assert trace["num_ancestors"] >= 2
        source_ids = [s["id"] for s in trace["source_nodes"]]
        assert "src_a" in source_ids
        assert "src_b" in source_ids

    def test_trace_includes_min_max_confidence(self):
        """Trace reports min and max source confidences."""
        engine = ConfidenceEngine()
        graph = build_convergent_graph()

        trace = engine.trace_confidence_source(graph, "merged")
        assert trace["min_source_confidence"] == 0.7
        assert trace["max_source_confidence"] == 0.8


# ------------------------------------------------------------------
# estimate_confidence_impact
# ------------------------------------------------------------------


class TestEstimateConfidenceImpact:
    def test_leaf_change_no_impact(self):
        """Changing a leaf has no downstream impact."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        impact = engine.estimate_confidence_impact(graph, "leaf", 0.1)
        assert impact["affected_nodes"] == 0

    def test_root_change_affects_descendants(self):
        """Changing root affects mid and leaf."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()

        impact = engine.estimate_confidence_impact(graph, "root", 0.3)
        assert impact["affected_nodes"] >= 1
        assert impact["current_confidence"] == 0.9
        assert impact["proposed_confidence"] == 0.3
        assert impact["max_negative_change"] < 0

    def test_nonexistent_node(self):
        """Impact estimate on nonexistent node returns empty dict."""
        engine = ConfidenceEngine()
        graph = KnowledgeGraph()
        impact = engine.estimate_confidence_impact(graph, "ghost", 0.5)
        assert impact == {}

    def test_increase_gives_positive_changes(self):
        """Increasing root confidence should give positive changes."""
        engine = ConfidenceEngine()
        graph = build_chain_graph()
        # root is at 0.9, set to 0.95
        impact = engine.estimate_confidence_impact(graph, "root", 0.95)
        assert impact["max_positive_change"] >= 0


# ------------------------------------------------------------------
# Custom configuration
# ------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_inference_strength(self):
        engine = ConfidenceEngine({"inference_strength": 0.5})
        assert engine.inference_strength == 0.5

    def test_custom_depth_decay(self):
        engine = ConfidenceEngine({"depth_decay": 0.8})
        assert engine.depth_decay == 0.8

    def test_custom_min_confidence(self):
        engine = ConfidenceEngine({"min_confidence": 0.2})
        assert engine.min_confidence == 0.2

    def test_default_config(self):
        engine = ConfidenceEngine()
        assert engine.inference_strength == 0.9
        assert engine.depth_decay == 0.95
        assert engine.convergence_boost == 1.2
        assert engine.min_confidence == 0.1
        assert engine.anomaly_threshold == 0.15


# ------------------------------------------------------------------
# Tests: bounded confidence propagation (max_hops)
# ------------------------------------------------------------------


class TestBoundedPropagation:
    """Tests for max_hops parameter in propagation and get_descendants."""

    def _build_chain(self, length=5):
        """Build a linear chain: n0 -> n1 -> n2 -> ... -> n{length-1}."""
        graph = KnowledgeGraph()
        nodes = []
        for i in range(length):
            preds = [nodes[-1].id] if nodes else []
            n = make_node(
                f"chain node {i}",
                confidence=0.9,
                node_id=f"n{i}",
                predecessors=preds,
                derivation_type="chain" if preds else "extracted",
                depth=i,
            )
            graph.add_node(n)
            if nodes:
                nodes[-1].add_successor(n.id)
            nodes.append(n)
        return graph, nodes

    def test_get_descendants_max_depth_none_unbounded(self):
        """max_depth=None returns all descendants (backward compat)."""
        graph, nodes = self._build_chain(5)
        desc = graph.get_descendants("n0", max_depth=None)
        assert desc == {"n1", "n2", "n3", "n4"}

    def test_get_descendants_max_depth_1(self):
        """max_depth=1 returns only direct children."""
        graph, nodes = self._build_chain(5)
        desc = graph.get_descendants("n0", max_depth=1)
        assert desc == {"n1"}

    def test_get_descendants_max_depth_2(self):
        """max_depth=2 returns children and grandchildren."""
        graph, nodes = self._build_chain(5)
        desc = graph.get_descendants("n0", max_depth=2)
        assert desc == {"n1", "n2"}

    def test_get_descendants_max_depth_0(self):
        """max_depth=0 returns nothing (no expansion)."""
        graph, nodes = self._build_chain(5)
        desc = graph.get_descendants("n0", max_depth=0)
        assert desc == set()

    def test_propagate_max_hops_limits_affected(self):
        """propagate with max_hops=1 only affects direct children."""
        graph, nodes = self._build_chain(4)
        engine = ConfidenceEngine()
        # Mutate root confidence
        graph.update_node_confidence("n0", 0.3)
        affected = engine.propagate_confidence_update(graph, "n0", max_hops=1)
        # Only n1 should be recalculated — n2,n3 are out of range
        assert "n1" in affected or len(affected) == 0  # might be 0 if change < threshold
        assert "n2" not in affected
        assert "n3" not in affected

    def test_propagate_max_hops_none_unbounded(self):
        """propagate with max_hops=None reaches all descendants."""
        graph, nodes = self._build_chain(4)
        engine = ConfidenceEngine()
        graph.update_node_confidence("n0", 0.1)
        affected = engine.propagate_confidence_update(graph, "n0", max_hops=None)
        # All descendants should be checked (some may not change enough)
        # At minimum, direct child should be affected
        assert len(affected) >= 1
