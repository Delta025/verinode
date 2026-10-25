"""
Integration tests for NeuroplasticityEngine — full maintenance cycles,
confidence propagation with real ConfidenceEngine, multi-cycle effects,
dry-run isolation, and graph serialization round-trips.

All tests are pure Python (no Ollama required).
"""

import copy
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, NodeStatus, Source
from src.core.confidence import ConfidenceEngine
from src.core.neuroplasticity import (
    NeuroplasticityEngine,
    MaintenanceReport,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_node(
    text,
    confidence=0.8,
    domain="chemistry",
    node_id=None,
    last_validated=None,
    access_count=0,
    sources=None,
    conflicts=None,
    predecessor_nodes=None,
):
    """Create a KnowledgeNode with configurable fields for testing."""
    node = KnowledgeNode(
        text=text,
        confidence=confidence,
        domain=domain,
        node_id=node_id,
        predecessor_nodes=predecessor_nodes,
    )
    if last_validated is not None:
        node.last_validated = last_validated
    if access_count > 0:
        node.access_count = access_count
    if sources:
        node.sources = sources
    if conflicts:
        node.conflicts_with = conflicts
        node.update_status()
    return node


def build_diverse_graph(now=None):
    """Build a graph with fresh, stale, popular, and conflicted nodes."""
    if now is None:
        now = datetime.now()

    graph = KnowledgeGraph()

    # Fresh, high-confidence (should not decay much)
    fresh = make_node(
        "Water boils at 100 degrees Celsius",
        confidence=0.95,
        node_id="fresh",
        last_validated=now - timedelta(days=2),
        access_count=10,
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.95)],
    )
    fresh.status = NodeStatus.ESTABLISHED

    # Stale node
    stale = make_node(
        "Gold melts at 1064 degrees Celsius",
        confidence=0.80,
        node_id="stale",
        last_validated=now - timedelta(days=120),
        access_count=1,
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.9)],
    )

    # Popular node (high access count)
    popular = make_node(
        "Iron is a metal",
        confidence=0.70,
        node_id="popular",
        last_validated=now - timedelta(days=10),
        access_count=25,
        sources=[Source(source_type="encyclopedia", title="Science Enc", quality_score=0.85)],
    )

    # Conflicted low-confidence node
    conflicted = make_node(
        "Iron melts at 100 degrees",
        confidence=0.25,
        node_id="conflicted",
        last_validated=now - timedelta(days=60),
        conflicts=["stale"],
        sources=[Source(source_type="web", title="BadSite", quality_score=0.2)],
    )

    graph.add_node(fresh)
    graph.add_node(stale)
    graph.add_node(popular)
    graph.add_node(conflicted)
    return graph


# ------------------------------------------------------------------
# TestFullMaintenanceCycle
# ------------------------------------------------------------------

class TestFullMaintenanceCycle:
    """Test complete maintenance cycles on diverse graphs."""

    def test_full_cycle_all_phases(self):
        """A diverse graph exercises all 3 phases: decay, strengthen, flag."""
        now = datetime.now()
        graph = build_diverse_graph(now=now)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            strengthen_access_threshold=3,
            doubt_low_confidence_threshold=0.3,
            doubt_stale_days_threshold=90,
        )

        report = engine.run_maintenance(graph, now=now)

        # Should have examined all nodes
        assert report.nodes_examined == 4

        # Stale node should be decayed (120 days > grace period of 7)
        assert report.nodes_decayed >= 1
        assert report.total_decay_amount > 0

        # Popular node has 25 accesses > threshold 3 → should be strengthened
        assert report.nodes_strengthened >= 1
        assert report.total_strength_boost > 0

        # Conflicted node (confidence 0.25, < 0.3 threshold) → flagged
        assert report.nodes_flagged >= 1
        flag_ids = {f.node_id for f in report.flags}
        assert "conflicted" in flag_ids

    def test_compound_decay_over_two_cycles(self):
        """Running maintenance twice with time gap should compound decay."""
        now = datetime.now()

        # Build graph with a moderately old node
        graph = KnowledgeGraph()
        node = make_node(
            "Test fact",
            confidence=0.80,
            node_id="test",
            last_validated=now - timedelta(days=60),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
        )

        # First cycle at now
        report1 = engine.run_maintenance(graph, now=now)
        conf_after_first = graph.get_node("test").confidence
        assert conf_after_first < 0.80

        # Second cycle 60 days later
        now2 = now + timedelta(days=60)
        report2 = engine.run_maintenance(graph, now=now2)
        conf_after_second = graph.get_node("test").confidence

        # Compound decay: second cycle should reduce further
        assert conf_after_second < conf_after_first
        assert report2.nodes_decayed >= 1


# ------------------------------------------------------------------
# TestConfidencePropagationIntegration
# ------------------------------------------------------------------

class TestConfidencePropagationIntegration:
    """Test propagation with real ConfidenceEngine (not mocked)."""

    def test_decay_triggers_propagation_with_real_engine(self):
        """Decaying a root node should trigger propagation to descendants."""
        now = datetime.now()
        graph = KnowledgeGraph()

        # Root node (stale)
        root = make_node(
            "Base fact",
            confidence=0.85,
            node_id="root",
            last_validated=now - timedelta(days=100),
        )
        graph.add_node(root)

        # Child node depending on root
        child = KnowledgeNode(
            text="Derived from base",
            confidence=0.75,
            domain="chemistry",
            node_id="child",
            predecessor_nodes=["root"],
        )
        child.last_validated = now - timedelta(days=5)  # Not stale itself
        graph.add_node(child)

        ce = ConfidenceEngine()
        engine = NeuroplasticityEngine(
            confidence_engine=ce,
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            propagation_max_hops=3,
        )

        # Record pre-maintenance confidence
        root_before = graph.get_node("root").confidence

        report = engine.run_maintenance(graph, now=now)

        # Root should have decayed
        assert graph.get_node("root").confidence < root_before
        # Maintenance should have run without errors
        assert report.nodes_decayed >= 1

    def test_strengthen_triggers_propagation_with_real_engine(self):
        """Strengthening a well-accessed root should complete without errors."""
        now = datetime.now()
        graph = KnowledgeGraph()

        root = make_node(
            "Popular fact",
            confidence=0.70,
            node_id="root",
            last_validated=now - timedelta(days=5),
            access_count=20,
        )
        graph.add_node(root)

        child = KnowledgeNode(
            text="Derived from popular",
            confidence=0.65,
            domain="chemistry",
            node_id="child",
            predecessor_nodes=["root"],
        )
        graph.add_node(child)

        ce = ConfidenceEngine()
        engine = NeuroplasticityEngine(
            confidence_engine=ce,
            strengthen_access_threshold=3,
            propagation_max_hops=3,
        )

        report = engine.run_maintenance(graph, now=now)

        # Root should have been strengthened
        assert report.nodes_strengthened >= 1
        assert graph.get_node("root").confidence >= 0.70


# ------------------------------------------------------------------
# TestMultiCycleEffects
# ------------------------------------------------------------------

class TestMultiCycleEffects:
    """Test effects of running multiple maintenance cycles."""

    def test_five_cycle_monotonic_decay(self):
        """Over 5 cycles, confidence should monotonically decrease for an unvalidated node."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Aging fact",
            confidence=0.90,
            node_id="aging",
            last_validated=now - timedelta(days=30),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
        )

        prev_conf = 0.90
        for i in range(5):
            cycle_time = now + timedelta(days=30 * i)
            engine.apply_staleness_decay(graph, now=cycle_time)
            current_conf = graph.get_node("aging").confidence
            assert current_conf <= prev_conf, (
                f"Cycle {i}: confidence {current_conf} > prev {prev_conf}"
            )
            prev_conf = current_conf

        # After 5 cycles (150 days total since last_validated),
        # confidence should be meaningfully lower
        assert graph.get_node("aging").confidence < 0.90

    def test_repeated_strengthening_caps_at_one(self):
        """Repeated strengthen_pathway calls should not exceed confidence 1.0."""
        graph = KnowledgeGraph()
        node = make_node(
            "Frequently accessed fact",
            confidence=0.70,
            node_id="popular",
            access_count=100,
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
            strengthen_max_boost=0.15,
        )

        # Strengthen many times
        for _ in range(20):
            engine.strengthen_pathway(graph, "popular")

        # Confidence is capped at 1.0 (max_boost is per-call, not lifetime)
        assert graph.get_node("popular").confidence <= 1.0
        # After many boosts from 0.70, should have reached 1.0
        assert graph.get_node("popular").confidence == 1.0

    def test_decay_then_strengthen_same_cycle(self):
        """A node that decays AND has high access should get both in one cycle."""
        now = datetime.now()
        graph = KnowledgeGraph()

        node = make_node(
            "Stale but popular",
            confidence=0.80,
            node_id="both",
            last_validated=now - timedelta(days=60),
            access_count=25,
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            strengthen_access_threshold=3,
        )

        report = engine.run_maintenance(graph, now=now)

        # Node should appear in both decay and strengthen details
        decay_ids = {r.node_id for r in report.decay_details}
        strength_ids = {r.node_id for r in report.strength_details}
        assert "both" in decay_ids
        assert "both" in strength_ids


# ------------------------------------------------------------------
# TestDryRunIsolation
# ------------------------------------------------------------------

class TestDryRunIsolation:
    """Ensure dry-run approach (copying nodes to fresh graph) is isolated."""

    def test_dry_run_does_not_modify_original(self):
        """Maintenance on a copy should leave the original graph unchanged."""
        now = datetime.now()
        graph = build_diverse_graph(now=now)

        # Snapshot original confidences
        original_confs = {
            nid: n.confidence for nid, n in graph.nodes.items()
        }

        # Create dry-run copy (same approach as CLI maintain --dry-run)
        graph_copy = KnowledgeGraph()
        for node in graph.nodes.values():
            graph_copy.add_node(copy.deepcopy(node))

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            strengthen_access_threshold=3,
        )
        report = engine.run_maintenance(graph_copy, now=now)

        # Verify copy was modified
        assert report.nodes_decayed > 0 or report.nodes_strengthened > 0

        # Verify original is unchanged
        for nid, original_conf in original_confs.items():
            assert graph.get_node(nid).confidence == original_conf, (
                f"Node {nid} confidence changed in original graph!"
            )


# ------------------------------------------------------------------
# TestGraphSerialization
# ------------------------------------------------------------------

class TestGraphSerialization:
    """Test that maintenance results survive graph save/load."""

    def test_decayed_confidences_persist_through_save_load(self):
        """Run maintenance, save, reload — decayed confidences should persist."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Fact that will decay",
            confidence=0.85,
            node_id="persist-test",
            last_validated=now - timedelta(days=100),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
        )
        engine.apply_staleness_decay(graph, now=now)
        decayed_conf = graph.get_node("persist-test").confidence
        assert decayed_conf < 0.85

        # Save and reload
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            graph.save(path)
            reloaded = KnowledgeGraph.load(path)
            assert abs(reloaded.get_node("persist-test").confidence - decayed_conf) < 1e-6
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# Run all
# ------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
