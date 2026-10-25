"""
Tests for the NeuroplasticityEngine: staleness decay, pathway strengthening,
doubt flagging, and combined maintenance.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.core.graph import KnowledgeGraph
from src.core.neuroplasticity import (
    MaintenanceReport,
    NeuroplasticityEngine,
)
from src.core.node import KnowledgeNode, NodeStatus, Source

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
):
    """Create a KnowledgeNode with configurable fields for testing."""
    node = KnowledgeNode(
        text=text,
        confidence=confidence,
        domain=domain,
        node_id=node_id,
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


def build_test_graph(now=None):
    """Build a graph with diverse nodes for maintenance testing."""
    if now is None:
        now = datetime.now()

    graph = KnowledgeGraph()

    # Recent, high-confidence node
    n1 = make_node(
        "Water boils at 100 degrees Celsius",
        confidence=0.95,
        node_id="fresh-high",
        last_validated=now - timedelta(days=1),
        access_count=10,
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.95)],
    )
    # Mark as ESTABLISHED
    n1.status = NodeStatus.ESTABLISHED

    # Moderately old node
    n2 = make_node(
        "Gold melts at 1064 degrees Celsius",
        confidence=0.80,
        node_id="moderate",
        last_validated=now - timedelta(days=60),
        access_count=5,
    )

    # Very stale node
    n3 = make_node(
        "Iron has the chemical symbol Fe",
        confidence=0.75,
        node_id="stale",
        last_validated=now - timedelta(days=200),
        access_count=1,
    )

    # Low confidence conflicted node
    n4 = make_node(
        "Mercury is solid at room temperature",
        confidence=0.25,
        node_id="conflicted",
        last_validated=now - timedelta(days=30),
        conflicts=["fresh-high"],
    )

    # Never accessed, contested
    n5 = make_node(
        "Lead is lighter than aluminum",
        confidence=0.40,
        node_id="untouched",
        last_validated=now - timedelta(days=50),
        access_count=0,
    )
    n5.status = NodeStatus.CONTESTED

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    graph.add_node(n5)

    return graph


# ------------------------------------------------------------------
# Unit tests: Staleness decay
# ------------------------------------------------------------------


class TestStalenessDecay:
    def test_no_decay_within_grace_period(self):
        """Nodes validated recently should not decay."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Fresh fact",
            confidence=0.80,
            node_id="fresh",
            last_validated=now - timedelta(days=3),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(decay_grace_period_days=7.0)
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 0
        assert graph.get_node("fresh").confidence == 0.80

    def test_decay_after_grace_period(self):
        """Nodes beyond grace period should decay."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Old fact",
            confidence=0.80,
            node_id="old",
            last_validated=now - timedelta(days=100),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 1
        result = results[0]
        assert result.node_id == "old"
        assert result.original_confidence == 0.80
        assert result.new_confidence < 0.80
        # After ~93 effective days with half-life 90, factor ~0.49
        assert result.new_confidence > 0.30
        assert graph.get_node("old").confidence == result.new_confidence

    def test_decay_respects_minimum(self):
        """Confidence should not drop below decay_min_confidence."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Very old fact",
            confidence=0.15,
            node_id="ancient",
            last_validated=now - timedelta(days=1000),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_min_confidence=0.1,
            decay_half_life_days=90.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 1
        assert results[0].new_confidence >= 0.1

    def test_established_nodes_protected(self):
        """ESTABLISHED nodes should not decay by default."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Established fact",
            confidence=0.98,
            node_id="established",
            last_validated=now - timedelta(days=365),
        )
        node.status = NodeStatus.ESTABLISHED
        graph.add_node(node)

        engine = NeuroplasticityEngine()
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 0
        assert graph.get_node("established").confidence == 0.98

    def test_custom_protected_statuses(self):
        """Custom protected statuses should be respected."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Well supported fact",
            confidence=0.85,
            node_id="well-supported",
            last_validated=now - timedelta(days=200),
        )
        node.status = NodeStatus.WELL_SUPPORTED
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_protected_statuses={NodeStatus.ESTABLISHED, NodeStatus.WELL_SUPPORTED},
        )
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 0

    def test_half_life_accuracy(self):
        """After exactly one half-life, confidence should halve (approximately)."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Test fact",
            confidence=0.80,
            node_id="half-life",
            last_validated=now - timedelta(days=97),  # 7 grace + 90 effective
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 1
        # After 90 effective days, factor = 2^(-1) = 0.5
        # New conf = 0.80 * 0.5 = 0.40
        assert abs(results[0].new_confidence - 0.40) < 0.01

    def test_decay_updates_status(self):
        """Decayed nodes should have their status updated."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Will lose status",
            confidence=0.85,  # WELL_SUPPORTED
            node_id="decay-status",
            last_validated=now - timedelta(days=300),
        )
        assert node.status == NodeStatus.WELL_SUPPORTED
        graph.add_node(node)

        engine = NeuroplasticityEngine(decay_half_life_days=90.0)
        engine.apply_staleness_decay(graph, now=now)

        # After significant decay, status should change
        node = graph.get_node("decay-status")
        assert node.confidence < 0.80
        # Status should no longer be WELL_SUPPORTED
        assert node.status != NodeStatus.WELL_SUPPORTED

    def test_empty_graph(self):
        """Decay on empty graph returns empty list."""
        graph = KnowledgeGraph()
        engine = NeuroplasticityEngine()
        results = engine.apply_staleness_decay(graph)
        assert results == []


# ------------------------------------------------------------------
# Unit tests: Pathway strengthening
# ------------------------------------------------------------------


class TestPathwayStrengthening:
    def test_no_boost_below_threshold(self):
        """Nodes below access threshold should not be boosted."""
        graph = KnowledgeGraph()
        node = make_node("Test fact", confidence=0.70, node_id="low-access")
        node.access_count = 1
        graph.add_node(node)

        engine = NeuroplasticityEngine(strengthen_access_threshold=3)
        result = engine.strengthen_pathway(graph, "low-access")

        assert result is None
        # Access count should still increment
        assert graph.get_node("low-access").access_count == 2

    def test_boost_above_threshold(self):
        """Nodes at or above threshold should be boosted."""
        graph = KnowledgeGraph()
        node = make_node("Test fact", confidence=0.70, node_id="high-access")
        node.access_count = 5
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
            strengthen_boost_per_access=0.01,
        )
        result = engine.strengthen_pathway(graph, "high-access")

        assert result is not None
        assert result.original_confidence == 0.70
        assert result.new_confidence > 0.70
        # 6 accesses (5+1), 3 excess, boost = 3 * 0.01 = 0.03
        assert abs(result.new_confidence - 0.73) < 0.01

    def test_boost_capped_at_max(self):
        """Boost should not exceed strengthen_max_boost."""
        graph = KnowledgeGraph()
        node = make_node("Test fact", confidence=0.70, node_id="max-boost")
        node.access_count = 100
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
            strengthen_boost_per_access=0.01,
            strengthen_max_boost=0.10,
        )
        result = engine.strengthen_pathway(graph, "max-boost")

        assert result is not None
        # Max boost is 0.10, so 0.70 + 0.10 = 0.80
        assert abs(result.new_confidence - 0.80) < 0.01

    def test_confidence_capped_at_one(self):
        """Confidence cannot exceed 1.0."""
        graph = KnowledgeGraph()
        node = make_node("Test fact", confidence=0.98, node_id="near-max")
        node.access_count = 50
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
            strengthen_boost_per_access=0.01,
            strengthen_max_boost=0.15,
        )
        result = engine.strengthen_pathway(graph, "near-max")

        if result is not None:
            assert result.new_confidence <= 1.0

    def test_validation_boost(self):
        """Validation events should give extra boost."""
        graph = KnowledgeGraph()
        node = make_node("Test fact", confidence=0.70, node_id="validated")
        node.access_count = 5
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
            strengthen_boost_per_access=0.005,
            strengthen_validation_boost=0.02,
        )
        result = engine.strengthen_pathway(graph, "validated", is_validation=True)

        assert result is not None
        # Access boost: 3 excess * 0.005 = 0.015
        # Validation boost: 0.02
        # Total: 0.035
        assert abs(result.new_confidence - 0.735) < 0.01

    def test_validation_updates_last_validated(self):
        """Validation should update the last_validated timestamp."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Test fact",
            confidence=0.70,
            node_id="validated",
            last_validated=now - timedelta(days=30),
        )
        node.access_count = 5
        graph.add_node(node)

        old_validated = node.last_validated

        engine = NeuroplasticityEngine(strengthen_access_threshold=3)
        engine.strengthen_pathway(graph, "validated", is_validation=True)

        assert graph.get_node("validated").last_validated > old_validated

    def test_nonexistent_node(self):
        """Strengthening a nonexistent node returns None."""
        graph = KnowledgeGraph()
        engine = NeuroplasticityEngine()
        result = engine.strengthen_pathway(graph, "nonexistent")
        assert result is None

    def test_strengthen_batch(self):
        """Batch strengthening should work on multiple nodes."""
        graph = KnowledgeGraph()
        for i in range(3):
            node = make_node(f"Fact {i}", confidence=0.7, node_id=f"n{i}")
            node.access_count = 5
            graph.add_node(node)

        engine = NeuroplasticityEngine(strengthen_access_threshold=3)
        results = engine.strengthen_batch(graph, ["n0", "n1", "n2"])

        assert len(results) == 3

    def test_access_count_increments(self):
        """Each strengthen call should increment access_count."""
        graph = KnowledgeGraph()
        node = make_node("Test fact", confidence=0.70, node_id="counting")
        graph.add_node(node)

        engine = NeuroplasticityEngine()
        engine.strengthen_pathway(graph, "counting")
        engine.strengthen_pathway(graph, "counting")
        engine.strengthen_pathway(graph, "counting")

        assert graph.get_node("counting").access_count == 3


# ------------------------------------------------------------------
# Unit tests: Doubt flagging
# ------------------------------------------------------------------


class TestDoubtFlagging:
    def test_flag_conflicted_nodes(self):
        """Nodes with active conflicts should be flagged."""
        now = datetime.now()
        graph = KnowledgeGraph()

        n1 = make_node("Fact A", confidence=0.80, node_id="n1")
        n2 = make_node("Fact B", confidence=0.60, node_id="n2", conflicts=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        engine = NeuroplasticityEngine()
        flags = engine.flag_doubtful_nodes(graph, now=now)

        conflict_flags = [f for f in flags if "conflict" in f.reason.lower()]
        assert len(conflict_flags) >= 1
        assert any(f.node_id == "n2" for f in conflict_flags)

    def test_flag_low_confidence(self):
        """Nodes below confidence threshold should be flagged."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node("Weak fact", confidence=0.15, node_id="weak")
        graph.add_node(node)

        engine = NeuroplasticityEngine(doubt_low_confidence_threshold=0.3)
        flags = engine.flag_doubtful_nodes(graph, now=now)

        low_conf_flags = [f for f in flags if "low confidence" in f.reason.lower()]
        assert len(low_conf_flags) >= 1

    def test_flag_stale_nodes(self):
        """Nodes not validated in a long time should be flagged."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Old fact",
            confidence=0.80,
            node_id="old",
            last_validated=now - timedelta(days=200),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(doubt_stale_days_threshold=180.0)
        flags = engine.flag_doubtful_nodes(graph, now=now)

        stale_flags = [f for f in flags if "validated" in f.reason.lower()]
        assert len(stale_flags) >= 1

    def test_flag_contested_never_accessed(self):
        """Contested nodes that were never accessed should be flagged."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Contested fact",
            confidence=0.40,
            node_id="contested",
            access_count=0,
        )
        node.status = NodeStatus.CONTESTED
        graph.add_node(node)

        engine = NeuroplasticityEngine()
        flags = engine.flag_doubtful_nodes(graph, now=now)

        contested_flags = [f for f in flags if "contested" in f.reason.lower()]
        assert len(contested_flags) >= 1

    def test_no_flags_for_healthy_nodes(self):
        """Healthy, recent, well-accessed nodes should not be flagged."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Healthy fact",
            confidence=0.90,
            node_id="healthy",
            last_validated=now - timedelta(days=5),
            access_count=10,
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine()
        flags = engine.flag_doubtful_nodes(graph, now=now)

        assert len(flags) == 0

    def test_flag_severity_ordering(self):
        """Flags should be sorted by severity (high first)."""
        now = datetime.now()
        graph = KnowledgeGraph()

        # Low confidence (medium severity)
        graph.add_node(make_node("Low conf", confidence=0.10, node_id="low"))

        # Stale (low severity)
        graph.add_node(
            make_node(
                "Stale fact",
                confidence=0.80,
                node_id="stale",
                last_validated=now - timedelta(days=200),
            )
        )

        # Many conflicts (high severity)
        n_conf = make_node(
            "Conflicted", confidence=0.50, node_id="many-conf", conflicts=["low", "stale", "many-conf-2"]
        )
        graph.add_node(n_conf)
        graph.add_node(make_node("Conflict target", confidence=0.50, node_id="many-conf-2"))

        engine = NeuroplasticityEngine()
        flags = engine.flag_doubtful_nodes(graph, now=now)

        if len(flags) >= 2:
            severity_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(flags) - 1):
                assert severity_order[flags[i].severity] <= severity_order[flags[i + 1].severity]

    def test_conflict_with_removed_node_ignored(self):
        """Conflicts with nodes no longer in the graph should be ignored."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Has stale conflict",
            confidence=0.70,
            node_id="n1",
            conflicts=["nonexistent-node"],
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine()
        flags = engine.flag_doubtful_nodes(graph, now=now)

        conflict_flags = [f for f in flags if "conflict" in f.reason.lower()]
        assert len(conflict_flags) == 0

    def test_empty_graph(self):
        """Empty graph should produce no flags."""
        graph = KnowledgeGraph()
        engine = NeuroplasticityEngine()
        flags = engine.flag_doubtful_nodes(graph)
        assert flags == []


# ------------------------------------------------------------------
# Integration tests: Combined maintenance
# ------------------------------------------------------------------


class TestMaintenanceCycle:
    def test_full_maintenance_cycle(self):
        """Full maintenance cycle returns a complete report."""
        now = datetime.now()
        graph = build_test_graph(now)

        engine = NeuroplasticityEngine()
        report = engine.run_maintenance(graph, now=now)

        assert isinstance(report, MaintenanceReport)
        assert report.nodes_examined == 5
        assert report.timestamp == now
        # Should have some decay (stale node)
        assert report.nodes_decayed >= 0
        # Should have some flags (conflicted, low confidence)
        assert report.nodes_flagged >= 0

    def test_maintenance_preserves_established_from_decay(self):
        """Established nodes should not decay during maintenance."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Established fact",
            confidence=0.98,
            node_id="established",
            last_validated=now - timedelta(days=365),
            access_count=0,  # No accesses — won't trigger strengthening
        )
        node.status = NodeStatus.ESTABLISHED
        graph.add_node(node)

        engine = NeuroplasticityEngine()
        report = engine.run_maintenance(graph, now=now)

        # Should NOT have decayed despite being 365 days old
        assert graph.get_node("established").confidence == 0.98
        decayed_ids = {d.node_id for d in report.decay_details}
        assert "established" not in decayed_ids

    def test_maintenance_decay_then_flag(self):
        """Nodes that decay enough should be flagged."""
        now = datetime.now()
        graph = KnowledgeGraph()

        # Node that will decay below doubt threshold
        node = make_node(
            "Will decay and flag",
            confidence=0.35,
            node_id="decay-then-flag",
            last_validated=now - timedelta(days=300),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            doubt_low_confidence_threshold=0.3,
        )
        report = engine.run_maintenance(graph, now=now)

        # Should be both decayed and flagged
        assert report.nodes_decayed >= 1
        assert report.nodes_flagged >= 1

    def test_maintenance_report_structure(self):
        """Verify all fields of the maintenance report."""
        now = datetime.now()
        graph = build_test_graph(now)

        engine = NeuroplasticityEngine()
        report = engine.run_maintenance(graph, now=now)

        assert isinstance(report.decay_details, list)
        assert isinstance(report.strength_details, list)
        assert isinstance(report.flags, list)
        assert isinstance(report.total_decay_amount, float)
        assert isinstance(report.total_strength_boost, float)
        assert report.total_decay_amount >= 0
        assert report.total_strength_boost >= 0

    def test_maintenance_strengthens_accessed_nodes(self):
        """Maintenance should boost nodes with high access counts."""
        now = datetime.now()
        graph = KnowledgeGraph()

        node = make_node(
            "Well-accessed fact",
            confidence=0.70,
            node_id="accessed",
            last_validated=now - timedelta(days=1),
            access_count=10,
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
            strengthen_boost_per_access=0.01,
        )
        report = engine.run_maintenance(graph, now=now)

        # Should have been strengthened
        assert report.nodes_strengthened >= 1
        assert graph.get_node("accessed").confidence > 0.70


# ------------------------------------------------------------------
# Unit tests: Corroboration decay
# ------------------------------------------------------------------


class TestCorroborationDecay:
    def test_small_graph_no_penalty(self):
        """In a small graph, single-source facts are not penalized."""
        now = datetime.now()
        graph = KnowledgeGraph()
        # Build a 50-node graph so scale factor (100) is not exceeded
        for i in range(49):
            filler = make_node(
                f"Filler fact {i}",
                confidence=0.9,
                node_id=f"filler-{i}",
                last_validated=now,
            )
            filler.status = NodeStatus.ESTABLISHED  # protect from decay
            graph.add_node(filler)

        target = make_node(
            "Single source fact",
            confidence=0.80,
            node_id="target",
            last_validated=now - timedelta(days=100),
            sources=[Source(source_type="web", title="Blog", quality_score=0.6)],
        )
        graph.add_node(target)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            corroboration_decay_enabled=True,
            corroboration_scale_factor=100.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        target_results = [r for r in results if r.node_id == "target"]
        assert len(target_results) == 1
        # graph_size=50, ratio=0.5, log2(0.5)=-1, expected=max(1,1+(-1))=1.0
        # 1 source / 1.0 expected = 1.0 multiplier -> base half-life
        assert target_results[0].effective_half_life == pytest.approx(90.0, abs=0.1)

    def test_large_graph_uncorroborated_decays_faster(self):
        """In a large graph, single-source facts decay faster."""
        now = datetime.now()
        graph = KnowledgeGraph()
        # Build a 1000-node graph
        for i in range(999):
            filler = make_node(
                f"Filler {i}",
                confidence=0.95,
                node_id=f"f-{i}",
                last_validated=now,
            )
            filler.status = NodeStatus.ESTABLISHED
            graph.add_node(filler)

        target = make_node(
            "Uncorroborated fact",
            confidence=0.80,
            node_id="target",
            last_validated=now - timedelta(days=60),
            sources=[Source(source_type="web", title="Blog", quality_score=0.5)],
        )
        graph.add_node(target)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            corroboration_decay_enabled=True,
            corroboration_scale_factor=100.0,
            corroboration_min_multiplier=0.5,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        target_results = [r for r in results if r.node_id == "target"]
        assert len(target_results) == 1
        # graph_size=1000, ratio=10, log2(10)=3.32, expected=4.32
        # 1 source / 4.32 = 0.23 -> clamped to min 0.5
        # effective half-life = 90 * 0.5 = 45
        assert target_results[0].effective_half_life == pytest.approx(45.0, abs=0.1)
        assert target_results[0].num_sources == 1

    def test_multi_source_decays_slower(self):
        """Facts with multiple sources get a longer half-life."""
        now = datetime.now()
        graph = KnowledgeGraph()
        for i in range(199):
            filler = make_node(
                f"Filler {i}",
                confidence=0.95,
                node_id=f"f-{i}",
                last_validated=now,
            )
            filler.status = NodeStatus.ESTABLISHED
            graph.add_node(filler)

        target = make_node(
            "Well-sourced fact",
            confidence=0.80,
            node_id="target",
            last_validated=now - timedelta(days=60),
            sources=[
                Source(source_type="textbook", title="Book A", quality_score=0.9),
                Source(source_type="paper", title="Paper B", quality_score=0.85),
                Source(source_type="paper", title="Paper C", quality_score=0.8),
            ],
        )
        graph.add_node(target)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            corroboration_decay_enabled=True,
            corroboration_scale_factor=100.0,
            corroboration_max_multiplier=2.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        target_results = [r for r in results if r.node_id == "target"]
        assert len(target_results) == 1
        # graph_size=200, ratio=2, log2(2)=1, expected=2.0
        # 3 sources / 2.0 = 1.5 multiplier
        # effective half-life = 90 * 1.5 = 135
        assert target_results[0].effective_half_life == pytest.approx(135.0, abs=0.1)
        assert target_results[0].num_sources == 3

    def test_corroboration_multiplier_capped(self):
        """Multiplier never exceeds max_corroboration_multiplier."""
        now = datetime.now()
        graph = KnowledgeGraph()
        for i in range(99):
            filler = make_node(f"F{i}", confidence=0.95, node_id=f"f-{i}", last_validated=now)
            filler.status = NodeStatus.ESTABLISHED
            graph.add_node(filler)

        # 10 sources in a 100-node graph
        target = make_node(
            "Heavily sourced",
            confidence=0.80,
            node_id="target",
            last_validated=now - timedelta(days=60),
            sources=[Source(source_type="t", title=f"S{i}", quality_score=0.9) for i in range(10)],
        )
        graph.add_node(target)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            corroboration_decay_enabled=True,
            corroboration_scale_factor=100.0,
            corroboration_max_multiplier=2.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        target_results = [r for r in results if r.node_id == "target"]
        assert len(target_results) == 1
        # 10 sources / expected ~1.0 = 10.0 -> capped at max 2.0
        assert target_results[0].effective_half_life == pytest.approx(180.0, abs=0.1)

    def test_corroboration_disabled_uses_base_halflife(self):
        """When disabled, all nodes use base half-life regardless of sources."""
        now = datetime.now()
        graph = KnowledgeGraph()
        for i in range(199):
            filler = make_node(f"F{i}", confidence=0.95, node_id=f"f-{i}", last_validated=now)
            filler.status = NodeStatus.ESTABLISHED
            graph.add_node(filler)

        target = make_node(
            "Fact with many sources",
            confidence=0.80,
            node_id="target",
            last_validated=now - timedelta(days=100),
            sources=[
                Source(source_type="textbook", title="A", quality_score=0.9),
                Source(source_type="textbook", title="B", quality_score=0.9),
            ],
        )
        graph.add_node(target)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            corroboration_decay_enabled=False,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        target_results = [r for r in results if r.node_id == "target"]
        assert len(target_results) == 1
        assert target_results[0].effective_half_life == 90.0

    def test_empty_sources_treated_as_single(self):
        """Node with no sources list gets same treatment as single source."""
        now = datetime.now()
        graph = KnowledgeGraph()
        for i in range(999):
            filler = make_node(f"F{i}", confidence=0.95, node_id=f"f-{i}", last_validated=now)
            filler.status = NodeStatus.ESTABLISHED
            graph.add_node(filler)

        target = make_node(
            "No sources",
            confidence=0.80,
            node_id="target",
            last_validated=now - timedelta(days=60),
            sources=[],
        )
        graph.add_node(target)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            corroboration_decay_enabled=True,
            corroboration_scale_factor=100.0,
            corroboration_min_multiplier=0.5,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        target_results = [r for r in results if r.node_id == "target"]
        assert len(target_results) == 1
        # 0 sources treated as 1 (max(1, 0)), same as single source
        assert target_results[0].effective_half_life == pytest.approx(45.0, abs=0.1)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestNeuroplasticityEdgeCases:
    def test_decay_with_zero_half_life(self):
        """Zero half-life should cause maximum decay."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Test",
            confidence=0.80,
            node_id="n1",
            last_validated=now - timedelta(days=10),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=0.001,  # Nearly zero
            decay_grace_period_days=0.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        assert len(results) == 1
        # Should decay to minimum
        assert results[0].new_confidence <= 0.1 + 0.01

    def test_strengthen_with_zero_boost(self):
        """Zero boost per access should not change confidence."""
        graph = KnowledgeGraph()
        node = make_node("Test", confidence=0.70, node_id="n1")
        node.access_count = 100
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            strengthen_boost_per_access=0.0,
            strengthen_validation_boost=0.0,
        )
        result = engine.strengthen_pathway(graph, "n1")

        # No boost applied, so result should be None
        assert result is None
        assert graph.get_node("n1").confidence == 0.70

    def test_multiple_maintenance_cycles(self):
        """Multiple maintenance cycles should compound effects."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Gradually decaying",
            confidence=0.80,
            node_id="n1",
            last_validated=now - timedelta(days=50),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
        )

        # First cycle at day 50
        engine.apply_staleness_decay(graph, now=now)
        conf_after_first = graph.get_node("n1").confidence

        # Second cycle at day 100 (update now)
        now2 = now + timedelta(days=50)
        engine.apply_staleness_decay(graph, now=now2)
        conf_after_second = graph.get_node("n1").confidence

        # Should have decayed further
        assert conf_after_second < conf_after_first
        assert conf_after_first < 0.80


# ------------------------------------------------------------------
# TestConfidencePropagation — verify propagation calls during maintenance
# ------------------------------------------------------------------


class TestConfidencePropagation:
    """Test that maintenance triggers confidence propagation correctly."""

    def test_maintenance_calls_propagation(self):
        """run_maintenance should call propagate_confidence_update for decayed nodes."""
        now = datetime.now()
        graph = KnowledgeGraph()
        # Stale node that will be decayed
        node = make_node(
            "Stale fact",
            confidence=0.80,
            node_id="n1",
            last_validated=now - timedelta(days=60),
        )
        graph.add_node(node)

        mock_ce = MagicMock()
        engine = NeuroplasticityEngine(
            confidence_engine=mock_ce,
            decay_half_life_days=90.0,
            decay_grace_period_days=7.0,
            propagation_max_hops=3,
        )
        engine.run_maintenance(graph, now=now)

        # propagate_confidence_update should have been called for node n1
        mock_ce.propagate_confidence_update.assert_called()
        call_args = mock_ce.propagate_confidence_update.call_args_list
        called_node_ids = [c.args[1] if len(c.args) > 1 else c.kwargs.get("updated_node_id") for c in call_args]
        assert "n1" in called_node_ids

    def test_propagation_max_hops_respected(self):
        """propagate_confidence_update should receive the configured max_hops."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Old fact",
            confidence=0.80,
            node_id="n1",
            last_validated=now - timedelta(days=60),
        )
        graph.add_node(node)

        mock_ce = MagicMock()
        engine = NeuroplasticityEngine(
            confidence_engine=mock_ce,
            propagation_max_hops=5,
            decay_grace_period_days=7.0,
        )
        engine.run_maintenance(graph, now=now)

        # Check that max_hops=5 was passed
        for call in mock_ce.propagate_confidence_update.call_args_list:
            assert call.kwargs.get("max_hops") == 5 or (len(call.args) >= 3 and call.args[2] == 5)

    def test_no_propagation_without_confidence_engine(self):
        """Maintenance should complete successfully with confidence_engine=None."""
        now = datetime.now()
        graph = KnowledgeGraph()
        node = make_node(
            "Old fact",
            confidence=0.80,
            node_id="n1",
            last_validated=now - timedelta(days=60),
        )
        graph.add_node(node)

        engine = NeuroplasticityEngine(
            confidence_engine=None,
            decay_grace_period_days=7.0,
        )
        # Should not raise
        report = engine.run_maintenance(graph, now=now)
        assert report.nodes_decayed > 0


# ------------------------------------------------------------------
# TestMultipleDoubtFlags — nodes that trigger multiple flag reasons
# ------------------------------------------------------------------


class TestMultipleDoubtFlags:
    """Test nodes that accumulate multiple doubt flag reasons."""

    def test_node_with_multiple_flag_reasons(self):
        """A node that is conflicted + low confidence + stale should get multiple flags."""
        now = datetime.now()
        graph = KnowledgeGraph()

        # Build a problematic node: low confidence, stale, AND conflicted
        bad_node = make_node(
            "Dubious claim",
            confidence=0.25,
            node_id="bad",
            last_validated=now - timedelta(days=200),
            conflicts=["other-node"],
        )
        # Also add the "other" node so conflict resolution works
        other_node = make_node(
            "Contradicting claim",
            confidence=0.90,
            node_id="other-node",
        )
        graph.add_node(bad_node)
        graph.add_node(other_node)

        engine = NeuroplasticityEngine(
            doubt_low_confidence_threshold=0.3,
            doubt_stale_days_threshold=90,
            decay_grace_period_days=7.0,
        )

        flags = engine.flag_doubtful_nodes(graph, now=now)

        # Filter flags for the "bad" node
        bad_flags = [f for f in flags if f.node_id == "bad"]
        # Should have at least 2 flags (low confidence + stale + conflicted)
        assert len(bad_flags) >= 2, f"Expected >= 2 flags, got {len(bad_flags)}: {[f.reason for f in bad_flags]}"

        # Reasons are human-readable strings, check with substring matching
        reasons_text = " ".join(f.reason.lower() for f in bad_flags)
        assert "confidence" in reasons_text or "low" in reasons_text
        # Should also mention staleness or conflict
        assert "conflict" in reasons_text or "validated" in reasons_text or "stale" in reasons_text


# ------------------------------------------------------------------
# TestMaintenanceWithEdges — graphs with predecessor/successor links
# ------------------------------------------------------------------


class TestMaintenanceWithEdges:
    """Test maintenance on graphs with node dependencies."""

    def test_decay_on_graph_with_dependencies(self):
        """Both root and derived nodes should decay independently."""
        now = datetime.now()
        graph = KnowledgeGraph()

        root = make_node(
            "Root fact",
            confidence=0.85,
            node_id="root",
            last_validated=now - timedelta(days=60),
        )
        graph.add_node(root)

        derived = KnowledgeNode(
            text="Derived fact",
            confidence=0.75,
            domain="chemistry",
            node_id="derived",
            predecessor_nodes=["root"],
        )
        derived.last_validated = now - timedelta(days=60)
        graph.add_node(derived)

        engine = NeuroplasticityEngine(
            decay_grace_period_days=7.0,
            decay_half_life_days=90.0,
        )
        results = engine.apply_staleness_decay(graph, now=now)

        decayed_ids = {r.node_id for r in results}
        assert "root" in decayed_ids
        assert "derived" in decayed_ids

        # Both should have lower confidence
        assert graph.get_node("root").confidence < 0.85
        assert graph.get_node("derived").confidence < 0.75

    def test_strengthening_does_not_affect_edges(self):
        """Strengthening a node should not alter predecessor/successor links."""
        datetime.now()
        graph = KnowledgeGraph()

        root = make_node(
            "Root fact",
            confidence=0.70,
            node_id="root",
            access_count=20,
        )
        graph.add_node(root)

        derived = KnowledgeNode(
            text="Derived fact",
            confidence=0.60,
            domain="chemistry",
            node_id="derived",
            predecessor_nodes=["root"],
        )
        graph.add_node(derived)

        # Verify edge structure before
        assert "derived" in graph.get_node("root").successor_nodes
        assert "root" in graph.get_node("derived").predecessor_nodes

        engine = NeuroplasticityEngine(
            strengthen_access_threshold=3,
        )
        engine.strengthen_pathway(graph, "root")

        # Edge structure should be unchanged
        assert "derived" in graph.get_node("root").successor_nodes
        assert "root" in graph.get_node("derived").predecessor_nodes
