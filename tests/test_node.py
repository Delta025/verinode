"""
Tests for KnowledgeNode: creation, property behavior, status transitions,
serialization (to_dict/from_dict), and edge cases.
"""

import pytest
import numpy as np
from datetime import datetime

from src.core.node import (
    KnowledgeNode, NodeType, NodeStatus, Source, Derivation,
)


# ------------------------------------------------------------------
# Node creation
# ------------------------------------------------------------------

class TestNodeCreation:
    def test_default_values(self):
        """Node with minimal args has sensible defaults."""
        node = KnowledgeNode(text="Hello world")
        assert node.text == "Hello world"
        assert node.node_type == NodeType.FACT
        assert node.confidence == 0.5
        assert node.domain == "general"
        assert node.embedding is None
        assert node.sources == []
        assert node.predecessor_nodes == []
        assert node.successor_nodes == []
        assert node.conflicts_with == []
        assert node.access_count == 0
        assert node.id  # UUID generated

    def test_custom_id(self):
        node = KnowledgeNode(text="Test", node_id="my-id")
        assert node.id == "my-id"

    def test_custom_type(self):
        node = KnowledgeNode(text="Test", node_type=NodeType.RULE)
        assert node.node_type == NodeType.RULE

    def test_sources_and_predecessors(self):
        src = Source(source_type="web", title="Wikipedia", quality_score=0.6)
        node = KnowledgeNode(
            text="Test",
            sources=[src],
            predecessor_nodes=["pred-1", "pred-2"],
        )
        assert len(node.sources) == 1
        assert node.predecessor_nodes == ["pred-1", "pred-2"]


# ------------------------------------------------------------------
# Confidence property
# ------------------------------------------------------------------

class TestConfidenceProperty:
    def test_getter(self):
        node = KnowledgeNode(text="Test", confidence=0.75)
        assert node.confidence == 0.75

    def test_setter_updates_bounds(self):
        node = KnowledgeNode(text="Test", confidence=0.5)
        node.confidence = 0.8
        assert node.confidence == 0.8
        assert abs(node.confidence_bounds[0] - 0.75) < 1e-9
        assert abs(node.confidence_bounds[1] - 0.85) < 1e-9

    def test_clamped_to_zero(self):
        node = KnowledgeNode(text="Test", confidence=-0.5)
        assert node.confidence == 0.0

    def test_clamped_to_one(self):
        node = KnowledgeNode(text="Test", confidence=1.5)
        assert node.confidence == 1.0

    def test_setter_clamped(self):
        node = KnowledgeNode(text="Test")
        node.confidence = 2.0
        assert node.confidence == 1.0
        node.confidence = -1.0
        assert node.confidence == 0.0

    def test_bounds_clamped_at_edges(self):
        """Confidence bounds shouldn't exceed [0, 1]."""
        node = KnowledgeNode(text="Test")
        node.confidence = 0.01
        assert node.confidence_bounds[0] >= 0.0
        node.confidence = 0.99
        assert node.confidence_bounds[1] <= 1.0


# ------------------------------------------------------------------
# Status transitions
# ------------------------------------------------------------------

class TestStatusTransitions:
    def test_established_at_high_confidence(self):
        node = KnowledgeNode(text="Test", confidence=0.96)
        assert node.status == NodeStatus.ESTABLISHED

    def test_well_supported(self):
        node = KnowledgeNode(text="Test", confidence=0.85)
        assert node.status == NodeStatus.WELL_SUPPORTED

    def test_provisional(self):
        node = KnowledgeNode(text="Test", confidence=0.65)
        assert node.status == NodeStatus.PROVISIONAL

    def test_contested_at_low_confidence(self):
        node = KnowledgeNode(text="Test", confidence=0.3)
        assert node.status == NodeStatus.CONTESTED

    def test_conflict_forces_contested(self):
        """Adding a conflict forces CONTESTED regardless of confidence."""
        node = KnowledgeNode(text="Test", confidence=0.99)
        assert node.status == NodeStatus.ESTABLISHED

        node.add_conflict("other-node")
        assert node.status == NodeStatus.CONTESTED

    def test_update_status_without_conflict(self):
        """update_status() recalculates from confidence when no conflicts."""
        node = KnowledgeNode(text="Test", confidence=0.5)
        node.confidence = 0.96
        node.update_status()
        assert node.status == NodeStatus.ESTABLISHED


# ------------------------------------------------------------------
# Graph relationship methods
# ------------------------------------------------------------------

class TestRelationships:
    def test_add_predecessor_dedup(self):
        node = KnowledgeNode(text="Test")
        node.add_predecessor("pred-1")
        node.add_predecessor("pred-1")
        assert node.predecessor_nodes == ["pred-1"]

    def test_add_successor_dedup(self):
        node = KnowledgeNode(text="Test")
        node.add_successor("succ-1")
        node.add_successor("succ-1")
        assert node.successor_nodes == ["succ-1"]

    def test_add_conflict_dedup(self):
        node = KnowledgeNode(text="Test")
        node.add_conflict("c-1")
        node.add_conflict("c-1")
        assert node.conflicts_with == ["c-1"]

    def test_record_access(self):
        node = KnowledgeNode(text="Test")
        assert node.access_count == 0
        node.record_access()
        node.record_access()
        assert node.access_count == 2


# ------------------------------------------------------------------
# Serialization round-trip
# ------------------------------------------------------------------

class TestSerialization:
    def test_round_trip_basic(self):
        """Basic to_dict -> from_dict preserves key fields."""
        node = KnowledgeNode(
            text="Water boils at 100°C",
            node_type=NodeType.FACT,
            confidence=0.85,
            domain="chemistry",
            node_id="test-node-001",
        )
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)

        assert restored.id == "test-node-001"
        assert restored.text == "Water boils at 100°C"
        assert restored.node_type == NodeType.FACT
        assert abs(restored.confidence - 0.85) < 0.001
        assert restored.domain == "chemistry"

    def test_round_trip_with_sources(self):
        node = KnowledgeNode(text="Test", node_id="src-test")
        node.add_source(Source(
            source_type="journal",
            title="Nature",
            author="Dr. Smith",
            quality_score=0.9,
        ))
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)

        assert len(restored.sources) == 1
        assert restored.sources[0].title == "Nature"
        assert restored.sources[0].quality_score == 0.9

    def test_round_trip_with_relationships(self):
        node = KnowledgeNode(text="Test", node_id="rel-test")
        node.predecessor_nodes = ["pred-1", "pred-2"]
        node.successor_nodes = ["succ-1"]
        node.conflicts_with = ["conflict-1"]

        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)

        assert restored.predecessor_nodes == ["pred-1", "pred-2"]
        assert restored.successor_nodes == ["succ-1"]
        assert restored.conflicts_with == ["conflict-1"]

    def test_round_trip_with_derivation(self):
        node = KnowledgeNode(
            text="Test",
            node_id="deriv-test",
            derivation=Derivation(
                method="inferred",
                derivation_type="chain",
                inference_depth=3,
                notes="Via syllogism",
            ),
        )
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)

        assert restored.derivation.method == "inferred"
        assert restored.derivation.derivation_type == "chain"
        assert restored.derivation.inference_depth == 3
        assert restored.derivation.notes == "Via syllogism"

    def test_round_trip_with_embedding(self):
        emb = np.random.rand(384).astype(np.float32)
        node = KnowledgeNode(text="Test", node_id="emb-test", embedding=emb)

        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)

        assert restored.embedding is not None
        assert np.allclose(restored.embedding, emb, atol=1e-6)

    def test_round_trip_without_embedding(self):
        node = KnowledgeNode(text="Test", node_id="no-emb")
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)
        assert restored.embedding is None

    def test_round_trip_preserves_metadata(self):
        node = KnowledgeNode(text="Test", node_id="meta-test")
        node.metadata = {"extraction_model": "qwen3:14b", "source_quote": "quoted text"}

        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)

        assert restored.metadata["extraction_model"] == "qwen3:14b"
        assert restored.metadata["source_quote"] == "quoted text"

    def test_round_trip_preserves_status(self):
        node = KnowledgeNode(text="Test", confidence=0.99, node_id="status-test")
        assert node.status == NodeStatus.ESTABLISHED

        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)
        assert restored.status == NodeStatus.ESTABLISHED


# ------------------------------------------------------------------
# Defensive from_dict parsing
# ------------------------------------------------------------------

class TestFromDictDefensive:
    def _base_dict(self):
        """Minimal valid node dict."""
        node = KnowledgeNode(text="Valid fact", confidence=0.8, node_id="valid")
        return node.to_dict()

    def test_missing_created_at(self):
        """Missing created_at uses default."""
        d = self._base_dict()
        del d['created_at']
        restored = KnowledgeNode.from_dict(d)
        assert isinstance(restored.created_at, datetime)

    def test_malformed_created_at(self):
        """Malformed date string doesn't crash."""
        d = self._base_dict()
        d['created_at'] = "not-a-date"
        restored = KnowledgeNode.from_dict(d)
        assert isinstance(restored.created_at, datetime)  # keeps default

    def test_missing_status(self):
        """Missing status uses default determination."""
        d = self._base_dict()
        del d['status']
        restored = KnowledgeNode.from_dict(d)
        # Should get default status based on confidence
        assert isinstance(restored.status, NodeStatus)

    def test_invalid_status(self):
        """Invalid status string doesn't crash."""
        d = self._base_dict()
        d['status'] = "invalid_status"
        restored = KnowledgeNode.from_dict(d)
        assert isinstance(restored.status, NodeStatus)

    def test_missing_derivation(self):
        """Missing derivation uses default."""
        d = self._base_dict()
        del d['derivation']
        restored = KnowledgeNode.from_dict(d)
        assert restored.derivation.method == "asserted"

    def test_empty_sources(self):
        d = self._base_dict()
        d['sources'] = []
        restored = KnowledgeNode.from_dict(d)
        assert restored.sources == []

    def test_missing_optional_fields(self):
        """Optional fields default gracefully."""
        d = self._base_dict()
        d.pop('successor_nodes', None)
        d.pop('conflicts_with', None)
        d.pop('metadata', None)
        d.pop('access_count', None)
        restored = KnowledgeNode.from_dict(d)
        assert restored.successor_nodes == []
        assert restored.conflicts_with == []
        assert restored.metadata == {}
        assert restored.access_count == 0


# ------------------------------------------------------------------
# Source and Derivation dataclasses
# ------------------------------------------------------------------

class TestSource:
    def test_str_with_all_fields(self):
        src = Source(
            source_type="journal",
            title="Nature",
            author="Einstein",
            date=datetime(2025, 1, 15),
        )
        s = str(src)
        assert "Nature" in s
        assert "Einstein" in s
        assert "2025" in s

    def test_str_minimal(self):
        src = Source(source_type="web", title="Wikipedia")
        assert str(src) == "Wikipedia"


class TestDerivation:
    def test_defaults(self):
        d = Derivation(method="extracted")
        assert d.inference_depth == 0
        assert d.derivation_type == "extracted"
        assert d.notes == ""


# ------------------------------------------------------------------
# __str__ and __repr__
# ------------------------------------------------------------------

class TestStringRepresentation:
    def test_str_truncates(self):
        long_text = "A" * 100
        node = KnowledgeNode(text=long_text, confidence=0.8)
        s = str(node)
        assert "..." in s
        assert "0.80" in s

    def test_repr_truncates(self):
        node = KnowledgeNode(text="A" * 100, confidence=0.75, node_id="repr-test")
        r = repr(node)
        assert "repr-test" in r
        assert "0.75" in r


# ------------------------------------------------------------------
# Edge labels
# ------------------------------------------------------------------

class TestEdgeLabels:
    """Test edge label storage and serialization on KnowledgeNode."""

    def test_add_predecessor_default_label(self):
        """add_predecessor without explicit label defaults to 'depends_on'."""
        node = KnowledgeNode(text="Test", confidence=0.8)
        node.add_predecessor("pred-1")
        assert node.get_edge_label("pred-1") == "depends_on"

    def test_add_predecessor_explicit_label(self):
        """add_predecessor with explicit label stores it."""
        node = KnowledgeNode(text="Test", confidence=0.8)
        node.add_predecessor("pred-1", "supports")
        assert node.get_edge_label("pred-1") == "supports"

    def test_edge_labels_round_trip(self):
        """Edge labels survive to_dict/from_dict round-trip."""
        node = KnowledgeNode(text="Test", confidence=0.8, node_id="test-labels")
        node.add_predecessor("pred-1", "supports")
        node.add_predecessor("pred-2", "elaborates")
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)
        assert restored.get_edge_label("pred-1") == "supports"
        assert restored.get_edge_label("pred-2") == "elaborates"

    def test_edge_labels_backward_compat(self):
        """Old serialized nodes without edge_labels key still work."""
        node = KnowledgeNode(text="Test", confidence=0.8, node_id="old")
        d = node.to_dict()
        del d["edge_labels"]  # Simulate old format
        restored = KnowledgeNode.from_dict(d)
        assert restored.edge_labels == {}
        assert restored.get_edge_label("any-id") == "depends_on"
