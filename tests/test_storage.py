"""
Unit tests for SQLite-backed knowledge graph storage.

No Ollama server needed — all tests use in-memory graphs and local SQLite files.
"""

import os
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph import KnowledgeGraph
from src.core.node import (
    Derivation,
    KnowledgeNode,
    NodeStatus,
    NodeType,
    Source,
)
from src.core.storage import (
    SCHEMA_VERSION,
    SqliteGraph,
    json_to_sqlite,
    sqlite_to_json,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _tmp_db() -> str:
    """Create a temporary .db file path and return it."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # SqliteGraph will create it
    return path


def _make_node(
    text="Water boils at 100 degrees Celsius",
    confidence=0.95,
    domain="chemistry",
    node_type=NodeType.FACT,
    sources=None,
    predecessor_nodes=None,
    embedding=None,
    derivation=None,
) -> KnowledgeNode:
    """Create a test node with sensible defaults."""
    if sources is None:
        sources = [Source(source_type="textbook", title="Chem 101", quality_score=0.9)]
    return KnowledgeNode(
        text=text,
        node_type=node_type,
        confidence=confidence,
        domain=domain,
        sources=sources,
        predecessor_nodes=predecessor_nodes or [],
        embedding=embedding,
        derivation=derivation,
    )


def _build_graph() -> KnowledgeGraph:
    """Create a small in-memory graph with 4 nodes for testing."""
    graph = KnowledgeGraph()

    n1 = _make_node(
        text="Water boils at 100 degrees Celsius at sea level",
        confidence=0.95,
    )
    n2 = _make_node(
        text="Water freezes at 0 degrees Celsius",
        confidence=0.9,
    )
    n3 = KnowledgeNode(
        text="Steam is produced when water boils",
        node_type=NodeType.INFERENCE,
        confidence=0.8,
        domain="chemistry",
        predecessor_nodes=[n1.id],
        derivation=Derivation(method="inferred", inference_depth=1, derivation_type="chain"),
    )
    n4 = _make_node(
        text="Gold is a metal",
        confidence=0.4,
        sources=[Source(source_type="web", title="RandomFacts.blog", quality_score=0.3)],
    )

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    return graph


# ------------------------------------------------------------------
# Schema and basic construction
# ------------------------------------------------------------------


class TestSchemaCreation:
    """Verify that SqliteGraph creates the expected tables."""

    def test_creates_tables(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            conn = sqlite3.connect(db_path)
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "nodes" in tables
            assert "sources" in tables
            assert "edges" in tables
            assert "conflicts" in tables
            assert "embeddings" in tables
            assert "changelog" in tables
            assert "schema_version" in tables
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_schema_version_stamped(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row[0] == SCHEMA_VERSION
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_empty_graph_has_no_nodes(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            assert len(graph) == 0
            assert graph.total_nodes == 0
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# add_node + persistence
# ------------------------------------------------------------------


class TestAddNode:
    """Test adding nodes and verifying SQLite persistence."""

    def test_add_single_node(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node()
            graph.add_node(node)

            assert len(graph) == 1
            assert node.id in graph

            # Verify it's in SQLite
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT id, text FROM nodes WHERE id = ?", (node.id,)).fetchone()
            assert row is not None
            assert row[1] == node.text
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_add_node_persists_sources(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node(
                sources=[
                    Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                    Source(source_type="paper", title="Research Paper", author="Smith", quality_score=0.85),
                ]
            )
            graph.add_node(node)

            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT source_type, title, author FROM sources WHERE node_id = ?",
                (node.id,),
            ).fetchall()
            assert len(rows) == 2
            titles = {r[1] for r in rows}
            assert "Chem 101" in titles
            assert "Research Paper" in titles
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_add_node_persists_edges(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            n1 = _make_node(text="Base fact")
            n2 = KnowledgeNode(
                text="Derived fact",
                node_type=NodeType.INFERENCE,
                confidence=0.8,
                predecessor_nodes=[n1.id],
                derivation=Derivation(method="inferred", inference_depth=1),
            )
            graph.add_node(n1)
            graph.add_node(n2)

            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT predecessor_id, successor_id FROM edges").fetchone()
            assert row is not None
            assert row[0] == n1.id
            assert row[1] == n2.id
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_add_node_persists_embedding(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            emb = np.random.randn(384).astype(np.float32)
            node = _make_node(embedding=emb)
            graph.add_node(node)

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE node_id = ?",
                (node.id,),
            ).fetchone()
            assert row is not None
            restored = np.frombuffer(row[0], dtype=np.float32)
            np.testing.assert_array_almost_equal(restored, emb)
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_add_node_logs_changelog(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node()
            graph.add_node(node)

            entries = graph.get_changelog()
            assert len(entries) == 1
            assert entries[0]["change_type"] == "add_node"
            assert entries[0]["node_id"] == node.id
            graph.close()
        finally:
            os.unlink(db_path)

    def test_add_duplicate_raises(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node()
            graph.add_node(node)
            with pytest.raises(ValueError, match="already exists"):
                graph.add_node(node)
            graph.close()
        finally:
            os.unlink(db_path)

    def test_node_without_embedding(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node(embedding=None)
            graph.add_node(node)

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE node_id = ?",
                (node.id,),
            ).fetchone()
            assert row is None
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# Round-trip: add → close → reopen → verify
# ------------------------------------------------------------------


class TestRoundTrip:
    """Test that data survives close + reopen."""

    def test_basic_round_trip(self):
        db_path = _tmp_db()
        try:
            # Write
            g1 = SqliteGraph(db_path)
            node = _make_node()
            g1.add_node(node)
            g1.close()

            # Read
            g2 = SqliteGraph(db_path)
            assert len(g2) == 1
            restored = g2.get_node(node.id)
            assert restored is not None
            assert restored.text == node.text
            assert abs(restored.confidence - node.confidence) < 1e-6
            assert restored.domain == node.domain
            assert restored.node_type == node.node_type
            assert restored.status == node.status
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_preserves_sources(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            node = _make_node(
                sources=[
                    Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                    Source(source_type="paper", title="Paper A", author="Jones", quality_score=0.8),
                ]
            )
            g1.add_node(node)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            assert len(restored.sources) == 2
            titles = {s.title for s in restored.sources}
            assert "Chem 101" in titles
            assert "Paper A" in titles
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_preserves_edges(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            n1 = _make_node(text="Base fact")
            n2 = KnowledgeNode(
                text="Derived fact",
                node_type=NodeType.INFERENCE,
                confidence=0.8,
                predecessor_nodes=[n1.id],
                derivation=Derivation(method="inferred", inference_depth=1),
            )
            g1.add_node(n1)
            g1.add_node(n2)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(n2.id)
            assert n1.id in restored.predecessor_nodes
            # Successor relationship should also be restored
            base = g2.get_node(n1.id)
            assert n2.id in base.successor_nodes
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_preserves_embedding(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            emb = np.random.randn(384).astype(np.float32)
            node = _make_node(embedding=emb)
            g1.add_node(node)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            assert restored.embedding is not None
            np.testing.assert_array_almost_equal(restored.embedding, emb)
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_preserves_derivation(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            node = _make_node(
                derivation=Derivation(
                    method="inferred",
                    inference_depth=2,
                    derivation_type="chain",
                    notes="test derivation",
                )
            )
            g1.add_node(node)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            assert restored.derivation.method == "inferred"
            assert restored.derivation.inference_depth == 2
            assert restored.derivation.derivation_type == "chain"
            assert restored.derivation.notes == "test derivation"
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_preserves_metadata(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            node = _make_node()
            node.metadata = {"key": "value", "count": 42}
            g1.add_node(node)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            assert restored.metadata == {"key": "value", "count": 42}
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_preserves_timestamps(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            node = _make_node()
            original_created = node.created_at
            g1.add_node(node)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            # Timestamps should match to the second (isoformat round-trip)
            assert abs((restored.created_at - original_created).total_seconds()) < 1
            g2.close()
        finally:
            os.unlink(db_path)

    def test_round_trip_multiple_nodes(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            nodes = [_make_node(text=f"Fact number {i}", confidence=0.5 + i * 0.1) for i in range(5)]
            for n in nodes:
                g1.add_node(n)
            g1.close()

            g2 = SqliteGraph(db_path)
            assert len(g2) == 5
            for n in nodes:
                restored = g2.get_node(n.id)
                assert restored is not None
                assert restored.text == n.text
            g2.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# update_node_confidence
# ------------------------------------------------------------------


class TestUpdateConfidence:
    """Test confidence updates are persisted."""

    def test_update_confidence(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            node = _make_node(confidence=0.5)
            g1.add_node(node)
            g1.update_node_confidence(node.id, 0.9)

            # In-memory
            assert abs(g1.nodes[node.id].confidence - 0.9) < 1e-6

            # In SQLite
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT confidence FROM nodes WHERE id = ?", (node.id,)).fetchone()
            assert abs(row[0] - 0.9) < 1e-6
            conn.close()
            g1.close()
        finally:
            os.unlink(db_path)

    def test_update_confidence_changes_status(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node(confidence=0.5)
            graph.add_node(node)
            assert node.status == NodeStatus.CONTESTED

            graph.update_node_confidence(node.id, 0.95)
            assert graph.nodes[node.id].status == NodeStatus.ESTABLISHED

            # Verify in SQLite
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT status FROM nodes WHERE id = ?", (node.id,)).fetchone()
            assert row[0] == "established"
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_update_confidence_logs_changelog(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node(confidence=0.5)
            graph.add_node(node)
            graph.update_node_confidence(node.id, 0.9)

            entries = graph.get_changelog()
            conf_entries = [e for e in entries if e["change_type"] == "update_confidence"]
            assert len(conf_entries) == 1
            assert conf_entries[0]["details"]["old"] == pytest.approx(0.5)
            assert conf_entries[0]["details"]["new"] == 0.9
            graph.close()
        finally:
            os.unlink(db_path)

    def test_update_confidence_survives_reopen(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            node = _make_node(confidence=0.5)
            g1.add_node(node)
            g1.update_node_confidence(node.id, 0.9)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            assert abs(restored.confidence - 0.9) < 1e-6
            g2.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# add_contradiction
# ------------------------------------------------------------------


class TestAddContradiction:
    """Test contradiction persistence."""

    def test_add_contradiction(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            n1 = _make_node(text="Water boils at 100C")
            n2 = _make_node(text="Water boils at 50C")
            graph.add_node(n1)
            graph.add_node(n2)
            graph.add_contradiction(n1.id, n2.id)

            # In-memory
            assert n2.id in graph.nodes[n1.id].conflicts_with
            assert n1.id in graph.nodes[n2.id].conflicts_with

            # In SQLite
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT node_id1, node_id2 FROM conflicts").fetchall()
            assert len(rows) == 1
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_contradiction_survives_reopen(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            n1 = _make_node(text="Water boils at 100C")
            n2 = _make_node(text="Water boils at 50C")
            g1.add_node(n1)
            g1.add_node(n2)
            g1.add_contradiction(n1.id, n2.id)
            g1.close()

            g2 = SqliteGraph(db_path)
            r1 = g2.get_node(n1.id)
            r2 = g2.get_node(n2.id)
            assert n2.id in r1.conflicts_with
            assert n1.id in r2.conflicts_with
            g2.close()
        finally:
            os.unlink(db_path)

    def test_contradiction_updates_status(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            n1 = _make_node(text="Fact A", confidence=0.95)
            n2 = _make_node(text="Fact B", confidence=0.95)
            graph.add_node(n1)
            graph.add_node(n2)
            graph.add_contradiction(n1.id, n2.id)

            # Both should now be contested
            assert graph.nodes[n1.id].status == NodeStatus.CONTESTED
            assert graph.nodes[n2.id].status == NodeStatus.CONTESTED
            graph.close()
        finally:
            os.unlink(db_path)

    def test_contradiction_logs_changelog(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            n1 = _make_node(text="Fact A")
            n2 = _make_node(text="Fact B")
            graph.add_node(n1)
            graph.add_node(n2)
            graph.add_contradiction(n1.id, n2.id)

            entries = graph.get_changelog()
            contra_entries = [e for e in entries if e["change_type"] == "add_contradiction"]
            assert len(contra_entries) == 1
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# remove_node
# ------------------------------------------------------------------


class TestRemoveNode:
    """Test node removal and cascade deletes."""

    def test_remove_node(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node()
            graph.add_node(node)
            graph.remove_node(node.id)

            # In-memory
            assert len(graph) == 0
            assert node.id not in graph

            # In SQLite
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT id FROM nodes WHERE id = ?", (node.id,)).fetchone()
            assert row is None
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_remove_cascades_sources(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node()
            graph.add_node(node)
            node_id = node.id
            graph.remove_node(node_id)

            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT id FROM sources WHERE node_id = ?", (node_id,)).fetchall()
            assert len(rows) == 0
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_remove_cascades_embeddings(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            emb = np.random.randn(384).astype(np.float32)
            node = _make_node(embedding=emb)
            graph.add_node(node)
            node_id = node.id
            graph.remove_node(node_id)

            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT node_id FROM embeddings WHERE node_id = ?", (node_id,)).fetchone()
            assert row is None
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_remove_cleans_conflicts(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            n1 = _make_node(text="Fact A")
            n2 = _make_node(text="Fact B")
            graph.add_node(n1)
            graph.add_node(n2)
            graph.add_contradiction(n1.id, n2.id)
            graph.remove_node(n1.id)

            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT * FROM conflicts").fetchall()
            assert len(rows) == 0
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_remove_logs_changelog(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node()
            graph.add_node(node)
            graph.remove_node(node.id)

            entries = graph.get_changelog()
            rm_entries = [e for e in entries if e["change_type"] == "remove_node"]
            assert len(rm_entries) == 1
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# Changelog
# ------------------------------------------------------------------


class TestChangelog:
    """Test changelog retrieval."""

    def test_changelog_ordered_newest_first(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            for i in range(5):
                graph.add_node(_make_node(text=f"Node {i}"))
            entries = graph.get_changelog()
            # IDs should be descending (newest first)
            ids = [e["id"] for e in entries]
            assert ids == sorted(ids, reverse=True)
            graph.close()
        finally:
            os.unlink(db_path)

    def test_changelog_limit(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            for i in range(10):
                graph.add_node(_make_node(text=f"Node {i}"))
            entries = graph.get_changelog(limit=3)
            assert len(entries) == 3
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# from_graph conversion
# ------------------------------------------------------------------


class TestFromGraph:
    """Test converting an in-memory KnowledgeGraph to SqliteGraph."""

    def test_from_graph_basic(self):
        db_path = _tmp_db()
        sg = None
        try:
            mem_graph = _build_graph()
            sg = SqliteGraph.from_graph(mem_graph, db_path)
            assert len(sg) == 4
            for node in mem_graph.nodes.values():
                restored = sg.get_node(node.id)
                assert restored is not None
                assert restored.text == node.text
        finally:
            if sg:
                sg.close()
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_from_graph_preserves_edges(self):
        db_path = _tmp_db()
        sg = None
        try:
            mem_graph = _build_graph()
            sg = SqliteGraph.from_graph(mem_graph, db_path)

            # Find the inference node
            for node in sg.nodes.values():
                if node.node_type == NodeType.INFERENCE:
                    assert len(node.predecessor_nodes) > 0
        finally:
            if sg:
                sg.close()
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_from_graph_rebuilds_indexes(self):
        db_path = _tmp_db()
        sg = None
        try:
            mem_graph = _build_graph()
            sg = SqliteGraph.from_graph(mem_graph, db_path)

            # Domain index
            assert "chemistry" in sg.domain_index
            assert len(sg.domain_index["chemistry"]) == 4

            # Type index
            assert len(sg.type_index[NodeType.FACT]) == 3
            assert len(sg.type_index[NodeType.INFERENCE]) == 1
        finally:
            if sg:
                sg.close()
            if os.path.exists(db_path):
                os.unlink(db_path)


# ------------------------------------------------------------------
# json_to_sqlite / sqlite_to_json
# ------------------------------------------------------------------


class TestConversion:
    """Test JSON ↔ SQLite round-trip conversion."""

    def test_json_to_sqlite(self):
        db_path = _tmp_db()
        json_path = _tmp_db().replace(".db", ".json")
        sg = None
        try:
            # Create JSON graph
            mem_graph = _build_graph()
            mem_graph.save(json_path)

            # Convert
            sg = json_to_sqlite(json_path, db_path)
            assert len(sg) == 4
        finally:
            if sg:
                sg.close()
            if os.path.exists(db_path):
                os.unlink(db_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_sqlite_to_json(self):
        db_path = _tmp_db()
        json_path = _tmp_db().replace(".db", ".json")
        try:
            # Create SQLite graph
            sg = SqliteGraph(db_path)
            for i in range(3):
                sg.add_node(_make_node(text=f"Fact {i}"))
            sg.close()

            # Convert
            sqlite_to_json(db_path, json_path)

            # Verify JSON
            loaded = KnowledgeGraph.load(json_path)
            assert len(loaded) == 3
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_json_sqlite_json_round_trip(self):
        """Full round-trip: JSON → SQLite → JSON preserves data."""
        db_path = _tmp_db()
        json_path1 = _tmp_db().replace(".db", "_1.json")
        json_path2 = _tmp_db().replace(".db", "_2.json")
        sg = None
        try:
            # Create original JSON
            mem_graph = _build_graph()
            mem_graph.save(json_path1)

            # JSON → SQLite
            sg = json_to_sqlite(json_path1, db_path)
            sg.close()
            sg = None

            # SQLite → JSON
            sqlite_to_json(db_path, json_path2)

            # Compare
            original = KnowledgeGraph.load(json_path1)
            restored = KnowledgeGraph.load(json_path2)

            assert len(original) == len(restored)
            for node_id in original.nodes:
                assert node_id in restored.nodes
                o = original.nodes[node_id]
                r = restored.nodes[node_id]
                assert o.text == r.text
                assert abs(o.confidence - r.confidence) < 1e-6
                assert o.domain == r.domain
                assert o.node_type == r.node_type
        finally:
            if sg:
                sg.close()
            for p in [db_path, json_path1, json_path2]:
                if os.path.exists(p):
                    os.unlink(p)


# ------------------------------------------------------------------
# Embedding persistence (public API)
# ------------------------------------------------------------------


class TestPersistEmbedding:
    """Test the persist_embedding() method."""

    def test_persist_embedding(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            node = _make_node(embedding=None)
            graph.add_node(node)

            # Persist an embedding externally
            emb = np.random.randn(384).astype(np.float32)
            graph.persist_embedding(node.id, emb)

            # Verify in SQLite
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE node_id = ?",
                (node.id,),
            ).fetchone()
            assert row is not None
            restored = np.frombuffer(row[0], dtype=np.float32)
            np.testing.assert_array_almost_equal(restored, emb)
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)

    def test_persist_embedding_overwrites(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            emb1 = np.ones(384, dtype=np.float32)
            node = _make_node(embedding=emb1)
            graph.add_node(node)

            # Overwrite
            emb2 = np.zeros(384, dtype=np.float32)
            graph.persist_embedding(node.id, emb2)

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE node_id = ?",
                (node.id,),
            ).fetchone()
            restored = np.frombuffer(row[0], dtype=np.float32)
            np.testing.assert_array_almost_equal(restored, emb2)
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# Large graph
# ------------------------------------------------------------------


class TestLargeGraph:
    """Performance check with a non-trivial number of nodes."""

    def test_hundred_nodes(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            ids = []
            for i in range(100):
                node = _make_node(
                    text=f"Scientific fact number {i} about various topics",
                    confidence=0.5 + (i % 50) * 0.01,
                )
                g1.add_node(node)
                ids.append(node.id)
            assert len(g1) == 100
            g1.close()

            # Reopen and verify
            g2 = SqliteGraph(db_path)
            assert len(g2) == 100
            for nid in ids:
                assert nid in g2
            g2.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# KnowledgeGraph API compatibility
# ------------------------------------------------------------------


class TestAPICompatibility:
    """Verify SqliteGraph works with all KnowledgeGraph methods."""

    def test_get_by_domain(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            graph.add_node(_make_node(text="Chem fact", domain="chemistry"))
            graph.add_node(_make_node(text="Phys fact", domain="physics"))

            chem = graph.get_by_domain("chemistry")
            assert len(chem) == 1
            assert chem[0].text == "Chem fact"
            graph.close()
        finally:
            os.unlink(db_path)

    def test_get_by_type(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            graph.add_node(_make_node(text="A fact", node_type=NodeType.FACT))
            graph.add_node(_make_node(text="A rule", node_type=NodeType.RULE))

            facts = graph.get_by_type(NodeType.FACT)
            assert len(facts) == 1
            assert facts[0].text == "A fact"
            graph.close()
        finally:
            os.unlink(db_path)

    def test_get_statistics(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            graph.add_node(_make_node(text="Fact 1", confidence=0.8))
            graph.add_node(_make_node(text="Fact 2", confidence=0.6))

            stats = graph.get_statistics()
            assert stats["total_nodes"] == 2
            assert "chemistry" in stats["nodes_by_domain"]
            graph.close()
        finally:
            os.unlink(db_path)

    def test_save_to_json(self):
        """SqliteGraph can still save to JSON via inherited method."""
        db_path = _tmp_db()
        json_path = _tmp_db().replace(".db", ".json")
        try:
            graph = SqliteGraph(db_path)
            graph.add_node(_make_node(text="Test node"))
            graph.save(json_path)  # Inherited from KnowledgeGraph

            loaded = KnowledgeGraph.load(json_path)
            assert len(loaded) == 1
            graph.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_isinstance_check(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            assert isinstance(graph, KnowledgeGraph)
            graph.close()
        finally:
            os.unlink(db_path)

    def test_get_predecessors_and_successors(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            n1 = _make_node(text="Base")
            n2 = KnowledgeNode(
                text="Derived",
                node_type=NodeType.INFERENCE,
                confidence=0.8,
                predecessor_nodes=[n1.id],
            )
            graph.add_node(n1)
            graph.add_node(n2)

            preds = graph.get_predecessors(n2.id)
            assert len(preds) == 1
            assert preds[0].id == n1.id

            succs = graph.get_successors(n1.id)
            assert len(succs) == 1
            assert succs[0].id == n2.id
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# Source with date
# ------------------------------------------------------------------


class TestSourceWithDate:
    """Verify Source date field serializes and deserializes correctly."""

    def test_source_date_round_trip(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            d = datetime(2025, 6, 15, 10, 30, 0)
            node = _make_node(
                sources=[
                    Source(
                        source_type="paper",
                        title="Research 2025",
                        author="Smith",
                        date=d,
                        url="https://example.com/paper",
                        quality_score=0.85,
                    )
                ]
            )
            graph.add_node(node)
            graph.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(node.id)
            src = restored.sources[0]
            assert src.author == "Smith"
            assert src.date is not None
            assert src.date.year == 2025
            assert src.date.month == 6
            assert src.url == "https://example.com/paper"
            g2.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_close_idempotent(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            graph.close()
            graph.close()  # Should not raise
        finally:
            os.unlink(db_path)

    def test_open_class_method(self):
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            g1.add_node(_make_node())
            g1.close()

            g2 = SqliteGraph.open(db_path)
            assert len(g2) == 1
            g2.close()
        finally:
            os.unlink(db_path)

    def test_remove_nonexistent_node(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            # Should not raise
            graph.remove_node("nonexistent-id")
            graph.close()
        finally:
            os.unlink(db_path)

    def test_update_nonexistent_node(self):
        db_path = _tmp_db()
        try:
            graph = SqliteGraph(db_path)
            # Should not raise
            graph.update_node_confidence("nonexistent-id", 0.9)
            graph.close()
        finally:
            os.unlink(db_path)


# ------------------------------------------------------------------
# Edge labels in SQLite
# ------------------------------------------------------------------


class TestEdgeLabelsStorage:
    """Test edge label persistence in SQLite."""

    def test_edge_labels_persisted(self):
        """Edge labels survive save and reopen."""
        db_path = _tmp_db()
        try:
            g1 = SqliteGraph(db_path)
            n1 = _make_node(text="Base fact")
            n2 = KnowledgeNode(
                text="Supporting fact",
                node_type=NodeType.FACT,
                confidence=0.8,
                domain="chemistry",
                predecessor_nodes=[n1.id],
            )
            n2.edge_labels[n1.id] = "supports"
            g1.add_node(n1)
            g1.add_node(n2)
            g1.close()

            g2 = SqliteGraph(db_path)
            restored = g2.get_node(n2.id)
            assert restored.get_edge_label(n1.id) == "supports"

            # Verify in SQLite directly
            import sqlite3

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT relation_type FROM edges WHERE predecessor_id = ? AND successor_id = ?",
                (n1.id, n2.id),
            ).fetchone()
            assert row[0] == "supports"
            conn.close()
            g2.close()
        finally:
            os.unlink(db_path)

    def test_schema_migration_v1_to_v2(self):
        """Opening a v1 database auto-migrates to v2 with relation_type column."""
        db_path = _tmp_db()
        try:
            import sqlite3 as _sqlite3

            # Create a minimal v1 database with the old schema (no relation_type)
            conn = _sqlite3.connect(db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    domain TEXT NOT NULL DEFAULT 'general',
                    status TEXT NOT NULL DEFAULT 'provisional',
                    derivation_method TEXT DEFAULT 'asserted',
                    derivation_depth INTEGER DEFAULT 0,
                    derivation_type TEXT DEFAULT 'extracted',
                    derivation_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_validated TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE TABLE edges (
                    predecessor_id TEXT NOT NULL,
                    successor_id TEXT NOT NULL,
                    PRIMARY KEY (predecessor_id, successor_id)
                );
                CREATE TABLE conflicts (
                    node_id1 TEXT NOT NULL,
                    node_id2 TEXT NOT NULL,
                    PRIMARY KEY (node_id1, node_id2)
                );
                CREATE TABLE sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT,
                    date TEXT,
                    url TEXT,
                    quality_score REAL DEFAULT 0.5
                );
                CREATE TABLE embeddings (
                    node_id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL
                );
                CREATE TABLE changelog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    node_id TEXT,
                    details TEXT
                );
                CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
                INSERT INTO schema_version VALUES (1);
            """)
            conn.close()

            # Open with SqliteGraph — should auto-migrate
            graph = SqliteGraph(db_path)

            # Verify the schema version was bumped
            conn = _sqlite3.connect(db_path)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row[0] == 2

            # Verify the relation_type column exists
            info = conn.execute("PRAGMA table_info(edges)").fetchall()
            col_names = [col[1] for col in info]
            assert "relation_type" in col_names
            conn.close()
            graph.close()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
