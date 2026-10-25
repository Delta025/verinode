"""
Calibration regression tests for scaling and performance.

Verifies that core operations complete within acceptable time limits
at various graph sizes. All tests are purely programmatic (no LLM needed).

Run with: pytest tests/test_calibration_scaling.py --run-calibration
"""

import os
import tempfile
import time

import pytest

from src.core.confidence import ConfidenceEngine
from src.core.dependency_detection import DependencyDetector
from src.core.embeddings import EmbeddingEngine
from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, Source
from src.core.storage import SqliteGraph

pytestmark = pytest.mark.calibration

# Reusable science sentences
SAMPLE_TEXTS = [
    "Water boils at 100 degrees Celsius at standard pressure.",
    "The Earth orbits the Sun once every 365.25 days.",
    "Light travels at approximately 300000 kilometers per second.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "DNA carries the genetic instructions for all living organisms.",
    "The periodic table organizes elements by atomic number.",
    "Gravity is proportional to mass and inversely proportional to distance squared.",
    "Cells are the basic building blocks of all living organisms.",
    "The speed of sound in air is approximately 343 meters per second.",
    "Mitochondria are the powerhouses of the cell.",
]


def _make_graph(n_nodes):
    """Create a graph with n_nodes nodes."""
    graph = KnowledgeGraph()
    for i in range(n_nodes):
        text = f"{SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]} (variant {i})"
        node = KnowledgeNode(
            text=text,
            node_type=NodeType.FACT,
            confidence=0.5 + 0.4 * (i % 10) / 10,
            domain="science",
            sources=[Source(source_type="test", title=f"source_{i % 5}", quality_score=0.8)],
        )
        graph.add_node(node)
    return graph


def _make_linear_chain(n_nodes):
    """Create a linear chain graph."""
    graph = KnowledgeGraph()
    prev = None
    for i in range(n_nodes):
        node = KnowledgeNode(
            text=f"Chain node {i}",
            node_type=NodeType.FACT,
            confidence=0.8,
            domain="science",
        )
        graph.add_node(node)
        if prev:
            node.add_predecessor(prev.id)
            prev.add_successor(node.id)
        prev = node
    return graph


# ── Search Latency ─────────────────────────────────────────────────────


class TestSearchLatency:
    @pytest.fixture(scope="class")
    def embed(self):
        return EmbeddingEngine()

    def test_search_200_nodes_under_100ms(self, embed):
        """Search on 200-node graph should complete under 100ms."""
        graph = _make_graph(200)
        embed.index_graph(graph)

        # Warm up
        embed.search("test query", k=5, threshold=0.0)

        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            results = embed.search("Water boils at high temperature", k=5, threshold=0.0)
            latencies.append(time.perf_counter() - t0)

        avg_ms = sum(latencies) / len(latencies) * 1000
        assert avg_ms < 100, f"Search avg {avg_ms:.1f}ms > 100ms on 200 nodes"
        assert len(results) > 0, "Search should return results"

    def test_search_returns_relevant_results(self, embed):
        """Search should return relevant results, not random ones."""
        graph = _make_graph(100)
        embed.index_graph(graph)

        results = embed.search("Water boils at high temperature", k=5, threshold=0.3)
        assert len(results) > 0, "Should find relevant results above threshold 0.3"

        # Top result should have reasonable similarity
        top_score = results[0]["score"]
        assert top_score > 0.5, f"Top result score {top_score:.3f} too low"


# ── Index Build Time ───────────────────────────────────────────────────


class TestIndexBuild:
    def test_index_200_nodes_under_30s(self):
        """Building index for 200 nodes should complete under 30s."""
        graph = _make_graph(200)
        embed = EmbeddingEngine()

        t0 = time.perf_counter()
        embed.index_graph(graph)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30, f"Index build took {elapsed:.1f}s > 30s for 200 nodes"

    def test_add_to_index_under_500ms(self):
        """Adding a single node to index should be under 500ms."""
        graph = _make_graph(100)
        embed = EmbeddingEngine()
        embed.index_graph(graph)

        node = KnowledgeNode(
            text="New fact about chemistry",
            node_type=NodeType.FACT,
            confidence=0.7,
            domain="science",
        )
        graph.add_node(node)

        t0 = time.perf_counter()
        embed.add_to_index(node)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5, f"Add-to-index took {elapsed:.3f}s > 500ms"


# ── Dependency Detection ───────────────────────────────────────────────


class TestDependencyDetection:
    def test_detect_50_nodes_under_30s(self):
        """Dependency detection on 50 nodes should complete under 30s."""
        graph = _make_graph(50)
        embed = EmbeddingEngine()
        embed.index_graph(graph)
        detector = DependencyDetector()

        t0 = time.perf_counter()
        deps = detector.detect_all(graph, embedding_engine=embed)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30, f"Dependency detection took {elapsed:.1f}s > 30s for 50 nodes"
        assert isinstance(deps, list)


# ── Confidence Propagation ─────────────────────────────────────────────


class TestConfidencePropagation:
    def test_chain_100_under_1s(self):
        """Propagation on 100-node chain should complete under 1s."""
        graph = _make_linear_chain(100)
        conf_engine = ConfidenceEngine()

        root_id = list(graph.nodes.keys())[0]
        graph.nodes[root_id].confidence = 0.5

        t0 = time.perf_counter()
        affected = conf_engine.propagate_confidence_update(graph, root_id)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"Propagation took {elapsed:.3f}s > 1s on 100-node chain"
        assert len(affected) > 0, "Should affect some nodes"

    def test_chain_200_under_2s(self):
        """Propagation on 200-node chain should complete under 2s."""
        graph = _make_linear_chain(200)
        conf_engine = ConfidenceEngine()

        root_id = list(graph.nodes.keys())[0]
        graph.nodes[root_id].confidence = 0.5

        t0 = time.perf_counter()
        conf_engine.propagate_confidence_update(graph, root_id)
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, f"Propagation took {elapsed:.3f}s > 2s on 200-node chain"

    def test_propagation_scales_linearly(self):
        """Propagation time should scale roughly linearly with chain length."""
        conf_engine = ConfidenceEngine()
        times = {}

        for length in [50, 100]:
            graph = _make_linear_chain(length)
            root_id = list(graph.nodes.keys())[0]
            graph.nodes[root_id].confidence = 0.5

            t0 = time.perf_counter()
            conf_engine.propagate_confidence_update(graph, root_id)
            times[length] = time.perf_counter() - t0

        # 100-node should not be more than 5x slower than 50-node
        # (generous bound to avoid flaky tests)
        if times[50] > 0:
            ratio = times[100] / times[50]
            assert ratio < 5.0, f"Propagation ratio {ratio:.1f}x (50→100 nodes) suggests > linear scaling"


# ── SQLite I/O ─────────────────────────────────────────────────────────


class TestSqliteIO:
    def test_save_200_nodes_under_10s(self):
        """Saving 200 nodes to SQLite should complete under 10s."""
        graph = _make_graph(200)
        embed = EmbeddingEngine()
        embed.index_graph(graph)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            t0 = time.perf_counter()
            sqlite_graph = SqliteGraph.from_graph(graph, db_path)
            elapsed = time.perf_counter() - t0
            sqlite_graph.close()

            assert elapsed < 10, f"SQLite save took {elapsed:.1f}s > 10s for 200 nodes"
        finally:
            os.unlink(db_path)

    def test_load_200_nodes_under_5s(self):
        """Loading 200 nodes from SQLite should complete under 5s."""
        graph = _make_graph(200)
        embed = EmbeddingEngine()
        embed.index_graph(graph)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            sqlite_graph = SqliteGraph.from_graph(graph, db_path)
            sqlite_graph.close()

            t0 = time.perf_counter()
            loaded = SqliteGraph.open(db_path)
            elapsed = time.perf_counter() - t0
            loaded.close()

            assert elapsed < 5, f"SQLite load took {elapsed:.1f}s > 5s for 200 nodes"
            assert len(loaded.nodes) == 200
        finally:
            os.unlink(db_path)

    def test_save_load_preserves_data(self):
        """SQLite round-trip should preserve all nodes."""
        graph = _make_graph(50)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            sqlite_graph = SqliteGraph.from_graph(graph, db_path)
            sqlite_graph.close()

            loaded = SqliteGraph.open(db_path)
            assert len(loaded.nodes) == 50

            # Check a sample node
            orig_ids = set(graph.nodes.keys())
            loaded_ids = set(loaded.nodes.keys())
            assert orig_ids == loaded_ids, "Node IDs should match after round-trip"
            loaded.close()
        finally:
            os.unlink(db_path)


# ── Node Throughput ────────────────────────────────────────────────────


class TestNodeThroughput:
    def test_add_500_nodes_under_1s(self):
        """Adding 500 nodes should complete under 1 second."""
        graph = KnowledgeGraph()

        t0 = time.perf_counter()
        for i in range(500):
            node = KnowledgeNode(
                text=f"Throughput node {i}",
                node_type=NodeType.FACT,
                confidence=0.7,
                domain="science",
            )
            graph.add_node(node)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"Adding 500 nodes took {elapsed:.3f}s > 1s"
        assert len(graph.nodes) == 500

    def test_get_statistics_500_nodes_under_100ms(self):
        """Statistics on 500-node graph should compute in < 100ms."""
        graph = _make_graph(500)

        t0 = time.perf_counter()
        stats = graph.get_statistics()
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Statistics took {elapsed:.3f}s > 100ms on 500 nodes"
        assert stats["total_nodes"] == 500
