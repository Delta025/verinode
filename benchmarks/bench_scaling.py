"""
Benchmark: Scaling & Performance

Measures how core operations scale with graph size:
1. Embedding search latency vs graph size (50, 100, 200, 500 nodes)
2. Dependency detection time (with and without embeddings)
3. Confidence propagation cost (linear chain vs branching graph)
4. SQLite save/load time vs graph size
5. Node addition throughput

All scenarios are purely programmatic (no LLM needed).

Usage:
    python -m benchmarks.bench_scaling
"""

import os
import sys
import time
import tempfile
import statistics

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, Source
from src.core.embeddings import EmbeddingEngine
from src.core.dependency_detection import DependencyDetector
from src.core.confidence import ConfidenceEngine
from src.core.storage import SqliteGraph
from src.core.metrics import BenchmarkResult, save_report

RESULTS_DIR = os.path.join("data", "calibration", "results")

# Science sentences for realistic node text
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
    "The human body has 206 bones in the adult skeleton.",
    "Electrons orbit the nucleus of an atom in defined energy levels.",
    "The pH scale measures how acidic or basic a solution is.",
    "Tectonic plates move causing earthquakes and volcanic eruptions.",
    "The atmosphere is composed primarily of nitrogen and oxygen.",
    "Momentum is conserved in a closed system.",
    "The human brain contains approximately 86 billion neurons.",
    "Enzymes are biological catalysts that speed up chemical reactions.",
    "The observable universe is approximately 93 billion light years in diameter.",
    "Chlorophyll absorbs light in the red and blue wavelengths.",
]


def _make_graph(n_nodes):
    """Create a graph with n_nodes realistic science nodes."""
    graph = KnowledgeGraph()
    for i in range(n_nodes):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        # Add slight variation to avoid exact duplicates
        text = f"{text} (variant {i})"
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
    """Create a linear chain: node0 → node1 → ... → nodeN."""
    graph = KnowledgeGraph()
    prev = None
    for i in range(n_nodes):
        node = KnowledgeNode(
            text=f"Chain node {i}: {SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]}",
            node_type=NodeType.FACT,
            confidence=0.8,
            domain="science",
        )
        graph.add_node(node)
        if prev is not None:
            node.add_predecessor(prev.id)
            prev.add_successor(node.id)
        prev = node
    return graph


def _make_branching_graph(depth, branching_factor=3):
    """Create a tree graph with given depth and branching factor."""
    graph = KnowledgeGraph()
    root = KnowledgeNode(
        text="Root fact of branching graph",
        node_type=NodeType.FACT,
        confidence=0.9,
        domain="science",
    )
    graph.add_node(root)

    current_level = [root]
    for d in range(depth):
        next_level = []
        for parent in current_level:
            for b in range(branching_factor):
                child = KnowledgeNode(
                    text=f"Branch d{d} b{b}: {SAMPLE_TEXTS[(d * branching_factor + b) % len(SAMPLE_TEXTS)]}",
                    node_type=NodeType.FACT,
                    confidence=0.7,
                    domain="science",
                )
                graph.add_node(child)
                child.add_predecessor(parent.id)
                parent.add_successor(child.id)
                next_level.append(child)
        current_level = next_level

    return graph, root


def bench_search_latency():
    """Measure embedding search latency at different graph sizes."""
    print("\n--- 1. Embedding Search Latency ---")
    sizes = [50, 100, 200, 500]
    results = {}

    embed = EmbeddingEngine()
    query = "Water boils at high temperature under normal conditions"

    for n in sizes:
        graph = _make_graph(n)
        embed.index_graph(graph)

        # Warm up
        embed.search(query, k=5, threshold=0.0)

        # Time 10 queries
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            embed.search(query, k=5, threshold=0.0)
            latencies.append(time.perf_counter() - t0)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        results[n] = {"avg_ms": avg * 1000, "p95_ms": p95 * 1000}
        print(f"  {n:>4} nodes: avg={avg * 1000:.2f}ms  p95={p95 * 1000:.2f}ms")

    return results


def bench_index_build():
    """Measure index build time at different graph sizes."""
    print("\n--- 2. Index Build Time ---")
    sizes = [50, 100, 200, 500]
    results = {}

    for n in sizes:
        graph = _make_graph(n)
        embed = EmbeddingEngine()

        t0 = time.perf_counter()
        embed.index_graph(graph)
        elapsed = time.perf_counter() - t0

        results[n] = {"build_time_s": elapsed}
        print(f"  {n:>4} nodes: {elapsed:.3f}s ({elapsed / n * 1000:.1f}ms/node)")

    return results


def bench_add_to_index():
    """Measure incremental add-to-index time."""
    print("\n--- 3. Incremental Add-to-Index ---")
    graph = _make_graph(100)
    embed = EmbeddingEngine()
    embed.index_graph(graph)

    # Add 50 more nodes one-by-one
    latencies = []
    for i in range(50):
        node = KnowledgeNode(
            text=f"New fact {i}: {SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]} additional",
            node_type=NodeType.FACT,
            confidence=0.7,
            domain="science",
        )
        graph.add_node(node)

        t0 = time.perf_counter()
        embed.add_to_index(node)
        latencies.append(time.perf_counter() - t0)

    avg = statistics.mean(latencies)
    last_10_avg = statistics.mean(latencies[-10:])
    results = {
        "avg_ms": avg * 1000,
        "last_10_avg_ms": last_10_avg * 1000,
        "total_s": sum(latencies),
    }
    print(f"  Add 50 nodes: avg={avg * 1000:.2f}ms/node  "
          f"last 10 avg={last_10_avg * 1000:.2f}ms/node  "
          f"total={sum(latencies):.2f}s")

    return results


def bench_dependency_detection():
    """Measure dependency detection time at different graph sizes."""
    print("\n--- 4. Dependency Detection Time ---")
    sizes = [20, 50, 100]
    results = {}

    for n in sizes:
        graph = _make_graph(n)
        embed = EmbeddingEngine()
        embed.index_graph(graph)
        detector = DependencyDetector()

        # With embeddings
        t0 = time.perf_counter()
        deps = detector.detect_all(graph, embedding_engine=embed)
        elapsed_embed = time.perf_counter() - t0

        results[n] = {
            "with_embeddings_s": elapsed_embed,
            "deps_found": len(deps),
            "per_node_ms": elapsed_embed / n * 1000,
        }
        print(f"  {n:>4} nodes (embeddings): {elapsed_embed:.3f}s  "
              f"{len(deps)} deps  {elapsed_embed / n * 1000:.1f}ms/node")

    return results


def bench_propagation():
    """Measure confidence propagation on different graph structures."""
    print("\n--- 5. Confidence Propagation ---")
    conf_engine = ConfidenceEngine()
    results = {}

    # Linear chain
    for length in [10, 50, 100, 200]:
        graph = _make_linear_chain(length)
        root_id = list(graph.nodes.keys())[0]
        graph.nodes[root_id].confidence = 0.5

        t0 = time.perf_counter()
        affected = conf_engine.propagate_confidence_update(graph, root_id)
        elapsed = time.perf_counter() - t0

        key = f"chain_{length}"
        results[key] = {"elapsed_s": elapsed, "affected": len(affected)}
        print(f"  Chain({length:>3}): {elapsed:.4f}s  {len(affected)} affected")

    # Branching tree
    for depth in [2, 3, 4]:
        graph, root = _make_branching_graph(depth, branching_factor=3)
        total = len(graph.nodes)
        root.confidence = 0.5

        t0 = time.perf_counter()
        affected = conf_engine.propagate_confidence_update(graph, root.id)
        elapsed = time.perf_counter() - t0

        key = f"tree_d{depth}_b3"
        results[key] = {"elapsed_s": elapsed, "affected": len(affected), "total": total}
        print(f"  Tree(d={depth}, b=3, n={total:>4}): {elapsed:.4f}s  "
              f"{len(affected)} affected")

    return results


def bench_sqlite_io():
    """Measure SQLite save and load times at different graph sizes."""
    print("\n--- 6. SQLite Save/Load ---")
    sizes = [50, 100, 200, 500]
    results = {}

    for n in sizes:
        graph = _make_graph(n)

        # Add embeddings
        embed = EmbeddingEngine()
        embed.index_graph(graph)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Save
            t0 = time.perf_counter()
            sqlite_graph = SqliteGraph.from_graph(graph, db_path)
            save_time = time.perf_counter() - t0
            sqlite_graph.close()

            db_size = os.path.getsize(db_path) / 1024  # KB

            # Load
            t0 = time.perf_counter()
            loaded = SqliteGraph.open(db_path)
            load_time = time.perf_counter() - t0
            loaded_count = len(loaded.nodes)
            loaded.close()

            results[n] = {
                "save_s": save_time,
                "load_s": load_time,
                "db_size_kb": db_size,
                "nodes_loaded": loaded_count,
            }
            print(f"  {n:>4} nodes: save={save_time:.3f}s  load={load_time:.3f}s  "
                  f"db={db_size:.0f}KB  loaded={loaded_count}")
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    return results


def bench_node_throughput():
    """Measure raw node addition throughput."""
    print("\n--- 7. Node Addition Throughput ---")
    results = {}

    for n in [100, 500, 1000]:
        graph = KnowledgeGraph()

        t0 = time.perf_counter()
        for i in range(n):
            node = KnowledgeNode(
                text=f"Throughput test node {i}",
                node_type=NodeType.FACT,
                confidence=0.7,
                domain="science",
            )
            graph.add_node(node)
        elapsed = time.perf_counter() - t0

        throughput = n / elapsed
        results[n] = {"elapsed_s": elapsed, "throughput_nodes_per_sec": throughput}
        print(f"  {n:>5} nodes: {elapsed:.4f}s  ({throughput:.0f} nodes/sec)")

    return results


def run():
    print("=" * 60)
    print("Benchmark: Scaling & Performance")
    print("=" * 60)

    all_results = {}
    all_results["search_latency"] = bench_search_latency()
    all_results["index_build"] = bench_index_build()
    all_results["add_to_index"] = bench_add_to_index()
    all_results["dependency_detection"] = bench_dependency_detection()
    all_results["propagation"] = bench_propagation()
    all_results["sqlite_io"] = bench_sqlite_io()
    all_results["node_throughput"] = bench_node_throughput()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Search (500 nodes): "
          f"{all_results['search_latency'].get(500, {}).get('avg_ms', 'N/A')}ms avg")
    print(f"Index build (500 nodes): "
          f"{all_results['index_build'].get(500, {}).get('build_time_s', 'N/A')}s")
    print(f"Dependency detection (100 nodes): "
          f"{all_results['dependency_detection'].get(100, {}).get('with_embeddings_s', 'N/A')}s")
    print(f"SQLite save (500 nodes): "
          f"{all_results['sqlite_io'].get(500, {}).get('save_s', 'N/A')}s")
    print(f"SQLite load (500 nodes): "
          f"{all_results['sqlite_io'].get(500, {}).get('load_s', 'N/A')}s")
    print(f"Node throughput (1000): "
          f"{all_results['node_throughput'].get(1000, {}).get('throughput_nodes_per_sec', 'N/A')} nodes/sec")

    # Build result
    result = BenchmarkResult(
        name="scaling",
        metrics={
            "search_avg_ms_500": all_results["search_latency"].get(500, {}).get("avg_ms", 0),
            "index_build_s_500": all_results["index_build"].get(500, {}).get("build_time_s", 0),
            "dep_detect_s_100": all_results["dependency_detection"].get(100, {}).get("with_embeddings_s", 0),
            "sqlite_save_s_500": all_results["sqlite_io"].get(500, {}).get("save_s", 0),
            "sqlite_load_s_500": all_results["sqlite_io"].get(500, {}).get("load_s", 0),
            "node_throughput_1000": all_results["node_throughput"].get(1000, {}).get("throughput_nodes_per_sec", 0),
        },
        thresholds={},
        details=[all_results],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "scaling_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
