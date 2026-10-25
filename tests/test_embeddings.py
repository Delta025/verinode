"""
Tests for the embedding engine.

Tests cover:
- Embedding generation (single and batch)
- Index build / add / remove
- Semantic search (text query and node similarity)
- Cosine similarity computation
- Domain filtering
- Graceful handling of edge cases (empty graph, missing nodes)

Run with:  python -m pytest tests/test_embeddings.py -v
Or:        python tests/test_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from core import EmbeddingEngine, KnowledgeGraph, KnowledgeNode, NodeType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """Module-scoped EmbeddingEngine so the sentence-transformers model is
    loaded exactly once for all tests in this file. This prevents transient
    httpx.ReadTimeout errors caused by repeated model initialization calls
    that each check HuggingFace for updates."""
    return EmbeddingEngine()


def _make_graph():
    """Build a small graph with diverse topics for search testing."""
    graph = KnowledgeGraph()
    nodes_data = [
        ("Water is composed of hydrogen and oxygen", "chemistry"),
        ("The boiling point of water is 100 degrees Celsius at sea level", "chemistry"),
        ("Photosynthesis converts sunlight into chemical energy", "biology"),
        ("Mitosis is the process of cell division", "biology"),
        ("Newton's second law states that force equals mass times acceleration", "physics"),
        ("The speed of light in a vacuum is approximately 300000 km/s", "physics"),
    ]
    nodes = []
    for text, domain in nodes_data:
        node = KnowledgeNode(text=text, node_type=NodeType.FACT, confidence=0.9, domain=domain)
        graph.add_node(node)
        nodes.append(node)
    return graph, nodes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_single_text(engine):
    """Embedding a single string returns a float32 vector of the right shape."""
    vec = engine.embed_text("Water is H2O")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    assert vec.shape[0] == engine.dimension


def test_embed_batch(engine):
    """Batch embedding returns a 2-D matrix with one row per input."""
    texts = ["Water is H2O", "The sky is blue", "Gravity pulls objects down"]
    vecs = engine.embed_texts(texts)
    assert vecs.shape == (3, engine.dimension)
    assert vecs.dtype == np.float32


def test_embed_node(engine):
    """embed_node sets the node's .embedding attribute."""
    node = KnowledgeNode(text="Iron rusts when exposed to water and oxygen", confidence=0.85)
    assert node.embedding is None
    vec = engine.embed_node(node)
    assert node.embedding is not None
    assert np.array_equal(vec, node.embedding)


def test_index_graph(engine):
    """index_graph indexes all nodes and returns the count."""
    graph, nodes = _make_graph()

    count = engine.index_graph(graph)
    assert count == len(nodes)
    assert engine.index_size == len(nodes)

    # Every node should now have an embedding
    for n in nodes:
        assert n.embedding is not None


def test_add_to_index(engine):
    """Adding a node after initial indexing increases the index size."""
    graph, _ = _make_graph()
    engine.index_graph(graph)
    old_size = engine.index_size

    new_node = KnowledgeNode(text="DNA carries genetic information", confidence=0.95, domain="biology")
    graph.add_node(new_node)
    engine.add_to_index(new_node)

    assert engine.index_size == old_size + 1


def test_remove_from_index(engine):
    """Removing a node decreases the index size."""
    graph, nodes = _make_graph()
    engine.index_graph(graph)
    old_size = engine.index_size

    engine.remove_from_index(nodes[0].id)
    assert engine.index_size == old_size - 1

    # Removing a non-existent ID is a no-op
    engine.remove_from_index("nonexistent-id")
    assert engine.index_size == old_size - 1


def test_search_returns_relevant_results(engine):
    """Semantic search returns results ordered by relevance."""
    graph, _ = _make_graph()
    engine.index_graph(graph)

    results = engine.search("What is water made of?", k=3, graph=graph)
    assert len(results) > 0
    assert len(results) <= 3

    # Top result should be the water-composition node
    top_text = graph.get_node(results[0]["node_id"]).text
    assert "water" in top_text.lower() or "hydrogen" in top_text.lower()


def test_search_domain_filter(engine):
    """Domain filter restricts results to matching domain."""
    graph, _ = _make_graph()
    engine.index_graph(graph)

    results = engine.search("energy from sunlight", k=5, domain="biology", graph=graph)
    for r in results:
        node = graph.get_node(r["node_id"])
        assert node.domain == "biology"


def test_search_threshold(engine):
    """Threshold parameter filters out low-scoring results."""
    graph, _ = _make_graph()
    engine.index_graph(graph)

    # Very high threshold should return fewer (possibly zero) results
    high = engine.search("water", k=10, threshold=0.99)
    low = engine.search("water", k=10, threshold=0.0)
    assert len(high) <= len(low)


def test_find_similar(engine):
    """find_similar returns neighbors of a given node, excluding itself."""
    graph, nodes = _make_graph()
    engine.index_graph(graph)

    water_node = nodes[0]  # "Water is composed of ..."
    results = engine.find_similar(water_node.id, k=2)
    assert len(results) > 0
    assert all(r["node_id"] != water_node.id for r in results)


def test_cosine_similarity(engine):
    """cosine_similarity returns value in [-1, 1] with expected ordering."""
    graph, nodes = _make_graph()
    engine.index_graph(graph)

    water_comp = nodes[0]  # chemistry: water composition
    water_bp = nodes[1]  # chemistry: water boiling point
    mitosis = nodes[3]  # biology: cell division

    sim_same_topic = engine.cosine_similarity(water_comp.id, water_bp.id)
    sim_diff_topic = engine.cosine_similarity(water_comp.id, mitosis.id)

    assert -1.0 <= sim_same_topic <= 1.0
    assert -1.0 <= sim_diff_topic <= 1.0
    # Same-topic nodes should be more similar than cross-topic
    assert sim_same_topic > sim_diff_topic


def test_empty_graph(engine):
    """Operations on an empty graph don't crash."""
    graph = KnowledgeGraph()

    count = engine.index_graph(graph)
    assert count == 0

    results = engine.search("anything", k=5)
    assert results == []


def test_find_similar_missing_node(engine):
    """find_similar with a non-indexed node returns empty list."""
    graph, _ = _make_graph()
    engine.index_graph(graph)

    results = engine.find_similar("nonexistent-id", k=3)
    assert results == []


# ---------------------------------------------------------------------------
# Runner (for standalone execution without pytest)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public accessor tests (get_embedding, has_embedding)
# ---------------------------------------------------------------------------


def test_get_embedding_returns_copy(engine):
    """get_embedding returns a copy of the stored vector."""
    graph, nodes = _make_graph()
    engine.index_graph(graph)
    nid = nodes[0].id
    vec = engine.get_embedding(nid)
    assert vec is not None
    assert vec.shape == (engine.dimension,)
    # Mutating the copy should not affect the stored embedding
    original = engine.get_embedding(nid)
    vec[0] = -999.0
    assert engine.get_embedding(nid)[0] != -999.0
    np.testing.assert_array_equal(original, engine.get_embedding(nid))


def test_get_embedding_missing_node(engine):
    """get_embedding returns None for a node not in the index."""
    assert engine.get_embedding("nonexistent-node-id") is None


def test_has_embedding(engine):
    """has_embedding correctly reports presence."""
    graph, nodes = _make_graph()
    engine.index_graph(graph)
    assert engine.has_embedding(nodes[0].id) is True
    assert engine.has_embedding("nonexistent-node-id") is False


def test_get_embedding_after_remove(engine):
    """get_embedding returns None after a node is removed from the index."""
    graph, nodes = _make_graph()
    engine.index_graph(graph)
    nid = nodes[0].id
    assert engine.get_embedding(nid) is not None
    engine.remove_from_index(nid)
    assert engine.get_embedding(nid) is None
    assert engine.has_embedding(nid) is False


def run_all_tests():
    print("\nRunning embedding engine tests...\n")
    e = EmbeddingEngine()
    test_embed_single_text(e)
    print("  embed_single_text passed")
    test_embed_batch(e)
    print("  embed_batch passed")
    test_embed_node(e)
    print("  embed_node passed")
    test_index_graph(e)
    print("  index_graph passed")
    test_add_to_index(e)
    print("  add_to_index passed")
    test_remove_from_index(e)
    print("  remove_from_index passed")
    test_search_returns_relevant_results(e)
    print("  search_returns_relevant_results passed")
    test_search_domain_filter(e)
    print("  search_domain_filter passed")
    test_search_threshold(e)
    print("  search_threshold passed")
    test_find_similar(e)
    print("  find_similar passed")
    test_cosine_similarity(e)
    print("  cosine_similarity passed")
    test_empty_graph(e)
    print("  empty_graph passed")
    test_find_similar_missing_node(e)
    print("  find_similar_missing_node passed")
    print("\nAll embedding engine tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
