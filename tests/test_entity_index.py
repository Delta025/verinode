"""Tests for the EntityIndex inverted index."""

from unittest.mock import MagicMock

from src.core.entity_index import EntityIndex, extract_content_words
from src.core.node import KnowledgeNode

# ------------------------------------------------------------------
# extract_content_words
# ------------------------------------------------------------------


class TestExtractContentWords:
    def test_basic_extraction(self):
        words = extract_content_words("Water boils at 100 degrees Celsius")
        assert "water" in words
        assert "boils" in words
        assert "100" in words
        assert "degrees" in words
        assert "celsius" in words
        # 'at' is a stopword
        assert "at" not in words

    def test_stopwords_excluded(self):
        words = extract_content_words("the cat is on the mat")
        assert "cat" in words
        assert "mat" in words
        assert "the" not in words
        assert "is" not in words
        assert "on" not in words

    def test_single_char_excluded(self):
        words = extract_content_words("I saw a B in class X")
        # single chars excluded
        assert "saw" in words
        assert "class" in words

    def test_empty_text(self):
        assert extract_content_words("") == set()

    def test_only_stopwords(self):
        assert extract_content_words("the is a an") == set()

    def test_punctuation_stripped(self):
        words = extract_content_words("Hello, world! Foo-bar (baz).")
        assert "hello" in words
        assert "world" in words
        assert "foo" in words
        assert "bar" in words
        assert "baz" in words


# ------------------------------------------------------------------
# EntityIndex
# ------------------------------------------------------------------


def _make_node(node_id: str, text: str) -> KnowledgeNode:
    return KnowledgeNode(node_id=node_id, text=text, domain="test")


class TestEntityIndex:
    def test_index_graph(self):
        graph = MagicMock()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        n2 = _make_node("n2", "Water freezes at 0 degrees Celsius")
        graph.nodes = {"n1": n1, "n2": n2}

        idx = EntityIndex()
        count = idx.index_graph(graph)
        assert count == 2
        assert idx.size == 2

    def test_find_matches_basic(self):
        graph = MagicMock()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        n2 = _make_node("n2", "Iron melts at 1538 degrees Celsius")
        n3 = _make_node("n3", "DNA is a double helix")
        graph.nodes = {"n1": n1, "n2": n2, "n3": n3}

        idx = EntityIndex()
        idx.index_graph(graph)

        # Query shares "water" and "boils" with n1
        matches = idx.find_matches("Water boils rapidly")
        assert len(matches) >= 1
        assert matches[0][0] == "n1"

    def test_find_matches_min_overlap(self):
        graph = MagicMock()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        graph.nodes = {"n1": n1}

        idx = EntityIndex()
        idx.index_graph(graph)

        # Only one shared word ("water") — below default min_overlap=2
        matches = idx.find_matches("Water is wet")
        # "water" and "wet" are content words but only "water" overlaps
        assert len(matches) == 0

    def test_find_matches_custom_min_overlap(self):
        graph = MagicMock()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        graph.nodes = {"n1": n1}

        idx = EntityIndex()
        idx.index_graph(graph)

        matches = idx.find_matches("Water is wet", min_overlap=1)
        assert len(matches) == 1
        assert matches[0][0] == "n1"

    def test_find_matches_exclude_ids(self):
        graph = MagicMock()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        n2 = _make_node("n2", "Water freezes at 0 degrees Celsius")
        graph.nodes = {"n1": n1, "n2": n2}

        idx = EntityIndex()
        idx.index_graph(graph)

        matches = idx.find_matches(
            "Water temperature in degrees Celsius",
            exclude_ids={"n1"},
        )
        node_ids = {m[0] for m in matches}
        assert "n1" not in node_ids
        assert "n2" in node_ids

    def test_find_matches_sorted_by_overlap(self):
        graph = MagicMock()
        # n1 has 3 overlapping words with query
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        # n2 has 2 overlapping words with query
        n2 = _make_node("n2", "Water freezes quickly")
        graph.nodes = {"n1": n1, "n2": n2}

        idx = EntityIndex()
        idx.index_graph(graph)

        matches = idx.find_matches("Water boils in degrees")
        assert len(matches) >= 1
        # n1 should have more overlap than n2
        if len(matches) >= 2:
            assert matches[0][1] >= matches[1][1]

    def test_add_node(self):
        idx = EntityIndex()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        idx.add_node(n1)
        assert idx.size == 1

        matches = idx.find_matches("Water boils rapidly")
        assert len(matches) == 1

    def test_add_node_update(self):
        idx = EntityIndex()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        idx.add_node(n1)

        # Update text
        n1_updated = _make_node("n1", "Iron melts at high temperatures")
        idx.add_node(n1_updated)
        assert idx.size == 1

        # Old text shouldn't match
        matches = idx.find_matches("Water boils rapidly")
        assert len(matches) == 0

        # New text should match
        matches = idx.find_matches("Iron melts easily")
        assert len(matches) == 1

    def test_remove_node(self):
        idx = EntityIndex()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        n2 = _make_node("n2", "Water freezes at 0 degrees Celsius")
        idx.add_node(n1)
        idx.add_node(n2)
        assert idx.size == 2

        idx.remove_node("n1")
        assert idx.size == 1

        matches = idx.find_matches("Water boils at 100 degrees")
        # Only n2 remains — shares "water" and "degrees" but not "boils" or "100"
        node_ids = {m[0] for m in matches}
        assert "n1" not in node_ids

    def test_remove_nonexistent_node(self):
        idx = EntityIndex()
        # Should not raise
        idx.remove_node("nonexistent")

    def test_empty_query(self):
        idx = EntityIndex()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        idx.add_node(n1)
        assert idx.find_matches("") == []

    def test_stopword_only_query(self):
        idx = EntityIndex()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        idx.add_node(n1)
        assert idx.find_matches("the is a an") == []

    def test_index_graph_clears_previous(self):
        graph1 = MagicMock()
        graph1.nodes = {"n1": _make_node("n1", "Alpha beta gamma")}

        graph2 = MagicMock()
        graph2.nodes = {"n2": _make_node("n2", "Delta epsilon zeta")}

        idx = EntityIndex()
        idx.index_graph(graph1)
        assert idx.size == 1

        idx.index_graph(graph2)
        assert idx.size == 1
        # Old data gone
        assert idx.find_matches("Alpha beta gamma") == []
        # New data present
        matches = idx.find_matches("Delta epsilon zeta")
        assert len(matches) == 1
        assert matches[0][0] == "n2"

    def test_node_with_empty_text(self):
        idx = EntityIndex()
        n1 = _make_node("n1", "")
        idx.add_node(n1)
        # Empty text yields no words, node not indexed
        assert idx.size == 0


# ------------------------------------------------------------------
# BM25 search
# ------------------------------------------------------------------


class TestBM25Search:
    def _build_index(self):
        """Build an index with nodes where 'planet' is common and 'neptune' is rare."""
        idx = EntityIndex()
        # 'planet' appears in many nodes; 'neptune' is rare
        idx.add_node(_make_node("n1", "Mercury is the closest planet to the Sun"))
        idx.add_node(_make_node("n2", "Venus is the second planet from the Sun"))
        idx.add_node(_make_node("n3", "Earth is the third planet from the Sun"))
        idx.add_node(_make_node("n4", "Mars is the fourth planet from the Sun"))
        idx.add_node(_make_node("n5", "Jupiter is the largest planet in the Solar System"))
        idx.add_node(_make_node("n6", "Saturn is known for its rings"))
        idx.add_node(_make_node("n7", "Uranus rotates on its side"))
        idx.add_node(_make_node("n8", "Neptune is the eighth planet from the Sun"))
        idx.add_node(_make_node("n9", "Neptune has 16 known moons"))
        idx.add_node(_make_node("n10", "The Sun is a yellow dwarf star"))
        return idx

    def test_bm25_basic(self):
        """BM25 search returns results sorted by score."""
        idx = self._build_index()
        results = idx.bm25_search("Neptune is a planet")
        assert len(results) > 0
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_idf_weighting(self):
        """Rare words like 'neptune' should boost ranking over common words like 'planet'."""
        idx = self._build_index()
        results = idx.bm25_search("Neptune is a planet", top_k=5)
        top_ids = [nid for nid, _ in results[:3]]
        # Neptune nodes should rank in the top 3 because "neptune" has high IDF
        assert "n8" in top_ids or "n9" in top_ids

    def test_bm25_empty_index(self):
        """BM25 on empty index returns empty list."""
        idx = EntityIndex()
        assert idx.bm25_search("Neptune is a planet") == []

    def test_bm25_empty_query(self):
        """BM25 with empty/stopword-only query returns empty list."""
        idx = self._build_index()
        assert idx.bm25_search("") == []
        assert idx.bm25_search("the is a an") == []

    def test_bm25_exclude_ids(self):
        """Excluded node IDs should not appear in results."""
        idx = self._build_index()
        results = idx.bm25_search("Neptune is a planet", exclude_ids={"n8"})
        result_ids = {nid for nid, _ in results}
        assert "n8" not in result_ids

    def test_bm25_top_k(self):
        """Results should be capped at top_k."""
        idx = self._build_index()
        results = idx.bm25_search("planet Sun Solar System", top_k=3)
        assert len(results) <= 3

    def test_bm25_incremental_add(self):
        """Adding a node updates BM25 statistics."""
        idx = EntityIndex()
        idx.add_node(_make_node("n1", "Water boils at 100 degrees"))
        assert idx._corpus_size == 1
        assert idx._doc_lengths["n1"] > 0
        old_total = idx._total_doc_length

        idx.add_node(_make_node("n2", "Ice melts at zero degrees"))
        assert idx._corpus_size == 2
        assert idx._total_doc_length > old_total

    def test_bm25_incremental_remove(self):
        """Removing a node updates BM25 statistics."""
        idx = EntityIndex()
        idx.add_node(_make_node("n1", "Water boils at 100 degrees"))
        idx.add_node(_make_node("n2", "Ice melts at zero degrees"))
        assert idx._corpus_size == 2

        idx.remove_node("n1")
        assert idx._corpus_size == 1
        assert "n1" not in idx._doc_lengths

    def test_doc_length_tracking(self):
        """Verify _doc_lengths, _total_doc_length, _corpus_size are correct."""
        idx = EntityIndex()
        n1 = _make_node("n1", "Water boils at 100 degrees Celsius")
        n2 = _make_node("n2", "Ice melts")
        idx.add_node(n1)
        idx.add_node(n2)

        from src.core.entity_index import extract_content_words

        words1 = extract_content_words(n1.text)
        words2 = extract_content_words(n2.text)
        assert idx._doc_lengths["n1"] == len(words1)
        assert idx._doc_lengths["n2"] == len(words2)
        assert idx._total_doc_length == len(words1) + len(words2)
        assert idx._corpus_size == 2

    def test_index_graph_resets_bm25_stats(self):
        """index_graph clears BM25 statistics before re-indexing."""
        idx = EntityIndex()
        idx.add_node(_make_node("n1", "Alpha beta gamma"))
        assert idx._corpus_size == 1

        graph = MagicMock()
        graph.nodes = {"n2": _make_node("n2", "Delta epsilon")}
        idx.index_graph(graph)
        assert idx._corpus_size == 1
        assert "n1" not in idx._doc_lengths
        assert "n2" in idx._doc_lengths

    def test_bm25_scores_positive(self):
        """All BM25 scores should be positive."""
        idx = self._build_index()
        results = idx.bm25_search("Neptune planet Sun")
        for _, score in results:
            assert score > 0.0

    def test_bm25_node_update_reindexes(self):
        """Updating a node via add_node should update BM25 stats correctly."""
        idx = EntityIndex()
        idx.add_node(_make_node("n1", "Water boils at 100 degrees"))
        old_dl = idx._doc_lengths["n1"]

        # Update with different text
        idx.add_node(_make_node("n1", "Short text"))
        assert idx._corpus_size == 1  # still one node
        assert idx._doc_lengths["n1"] != old_dl  # length changed
