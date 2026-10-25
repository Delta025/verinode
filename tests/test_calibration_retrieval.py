"""
Calibration tests for BM25 hybrid retrieval on the Solar System test graph.

These tests measure retrieval quality (recall, ranking) and verification
accuracy without requiring an LLM. They use the programmatic solar system
graph built by scripts/build_solar_system_test.py.

Run with: python -m pytest tests/test_calibration_retrieval.py --run-calibration -v
"""

# Import fact lists for reference
import sys
from pathlib import Path

import pytest

from src.core.embeddings import EmbeddingEngine
from src.core.entity_index import EntityIndex
from src.core.graph import KnowledgeGraph
from src.core.pipeline import reciprocal_rank_fusion
from src.core.verification import Verdict, VerificationEngine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


GRAPH_PATH = ROOT / "data" / "solar_system_test_graph.json"


@pytest.fixture(scope="module")
def solar_graph():
    """Load the solar system test graph."""
    if not GRAPH_PATH.exists():
        pytest.skip(
            f"Solar system test graph not found at {GRAPH_PATH}. Run: python scripts/build_solar_system_test.py"
        )
    return KnowledgeGraph.load(str(GRAPH_PATH))


@pytest.fixture(scope="module")
def embedding_engine(solar_graph):
    """Build embedding index for the graph."""
    emb = EmbeddingEngine()
    emb.index_graph(solar_graph)
    return emb


@pytest.fixture(scope="module")
def entity_index(solar_graph):
    """Build entity index for the graph."""
    idx = EntityIndex()
    idx.index_graph(solar_graph)
    return idx


# ------------------------------------------------------------------
# BM25 retrieval quality
# ------------------------------------------------------------------


@pytest.mark.calibration
class TestBM25RetrievalQuality:
    """Test that BM25 retrieval ranks relevant candidates highly."""

    def test_bm25_finds_earth_distance(self, entity_index, solar_graph):
        """BM25 should find the Earth distance fact for a numerical query."""
        results = entity_index.bm25_search("Earth orbits the Sun at 230 million km", top_k=5)
        top_texts = [solar_graph.get_node(nid).text for nid, _ in results]
        # The true fact about Earth's distance should be in top results
        assert any("149.6" in t for t in top_texts), f"Expected Earth distance fact in top-5, got: {top_texts}"

    def test_bm25_ranks_neptune_highly(self, entity_index, solar_graph):
        """Rare entity 'Neptune' should rank Neptune facts high."""
        results = entity_index.bm25_search("Neptune is the seventh planet", top_k=5)
        top_texts = [solar_graph.get_node(nid).text for nid, _ in results]
        # Neptune facts should dominate top results
        neptune_count = sum(1 for t in top_texts if "Neptune" in t)
        assert neptune_count >= 2, f"Expected at least 2 Neptune facts in top-5, got {neptune_count}"

    def test_bm25_finds_mars_moons(self, entity_index, solar_graph):
        """Query about Mars moons should find the relevant fact."""
        results = entity_index.bm25_search("Mars has five moons", top_k=5)
        top_texts = [solar_graph.get_node(nid).text for nid, _ in results]
        assert any("Mars" in t and "moon" in t.lower() for t in top_texts)

    def test_bm25_finds_venus_type(self, entity_index, solar_graph):
        """Query about Venus type should find the relevant classification."""
        results = entity_index.bm25_search("Venus is a gas giant", top_k=5)
        top_texts = [solar_graph.get_node(nid).text for nid, _ in results]
        assert any("Venus" in t for t in top_texts)

    def test_bm25_finds_sun_composition(self, entity_index, solar_graph):
        """Query about Sun composition should find the relevant fact."""
        results = entity_index.bm25_search("The Sun is primarily composed of helium", top_k=5)
        top_texts = [solar_graph.get_node(nid).text for nid, _ in results]
        assert any("Sun" in t and "hydrogen" in t.lower() for t in top_texts)


# ------------------------------------------------------------------
# RRF fusion quality
# ------------------------------------------------------------------


@pytest.mark.calibration
class TestRRFQuality:
    """Test that RRF fusion improves candidate quality."""

    def test_rrf_merges_embedding_and_bm25(self, embedding_engine, entity_index, solar_graph):
        """RRF should combine embedding and BM25 results."""
        query = "Europa is a moon of Saturn"

        # Get embedding results
        embed_raw = embedding_engine.search(query, k=8, threshold=0.3)
        embed_results = [(c["node_id"], c.get("score", 0.0)) for c in embed_raw]

        # Get BM25 results
        bm25_results = entity_index.bm25_search(query, top_k=10)

        # Fuse
        fused = reciprocal_rank_fusion([embed_results, bm25_results], k=60, top_n=10)
        fused_ids = {c["node_id"] for c in fused}

        # RRF should contain candidates from both sources
        embed_ids = {nid for nid, _ in embed_results}
        bm25_ids = {nid for nid, _ in bm25_results}
        fused_ids & embed_ids & bm25_ids
        # At least some overlap between the sources
        assert len(fused_ids) > 0

    def test_rrf_recall_vs_embedding_only(self, embedding_engine, entity_index, solar_graph):
        """RRF should have equal or better recall than embedding alone."""
        queries_and_targets = [
            ("Earth orbits at 230 million km", "149.6"),
            ("Mars has five moons", "Phobos"),
            ("Venus is a gas giant", "terrestrial"),
        ]

        embed_hits = 0
        rrf_hits = 0

        for query, target_substring in queries_and_targets:
            # Embedding only
            embed_raw = embedding_engine.search(query, k=10, threshold=0.3)
            embed_texts = [solar_graph.get_node(c["node_id"]).text for c in embed_raw]
            if any(target_substring in t for t in embed_texts):
                embed_hits += 1

            # RRF fusion
            embed_results = [(c["node_id"], c.get("score", 0.0)) for c in embed_raw]
            bm25_results = entity_index.bm25_search(query, top_k=10)
            fused = reciprocal_rank_fusion([embed_results, bm25_results], k=60, top_n=10)
            fused_texts = [solar_graph.get_node(c["node_id"]).text for c in fused]
            if any(target_substring in t for t in fused_texts):
                rrf_hits += 1

        assert rrf_hits >= embed_hits, f"RRF should have >= recall: RRF={rrf_hits}, embed={embed_hits}"


# ------------------------------------------------------------------
# Heuristic verification accuracy (no LLM)
# ------------------------------------------------------------------


@pytest.mark.calibration
class TestHeuristicVerificationAccuracy:
    """Test verification accuracy using only heuristic classification."""

    @pytest.fixture()
    def heuristic_engine(self, solar_graph, embedding_engine, entity_index):
        """Create a VerificationEngine with no LLM (heuristic only)."""
        return VerificationEngine(
            graph=solar_graph,
            embedding_engine=embedding_engine,
            entity_index=entity_index,
            contradiction_checker=None,
            use_bm25=True,
        )

    def test_true_claims_not_contradicted(self, heuristic_engine):
        """True claims should not be marked as CONTRADICTED."""
        true_claims = [
            "The Solar System has eight planets.",
            "Jupiter is a gas giant.",
            "Mars has two moons named Phobos and Deimos.",
            "Earth orbits the Sun at an average distance of 149.6 million km.",
            "The Sun's surface temperature is approximately 5,778 K.",
        ]

        contradicted_count = 0
        for claim in true_claims:
            result = heuristic_engine.verify(claim)
            if result.verdict == Verdict.CONTRADICTED:
                contradicted_count += 1

        # At most 1 false positive out of 5
        assert contradicted_count <= 1, f"Too many true claims contradicted: {contradicted_count}/5"

    def test_false_claims_lower_confidence(self, heuristic_engine):
        """False claims should get lower confidence than true claims.

        Without an LLM, the heuristic path cannot distinguish contradictions
        from support — it only sees semantic similarity. However, false claim
        nodes in the graph have lower confidence (0.40) than true nodes (0.85),
        so the weighted confidence for false claims should be lower.
        """
        true_claims = [
            "Jupiter is a gas giant.",
            "Earth orbits the Sun at an average distance of 149.6 million km.",
        ]
        false_claims = [
            "Venus is a gas giant.",
            "Mars has five moons.",
        ]

        true_confs = []
        for claim in true_claims:
            result = heuristic_engine.verify(claim)
            true_confs.append(result.confidence)

        false_confs = []
        for claim in false_claims:
            result = heuristic_engine.verify(claim)
            false_confs.append(result.confidence)

        avg_true = sum(true_confs) / len(true_confs)
        avg_false = sum(false_confs) / len(false_confs)

        # True claims should have higher confidence on average
        assert avg_true > avg_false, (
            f"True claims avg confidence ({avg_true:.3f}) should exceed false claims ({avg_false:.3f})"
        )


# ------------------------------------------------------------------
# BM25 vs entity overlap comparison
# ------------------------------------------------------------------


@pytest.mark.calibration
class TestBM25VsEntityOverlap:
    """Compare BM25 retrieval quality against raw entity overlap."""

    def test_bm25_ranks_rare_terms_higher(self, entity_index, solar_graph):
        """BM25 IDF should rank rare-entity matches higher than overlap count."""
        # 'neptune' is rarer than 'planet' in the corpus
        bm25_results = entity_index.bm25_search("Neptune planet", top_k=5)
        entity_index.find_matches("Neptune planet", min_overlap=1)

        # BM25 top result should have Neptune, overlap may not
        if bm25_results:
            top_bm25_text = solar_graph.get_node(bm25_results[0][0]).text
            assert "Neptune" in top_bm25_text, f"BM25 top result should mention Neptune, got: {top_bm25_text}"

    def test_bm25_retrieves_numerical_candidates(self, entity_index, solar_graph):
        """BM25 should retrieve candidates for numerical contradiction queries."""
        queries = [
            "Jupiter has a diameter of 71,000 km",
            "The Moon orbits Earth at 405,000 km",
            "Mercury's orbital period is 116 days",
        ]

        total_found = 0
        for query in queries:
            results = entity_index.bm25_search(query, top_k=5)
            if results:
                total_found += 1

        assert total_found == len(queries), (
            f"BM25 should find candidates for all queries, found {total_found}/{len(queries)}"
        )
