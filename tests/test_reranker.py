"""
Tests for the CrossEncoderReranker: re-ranking, score normalization,
integration with VerificationEngine and Pipeline.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from src.core.reranker import CrossEncoderReranker, RerankResult
from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, Source
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import ReasoningEngine
from src.core.verification import VerificationEngine, Verdict
from src.core.llm_contradiction import RelevanceResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_node(text, confidence=0.9, domain="chemistry", node_id=None):
    """Create a KnowledgeNode for testing."""
    return KnowledgeNode(
        text=text,
        confidence=confidence,
        domain=domain,
        node_id=node_id,
    )


def make_mock_reranker(scores):
    """
    Create a mock CrossEncoderReranker that returns predictable scores.

    Args:
        scores: List of raw cross-encoder scores (one per candidate).
    """
    reranker = MagicMock(spec=CrossEncoderReranker)

    def mock_rerank(query, candidates, graph=None, *, top_n=None):
        """Simulate reranking by assigning scores in order."""
        # Normalize scores to [0, 1] using sigmoid
        normalized = [1.0 / (1.0 + np.exp(-s)) for s in scores]

        results = []
        for i, cand in enumerate(candidates):
            if i >= len(normalized):
                break
            result = dict(cand)
            result["original_score"] = cand.get("score", 0.5)
            result["rerank_score"] = normalized[i]
            result["score"] = 0.3 * result["original_score"] + 0.7 * normalized[i]
            results.append(result)

        results.sort(key=lambda r: r["score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    reranker.rerank.side_effect = mock_rerank
    return reranker


# ------------------------------------------------------------------
# Unit tests: CrossEncoderReranker
# ------------------------------------------------------------------

class TestCrossEncoderRerankerInit:
    def test_default_parameters(self):
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.weight_original == 0.3
        assert reranker.weight_rerank == 0.7
        assert reranker.min_rerank_score is None
        assert reranker.batch_size == 32
        assert reranker._model is None

    def test_custom_parameters(self):
        reranker = CrossEncoderReranker(
            model_name="custom-model",
            weight_original=0.5,
            weight_rerank=0.5,
            min_rerank_score=0.2,
            batch_size=16,
        )
        assert reranker.model_name == "custom-model"
        assert reranker.weight_original == 0.5
        assert reranker.weight_rerank == 0.5
        assert reranker.min_rerank_score == 0.2
        assert reranker.batch_size == 16


class TestNormalizeScores:
    def test_positive_scores(self):
        reranker = CrossEncoderReranker()
        normalized = reranker.normalize_scores([5.0, 0.0, -5.0])
        assert len(normalized) == 3
        # Sigmoid: large positive -> near 1, zero -> 0.5, large negative -> near 0
        assert normalized[0] > 0.99
        assert abs(normalized[1] - 0.5) < 0.01
        assert normalized[2] < 0.01

    def test_empty_scores(self):
        reranker = CrossEncoderReranker()
        assert reranker.normalize_scores([]) == []

    def test_extreme_scores_clipped(self):
        reranker = CrossEncoderReranker()
        normalized = reranker.normalize_scores([100.0, -100.0])
        assert len(normalized) == 2
        # Should not overflow — clipped to [-20, 20]
        assert normalized[0] == pytest.approx(1.0, abs=1e-8)
        assert normalized[1] == pytest.approx(0.0, abs=1e-8)

    def test_all_same_scores(self):
        reranker = CrossEncoderReranker()
        normalized = reranker.normalize_scores([0.0, 0.0, 0.0])
        for s in normalized:
            assert abs(s - 0.5) < 0.01


class TestScorePairs:
    def test_score_pairs_calls_model(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.5, -1.2, 0.8])
        reranker._model = mock_model

        pairs = [
            ("claim A", "text A"),
            ("claim B", "text B"),
            ("claim C", "text C"),
        ]
        scores = reranker.score_pairs(pairs)

        assert len(scores) == 3
        assert scores[0] == pytest.approx(3.5)
        assert scores[1] == pytest.approx(-1.2)
        assert scores[2] == pytest.approx(0.8)
        mock_model.predict.assert_called_once()

    def test_score_pairs_empty(self):
        reranker = CrossEncoderReranker()
        assert reranker.score_pairs([]) == []


class TestRerank:
    def test_rerank_sorts_by_combined_score(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        # Candidate 0: low bi-encoder but high cross-encoder
        # Candidate 1: high bi-encoder but low cross-encoder
        mock_model.predict.return_value = np.array([5.0, -2.0])
        reranker._model = mock_model

        graph = KnowledgeGraph()
        graph.add_node(make_node("Gold melts at 1064 degrees Celsius", node_id="n1"))
        graph.add_node(make_node("Gold is a chemical element", node_id="n2"))

        candidates = [
            {"node_id": "n1", "score": 0.60},
            {"node_id": "n2", "score": 0.90},
        ]

        reranked = reranker.rerank("Gold melts at 500 degrees", candidates, graph)

        assert len(reranked) == 2
        # n1 should now be first (high rerank score)
        assert reranked[0]["node_id"] == "n1"
        assert reranked[0]["rerank_score"] > 0.9
        assert reranked[0]["original_score"] == 0.60
        # n2 should be second (low rerank score)
        assert reranked[1]["node_id"] == "n2"
        assert reranked[1]["rerank_score"] < 0.2

    def test_rerank_top_n(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.0, 3.0, 1.0, -1.0])
        reranker._model = mock_model

        graph = KnowledgeGraph()
        for i in range(4):
            graph.add_node(make_node(f"Fact {i}", node_id=f"n{i}"))

        candidates = [{"node_id": f"n{i}", "score": 0.5} for i in range(4)]

        reranked = reranker.rerank("query", candidates, graph, top_n=2)
        assert len(reranked) == 2

    def test_rerank_min_score_filter(self):
        reranker = CrossEncoderReranker(min_rerank_score=0.5)
        mock_model = MagicMock()
        # Score 5.0 -> sigmoid ~0.993 (above threshold)
        # Score -5.0 -> sigmoid ~0.007 (below threshold)
        mock_model.predict.return_value = np.array([5.0, -5.0])
        reranker._model = mock_model

        graph = KnowledgeGraph()
        graph.add_node(make_node("Relevant fact", node_id="n1"))
        graph.add_node(make_node("Irrelevant fact", node_id="n2"))

        candidates = [
            {"node_id": "n1", "score": 0.7},
            {"node_id": "n2", "score": 0.7},
        ]

        reranked = reranker.rerank("query", candidates, graph)
        assert len(reranked) == 1
        assert reranked[0]["node_id"] == "n1"

    def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", [], None) == []

    def test_rerank_missing_text_skips_candidate(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.0])
        reranker._model = mock_model

        graph = KnowledgeGraph()
        graph.add_node(make_node("Known fact", node_id="n1"))

        candidates = [
            {"node_id": "n1", "score": 0.8},
            {"node_id": "n-missing", "score": 0.7},  # Not in graph, no text
        ]

        reranked = reranker.rerank("query", candidates, graph)
        assert len(reranked) == 1
        assert reranked[0]["node_id"] == "n1"

    def test_rerank_with_text_in_candidate(self):
        """If candidate has 'text' key, no graph needed."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([4.0, 1.0])
        reranker._model = mock_model

        candidates = [
            {"node_id": "n1", "score": 0.5, "text": "Fact A"},
            {"node_id": "n2", "score": 0.5, "text": "Fact B"},
        ]

        reranked = reranker.rerank("query", candidates, graph=None)
        assert len(reranked) == 2
        assert reranked[0]["node_id"] == "n1"

    def test_rerank_preserves_extra_keys(self):
        """Reranking preserves extra candidate dict keys like 'source'."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.0])
        reranker._model = mock_model

        candidates = [
            {"node_id": "n1", "score": 0.8, "source": "entity_overlap", "custom_key": 42},
        ]
        graph = KnowledgeGraph()
        graph.add_node(make_node("Fact", node_id="n1"))

        reranked = reranker.rerank("query", candidates, graph)
        assert reranked[0]["source"] == "entity_overlap"
        assert reranked[0]["custom_key"] == 42


class TestRerankWithDetails:
    def test_returns_rerank_result_objects(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.0, -1.0])
        reranker._model = mock_model

        graph = KnowledgeGraph()
        graph.add_node(make_node("Fact A", node_id="n1"))
        graph.add_node(make_node("Fact B", node_id="n2"))

        candidates = [
            {"node_id": "n1", "score": 0.8},
            {"node_id": "n2", "score": 0.7},
        ]

        details = reranker.rerank_with_details("query", candidates, graph)
        assert len(details) == 2
        assert isinstance(details[0], RerankResult)
        assert details[0].node_id in ("n1", "n2")
        assert details[0].combined_score > 0
        assert details[0].rerank_score > 0


# ------------------------------------------------------------------
# Integration: VerificationEngine + Reranker
# ------------------------------------------------------------------

class TestVerificationWithReranker:
    def test_reranker_used_in_gather_candidates(self):
        """VerificationEngine._gather_candidates uses reranker when provided."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Gold melts at 1064 degrees Celsius",
            confidence=0.95,
            node_id="gold-melt",
        ))
        graph.add_node(make_node(
            "Gold is a chemical element with symbol Au",
            confidence=0.90,
            node_id="gold-element",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "gold-element", "score": 0.88},
            {"node_id": "gold-melt", "score": 0.72},
        ]

        # Reranker should prioritize gold-melt over gold-element for melting claims
        mock_reranker = make_mock_reranker([5.0, -2.0])

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            reranker=mock_reranker,
        )

        candidates = engine._gather_candidates("Gold melts at 500 degrees", k=10)

        mock_reranker.rerank.assert_called_once()
        # After reranking, gold-melt should be scored higher
        assert len(candidates) == 2

    def test_reranker_retrieves_more_initial_candidates(self):
        """When reranker is present, embedding search retrieves k*3 candidates."""
        graph = KnowledgeGraph()
        for i in range(5):
            graph.add_node(make_node(f"Fact {i}", node_id=f"n{i}"))

        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": f"n{i}", "score": 0.8 - i * 0.05} for i in range(5)
        ]

        mock_reranker = make_mock_reranker([5.0, 4.0, 3.0, 2.0, 1.0])

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            reranker=mock_reranker,
        )

        engine._gather_candidates("test query", k=5)

        # Should have called search with k=15 (k*3) instead of k=5
        call_kwargs = mock_emb.search.call_args
        assert call_kwargs[1].get("k", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None) == 15

    def test_reranker_failure_falls_back_gracefully(self):
        """If reranker raises an exception, falls back to original ordering."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="node-1",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.side_effect = RuntimeError("Model load failed")

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            reranker=mock_reranker,
        )

        # Should not raise — falls back to original order
        candidates = engine._gather_candidates("Water boils", k=10)
        assert len(candidates) >= 1

    def test_verify_with_reranker_detects_contradiction(self):
        """Full verify flow with reranker correctly detects contradictions."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Gold melts at 1064 degrees Celsius",
            confidence=0.95,
            node_id="gold-melt",
        ))
        graph.add_node(make_node(
            "Gold is a dense yellow metal",
            confidence=0.90,
            node_id="gold-desc",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 2
        mock_emb.search.return_value = [
            {"node_id": "gold-desc", "score": 0.80},
            {"node_id": "gold-melt", "score": 0.75},
        ]

        # Reranker boosts gold-melt to top for melting point claims
        mock_reranker = make_mock_reranker([-2.0, 5.0])

        mock_checker = MagicMock()
        mock_checker.check_relevance_batch.return_value = [
            RelevanceResult(
                claim="Gold melts at 500 degrees Celsius",
                knowledge="Gold melts at 1064 degrees Celsius",
                relationship="contradicts",
                confidence=0.95,
                explanation="Numerical contradiction: 500C vs 1064C",
            ),
            RelevanceResult(
                claim="Gold melts at 500 degrees Celsius",
                knowledge="Gold is a dense yellow metal",
                relationship="irrelevant",
                confidence=0.85,
                explanation="Different property",
            ),
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            reranker=mock_reranker,
            contradiction_checker=mock_checker,
        )

        result = engine.verify("Gold melts at 500 degrees Celsius")
        assert result.verdict == Verdict.CONTRADICTED

    def test_no_reranker_works_as_before(self):
        """Without reranker, verification works identically to before."""
        graph = KnowledgeGraph()
        graph.add_node(make_node(
            "Water boils at 100 degrees Celsius",
            confidence=0.95,
            node_id="node-1",
        ))

        mock_emb = MagicMock()
        mock_emb.index_size = 1
        mock_emb.search.return_value = [
            {"node_id": "node-1", "score": 0.85},
        ]

        engine = VerificationEngine(
            graph,
            embedding_engine=mock_emb,
            reranker=None,  # Explicitly no reranker
        )

        result = engine.verify("Water boils at 100 degrees Celsius")
        assert result.verdict == Verdict.SUPPORTED


# ------------------------------------------------------------------
# Integration: Pipeline + Reranker
# ------------------------------------------------------------------

class TestPipelineWithReranker:
    def test_pipeline_accepts_reranker(self):
        """Pipeline constructor accepts reranker parameter."""
        from src.core.pipeline import Pipeline
        from src.core.embeddings import EmbeddingEngine

        graph = KnowledgeGraph()
        mock_extractor = MagicMock()
        mock_reranker = MagicMock(spec=CrossEncoderReranker)

        pipeline = Pipeline(
            graph,
            extractor=mock_extractor,
            reranker=mock_reranker,
            use_llm_contradiction=False,
        )

        assert pipeline.reranker is mock_reranker

    def test_pipeline_reranker_none_by_default(self):
        """Pipeline has no reranker by default."""
        from src.core.pipeline import Pipeline

        graph = KnowledgeGraph()
        mock_extractor = MagicMock()

        pipeline = Pipeline(
            graph,
            extractor=mock_extractor,
            use_llm_contradiction=False,
        )

        assert pipeline.reranker is None


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestRerankerEdgeCases:
    def test_single_candidate(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.5])
        reranker._model = mock_model

        candidates = [{"node_id": "n1", "score": 0.8, "text": "Fact A"}]
        reranked = reranker.rerank("query", candidates)
        assert len(reranked) == 1
        assert reranked[0]["node_id"] == "n1"
        assert "rerank_score" in reranked[0]

    def test_all_candidates_below_min_score(self):
        reranker = CrossEncoderReranker(min_rerank_score=0.9)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-5.0, -3.0])
        reranker._model = mock_model

        candidates = [
            {"node_id": "n1", "score": 0.8, "text": "Fact A"},
            {"node_id": "n2", "score": 0.7, "text": "Fact B"},
        ]

        reranked = reranker.rerank("query", candidates)
        assert len(reranked) == 0

    def test_no_graph_no_text_returns_originals(self):
        """When no graph and no text in candidates, returns originals unchanged."""
        reranker = CrossEncoderReranker()
        # Don't set _model — shouldn't need it since all candidates will be skipped

        candidates = [
            {"node_id": "n1", "score": 0.8},
            {"node_id": "n2", "score": 0.7},
        ]

        # All candidates lack text and no graph provided — should return originals
        reranked = reranker.rerank("query", candidates, graph=None)
        assert reranked == candidates

    def test_combined_score_weights(self):
        """Verify combined score uses correct weights."""
        reranker = CrossEncoderReranker(weight_original=0.4, weight_rerank=0.6)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.0])  # sigmoid(0) = 0.5
        reranker._model = mock_model

        candidates = [{"node_id": "n1", "score": 1.0, "text": "Fact"}]

        reranked = reranker.rerank("query", candidates)
        # combined = 0.4 * 1.0 + 0.6 * 0.5 = 0.7
        assert abs(reranked[0]["score"] - 0.7) < 0.01
