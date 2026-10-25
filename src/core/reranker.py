"""
Cross-encoder re-ranker for improved retrieval precision.

Bi-encoder models (like all-MiniLM-L6-v2) embed query and candidates
independently, which is fast but imprecise for subtle semantic
distinctions. Cross-encoders score (query, candidate) pairs jointly,
giving much better relevance estimates at the cost of being slower
(cannot be pre-indexed).

Typical usage in the retrieval pipeline:
    1. Bi-encoder retrieves top-k candidates (fast, broad recall).
    2. Cross-encoder re-ranks those k candidates (slower, precise).
    3. Top-n after re-ranking proceed to LLM classification.

This module provides:
- ``CrossEncoderReranker``: Wraps a sentence-transformers CrossEncoder
  model. Scores (claim, candidate_text) pairs and returns candidates
  sorted by joint relevance.
- ``rerank_candidates()``: Convenience function that takes the
  standard candidate dict format from EmbeddingEngine.search() and
  returns re-ranked candidates with updated scores.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_cross_encoder_cls = None


def _get_cross_encoder():
    """Lazy-load the CrossEncoder class from sentence-transformers."""
    global _cross_encoder_cls
    if _cross_encoder_cls is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder_cls = CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder re-ranking. "
                "Install with: pip install sentence-transformers"
            )
    return _cross_encoder_cls


@dataclass
class RerankResult:
    """Result of re-ranking a single candidate."""
    node_id: str
    original_score: float
    rerank_score: float
    combined_score: float


class CrossEncoderReranker:
    """
    Re-ranks retrieval candidates using a cross-encoder model.

    Cross-encoders process (query, candidate) pairs jointly through
    a transformer, producing a single relevance score. This is much
    more accurate than independent bi-encoder embeddings but cannot
    be pre-indexed, so it is used as a second-stage re-ranker on a
    small set of candidates.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    - 6-layer MiniLM, fast (~5ms per pair on CPU)
    - Trained on MS MARCO passage retrieval
    - Good general-purpose relevance scoring

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(
            query="Gold melts at 500 degrees Celsius",
            candidates=[
                {"node_id": "n1", "score": 0.88},
                {"node_id": "n2", "score": 0.72},
            ],
            graph=graph,
        )
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        weight_original: float = 0.3,
        weight_rerank: float = 0.7,
        min_rerank_score: Optional[float] = None,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: Cross-encoder model to load. Must be compatible
                with sentence-transformers CrossEncoder.
            weight_original: Weight for the original bi-encoder score
                in the combined score (default 0.3).
            weight_rerank: Weight for the cross-encoder score in the
                combined score (default 0.7).
            min_rerank_score: If set, candidates below this cross-encoder
                score are dropped entirely.
            batch_size: Batch size for cross-encoder predictions.
        """
        self.model_name = model_name
        self.weight_original = weight_original
        self.weight_rerank = weight_rerank
        self.min_rerank_score = min_rerank_score
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            CrossEncoder = _get_cross_encoder()
            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder model: %s", self.model_name)
        return self._model

    def score_pairs(
        self, pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Score a list of (query, candidate_text) pairs.

        Args:
            pairs: List of (query, candidate_text) tuples.

        Returns:
            List of relevance scores (higher = more relevant).
            Scores are raw logits from the cross-encoder, typically
            in the range [-10, +10] for ms-marco models.
        """
        if not pairs:
            return []

        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # Ensure we return a plain list of floats
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()

        return [float(s) for s in scores]

    def normalize_scores(self, raw_scores: List[float]) -> List[float]:
        """
        Normalize raw cross-encoder scores to [0, 1] range.

        Uses sigmoid normalization to map logits to probabilities.

        Args:
            raw_scores: Raw cross-encoder scores (logits).

        Returns:
            Normalized scores in [0, 1].
        """
        if not raw_scores:
            return []

        scores_array = np.array(raw_scores, dtype=np.float64)
        # Sigmoid: 1 / (1 + exp(-x))
        # Clip to avoid overflow in exp
        scores_clipped = np.clip(scores_array, -20, 20)
        normalized = 1.0 / (1.0 + np.exp(-scores_clipped))
        return normalized.tolist()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        graph=None,
        *,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieval candidates using the cross-encoder.

        Takes the standard candidate format from EmbeddingEngine.search()
        (list of dicts with ``node_id`` and ``score``) and returns them
        re-sorted by combined (bi-encoder + cross-encoder) score.

        Args:
            query: The query/claim text.
            candidates: List of candidate dicts, each with at least
                ``node_id`` and ``score`` keys.
            graph: KnowledgeGraph to look up node texts. If None,
                candidates must include a ``text`` key.
            top_n: If set, return only the top N re-ranked candidates.

        Returns:
            Re-ranked candidates with updated ``score`` (combined),
            plus ``rerank_score`` and ``original_score`` keys added.
        """
        if not candidates:
            return []

        # Build (query, candidate_text) pairs
        pairs = []
        valid_candidates = []

        for cand in candidates:
            node_id = cand["node_id"]
            # Try to get text from candidate dict or graph
            text = cand.get("text")
            if text is None and graph is not None:
                node = graph.get_node(node_id)
                if node:
                    text = node.text
            if text is None:
                logger.warning(
                    "Cannot re-rank candidate %s: no text available",
                    node_id,
                )
                continue

            pairs.append((query, text))
            valid_candidates.append(cand)

        if not pairs:
            return candidates  # Return originals if we couldn't get text

        # Score with cross-encoder
        raw_scores = self.score_pairs(pairs)
        normalized_scores = self.normalize_scores(raw_scores)

        # Build results with combined scores
        results = []
        for i, cand in enumerate(valid_candidates):
            original_score = cand.get("score", 0.5)
            rerank_score = normalized_scores[i]

            # Combined score: weighted average
            combined = (
                self.weight_original * original_score
                + self.weight_rerank * rerank_score
            )

            # Apply minimum score filter
            if self.min_rerank_score is not None and rerank_score < self.min_rerank_score:
                logger.debug(
                    "Dropped candidate %s: rerank_score=%.3f < min=%.3f",
                    cand["node_id"], rerank_score, self.min_rerank_score,
                )
                continue

            result = dict(cand)  # Copy original dict
            result["original_score"] = original_score
            result["rerank_score"] = rerank_score
            result["score"] = combined  # Overwrite with combined score
            results.append(result)

        # Sort by combined score (descending)
        results.sort(key=lambda r: r["score"], reverse=True)

        # Apply top_n limit
        if top_n is not None:
            results = results[:top_n]

        return results

    def rerank_with_details(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        graph=None,
        *,
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Re-rank candidates and return structured RerankResult objects.

        Useful when the caller needs the detailed score breakdown
        without the full candidate dict.

        Args:
            query: The query/claim text.
            candidates: List of candidate dicts.
            graph: KnowledgeGraph for text lookup.
            top_n: Max results to return.

        Returns:
            List of RerankResult dataclass instances, sorted by
            combined score (descending).
        """
        reranked = self.rerank(query, candidates, graph, top_n=top_n)

        return [
            RerankResult(
                node_id=r["node_id"],
                original_score=r["original_score"],
                rerank_score=r["rerank_score"],
                combined_score=r["score"],
            )
            for r in reranked
        ]
