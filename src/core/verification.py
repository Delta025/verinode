"""
Verification engine for claim verification and retraction simulation.

Provides two core capabilities:
1. **Claim verification** (``verify``): Check whether an external claim
   is supported, contradicted, or unresolvable against the knowledge
   graph — without modifying anything. Combines embedding search,
   entity-overlap retrieval, heuristic filtering, optional LLM
   contradiction detection, backward chaining, and confidence scoring.

2. **Retraction simulation** (``simulate_retraction``): "What-if"
   analysis showing how the graph's confidence landscape would change
   if a specific node were retracted (set to confidence 0). Uses
   temporary mutation + restore to compute exact cascading impacts
   without permanently altering the graph.

Both operations are strictly read-only from the caller's perspective.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .confidence import ConfidenceEngine
from .dependency_detection import DependencyDetector
from .entity_index import EntityIndex, extract_content_words
from .graph import KnowledgeGraph
from .node import KnowledgeNode
from .pipeline import reciprocal_rank_fusion
from .reasoning import ReasoningChain, ReasoningEngine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


class Verdict(Enum):
    """Outcome of verifying a claim against the knowledge graph."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class SourceEvidence:
    """A single piece of evidence (supporting or contradicting)."""

    node_id: str
    node_text: str
    confidence: float
    relationship: str  # "supports" or "contradicts"
    explanation: str = ""
    chains: List[ReasoningChain] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete result of verifying a claim against the graph."""

    claim: str
    verdict: Verdict
    confidence: float
    explanation: str
    supporting_evidence: List[SourceEvidence] = field(default_factory=list)
    contradicting_evidence: List[SourceEvidence] = field(default_factory=list)
    coverage: float = 0.0  # fraction of claim addressed by evidence
    num_candidates_checked: int = 0


@dataclass
class RetractedNodeImpact:
    """Impact of retraction on a single downstream node."""

    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    change: float  # new - original
    hops_from_retracted: int


@dataclass
class RetractionSimulation:
    """Complete result of a retraction what-if analysis."""

    retracted_node_id: str
    retracted_node_text: str
    original_confidence: float
    affected_nodes: List[RetractedNodeImpact] = field(default_factory=list)
    total_affected: int = 0
    max_confidence_drop: float = 0.0
    avg_confidence_drop: float = 0.0
    orphaned_nodes: int = 0  # nodes that would lose all predecessors


# _STOP_WORDS and _extract_content_words moved to entity_index module


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class VerificationEngine:
    """
    Verifies external claims against the knowledge graph.

    Read-only operations: nothing is added, modified, or deleted
    in the graph.

    Usage:
        engine = VerificationEngine(graph, embedding_engine=emb)
        result = engine.verify("Water boils at 50 degrees Celsius")
        print(result.verdict, result.confidence)

        sim = engine.simulate_retraction(node_id)
        print(sim.total_affected, sim.max_confidence_drop)
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedding_engine=None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        contradiction_checker=None,
        reranker=None,
        entity_index: Optional[EntityIndex] = None,
        *,
        max_candidates: int = 20,
        min_similarity: float = 0.45,
        llm_contradiction_confidence: float = 0.5,
        use_source_quality: bool = True,
        source_quality_weight: float = 0.3,
        use_bm25: bool = True,
    ):
        """
        Args:
            graph: Knowledge graph to verify against.
            embedding_engine: Optional EmbeddingEngine for semantic search.
            confidence_engine: ConfidenceEngine (created if None).
            reasoning_engine: ReasoningEngine (created if None).
            contradiction_checker: Optional ContradictionChecker for LLM verification.
            reranker: Optional CrossEncoderReranker for improved candidate ranking.
            max_candidates: Max candidate nodes to consider.
            min_similarity: Minimum embedding similarity threshold.
            llm_contradiction_confidence: Minimum LLM confidence to trust contradiction.
            use_source_quality: Whether to factor source quality into evidence weighting.
            source_quality_weight: How much source quality affects the final weight.
                0.0 = source quality has no effect (backward-compatible).
                0.3 (default) = source quality contributes 30% to the evidence weight.
                1.0 = source quality fully scales the evidence weight.
            use_bm25: Use BM25 + RRF hybrid retrieval instead of raw entity overlap.
        """
        self.graph = graph
        self.embedding_engine = embedding_engine
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.reasoning_engine = reasoning_engine or ReasoningEngine()
        self.contradiction_checker = contradiction_checker
        self.reranker = reranker

        # Entity-overlap inverted index
        if entity_index is not None:
            self.entity_index = entity_index
        else:
            self.entity_index = EntityIndex()
            if graph.nodes:
                self.entity_index.index_graph(graph)

        self.max_candidates = max_candidates
        self.min_similarity = min_similarity
        self.use_bm25 = use_bm25
        self.llm_contradiction_confidence = llm_contradiction_confidence
        self.use_source_quality = use_source_quality
        self.source_quality_weight = source_quality_weight

    # ------------------------------------------------------------------
    # Source quality weighting
    # ------------------------------------------------------------------

    def _get_source_quality(self, node: "KnowledgeNode") -> float:
        """
        Get the best source quality for a node.

        Returns the maximum quality_score across all sources.
        If the node has no sources, returns 0.5 (neutral).
        """
        if not node.sources:
            return 0.5
        return max(s.quality_score for s in node.sources)

    def _weighted_confidence(self, node: "KnowledgeNode", sim_score: float) -> float:
        """
        Compute evidence confidence weighted by similarity and source quality.

        Base formula (no source quality):
            weighted = node.confidence * sim_score

        With source quality (use_source_quality=True):
            quality = max(source.quality_score for source in node.sources)
            quality_factor = (1 - w) + w * quality
            weighted = node.confidence * sim_score * quality_factor

        where w = source_quality_weight (default 0.3).

        This means:
        - High-quality source (0.95): factor = 0.7 + 0.3*0.95 = 0.985 (nearly full weight)
        - Medium source (0.70):       factor = 0.7 + 0.3*0.70 = 0.91  (slight reduction)
        - Low-quality source (0.50):  factor = 0.7 + 0.3*0.50 = 0.85  (moderate reduction)
        - No source (default 0.50):   factor = 0.7 + 0.3*0.50 = 0.85  (same as low)

        The weighting is deliberately mild — source quality contributes 30%
        by default — so it nudges verdicts without overriding strong evidence.
        """
        base = node.confidence * sim_score

        if not self.use_source_quality:
            return base

        quality = self._get_source_quality(node)
        w = self.source_quality_weight
        quality_factor = (1.0 - w) + w * quality
        return base * quality_factor

    # ------------------------------------------------------------------
    # Claim verification
    # ------------------------------------------------------------------

    def verify(self, claim: str, k: int = 10) -> VerificationResult:
        """
        Verify a claim against the knowledge graph.

        Algorithm:
        1. Find candidate nodes via embedding search + entity overlap.
        2. Heuristic filter: score each candidate for contradiction signals.
        3. (Optional) LLM batch verification for precise classification.
        4. Classify evidence as supporting or contradicting.
        5. Build backward chains for evidence provenance.
        6. Compute verdict, confidence, and coverage.

        Args:
            claim: The factual claim to verify.
            k: Number of candidates from embedding search.

        Returns:
            VerificationResult with verdict, confidence, evidence, etc.
        """
        if len(self.graph) == 0:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                explanation="The knowledge graph is empty.",
            )

        # Step 1: Gather candidates
        candidates = self._gather_candidates(claim, k)

        if not candidates:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                explanation="No relevant information found in the knowledge graph.",
            )

        # Step 2-3: Classify evidence
        supporting, contradicting = self._classify_evidence(claim, candidates)

        num_checked = len(candidates)

        # Step 4: Build backward chains for provenance
        for evidence in supporting + contradicting:
            evidence.chains = self.reasoning_engine.backward_chain(self.graph, evidence.node_id, max_depth=3)

        # Step 5: Compute verdict
        result = self._compute_verdict(claim, supporting, contradicting, num_checked)

        return result

    def _gather_candidates(self, claim: str, k: int) -> List[Dict[str, Any]]:
        """
        Gather candidate nodes via hybrid retrieval.

        When use_bm25=True (default), combines embedding search with
        BM25 retrieval via Reciprocal Rank Fusion (RRF). When False,
        uses legacy embedding + raw entity overlap.

        When a cross-encoder reranker is available, candidates are
        re-scored using joint (claim, candidate) encoding before
        the top-N cap.
        """
        if self.use_bm25:
            # Hybrid: embedding + BM25 fused via RRF
            embed_results = []
            if self.embedding_engine and self.embedding_engine.index_size > 0:
                retrieval_k = k * 3 if self.reranker else k
                raw = self.embedding_engine.search(claim, k=retrieval_k, threshold=self.min_similarity)
                embed_results = [(c["node_id"], c.get("score", 0.0)) for c in raw]

            bm25_results = self.entity_index.bm25_search(claim, top_k=k * 2)

            rrf_top = self.max_candidates * 2 if self.reranker else self.max_candidates
            candidates = reciprocal_rank_fusion(
                [embed_results, bm25_results],
                k=60,
                top_n=rrf_top,
            )
        else:
            # Legacy path: embedding + raw entity overlap
            candidates = []
            if self.embedding_engine and self.embedding_engine.index_size > 0:
                retrieval_k = k * 3 if self.reranker else k
                results = self.embedding_engine.search(claim, k=retrieval_k, threshold=self.min_similarity)
                candidates.extend(results)

            seen_ids = {c["node_id"] for c in candidates}
            query_words = extract_content_words(claim)
            entity_matches = self.entity_index.find_matches(
                claim,
                min_overlap=2,
                exclude_ids=seen_ids,
            )
            for node_id, overlap_count in entity_matches:
                candidates.append(
                    {
                        "node_id": node_id,
                        "score": overlap_count / max(len(query_words), 1),
                        "source": "entity_overlap",
                    }
                )

        # Cross-encoder re-ranking (when available)
        if self.reranker and candidates:
            try:
                candidates = self.reranker.rerank(
                    query=claim,
                    candidates=candidates,
                    graph=self.graph,
                    top_n=self.max_candidates,
                )
                logger.debug(
                    "Re-ranked %d candidates for claim: %s",
                    len(candidates),
                    claim[:60],
                )
                return candidates
            except Exception as e:
                logger.warning(
                    "Cross-encoder reranking failed, falling back to original order: %s",
                    e,
                )

        # Cap candidates (no reranker or reranker failed)
        return candidates[: self.max_candidates]

    # Similarity threshold below which candidates are considered
    # "irrelevant" and excluded from the verdict (heuristic/legacy path).
    RELEVANCE_THRESHOLD = 0.65

    def _classify_evidence(
        self,
        claim: str,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[SourceEvidence], List[SourceEvidence]]:
        """
        Classify candidates as supporting, contradicting, or irrelevant.

        Irrelevant candidates are excluded from both lists.
        Uses 3-way LLM classification when available, otherwise
        heuristic scoring with similarity-based relevance filtering.
        """
        # Collect candidate info with similarity scores
        candidate_info: List[Tuple[KnowledgeNode, float]] = []
        for cand in candidates:
            node = self.graph.get_node(cand["node_id"])
            if not node:
                continue
            sim_score = cand.get("score", 0.5)
            candidate_info.append((node, sim_score))

        if not candidate_info:
            return [], []

        # LLM 3-way classification (when available)
        if self.contradiction_checker is not None:
            return self._llm_classify_3way(claim, candidate_info)
        else:
            return self._heuristic_classify_with_relevance(claim, candidate_info)

    def _llm_classify_3way(
        self,
        claim: str,
        candidate_info: List[Tuple[KnowledgeNode, float]],
    ) -> Tuple[List[SourceEvidence], List[SourceEvidence]]:
        """Use LLM 3-way classification: contradicts/supports/irrelevant."""
        pairs = [(claim, node.text) for node, _ in candidate_info]
        if not pairs:
            return [], []

        results = self.contradiction_checker.check_relevance_batch(pairs, max_workers=4)

        contradicting: List[SourceEvidence] = []
        supporting: List[SourceEvidence] = []

        for i, result in enumerate(results):
            node, sim_score = candidate_info[i]
            weighted_conf = self._weighted_confidence(node, sim_score)

            if result.relationship == "contradicts" and result.confidence >= self.llm_contradiction_confidence:
                contradicting.append(
                    SourceEvidence(
                        node_id=node.id,
                        node_text=node.text,
                        confidence=weighted_conf,
                        relationship="contradicts",
                        explanation=f"LLM: {result.explanation}",
                    )
                )
            elif result.relationship == "supports" and result.confidence >= 0.5:
                supporting.append(
                    SourceEvidence(
                        node_id=node.id,
                        node_text=node.text,
                        confidence=weighted_conf,
                        relationship="supports",
                        explanation=f"LLM: {result.explanation}",
                    )
                )
            else:
                logger.debug(
                    "Excluded irrelevant candidate %s (relationship=%s, conf=%.2f)",
                    node.id,
                    result.relationship,
                    result.confidence,
                )

        return supporting, contradicting

    def _heuristic_classify_with_relevance(
        self,
        claim: str,
        candidate_info: List[Tuple[KnowledgeNode, float]],
    ) -> Tuple[List[SourceEvidence], List[SourceEvidence]]:
        """
        Heuristic classification with similarity-based relevance filtering.

        Candidates below ``RELEVANCE_THRESHOLD`` are excluded as irrelevant.
        """
        contradicting: List[SourceEvidence] = []
        supporting: List[SourceEvidence] = []

        for node, sim_score in candidate_info:
            if sim_score < self.RELEVANCE_THRESHOLD:
                logger.debug(
                    "Excluded low-relevance candidate %s (sim=%.2f)",
                    node.id,
                    sim_score,
                )
                continue

            score = DependencyDetector._contradiction_score(claim.lower(), node.text.lower(), sim_score)
            weighted_conf = self._weighted_confidence(node, sim_score)

            if score > 0.5:
                contradicting.append(
                    SourceEvidence(
                        node_id=node.id,
                        node_text=node.text,
                        confidence=weighted_conf,
                        relationship="contradicts",
                        explanation="Negation mismatch (heuristic)",
                    )
                )
            else:
                supporting.append(
                    SourceEvidence(
                        node_id=node.id,
                        node_text=node.text,
                        confidence=weighted_conf,
                        relationship="supports",
                    )
                )

        return supporting, contradicting

    def _compute_verdict(
        self,
        claim: str,
        supporting: List[SourceEvidence],
        contradicting: List[SourceEvidence],
        num_checked: int,
    ) -> VerificationResult:
        """Compute the final verdict from classified evidence."""
        support_conf = max((e.confidence for e in supporting), default=0.0)
        contra_conf = max((e.confidence for e in contradicting), default=0.0)

        # Coverage: fraction of candidates that provided non-irrelevant evidence
        total_evidence = len(supporting) + len(contradicting)
        coverage = min(1.0, total_evidence / max(num_checked, 1))

        # Reduced from 0.1 — relevance filtering already removes noise
        MARGIN = 0.05

        # Determine verdict
        if not supporting and not contradicting:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            confidence = 0.0
            explanation = "No relevant evidence found in the knowledge graph."
        elif contradicting and not supporting:
            verdict = Verdict.CONTRADICTED
            confidence = contra_conf
            explanation = f"Contradicted by {len(contradicting)} node(s) (max confidence: {contra_conf:.2f})."
        elif supporting and not contradicting:
            verdict = Verdict.SUPPORTED
            confidence = support_conf
            explanation = f"Supported by {len(supporting)} node(s) (max confidence: {support_conf:.2f})."
        elif contra_conf > support_conf + MARGIN:
            # Contradicting evidence is significantly stronger
            verdict = Verdict.CONTRADICTED
            confidence = contra_conf
            explanation = (
                f"Contradicted by {len(contradicting)} node(s) "
                f"(max conf: {contra_conf:.2f}) despite "
                f"{len(supporting)} supporting node(s) "
                f"(max conf: {support_conf:.2f})."
            )
        elif support_conf > contra_conf + MARGIN:
            # Supporting evidence is significantly stronger
            verdict = Verdict.SUPPORTED
            confidence = support_conf
            explanation = (
                f"Supported by {len(supporting)} node(s) "
                f"(max conf: {support_conf:.2f}) despite "
                f"{len(contradicting)} contradicting node(s) "
                f"(max conf: {contra_conf:.2f})."
            )
        else:
            # Within margin — use count-based tiebreaker
            contra_ratio = len(contradicting) / max(total_evidence, 1)
            if contra_ratio > 0.5:
                verdict = Verdict.CONTRADICTED
                confidence = contra_conf
                explanation = (
                    f"Contradicted: {len(contradicting)} contradicting vs "
                    f"{len(supporting)} supporting node(s) at similar confidence."
                )
            elif contra_ratio < 0.5:
                verdict = Verdict.SUPPORTED
                confidence = support_conf
                explanation = (
                    f"Supported: {len(supporting)} supporting vs "
                    f"{len(contradicting)} contradicting node(s) at similar confidence."
                )
            else:
                verdict = Verdict.UNCERTAIN
                confidence = max(support_conf, contra_conf)
                explanation = (
                    f"Evidence is evenly split: {len(supporting)} supporting "
                    f"(max conf: {support_conf:.2f}) and "
                    f"{len(contradicting)} contradicting "
                    f"(max conf: {contra_conf:.2f}) node(s)."
                )

        return VerificationResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            coverage=coverage,
            num_candidates_checked=num_checked,
        )

    # ------------------------------------------------------------------
    # Retraction simulation
    # ------------------------------------------------------------------

    def simulate_retraction(self, node_id: str) -> RetractionSimulation:
        """
        Simulate what would happen if a node were retracted (set to confidence 0).

        Uses temporary mutation + restore:
        1. Record original confidences for the node and all descendants.
        2. Set the node's confidence to 0.
        3. Recalculate all descendant confidences via ConfidenceEngine.
        4. Record the impact (delta per node).
        5. Restore all original confidences.

        The graph is left unchanged after this operation.

        Args:
            node_id: ID of the node to hypothetically retract.

        Returns:
            RetractionSimulation with impact analysis.
        """
        node = self.graph.get_node(node_id)
        if not node:
            return RetractionSimulation(
                retracted_node_id=node_id,
                retracted_node_text="(node not found)",
                original_confidence=0.0,
            )

        original_confidence = node.confidence

        # Find all descendants with hop distances
        descendants_with_hops = self._bfs_with_hops(node_id)

        if not descendants_with_hops:
            return RetractionSimulation(
                retracted_node_id=node_id,
                retracted_node_text=node.text,
                original_confidence=original_confidence,
            )

        # Save original confidences for restoration
        originals: Dict[str, float] = {node_id: original_confidence}
        for desc_id, _ in descendants_with_hops:
            desc = self.graph.get_node(desc_id)
            if desc:
                originals[desc_id] = desc.confidence

        # Temporarily set retracted node to confidence 0
        # Wrap in try/finally to guarantee confidence restoration even on error
        node.confidence = 0.0

        try:
            # Recalculate descendant confidences in topological order
            desc_ids = {d_id for d_id, _ in descendants_with_hops}
            sorted_ids = self.confidence_engine._topological_sort(self.graph, desc_ids)

            new_confidences: Dict[str, float] = {}
            for desc_id in sorted_ids:
                desc = self.graph.get_node(desc_id)
                if desc:
                    new_conf = self.confidence_engine.calculate_node_confidence(desc, self.graph)
                    new_confidences[desc_id] = new_conf
                    # Temporarily set for downstream calculations
                    desc.confidence = new_conf

            # Build impact list
            affected: List[RetractedNodeImpact] = []
            orphaned = 0
            hop_map = dict(descendants_with_hops)

            for desc_id in sorted_ids:
                desc = self.graph.get_node(desc_id)
                if not desc:
                    continue

                orig_conf = originals.get(desc_id, 0.0)
                new_conf = new_confidences.get(desc_id, orig_conf)
                change = new_conf - orig_conf

                # Check if this node would be orphaned
                if node_id in desc.predecessor_nodes:
                    remaining_preds = [
                        pid for pid in desc.predecessor_nodes if pid != node_id and pid in self.graph.nodes
                    ]
                    if not remaining_preds and not desc.sources:
                        orphaned += 1

                if abs(change) > 0.001:  # Only include significant changes
                    affected.append(
                        RetractedNodeImpact(
                            node_id=desc_id,
                            node_text=desc.text,
                            original_confidence=orig_conf,
                            new_confidence=new_conf,
                            change=change,
                            hops_from_retracted=hop_map.get(desc_id, 0),
                        )
                    )
        finally:
            # Restore ALL original confidences — must run even on exception
            for nid, orig_conf in originals.items():
                n = self.graph.get_node(nid)
                if n:
                    n.confidence = orig_conf

        # Compute summary stats
        drops = [a.change for a in affected if a.change < 0]
        max_drop = min(drops) if drops else 0.0
        avg_drop = sum(drops) / len(drops) if drops else 0.0

        return RetractionSimulation(
            retracted_node_id=node_id,
            retracted_node_text=node.text,
            original_confidence=original_confidence,
            affected_nodes=affected,
            total_affected=len(affected),
            max_confidence_drop=max_drop,
            avg_confidence_drop=avg_drop,
            orphaned_nodes=orphaned,
        )

    def _bfs_with_hops(self, start_id: str) -> List[Tuple[str, int]]:
        """
        BFS from start_id collecting (node_id, hop_distance) pairs.

        Does NOT include start_id itself in the result.
        """
        result: List[Tuple[str, int]] = []
        visited = {start_id}
        queue: deque = deque()

        # Seed with direct successors
        node = self.graph.get_node(start_id)
        if not node:
            return result

        for succ_id in node.successor_nodes:
            if succ_id in self.graph.nodes and succ_id not in visited:
                visited.add(succ_id)
                queue.append((succ_id, 1))
                result.append((succ_id, 1))

        while queue:
            current_id, hops = queue.popleft()
            current = self.graph.get_node(current_id)
            if not current:
                continue

            for succ_id in current.successor_nodes:
                if succ_id in self.graph.nodes and succ_id not in visited:
                    visited.add(succ_id)
                    queue.append((succ_id, hops + 1))
                    result.append((succ_id, hops + 1))

        return result
