"""
End-to-end ingestion pipeline with truth verification.

Orchestrates the full flow:
  text input -> fact extraction (Ollama) -> verification against
  existing graph -> node creation -> dependency detection -> graph
  wiring -> confidence propagation

Implements Layers 1 and 3 of the Truth Verification Strategy:
  - Layer 1: Source quality gating (untrusted sources → PROVISIONAL)
  - Layer 3: Ingestion gating (verify new facts against existing graph)

Key behaviours:
  - **Contradiction detection** (runs first): Multi-strategy retrieval
    finds candidates via embedding similarity AND entity-word overlap,
    then heuristic + LLM verification confirms true contradictions.
    Contradiction check runs before deduplication so that near-duplicates
    that are actually contradictions get flagged, not silently merged.
  - **Deduplication**: If an extracted fact is semantically near-identical
    to an existing node (cosine similarity ≥ duplicate_threshold) and is
    NOT a contradiction, the existing node gets a convergent-evidence
    confidence boost instead of creating a duplicate.
  - **Source quality enforcement**: Nodes from low-quality sources
    receive capped confidence, keeping them PROVISIONAL until
    corroborated.
  - **Quality filtering**: Extracted facts below a minimum confidence
    are silently dropped.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
import logging

import numpy as np

from .node import KnowledgeNode, NodeType, NodeStatus, Source, SourceTier, Derivation
from .graph import KnowledgeGraph
from .confidence import ConfidenceEngine
from .embeddings import EmbeddingEngine
from .fact_extraction import FactExtractionEngine, ExtractionResult, ExtractedFact
from .dependency_detection import DependencyDetector, DetectedDependency, RelationType
from .llm_contradiction import create_contradiction_checker, ContradictionChecker
from .entity_index import EntityIndex, extract_content_words
from .unit_normalizer import normalize_units

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Reciprocal Rank Fusion
# ------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Each ranked list can contain either (node_id, score) tuples or
    dicts with a "node_id" key. Results are merged by node_id and
    ordered by RRF score = sum(1 / (k + rank + 1)).

    The output ``score`` preserves the maximum original score across
    all lists (for downstream similarity-based weighting), while
    RRF ordering determines which candidates make the top-N cut.

    Args:
        ranked_lists: List of ranked results.
        k: RRF constant (default 60).
        top_n: Number of results to return.

    Returns:
        List of candidate dicts with node_id and score keys,
        ordered by RRF rank.
    """
    rrf_scores: Dict[str, float] = {}
    best_scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            if isinstance(item, dict):
                nid = item["node_id"]
                orig = item.get("score", 0.5)
            else:
                nid = item[0]
                orig = item[1] if len(item) > 1 else 0.5
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            best_scores[nid] = max(best_scores.get(nid, 0.0), orig)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Normalize best_scores to [0, 1] — BM25 scores can exceed 1.0
    # and downstream evidence weighting assumes similarity-scale scores.
    max_score = max(best_scores.values()) if best_scores else 1.0
    if max_score > 1.0:
        for nid in best_scores:
            best_scores[nid] = best_scores[nid] / max_score

    return [
        {"node_id": nid, "score": best_scores.get(nid, 0.5)}
        for nid, _rrf in sorted_ids[:top_n]
    ]


# ------------------------------------------------------------------
# Result data structures
# ------------------------------------------------------------------

@dataclass
class IngestionVerdict:
    """Verdict for a single extracted fact during ingestion."""
    fact: ExtractedFact
    action: str  # "added", "duplicate", "flagged", "rejected"
    node_id: Optional[str] = None       # ID of created or existing node
    reason: str = ""
    contradiction_node_ids: List[str] = field(default_factory=list)
    duplicate_node_id: Optional[str] = None
    confidence_before: Optional[float] = None  # for duplicates
    confidence_after: Optional[float] = None   # for duplicates


@dataclass
class IngestionResult:
    """Result of ingesting one text passage."""
    source_text: str
    extraction_model: str
    verdicts: List[IngestionVerdict] = field(default_factory=list)
    nodes_added: int = 0
    duplicates_merged: int = 0
    facts_flagged: int = 0
    facts_rejected: int = 0
    dependencies_added: int = 0
    nodes_propagated: int = 0

    @property
    def total_extracted(self) -> int:
        return len(self.verdicts)

    @property
    def summary(self) -> str:
        parts = [
            f"Extracted {self.total_extracted} fact(s)",
            f"added {self.nodes_added}",
            f"merged {self.duplicates_merged} duplicate(s)",
        ]
        if self.facts_flagged:
            parts.append(f"flagged {self.facts_flagged}")
        if self.facts_rejected:
            parts.append(f"rejected {self.facts_rejected}")
        parts.append(f"wired {self.dependencies_added} dependenc(ies)")
        return ", ".join(parts)


@dataclass
class BatchIngestionResult:
    """Result of ingesting multiple text passages."""
    results: List[IngestionResult] = field(default_factory=list)

    @property
    def total_added(self) -> int:
        return sum(r.nodes_added for r in self.results)

    @property
    def total_duplicates(self) -> int:
        return sum(r.duplicates_merged for r in self.results)

    @property
    def total_flagged(self) -> int:
        return sum(r.facts_flagged for r in self.results)

    @property
    def total_rejected(self) -> int:
        return sum(r.facts_rejected for r in self.results)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class Pipeline:
    """
    End-to-end ingestion pipeline with truth verification.

    Connects fact extraction, embedding search, dependency detection,
    and confidence propagation into a single workflow with built-in
    verification gating.

    Usage:
        graph = KnowledgeGraph()
        pipeline = Pipeline(graph)

        # Ingest a single passage
        result = pipeline.ingest(
            "Water boils at 100°C at sea level.",
            source=Source(source_type="textbook", title="Chem 101",
                          quality_score=0.9),
            domain="chemistry",
        )

        # Batch ingest
        batch = pipeline.ingest_batch(texts, source=source)

    Configuration:
        duplicate_threshold: Cosine similarity above which two facts
            are considered duplicates (default 0.87, Optuna-tuned).
        contradiction_threshold: Confidence above which an existing
            contradicting node causes a new fact to be flagged
            (default 0.52, Optuna-tuned).
        min_extraction_confidence: Minimum LLM-assigned confidence
            to accept a fact (default 0.46, Optuna-tuned).
        source_quality_cap: When source quality is below this, node
            confidence is capped at source quality (default 0.54,
            Optuna-tuned).
        reject_contradictions: If True, contradictions against
            high-confidence nodes are rejected outright. If False,
            they are flagged but still added as CONTESTED (default
            False).
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        extractor: Optional[FactExtractionEngine] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        dependency_detector: Optional[DependencyDetector] = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        contradiction_checker: Optional[ContradictionChecker] = None,
        reranker=None,
        entity_index: Optional[EntityIndex] = None,
        *,
        duplicate_threshold: float = 0.87,
        contradiction_threshold: float = 0.52,
        min_extraction_confidence: float = 0.46,
        source_quality_cap: float = 0.54,
        reject_contradictions: bool = False,
        use_llm_contradiction: bool = True,
        llm_contradiction_confidence: float = 0.7,
        # Quote verification
        verify_quotes: bool = True,
        quote_penalty: float = 0.7,
        min_quote_overlap: float = 0.6,
        # Tiered ingestion
        use_tiered_ingestion: bool = False,
        # Dependency detection
        detect_dependencies: bool = True,
        # Content pre-filters for contradiction detection
        min_content_overlap: int = 1,
        use_unit_normalization: bool = True,
        # BM25 hybrid retrieval
        use_bm25: bool = True,
    ):
        self.graph = graph
        self.extractor = extractor or FactExtractionEngine()
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.detector = dependency_detector or DependencyDetector()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.reranker = reranker

        # Entity-overlap inverted index
        if entity_index is not None:
            self.entity_index = entity_index
        else:
            self.entity_index = EntityIndex()
            if graph.nodes:
                self.entity_index.index_graph(graph)

        # LLM-assisted contradiction detection
        if contradiction_checker is not None:
            self.contradiction_checker = contradiction_checker
        elif use_llm_contradiction:
            self.contradiction_checker = create_contradiction_checker()
        else:
            self.contradiction_checker = None

        # Verification thresholds
        self.duplicate_threshold = duplicate_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_extraction_confidence = min_extraction_confidence
        self.source_quality_cap = source_quality_cap
        self.reject_contradictions = reject_contradictions
        self.llm_contradiction_confidence = llm_contradiction_confidence

        # Quote verification
        self.verify_quotes = verify_quotes
        self.quote_penalty = quote_penalty
        self.min_quote_overlap = min_quote_overlap

        # Tiered ingestion
        self.use_tiered_ingestion = use_tiered_ingestion

        # Dependency detection
        self.detect_dependencies = detect_dependencies

        # Content pre-filters for contradiction detection
        self.min_content_overlap = min_content_overlap
        self.use_unit_normalization = use_unit_normalization

        # BM25 hybrid retrieval
        self.use_bm25 = use_bm25

        # Retrieval strategies for contradiction candidate discovery
        self.max_contradiction_candidates = 20  # cap total candidates for LLM calls
        self.entity_retrieval = True           # entity-overlap retrieval (fallback when use_bm25=False)
        self.max_entity_candidates = 10        # cap entity overlap results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        source: Optional[Source] = None,
        domain: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest a single text passage into the knowledge graph.

        Steps:
        1. Extract facts via Ollama
        2. Quality-filter (drop low-confidence extractions)
        3. For each fact: dedup → contradiction check → add or flag
        4. Detect and wire dependencies for new nodes
        5. Propagate confidence

        Args:
            text: Natural-language text to ingest.
            source: Optional Source metadata to attach.
            domain: Optional domain hint for extraction.

        Returns:
            IngestionResult describing what happened.
        """
        # 1. Extract
        extraction = self.extractor.extract(text, domain=domain)
        result = IngestionResult(
            source_text=text,
            extraction_model=extraction.model,
        )

        # 2. Quality filter
        facts = [
            f for f in extraction.facts
            if f.confidence >= self.min_extraction_confidence
        ]

        # 2.5 Quote verification
        if self.verify_quotes:
            verified_facts = []
            for fact in facts:
                if self._verify_quote(fact, text):
                    fact.metadata["quote_verified"] = True
                    verified_facts.append(fact)
                else:
                    fact.confidence *= self.quote_penalty
                    fact.metadata["quote_verified"] = False
                    logger.info(
                        "Quote verification failed for: %s (confidence penalized to %.2f)",
                        fact.text[:60], fact.confidence,
                    )
                    # Re-check after penalty: might fall below quality threshold
                    if fact.confidence >= self.min_extraction_confidence:
                        verified_facts.append(fact)
            facts = verified_facts

        # Track new node IDs for dependency detection + propagation
        new_node_ids: List[str] = []

        # 3. Per-fact verification
        for fact in facts:
            verdict = self._process_fact(fact, source, extraction.model)
            result.verdicts.append(verdict)

            if verdict.action == "added":
                result.nodes_added += 1
                new_node_ids.append(verdict.node_id)
            elif verdict.action == "duplicate":
                result.duplicates_merged += 1
            elif verdict.action == "flagged":
                result.facts_flagged += 1
                if verdict.node_id:
                    new_node_ids.append(verdict.node_id)
            elif verdict.action == "rejected":
                result.facts_rejected += 1

        # 4. Dependency detection for new nodes
        if self.detect_dependencies and new_node_ids:
            # Ensure the embedding index includes the new nodes
            self._ensure_indexed(new_node_ids)

            for nid in new_node_ids:
                node = self.graph.get_node(nid)
                if not node:
                    continue
                deps = self.detector.detect_for_node(
                    node, self.graph,
                    embedding_engine=self.embedding_engine,
                    contradiction_checker=self.contradiction_checker,
                )
                added = self.detector.apply_dependencies(deps, self.graph)
                result.dependencies_added += added

        # 5. Confidence propagation
        propagated: Set[str] = set()
        for nid in new_node_ids:
            affected = self.confidence_engine.propagate_confidence_update(
                self.graph, nid, max_hops=5
            )
            propagated.update(affected)
        result.nodes_propagated = len(propagated)

        return result

    def ingest_batch(
        self,
        texts: List[str],
        source: Optional[Source] = None,
        sources: Optional[List[Source]] = None,
        domain: Optional[str] = None,
    ) -> BatchIngestionResult:
        """
        Ingest multiple text passages sequentially.

        Each passage is ingested fully (including dependency detection
        and confidence propagation) before the next one starts. This
        ensures that later passages benefit from facts added by earlier
        ones during deduplication and contradiction checking.

        When ``use_tiered_ingestion`` is enabled and per-text ``sources``
        are provided, texts are sorted by source tier (authoritative
        first) before processing, so that high-quality facts form a
        "bedrock" that lower-quality sources are checked against.

        Args:
            texts: List of text passages to ingest.
            source: Optional Source metadata (same for all).
            sources: Optional per-text sources (must match texts length).
                Overrides ``source`` when provided.
            domain: Optional domain hint (same for all).

        Returns:
            BatchIngestionResult with per-passage results.
        """
        batch = BatchIngestionResult()

        # Build (text, source) pairs
        if sources is not None:
            if len(sources) != len(texts):
                raise ValueError(
                    f"sources length ({len(sources)}) must match "
                    f"texts length ({len(texts)})"
                )
            pairs = list(zip(texts, sources))
        else:
            pairs = [(t, source) for t in texts]

        # Sort by source tier when tiered ingestion is enabled
        if self.use_tiered_ingestion:
            pairs.sort(
                key=lambda p: SourceTier.from_quality_score(
                    p[1].quality_score if p[1] else 0.0
                ).sort_priority
            )

        for text, src in pairs:
            r = self.ingest(text, source=src, domain=domain)
            batch.results.append(r)

        # Rebuild the FAISS index once after the full batch instead of
        # per-node.  Individual add_to_index() calls during ingest() defer
        # the rebuild via the dirty flag; this single flush converts batch
        # ingestion from O(N**2) to O(N) for the FAISS work.
        self.embedding_engine.flush_index()

        return batch

    def build_index(self) -> None:
        """
        (Re)build the embedding index from all nodes in the graph.

        Call this after loading a pre-existing graph from disk, before
        ingesting new text, so that deduplication and contradiction
        checks have embeddings to work with.
        """
        if len(self.graph) > 0:
            self.embedding_engine.index_graph(self.graph)

    def rebuild_contradictions(self) -> dict:
        """Re-run contradiction checking on all existing nodes.

        Clears all existing conflicts and confidence penalties, then
        re-checks every node against the graph using the current
        contradiction checker and prompts.  Useful after improving
        prompts or switching from binary to 3-way relevance checking
        without needing to re-extract facts from sources.

        Returns:
            dict with keys: nodes_checked, conflicts_found,
            conflicts_cleared.
        """
        nodes = list(self.graph.nodes.values())

        # 1. Clear existing conflicts and restore confidence
        conflicts_cleared = 0
        for node in nodes:
            if node.conflicts_with:
                conflicts_cleared += len(node.conflicts_with)
                node.conflicts_with = []
            # Reset confidence floor from conflict penalty (0.3 cap)
            # to a reasonable value based on source tier
            if node.confidence <= 0.30:
                max_source_quality = max(
                    (s.quality_score for s in node.sources), default=0.5
                )
                tier = SourceTier.from_quality_score(max_source_quality)
                tier_cap = tier.confidence_cap
                # Restore to source-tier-based default
                node.confidence = min(0.7, tier_cap)
            if node.status == NodeStatus.CONTESTED:
                node.status = NodeStatus.PROVISIONAL

        # 2. Clear checker cache
        if self.contradiction_checker:
            self.contradiction_checker.clear_cache()

        # 3. Re-check each node
        conflicts_found = 0
        for node in nodes:
            contradictions = self._find_contradictions(node.text, exclude_node_id=node.id)
            if contradictions:
                # Apply conflict edges and confidence penalties
                for c in contradictions:
                    cid = c["node_id"]
                    if cid not in node.conflicts_with:
                        node.conflicts_with.append(cid)
                    contra_node = self.graph.get_node(cid)
                    if contra_node and node.id not in contra_node.conflicts_with:
                        contra_node.conflicts_with.append(node.id)
                    conflicts_found += 1

                # Apply confidence penalty to the lower-confidence side
                for c in contradictions:
                    contra_node = self.graph.get_node(c["node_id"])
                    if contra_node:
                        if node.confidence <= contra_node.confidence:
                            node.confidence = min(node.confidence, 0.3)
                            node.status = NodeStatus.CONTESTED
                        else:
                            contra_node.confidence = min(contra_node.confidence, 0.3)
                            contra_node.status = NodeStatus.CONTESTED

        return {
            "nodes_checked": len(nodes),
            "conflicts_found": conflicts_found,
            "conflicts_cleared": conflicts_cleared,
        }

    # ------------------------------------------------------------------
    # Internal: per-fact processing
    # ------------------------------------------------------------------

    def _process_fact(
        self,
        fact: ExtractedFact,
        source: Optional[Source],
        model: str,
    ) -> IngestionVerdict:
        """
        Process a single extracted fact: contradict-check, dedup, add.

        Contradiction detection runs BEFORE dedup so that a fact which
        is both a near-duplicate and a contradiction gets flagged rather
        than silently merged. This prevents dedup from "stealing"
        contradiction detections (e.g. "born on the 22nd" matching
        "born on the 21st" as a duplicate instead of a contradiction).
        """
        # --- Contradiction check (runs first) ---
        contradictions = self._find_contradictions(fact.text)
        if contradictions:
            return self._handle_contradictions(
                fact, contradictions, source, model
            )

        # --- Deduplication (only if no contradiction) ---
        dup_id = self._find_duplicate(fact.text)
        if dup_id:
            return self._merge_duplicate(fact, dup_id, source)

        # --- Normal add ---
        return self._add_fact(fact, source, model)

    def _find_duplicate(self, text: str) -> Optional[str]:
        """
        Search for a near-identical existing node via embedding similarity.
        Returns the node ID of the duplicate, or None.
        """
        if self.embedding_engine.index_size == 0:
            return None

        results = self.embedding_engine.search(
            text, k=1, threshold=self.duplicate_threshold
        )
        if results:
            return results[0]["node_id"]
        return None

    def _merge_duplicate(
        self,
        fact: ExtractedFact,
        existing_id: str,
        source: Optional[Source],
    ) -> IngestionVerdict:
        """
        Merge a duplicate fact into the existing node as convergent
        evidence (noisy-OR confidence boost).
        """
        existing = self.graph.get_node(existing_id)
        if not existing:
            # Should not happen, but fall through to normal add
            return self._add_fact(fact, source, "")

        old_conf = existing.confidence

        # Noisy-OR: combined = 1 - (1 - old) * (1 - new)
        new_conf = 1.0 - (1.0 - old_conf) * (1.0 - fact.confidence)
        new_conf = max(0.0, min(0.99, new_conf))

        self.graph.update_node_confidence(existing_id, new_conf)

        if source:
            existing.add_source(source)

        return IngestionVerdict(
            fact=fact,
            action="duplicate",
            node_id=existing_id,
            duplicate_node_id=existing_id,
            reason=f"Near-duplicate of existing node (boosted confidence)",
            confidence_before=old_conf,
            confidence_after=new_conf,
        )

    # _extract_content_words moved to entity_index.extract_content_words

    def _find_entity_matches(
        self,
        text: str,
        exclude_ids: Set[str],
    ) -> List[Dict]:
        """
        Find graph nodes sharing significant content words with the query.

        This catches contradictions that embedding search misses because
        contradicting statements often land in different semantic
        neighborhoods (e.g. "X is a cat" vs "X is a singer" have low
        cosine similarity despite sharing the subject entity).

        Requires >=2 shared content words to avoid noise. Results are
        sorted by overlap count (descending), capped at
        max_entity_candidates.

        Args:
            text: The new fact text to find matches for.
            exclude_ids: Node IDs already in the candidate set.

        Returns:
            List of candidate dicts with node_id and score keys.
        """
        # Use inverted index for O(k) lookup instead of O(N) scan
        all_matches = self.entity_index.find_matches(
            text, min_overlap=2, exclude_ids=exclude_ids,
        )
        if not all_matches:
            return []

        # Apply confidence threshold filter
        matches: List[Tuple[str, int]] = []
        for node_id, overlap_count in all_matches:
            node = self.graph.get_node(node_id)
            if node and node.confidence >= self.contradiction_threshold:
                matches.append((node_id, overlap_count))

        # Cap results
        max_entity = self.max_entity_candidates
        matches = matches[:max_entity]

        # Compute embedding similarity for entity matches if possible
        entity_candidates = []
        query_vec = None
        if matches and self.embedding_engine.index_size > 0:
            try:
                query_vec = self.embedding_engine.embed_text(text)
            except Exception:
                logger.warning("Failed to embed text for entity match scoring", exc_info=True)
                query_vec = None

        for node_id, overlap_count in matches:
            score = 0.5  # neutral default
            if query_vec is not None:
                node_vec = self.embedding_engine.get_embedding(node_id)
                if node_vec is not None:
                    dot = float(np.dot(query_vec, node_vec))
                    norm_q = float(np.linalg.norm(query_vec))
                    norm_n = float(np.linalg.norm(node_vec))
                    score = dot / (norm_q * norm_n) if (norm_q * norm_n) > 0 else 0.0

            entity_candidates.append({
                "node_id": node_id,
                "score": score,
                "source": "entity_overlap",
                "overlap": overlap_count,
            })

        return entity_candidates

    def _apply_content_filters(
        self, text: str, candidates: List[Dict],
    ) -> List[Dict]:
        """
        Filter candidates using content-word overlap.

        Removes candidates that share too few content words with the query
        text (different-subject filter). A cheap pre-computation that
        eliminates candidates before the expensive LLM call.

        Args:
            text: Query text to compare against.
            candidates: List of candidate dicts with ``"node_id"`` keys.

        Returns:
            Filtered list of candidates.
        """
        if self.min_content_overlap <= 0:
            return candidates  # filter disabled

        query_words = extract_content_words(text)
        if not query_words:
            return candidates  # can't filter without words

        filtered = []
        for cand in candidates:
            node = self.graph.get_node(cand["node_id"])
            if not node:
                continue
            cand_words = extract_content_words(node.text)

            overlap = len(query_words & cand_words)
            if overlap < self.min_content_overlap:
                continue

            filtered.append(cand)

        if len(filtered) < len(candidates):
            logger.debug(
                "Content filters: %d -> %d candidates",
                len(candidates), len(filtered),
            )
        return filtered

    def _find_contradictions(
        self, text: str, exclude_node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find existing high-confidence nodes that may contradict the
        new fact text.

        Args:
            text: The fact text to check against existing nodes.
            exclude_node_id: Optional node ID to exclude from candidates
                (used by rebuild_contradictions to avoid self-matches).

        Returns list of dicts with keys:
            node_id (str), reason (str)

        Retrieval uses multiple strategies to maximize candidate recall:
          Phase 0: Embedding search — top-k by cosine similarity.
          Phase 0.5a: Entity overlap retrieval — finds nodes sharing 2+
                      content words, catching contradictions where embedding
                      similarity is low (e.g. "X is a cat" vs "X is a singer").
          Phase 0.5b: (Optional) Neighbor expansion — follow graph edges
                      from top-k hits.
          Phase 1: Heuristic pre-filter (fast) — finds negation-based candidates.
          Phase 2: LLM verification — verifies heuristic hits (removes false
                   positives) and checks remaining candidates the heuristic
                   missed (catches numerical / semantic contradictions).

        When no LLM is available, only Phases 0–1 run (backward-compatible).
        """
        if self.embedding_engine.index_size == 0 and len(self.graph) == 0:
            return []

        if self.use_bm25:
            # Hybrid retrieval: embedding + BM25 fused via RRF
            embed_results = []
            if self.embedding_engine.index_size > 0:
                raw = self.embedding_engine.search(text, k=8, threshold=0.45)
                embed_results = [(c["node_id"], c.get("score", 0.0)) for c in raw]

            exclude = {exclude_node_id} if exclude_node_id else None
            bm25_results = self.entity_index.bm25_search(
                text, top_k=12, exclude_ids=exclude,
            )

            candidates = reciprocal_rank_fusion(
                [embed_results, bm25_results],
                k=60,
                top_n=self.max_contradiction_candidates,
            )
        else:
            # Legacy path: embedding + entity overlap (no BM25)
            candidates = []
            if self.embedding_engine.index_size > 0:
                candidates = self.embedding_engine.search(
                    text, k=8, threshold=0.45
                )

            if self.entity_retrieval:
                seen_ids = {c["node_id"] for c in candidates}
                entity_matches = self._find_entity_matches(text, seen_ids)
                candidates = candidates + entity_matches

        # Phase 0.75: Cross-encoder re-ranking (when available)
        # Re-scores candidates using joint (text, candidate) encoding
        # for more accurate relevance ranking before the cap.
        if self.reranker and candidates:
            try:
                candidates = self.reranker.rerank(
                    query=text,
                    candidates=candidates,
                    graph=self.graph,
                    top_n=self.max_contradiction_candidates,
                )
                logger.debug(
                    "Re-ranked %d pipeline candidates for: %s",
                    len(candidates), text[:60],
                )
            except Exception as e:
                logger.warning(
                    "Pipeline cross-encoder reranking failed, "
                    "falling back to original order: %s", e,
                )

        # Cap total candidates
        if len(candidates) > self.max_contradiction_candidates:
            candidates = candidates[:self.max_contradiction_candidates]

        # Exclude self (used by rebuild_contradictions)
        if exclude_node_id:
            candidates = [c for c in candidates if c["node_id"] != exclude_node_id]

        # Phase 0.9: Content pre-filters (content word overlap gate)
        candidates = self._apply_content_filters(text, candidates)

        # Phase 1: Heuristic pre-filter
        heuristic_hits: List[str] = []          # node IDs flagged by heuristic
        heuristic_misses: List[Dict] = []       # candidates heuristic did NOT flag

        for cand in candidates:
            node = self.graph.get_node(cand["node_id"])
            if not node or node.confidence < self.contradiction_threshold:
                continue

            sim_score = cand.get("score", 0.5)
            score = DependencyDetector._contradiction_score(
                text.lower(), node.text.lower(), sim_score
            )
            if score > 0.5:
                heuristic_hits.append(node.id)
            else:
                heuristic_misses.append(cand)

        # Phase 2: LLM verification (when available)
        # Uses batch parallelism when multiple candidates need checking.
        if self.contradiction_checker is not None:
            verified: List[Dict[str, Any]] = []

            # Collect all pairs for batch LLM checking
            # Track which are heuristic hits (fallback to heuristic on LLM failure)
            all_pairs: List[tuple] = []         # (text, node_text)
            pair_node_ids: List[str] = []       # corresponding node IDs
            pair_is_heuristic: List[bool] = []  # True = heuristic hit (trust on LLM failure)

            for nid in heuristic_hits:
                node = self.graph.get_node(nid)
                if not node:
                    continue
                all_pairs.append((text, node.text))
                pair_node_ids.append(nid)
                pair_is_heuristic.append(True)

            for cand in heuristic_misses:
                node = self.graph.get_node(cand["node_id"])
                if not node:
                    continue
                all_pairs.append((text, node.text))
                pair_node_ids.append(node.id)
                pair_is_heuristic.append(False)

            if not all_pairs:
                return []

            # Phase 2: Unit normalization for LLM pairs
            if self.use_unit_normalization and all_pairs:
                llm_pairs = [
                    (normalize_units(a), normalize_units(b))
                    for a, b in all_pairs
                ]
            else:
                llm_pairs = all_pairs

            # 3-way relevance classification (filters false positives)
            relevance_results = self.contradiction_checker.check_relevance_batch(
                llm_pairs, max_workers=4
            )

            for i, result in enumerate(relevance_results):
                nid = pair_node_ids[i]
                is_heuristic = pair_is_heuristic[i]

                if result.relationship == "contradicts" and result.confidence >= self.llm_contradiction_confidence:
                    verified.append({
                        "node_id": nid,
                        "reason": f"LLM (contradiction): {result.explanation}",
                    })
                elif is_heuristic and result.confidence == 0.0 and result.explanation and "failed" in result.explanation.lower():
                    verified.append({
                        "node_id": nid,
                        "reason": "Negation mismatch (heuristic, LLM unavailable)",
                    })

            return verified

        # No LLM: return heuristic results only
        return [
            {"node_id": nid, "reason": "Negation mismatch on semantically similar content"}
            for nid in heuristic_hits
        ]

    def _handle_contradictions(
        self,
        fact: ExtractedFact,
        contradictions: List[Dict[str, Any]],
        source: Optional[Source],
        model: str,
    ) -> IngestionVerdict:
        """Handle a fact that contradicts existing high-confidence nodes."""
        contradiction_ids = [c["node_id"] for c in contradictions]

        contra_parts = []
        for c in contradictions:
            cn = self.graph.get_node(c["node_id"])
            if cn:
                part = f"'{cn.text}' (conf={cn.confidence:.2f})"
                if c.get("reason"):
                    part += f" [{c['reason']}]"
                contra_parts.append(part)

        reason = (
            f"Contradicts existing high-confidence node(s): "
            + "; ".join(contra_parts)
        )

        if self.reject_contradictions:
            return IngestionVerdict(
                fact=fact,
                action="rejected",
                reason=reason,
                contradiction_node_ids=contradiction_ids,
            )

        # Add as CONTESTED with low confidence
        node = self._create_node(fact, source, model)
        node.confidence = min(node.confidence, 0.3)
        node.update_status()  # will become PROVISIONAL or CONTESTED
        self.graph.add_node(node)
        self._index_node(node)

        # Record contradictions
        for cid in contradiction_ids:
            self.graph.add_contradiction(node.id, cid)

        return IngestionVerdict(
            fact=fact,
            action="flagged",
            node_id=node.id,
            reason=reason,
            contradiction_node_ids=contradiction_ids,
        )

    def _add_fact(
        self,
        fact: ExtractedFact,
        source: Optional[Source],
        model: str,
    ) -> IngestionVerdict:
        """Add a fact as a new node (no dedup, no contradiction)."""
        node = self._create_node(fact, source, model)

        # Layer 1: Source quality gating
        if source and source.quality_score < self.source_quality_cap:
            node.confidence = min(node.confidence, source.quality_score)
            node.update_status()

        # Layer 1b: Tiered confidence cap
        if self.use_tiered_ingestion and source:
            tier_cap = source.tier.confidence_cap
            if tier_cap < 1.0 and node.confidence > tier_cap:
                node.confidence = tier_cap
                node.update_status()

        self.graph.add_node(node)
        self._index_node(node)

        return IngestionVerdict(
            fact=fact,
            action="added",
            node_id=node.id,
        )

    # ------------------------------------------------------------------
    # Internal: node creation and indexing
    # ------------------------------------------------------------------

    def _create_node(
        self,
        fact: ExtractedFact,
        source: Optional[Source],
        model: str,
    ) -> KnowledgeNode:
        """Create a KnowledgeNode from an ExtractedFact."""
        node_type_map = {v.value: v for v in NodeType}
        nt = node_type_map.get(fact.fact_type, NodeType.FACT)

        node = KnowledgeNode(
            text=fact.text,
            node_type=nt,
            confidence=fact.confidence,
            domain=fact.domain,
            derivation=Derivation(
                method="extracted",
                derivation_type="extracted",
                notes=f"Pipeline extraction via {model}",
            ),
        )
        if source:
            node.add_source(source)

        node.metadata["source_quote"] = fact.source_quote
        node.metadata["extraction_model"] = model
        if "quote_verified" in fact.metadata:
            node.metadata["quote_verified"] = fact.metadata["quote_verified"]
        return node

    def _index_node(self, node: KnowledgeNode) -> None:
        """Add a single node to the embedding and entity indexes."""
        try:
            self.embedding_engine.add_to_index(node)
        except Exception:
            # If embedding fails (e.g. model not loaded), log and continue.
            # The pipeline still works, just without dedup/contradiction
            # checking for this node.
            logger.warning(f"Could not index node {node.id}", exc_info=True)
        self.entity_index.add_node(node)

    def _verify_quote(
        self,
        fact: ExtractedFact,
        source_text: str,
    ) -> bool:
        """
        Verify that the fact's source_quote actually appears in the source text.

        Uses token-overlap ratio for fuzzy matching (LLMs often paraphrase
        slightly or change punctuation).

        Args:
            fact: The extracted fact with a source_quote field.
            source_text: The original text that was ingested.

        Returns:
            True if the quote is verified (or empty), False if it fails.
        """
        quote = fact.source_quote.strip()
        if not quote:
            # No quote to verify -- pass through
            return True

        # Fast path: exact substring match (case-insensitive)
        if quote.lower() in source_text.lower():
            return True

        # Fuzzy path: token-overlap ratio
        quote_tokens = set(quote.lower().split())
        source_tokens = set(source_text.lower().split())

        if not quote_tokens:
            return True

        overlap = len(quote_tokens & source_tokens)
        ratio = overlap / len(quote_tokens)

        return ratio >= self.min_quote_overlap

    def _ensure_indexed(self, node_ids: List[str]) -> None:
        """Make sure a set of nodes are in the embedding index."""
        for nid in node_ids:
            if not self.embedding_engine.has_embedding(nid):
                node = self.graph.get_node(nid)
                if node:
                    self._index_node(node)
