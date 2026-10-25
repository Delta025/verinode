"""
Automatic dependency detection between knowledge nodes.

Uses a combination of:
1. Semantic similarity (via EmbeddingEngine) to find candidate pairs.
2. Linguistic/structural heuristics (shared entities, causal language,
   definitional patterns) to classify relationships.

Detected relationship types:
- DEPENDS_ON:  B logically depends on A (A is a prerequisite for B).
- SUPPORTS:    A provides evidence for B.
- CONTRADICTS: A and B make conflicting claims.
- ELABORATES:  B adds detail to A.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple

from .graph import KnowledgeGraph
from .node import KnowledgeNode, NodeType

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships that can be detected between nodes."""

    DEPENDS_ON = "depends_on"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"


@dataclass
class DetectedDependency:
    """A detected dependency between two nodes."""

    source_id: str  # The node that is depended upon (prerequisite)
    target_id: str  # The node that depends on the source
    relation: RelationType
    confidence: float  # How confident we are in this detection [0, 1]
    evidence: str  # Explanation of why this dependency was detected


# ------------------------------------------------------------------
# Linguistic patterns
# ------------------------------------------------------------------

# Causal/dependency indicators: "because", "therefore", "since", "due to", etc.
_CAUSAL_PATTERNS = [
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bsince\b",
    r"\bdue to\b",
    r"\bas a result\b",
    r"\bconsequently\b",
    r"\bimplies\b",
    r"\bleads to\b",
    r"\bcauses\b",
    r"\bresults in\b",
]

_NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoesn't\b",
    r"\bwon't\b",
    r"\bneither\b",
    r"\bfalse\b",
    r"\bincorrect\b",
]

_DEFINITIONAL_PATTERNS = [
    r"\bis defined as\b",
    r"\brefers to\b",
    r"\bis a\b",
    r"\bis the\b",
    r"\bare called\b",
    r"\bknown as\b",
    r"\bmeans\b",
]

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "it",
        "its",
        "that",
        "this",
        "these",
        "those",
    }
)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class DependencyDetector:
    """
    Detects logical dependencies between knowledge nodes.

    Works in three stages:
    1. **Candidate selection** – Use semantic similarity to find pairs
       of nodes that are likely related.
    2. **Relationship classification** – Apply linguistic heuristics to
       classify the relationship type.
    3. **Graph wiring** – Optionally add detected dependencies to the
       knowledge graph.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        dependency_confidence_threshold: float = 0.4,
        llm_contradiction_confidence: float = 0.5,
        embedding_engine=None,
    ):
        """
        Args:
            similarity_threshold: Minimum cosine similarity to consider
                two nodes as candidates for a relationship.
            dependency_confidence_threshold: Minimum confidence to
                accept a detected dependency.
            llm_contradiction_confidence: Minimum LLM confidence to
                accept a contradiction verdict (default 0.5).
            embedding_engine: Optional pre-initialised EmbeddingEngine.
        """
        self.similarity_threshold = similarity_threshold
        self.dependency_confidence_threshold = dependency_confidence_threshold
        self.llm_contradiction_confidence = llm_contradiction_confidence
        self._embedding_engine = embedding_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(
        self,
        graph: KnowledgeGraph,
        embedding_engine=None,
        contradiction_checker=None,
    ) -> List[DetectedDependency]:
        """
        Detect dependencies across all node pairs in the graph.

        Uses the embedding engine for candidate selection when available,
        falling back to pairwise comparison otherwise.

        Args:
            graph: The knowledge graph to analyse.
            embedding_engine: Optional EmbeddingEngine override.
            contradiction_checker: Optional ContradictionChecker for
                LLM-assisted contradiction detection.

        Returns:
            List of DetectedDependency objects.
        """
        engine = embedding_engine or self._embedding_engine
        node_ids = list(graph.nodes.keys())

        if len(node_ids) < 2:
            return []

        candidates = self._find_candidates(graph, node_ids, engine)
        dependencies = []

        for src_id, tgt_id, similarity in candidates:
            src = graph.get_node(src_id)
            tgt = graph.get_node(tgt_id)
            if not src or not tgt:
                continue

            dep = self._classify_relationship(
                src,
                tgt,
                similarity,
                contradiction_checker=contradiction_checker,
            )
            if dep and dep.confidence >= self.dependency_confidence_threshold:
                dependencies.append(dep)

        return dependencies

    def detect_for_node(
        self,
        node: KnowledgeNode,
        graph: KnowledgeGraph,
        embedding_engine=None,
        k: int = 10,
        contradiction_checker=None,
    ) -> List[DetectedDependency]:
        """
        Detect dependencies for a single node against the rest of the graph.

        Args:
            node: The target node to find dependencies for.
            graph: The knowledge graph.
            embedding_engine: Optional EmbeddingEngine.
            k: Number of nearest neighbors to check.
            contradiction_checker: Optional ContradictionChecker for
                LLM-assisted contradiction detection.

        Returns:
            List of DetectedDependency objects.
        """
        engine = embedding_engine or self._embedding_engine
        dependencies = []

        if engine and node.id in engine._id_to_position:
            neighbors = engine.find_similar(node.id, k=k, threshold=self.similarity_threshold)
            candidates = [(n["node_id"], n["score"]) for n in neighbors]
        else:
            # Fallback: compare against all nodes using text similarity
            candidates = []
            for nid in graph.nodes:
                if nid == node.id:
                    continue
                other = graph.get_node(nid)
                if other:
                    candidates.append((nid, self._text_similarity(node.text, other.text)))

        for cand_id, similarity in candidates:
            other = graph.get_node(cand_id)
            if not other:
                continue

            dep = self._classify_relationship(
                other,
                node,
                similarity,
                contradiction_checker=contradiction_checker,
            )
            if dep and dep.confidence >= self.dependency_confidence_threshold:
                dependencies.append(dep)

        return dependencies

    def apply_dependencies(
        self,
        dependencies: List[DetectedDependency],
        graph: KnowledgeGraph,
    ) -> int:
        """
        Wire detected dependencies into the graph as predecessor/successor edges.

        Wires DEPENDS_ON, SUPPORTS, and ELABORATES relations as graph edges
        with their relation type labels. CONTRADICTS relations are recorded
        via add_contradiction. Edges that would create cycles are skipped.

        Args:
            dependencies: Output of detect_all or detect_for_node.
            graph: The graph to modify.

        Returns:
            Number of new edges added.
        """
        added = 0
        for dep in dependencies:
            src = graph.get_node(dep.source_id)
            tgt = graph.get_node(dep.target_id)
            if not src or not tgt:
                continue

            if dep.relation in (RelationType.DEPENDS_ON, RelationType.SUPPORTS, RelationType.ELABORATES):
                if dep.source_id not in tgt.predecessor_nodes:
                    if graph.would_create_cycle(dep.source_id, dep.target_id):
                        logger.warning(
                            "Skipping dependency %s -> %s (%s): would create cycle",
                            dep.source_id,
                            dep.target_id,
                            dep.relation.value,
                        )
                        continue
                    tgt.add_predecessor(dep.source_id, dep.relation.value)
                    src.add_successor(dep.target_id)
                    added += 1

            elif dep.relation == RelationType.CONTRADICTS:
                graph.add_contradiction(dep.source_id, dep.target_id)
                added += 1

        return added

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        graph: KnowledgeGraph,
        node_ids: List[str],
        embedding_engine=None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find candidate pairs that might have a relationship.

        Returns list of (source_id, target_id, similarity_score) tuples.
        """
        candidates = []
        seen = set()

        if embedding_engine and embedding_engine.index_size > 0:
            # Use embedding-based nearest-neighbor search
            for nid in node_ids:
                if nid not in embedding_engine._id_to_position:
                    continue
                neighbors = embedding_engine.find_similar(
                    nid,
                    k=min(10, len(node_ids) - 1),
                    threshold=self.similarity_threshold,
                )
                for nb in neighbors:
                    pair = tuple(sorted([nid, nb["node_id"]]))
                    if pair not in seen:
                        seen.add(pair)
                        candidates.append((nid, nb["node_id"], nb["score"]))
        else:
            # Brute-force pairwise text similarity
            for i, id_a in enumerate(node_ids):
                node_a = graph.get_node(id_a)
                if not node_a:
                    continue
                for id_b in node_ids[i + 1 :]:
                    node_b = graph.get_node(id_b)
                    if not node_b:
                        continue
                    sim = self._text_similarity(node_a.text, node_b.text)
                    if sim >= self.similarity_threshold:
                        candidates.append((id_a, id_b, sim))

        return candidates

    # ------------------------------------------------------------------
    # Relationship classification
    # ------------------------------------------------------------------

    def _classify_relationship(
        self,
        source: KnowledgeNode,
        target: KnowledgeNode,
        similarity: float,
        contradiction_checker=None,
    ) -> Optional[DetectedDependency]:
        """
        Classify the relationship between two nodes.

        When a contradiction_checker (ContradictionChecker) is provided and
        similarity is above 0.3, the LLM is consulted for contradiction
        detection.  The LLM result overrides the heuristic: it can promote a
        non-contradiction to a contradiction (catching numerical / semantic
        cases the heuristic misses) or demote a heuristic hit that turns out
        to be a false positive.

        Returns None if no meaningful relationship detected.
        """
        src_text = source.text.lower()
        tgt_text = target.text.lower()

        # 1. Check for contradiction
        # --- heuristic first (fast) ---
        contradiction_score = self._contradiction_score(src_text, tgt_text, similarity)
        heuristic_says_contradiction = contradiction_score > 0.5

        # --- LLM override when available ---
        if contradiction_checker is not None and similarity > 0.3:
            try:
                llm_result = contradiction_checker.check_relevance(source.text, target.text)
                if (
                    llm_result.relationship == "contradicts"
                    and llm_result.confidence >= self.llm_contradiction_confidence
                ):
                    # LLM says contradiction → use its verdict
                    return DetectedDependency(
                        source_id=source.id,
                        target_id=target.id,
                        relation=RelationType.CONTRADICTS,
                        confidence=min(1.0, llm_result.confidence),
                        evidence=f"LLM (contradicts): {llm_result.explanation}",
                    )
                elif heuristic_says_contradiction and llm_result.relationship != "contradicts":
                    # LLM overrides heuristic false positive — fall through
                    heuristic_says_contradiction = False
            except Exception:
                logger.warning(
                    "LLM classification failed for %s→%s, falling back to heuristic",
                    source.id,
                    target.id,
                    exc_info=True,
                )

        if heuristic_says_contradiction:
            return DetectedDependency(
                source_id=source.id,
                target_id=target.id,
                relation=RelationType.CONTRADICTS,
                confidence=min(1.0, contradiction_score),
                evidence="Negation mismatch on semantically similar content",
            )

        # 2. Check for definitional dependency
        if self._is_definitional(source) and self._references_concept(target, source):
            return DetectedDependency(
                source_id=source.id,
                target_id=target.id,
                relation=RelationType.DEPENDS_ON,
                confidence=min(1.0, similarity + 0.2),
                evidence="Target references concept defined in source",
            )

        # 3. Check for causal dependency
        causal_score = self._causal_score(src_text, tgt_text)
        if causal_score > 0.3 and similarity > self.similarity_threshold:
            return DetectedDependency(
                source_id=source.id,
                target_id=target.id,
                relation=RelationType.DEPENDS_ON,
                confidence=min(1.0, (similarity + causal_score) / 2),
                evidence="Causal language connects these facts",
            )

        # 4. Check for entity overlap (shared key nouns)
        shared_entities = self._shared_entities(src_text, tgt_text)
        if len(shared_entities) >= 2 and similarity > self.similarity_threshold:
            # Determine direction: more general fact supports more specific
            if self._is_more_specific(target, source):
                return DetectedDependency(
                    source_id=source.id,
                    target_id=target.id,
                    relation=RelationType.ELABORATES,
                    confidence=min(1.0, similarity * 0.8),
                    evidence=f"Shared entities: {', '.join(list(shared_entities)[:5])}",
                )
            elif self._is_more_specific(source, target):
                return DetectedDependency(
                    source_id=target.id,
                    target_id=source.id,
                    relation=RelationType.ELABORATES,
                    confidence=min(1.0, similarity * 0.8),
                    evidence=f"Shared entities: {', '.join(list(shared_entities)[:5])}",
                )

        # 5. High similarity with same domain suggests support
        if similarity >= self.similarity_threshold and source.domain == target.domain:
            return DetectedDependency(
                source_id=source.id,
                target_id=target.id,
                relation=RelationType.SUPPORTS,
                confidence=similarity * 0.7,
                evidence="Semantic similarity in same domain",
            )

        return None

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Simple word-overlap Jaccard similarity as a fallback."""
        words_a = set(text_a.lower().split()) - _STOP_WORDS
        words_b = set(text_b.lower().split()) - _STOP_WORDS
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _contradiction_score(text_a: str, text_b: str, similarity: float) -> float:
        """Score how likely two texts contradict each other.

        Uses negation asymmetry and numerical disagreement heuristics.
        """
        if similarity < 0.3:
            return 0.0  # Too dissimilar to be contradictions

        score = 0.0

        # Negation asymmetry: one statement negated, the other not
        neg_a = sum(1 for p in _NEGATION_PATTERNS if re.search(p, text_a))
        neg_b = sum(1 for p in _NEGATION_PATTERNS if re.search(p, text_b))
        if (neg_a > 0) != (neg_b > 0):
            score = max(score, similarity * 0.9)

        # Numerical disagreement: same topic but different numbers
        nums_a = set(re.findall(r"\b\d+(?:\.\d+)?\b", text_a))
        nums_b = set(re.findall(r"\b\d+(?:\.\d+)?\b", text_b))
        if nums_a and nums_b and similarity > 0.5:
            # Both texts have numbers and are semantically similar
            # Check if they share some numbers but differ on others
            nums_a & nums_b
            diff_a = nums_a - nums_b
            diff_b = nums_b - nums_a
            if diff_a and diff_b:
                # Different numbers in similar texts — possible contradiction
                score = max(score, similarity * 0.7)

        # Antonym pairs: common mutually exclusive terms
        antonym_pairs = [
            (r"\btrue\b", r"\bfalse\b"),
            (r"\bincrease\b", r"\bdecrease\b"),
            (r"\babove\b", r"\bbelow\b"),
            (r"\bbefore\b", r"\bafter\b"),
            (r"\blarger\b", r"\bsmaller\b"),
            (r"\bmore\b", r"\bless\b"),
            (r"\bhigher\b", r"\blower\b"),
            (r"\bpositive\b", r"\bnegative\b"),
            (r"\bsuccess\b", r"\bfailure?\b"),
            (r"\bsafe\b", r"\bdangerous\b"),
        ]
        for pat_x, pat_y in antonym_pairs:
            if (re.search(pat_x, text_a) and re.search(pat_y, text_b)) or (
                re.search(pat_y, text_a) and re.search(pat_x, text_b)
            ):
                score = max(score, similarity * 0.8)
                break

        return score

    @staticmethod
    def _is_definitional(node: KnowledgeNode) -> bool:
        """Check if a node is a definition."""
        if node.node_type == NodeType.DEFINITION:
            return True
        text = node.text.lower()
        return any(re.search(p, text) for p in _DEFINITIONAL_PATTERNS)

    @staticmethod
    def _references_concept(node: KnowledgeNode, definition_node: KnowledgeNode) -> bool:
        """Check if node references the concept defined by definition_node."""
        # Extract the subject being defined (first few key words)
        def_text = definition_node.text.lower()
        node_text = node.text.lower()

        # Simple approach: find the subject (words before "is defined as" / "is a" etc.)
        for pattern in _DEFINITIONAL_PATTERNS:
            match = re.search(pattern, def_text)
            if match:
                subject = def_text[: match.start()].strip()
                # Check if any substantial word from subject appears in node_text
                subject_words = set(subject.split()) - {"the", "a", "an"}
                if subject_words and any(w in node_text for w in subject_words if len(w) > 2):
                    return True
        return False

    @staticmethod
    def _causal_score(text_a: str, text_b: str) -> float:
        """Score causal language in the pair."""
        combined = text_a + " " + text_b
        matches = sum(1 for p in _CAUSAL_PATTERNS if re.search(p, combined))
        return min(1.0, matches * 0.3)

    @staticmethod
    def _shared_entities(text_a: str, text_b: str) -> Set[str]:
        """Find shared non-stop-word tokens (min 4 chars) between texts."""
        words_a = {w for w in text_a.lower().split() if len(w) >= 4} - _STOP_WORDS
        words_b = {w for w in text_b.lower().split() if len(w) >= 4} - _STOP_WORDS
        return words_a & words_b

    @staticmethod
    def _is_more_specific(candidate: KnowledgeNode, reference: KnowledgeNode) -> bool:
        """Heuristic: longer text with more detail is more specific."""
        return len(candidate.text) > len(reference.text) * 1.3
