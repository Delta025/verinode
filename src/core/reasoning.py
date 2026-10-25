"""
Reasoning engine for the knowledge graph.

Provides multi-hop inference, query answering, hypothesis testing,
and explanation generation over the knowledge graph.

Key capabilities:
- Forward chaining: derive new conclusions from existing facts.
- Backward chaining: find evidence chains that support or refute a claim.
- Query answering: find the best answer to a natural-language question
  using semantic search + graph traversal.
- Explanation: generate human-readable reasoning traces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

from .confidence import ConfidenceEngine
from .graph import KnowledgeGraph

# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


class InferenceStrategy(Enum):
    """Strategy for combining evidence along a chain."""

    WEAKEST_LINK = "weakest_link"  # min(confidences)
    PRODUCT = "product"  # product of confidences
    AVERAGE = "average"  # mean of confidences


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    node_id: str
    node_text: str
    confidence: float
    step_type: str  # "premise", "inference", "conclusion"
    explanation: str = ""


@dataclass
class ReasoningChain:
    """A complete chain of reasoning from premises to conclusion."""

    steps: List[ReasoningStep]
    conclusion_confidence: float
    strategy: InferenceStrategy
    is_valid: bool = True
    explanation: str = ""

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def premises(self) -> List[ReasoningStep]:
        return [s for s in self.steps if s.step_type == "premise"]

    @property
    def conclusion(self) -> Optional[ReasoningStep]:
        conclusions = [s for s in self.steps if s.step_type == "conclusion"]
        return conclusions[-1] if conclusions else None


@dataclass
class QueryResult:
    """Result of a query against the knowledge graph."""

    query: str
    answer_node_id: Optional[str]
    answer_text: str
    confidence: float
    supporting_chains: List[ReasoningChain]
    related_node_ids: List[str] = field(default_factory=list)


@dataclass
class HypothesisResult:
    """Result of testing a hypothesis against the graph."""

    hypothesis: str
    supported: bool
    confidence: float
    supporting_evidence: List[ReasoningChain]
    contradicting_evidence: List[ReasoningChain]
    explanation: str = ""


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class ReasoningEngine:
    """
    Multi-hop reasoning over the knowledge graph.

    Combines graph traversal, confidence propagation, and optional
    semantic search to answer queries and derive conclusions.
    """

    def __init__(
        self,
        confidence_engine: Optional[ConfidenceEngine] = None,
        max_chain_depth: int = 5,
        min_confidence: float = 0.1,
        inference_strategy: InferenceStrategy = InferenceStrategy.WEAKEST_LINK,
    ):
        """
        Args:
            confidence_engine: ConfidenceEngine instance (created if None).
            max_chain_depth: Maximum depth for chain traversal.
            min_confidence: Minimum confidence to include in results.
            inference_strategy: Default strategy for combining evidence.
        """
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.max_chain_depth = max_chain_depth
        self.min_confidence = min_confidence
        self.inference_strategy = inference_strategy

    # ------------------------------------------------------------------
    # Forward chaining
    # ------------------------------------------------------------------

    def forward_chain(
        self,
        graph: KnowledgeGraph,
        start_node_id: str,
        max_depth: Optional[int] = None,
    ) -> List[ReasoningChain]:
        """
        Forward chain from a node to find all conclusions it supports.

        Traverses the successor graph from start_node_id, building
        reasoning chains along each path.

        Args:
            graph: Knowledge graph.
            start_node_id: Starting node ID.
            max_depth: Override max chain depth.

        Returns:
            List of reasoning chains, one per reachable leaf.
        """
        max_depth = max_depth or self.max_chain_depth
        start = graph.get_node(start_node_id)
        if not start:
            return []

        chains = []
        self._forward_dfs(
            graph,
            start_node_id,
            current_chain=[
                ReasoningStep(
                    node_id=start.id,
                    node_text=start.text,
                    confidence=start.confidence,
                    step_type="premise",
                    explanation="Starting premise",
                )
            ],
            visited={start_node_id},
            chains=chains,
            depth=0,
            max_depth=max_depth,
        )
        return chains

    def _forward_dfs(
        self,
        graph: KnowledgeGraph,
        node_id: str,
        current_chain: List[ReasoningStep],
        visited: Set[str],
        chains: List[ReasoningChain],
        depth: int,
        max_depth: int,
    ):
        """DFS helper for forward chaining."""
        node = graph.get_node(node_id)
        if not node:
            return

        successors = [sid for sid in node.successor_nodes if sid in graph.nodes and sid not in visited]

        if not successors or depth >= max_depth:
            # Leaf or max depth — emit chain
            if len(current_chain) > 1:
                # Mark last step as conclusion
                chain_copy = list(current_chain)
                chain_copy[-1] = ReasoningStep(
                    node_id=chain_copy[-1].node_id,
                    node_text=chain_copy[-1].node_text,
                    confidence=chain_copy[-1].confidence,
                    step_type="conclusion",
                    explanation=chain_copy[-1].explanation,
                )
                conf = self._chain_confidence(chain_copy)
                chains.append(
                    ReasoningChain(
                        steps=chain_copy,
                        conclusion_confidence=conf,
                        strategy=self.inference_strategy,
                        is_valid=conf >= self.min_confidence,
                        explanation=self._build_explanation(chain_copy),
                    )
                )
            return

        for succ_id in successors:
            succ = graph.get_node(succ_id)
            if not succ:
                continue

            step = ReasoningStep(
                node_id=succ.id,
                node_text=succ.text,
                confidence=succ.confidence,
                step_type="inference",
                explanation=f"Derived from {node.text[:40]}...",
            )

            visited.add(succ_id)
            current_chain.append(step)
            self._forward_dfs(graph, succ_id, current_chain, visited, chains, depth + 1, max_depth)
            current_chain.pop()
            visited.discard(succ_id)

    # ------------------------------------------------------------------
    # Backward chaining
    # ------------------------------------------------------------------

    def backward_chain(
        self,
        graph: KnowledgeGraph,
        target_node_id: str,
        max_depth: Optional[int] = None,
    ) -> List[ReasoningChain]:
        """
        Backward chain from a conclusion to find its supporting evidence.

        Traverses the predecessor graph to build evidence chains.

        Args:
            graph: Knowledge graph.
            target_node_id: The conclusion node to explain.
            max_depth: Override max chain depth.

        Returns:
            List of reasoning chains ending at target_node_id.
        """
        max_depth = max_depth or self.max_chain_depth
        target = graph.get_node(target_node_id)
        if not target:
            return []

        chains = []
        self._backward_dfs(
            graph,
            target_node_id,
            current_chain=[
                ReasoningStep(
                    node_id=target.id,
                    node_text=target.text,
                    confidence=target.confidence,
                    step_type="conclusion",
                    explanation="Conclusion to be explained",
                )
            ],
            visited={target_node_id},
            chains=chains,
            depth=0,
            max_depth=max_depth,
        )
        # Reverse the chains so they read premise -> conclusion
        for chain in chains:
            chain.steps.reverse()
        return chains

    def _backward_dfs(
        self,
        graph: KnowledgeGraph,
        node_id: str,
        current_chain: List[ReasoningStep],
        visited: Set[str],
        chains: List[ReasoningChain],
        depth: int,
        max_depth: int,
    ):
        """DFS helper for backward chaining."""
        node = graph.get_node(node_id)
        if not node:
            return

        predecessors = [pid for pid in node.predecessor_nodes if pid in graph.nodes and pid not in visited]

        if not predecessors or depth >= max_depth:
            # Root or max depth — emit chain
            if len(current_chain) > 1:
                chain_copy = list(current_chain)
                # Mark last appended (which is root after reversal) as premise
                chain_copy[-1] = ReasoningStep(
                    node_id=chain_copy[-1].node_id,
                    node_text=chain_copy[-1].node_text,
                    confidence=chain_copy[-1].confidence,
                    step_type="premise",
                    explanation="Base evidence",
                )
                conf = self._chain_confidence(chain_copy)
                chains.append(
                    ReasoningChain(
                        steps=chain_copy,
                        conclusion_confidence=conf,
                        strategy=self.inference_strategy,
                        is_valid=conf >= self.min_confidence,
                        explanation="",  # Built after reversal
                    )
                )
            return

        for pred_id in predecessors:
            pred = graph.get_node(pred_id)
            if not pred:
                continue

            step = ReasoningStep(
                node_id=pred.id,
                node_text=pred.text,
                confidence=pred.confidence,
                step_type="inference",
                explanation=f"Supports {node.text[:40]}...",
            )

            visited.add(pred_id)
            current_chain.append(step)
            self._backward_dfs(graph, pred_id, current_chain, visited, chains, depth + 1, max_depth)
            current_chain.pop()
            visited.discard(pred_id)

    # ------------------------------------------------------------------
    # Query answering
    # ------------------------------------------------------------------

    def query(
        self,
        graph: KnowledgeGraph,
        question: str,
        embedding_engine=None,
        k: int = 5,
    ) -> QueryResult:
        """
        Answer a natural-language question using the knowledge graph.

        Uses semantic search (if embedding_engine provided) to find
        relevant nodes, then builds supporting reasoning chains.

        Args:
            graph: Knowledge graph.
            question: Natural language question.
            embedding_engine: Optional EmbeddingEngine for semantic search.
            k: Number of candidate nodes to consider.

        Returns:
            QueryResult with answer, confidence, and supporting chains.
        """
        # Find candidate answer nodes
        candidates = self._find_candidates(graph, question, embedding_engine, k)

        if not candidates:
            return QueryResult(
                query=question,
                answer_node_id=None,
                answer_text="No relevant information found in the knowledge graph.",
                confidence=0.0,
                supporting_chains=[],
            )

        # Build reasoning chains for each candidate
        best_candidate = None
        best_score = -1.0
        best_chains = []

        for node_id, search_score in candidates:
            node = graph.get_node(node_id)
            if not node:
                continue

            chains = self.backward_chain(graph, node_id)
            # Score = search relevance * node confidence * chain support
            chain_boost = min(1.0, len(chains) * 0.1 + 0.5) if chains else 0.5
            combined_score = search_score * node.confidence * chain_boost

            if combined_score > best_score:
                best_score = combined_score
                best_candidate = node
                best_chains = chains

        if not best_candidate:
            return QueryResult(
                query=question,
                answer_node_id=None,
                answer_text="No relevant information found.",
                confidence=0.0,
                supporting_chains=[],
            )

        # Build explanations for chains
        for chain in best_chains:
            chain.explanation = self._build_explanation(chain.steps)

        return QueryResult(
            query=question,
            answer_node_id=best_candidate.id,
            answer_text=best_candidate.text,
            confidence=min(1.0, best_score),
            supporting_chains=best_chains,
            related_node_ids=[c for c, _ in candidates if c != best_candidate.id],
        )

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    def test_hypothesis(
        self,
        graph: KnowledgeGraph,
        hypothesis: str,
        embedding_engine=None,
        k: int = 10,
    ) -> HypothesisResult:
        """
        Test a hypothesis against the knowledge graph.

        Searches for both supporting and contradicting evidence.

        Args:
            graph: Knowledge graph.
            hypothesis: The claim to test.
            embedding_engine: Optional EmbeddingEngine.
            k: Number of candidates to consider.

        Returns:
            HypothesisResult with verdict, confidence, and evidence.
        """
        candidates = self._find_candidates(graph, hypothesis, embedding_engine, k)
        contradicting_ids = self._find_contradicting_nodes(graph, hypothesis, candidates)

        supporting: List[ReasoningChain] = []
        contradicting: List[ReasoningChain] = []

        for node_id, score in candidates:
            node = graph.get_node(node_id)
            if not node:
                continue

            chains = self.backward_chain(graph, node_id)

            # Check if this node contradicts or supports the hypothesis
            if node_id in contradicting_ids:
                for chain in chains:
                    chain.explanation = self._build_explanation(chain.steps)
                contradicting.extend(chains)
                # Also count standalone contradictions
                if not chains:
                    contradicting.append(
                        ReasoningChain(
                            steps=[
                                ReasoningStep(
                                    node_id=node.id,
                                    node_text=node.text,
                                    confidence=node.confidence,
                                    step_type="premise",
                                    explanation="Contradicting evidence",
                                )
                            ],
                            conclusion_confidence=node.confidence,
                            strategy=self.inference_strategy,
                            explanation=f"Contradicts hypothesis: {node.text}",
                        )
                    )
            else:
                for chain in chains:
                    chain.explanation = self._build_explanation(chain.steps)
                supporting.extend(chains)
                if not chains:
                    supporting.append(
                        ReasoningChain(
                            steps=[
                                ReasoningStep(
                                    node_id=node.id,
                                    node_text=node.text,
                                    confidence=node.confidence,
                                    step_type="premise",
                                    explanation="Supporting evidence",
                                )
                            ],
                            conclusion_confidence=node.confidence,
                            strategy=self.inference_strategy,
                            explanation=f"Supports hypothesis: {node.text}",
                        )
                    )

        # Compute overall confidence
        support_conf = max((c.conclusion_confidence for c in supporting), default=0.0)
        contra_conf = max((c.conclusion_confidence for c in contradicting), default=0.0)

        if support_conf > 0 or contra_conf > 0:
            net_confidence = support_conf - contra_conf * 0.5
            net_confidence = max(0.0, min(1.0, net_confidence))
        else:
            net_confidence = 0.0

        is_supported = support_conf > contra_conf and net_confidence > 0.3

        explanation_parts = []
        if supporting:
            explanation_parts.append(
                f"Found {len(supporting)} supporting evidence chain(s) (max confidence: {support_conf:.2f})"
            )
        if contradicting:
            explanation_parts.append(
                f"Found {len(contradicting)} contradicting evidence chain(s) (max confidence: {contra_conf:.2f})"
            )
        if not supporting and not contradicting:
            explanation_parts.append("No relevant evidence found in the knowledge graph")

        return HypothesisResult(
            hypothesis=hypothesis,
            supported=is_supported,
            confidence=net_confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            explanation=". ".join(explanation_parts) + ".",
        )

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_node(
        self,
        graph: KnowledgeGraph,
        node_id: str,
    ) -> str:
        """
        Generate a human-readable explanation of why a node has its
        current confidence score.

        Args:
            graph: Knowledge graph.
            node_id: Node to explain.

        Returns:
            Human-readable explanation string.
        """
        node = graph.get_node(node_id)
        if not node:
            return "Node not found."

        lines = [f"Fact: {node.text}"]
        lines.append(f"Confidence: {node.confidence:.2f} (status: {node.status.value})")

        # Sources
        if node.sources:
            lines.append(f"Sources ({len(node.sources)}):")
            for s in node.sources:
                lines.append(f"  - {s} (quality: {s.quality_score:.2f})")

        # Dependencies
        if node.predecessor_nodes:
            lines.append(f"Depends on {len(node.predecessor_nodes)} fact(s):")
            for pid in node.predecessor_nodes:
                pred = graph.get_node(pid)
                if pred:
                    lines.append(f"  - {pred.text[:60]} (conf: {pred.confidence:.2f})")

        # Backward chains
        chains = self.backward_chain(graph, node_id, max_depth=3)
        if chains:
            lines.append(f"Evidence chains ({len(chains)}):")
            for i, chain in enumerate(chains):
                lines.append(f"  Chain {i + 1} (confidence: {chain.conclusion_confidence:.2f}):")
                for step in chain.steps:
                    marker = ">>" if step.step_type == "conclusion" else "  "
                    lines.append(f"    {marker} {step.node_text[:60]} [{step.confidence:.2f}]")

        # Conflicts
        if node.conflicts_with:
            lines.append(f"Conflicts with {len(node.conflicts_with)} node(s):")
            for cid in node.conflicts_with:
                conflict = graph.get_node(cid)
                if conflict:
                    lines.append(f"  - {conflict.text[:60]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        graph: KnowledgeGraph,
        query_text: str,
        embedding_engine=None,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find candidate nodes matching a query. Returns (node_id, score) pairs."""
        if embedding_engine and embedding_engine.index_size > 0:
            results = embedding_engine.search(query_text, k=k, graph=graph)
            return [(r["node_id"], r["score"]) for r in results]

        # Fallback: keyword search
        query_words = set(query_text.lower().split())
        scored = []
        for node_id, node in graph.nodes.items():
            node_words = set(node.text.lower().split())
            overlap = len(query_words & node_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _find_contradicting_nodes(
        self,
        graph: KnowledgeGraph,
        hypothesis: str,
        candidates: List[Tuple[str, float]],
    ) -> Set[str]:
        """Identify candidate nodes that contradict the hypothesis."""
        contradicting = set()
        hyp_lower = hypothesis.lower()
        has_negation = any(neg in hyp_lower for neg in ["not", "never", "no ", "cannot", "doesn't", "don't", "isn't"])

        for node_id, _ in candidates:
            node = graph.get_node(node_id)
            if not node:
                continue

            # If the node is in the conflicts list of any other candidate
            if node.conflicts_with:
                contradicting.add(node_id)
                continue

            # Check negation mismatch
            node_lower = node.text.lower()
            node_has_neg = any(
                neg in node_lower for neg in ["not", "never", "no ", "cannot", "doesn't", "don't", "isn't"]
            )
            if has_negation != node_has_neg:
                contradicting.add(node_id)

        return contradicting

    def _chain_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence for a reasoning chain."""
        if not steps:
            return 0.0

        confidences = [s.confidence for s in steps]

        if self.inference_strategy == InferenceStrategy.WEAKEST_LINK:
            base = min(confidences)
        elif self.inference_strategy == InferenceStrategy.PRODUCT:
            base = 1.0
            for c in confidences:
                base *= c
        else:  # AVERAGE
            base = sum(confidences) / len(confidences)

        # Apply depth decay
        depth_factor = 0.95 ** max(0, len(steps) - 1)
        return max(self.min_confidence, min(1.0, base * depth_factor))

    @staticmethod
    def _build_explanation(steps: List[ReasoningStep]) -> str:
        """Build a human-readable explanation from chain steps."""
        if not steps:
            return ""

        parts = []
        for i, step in enumerate(steps):
            if step.step_type == "premise":
                parts.append(f"Given that: {step.node_text}")
            elif step.step_type == "conclusion":
                parts.append(f"Therefore: {step.node_text} (confidence: {step.confidence:.2f})")
            else:
                parts.append(f"And since: {step.node_text}")

        return " -> ".join(parts)
