"""
LLM Memory Substrate — structured memory API for LLM consumption.

Provides a thin facade over existing engines, reformatting results
into flat, LLM-consumable JSON with explicit confidence scores,
source provenance, and reasoning chains.

Usage:
    memory = LLMMemory(graph, embedding_engine=emb, reasoning_engine=re,
                       confidence_engine=ce)
    result = memory.query("What is the boiling point of water?")
    facts = memory.recall("water boiling", k=3)
    verdict = memory.verify("Water boils at 50 degrees Celsius")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .confidence import ConfidenceEngine
from .graph import KnowledgeGraph
from .node import KnowledgeNode, Source
from .reasoning import ReasoningEngine

# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class MemoryFact:
    """Single fact with full provenance for LLM consumption."""

    node_id: str
    text: str
    confidence: float
    domain: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: Optional[List[Dict[str, Any]]] = None


@dataclass
class MemoryQueryResult:
    """Structured query result for LLM consumption."""

    query: str
    facts: List[MemoryFact] = field(default_factory=list)
    summary: str = ""
    overall_confidence: float = 0.0
    coverage: str = "none"  # "high" | "medium" | "low" | "none"
    total_graph_nodes: int = 0


@dataclass
class MemoryIngestResult:
    """Result of LLM submitting text for ingestion."""

    action: str  # "added" | "duplicate" | "contradicted" | "rejected"
    node_id: Optional[str] = None
    confidence: float = 0.0
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _source_to_dict(source: Source) -> Dict[str, Any]:
    """Convert a Source to a flat dictionary."""
    return {
        "title": source.title,
        "type": source.source_type,
        "quality": source.quality_score,
        "author": source.author,
        "date": source.date.isoformat() if source.date and hasattr(source.date, "isoformat") else None,
    }


def _node_to_memory_fact(
    node: KnowledgeNode,
    chains: Optional[List] = None,
) -> MemoryFact:
    """Convert a KnowledgeNode to a MemoryFact."""
    reasoning_chain = None
    if chains:
        reasoning_chain = [
            {
                "text": step.node_text,
                "confidence": step.confidence,
                "type": step.step_type,
            }
            for chain in chains
            for step in chain.steps
        ]

    return MemoryFact(
        node_id=node.id,
        text=node.text,
        confidence=node.confidence,
        domain=node.domain,
        sources=[_source_to_dict(s) for s in node.sources],
        reasoning_chain=reasoning_chain,
    )


def _compute_coverage(fact_count: int, overall_confidence: float) -> str:
    """Determine coverage level from fact count and confidence."""
    if fact_count == 0:
        return "none"
    if fact_count >= 3 and overall_confidence >= 0.7:
        return "high"
    if fact_count >= 1 and overall_confidence >= 0.4:
        return "medium"
    return "low"


# ------------------------------------------------------------------
# LLM Memory facade
# ------------------------------------------------------------------


class LLMMemory:
    """
    Facade for LLM programmatic access to the knowledge graph.

    All methods delegate to existing engines and reformat output
    into flat, JSON-serializable structures.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedding_engine=None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        pipeline=None,
        verification_engine=None,
    ):
        self.graph = graph
        self.embedding_engine = embedding_engine
        self.reasoning_engine = reasoning_engine or ReasoningEngine()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.pipeline = pipeline
        self.verification_engine = verification_engine

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        domain: Optional[str] = None,
        k: int = 10,
        include_chains: bool = False,
    ) -> MemoryQueryResult:
        """
        Query the graph for facts relevant to a question.

        Uses ReasoningEngine.query() internally and reformats the
        result into a flat MemoryQueryResult.

        Args:
            question: Natural-language question.
            domain: Optional domain filter.
            k: Number of candidate nodes.
            include_chains: Whether to include reasoning chains per fact.

        Returns:
            MemoryQueryResult with facts, confidence, and coverage.
        """
        total_nodes = len(self.graph)

        if total_nodes == 0:
            return MemoryQueryResult(
                query=question,
                coverage="none",
                total_graph_nodes=0,
            )

        # Use reasoning engine for semantic search + backward chaining
        qr = self.reasoning_engine.query(
            self.graph,
            question,
            embedding_engine=self.embedding_engine,
            k=k,
        )

        # Collect fact nodes: best answer + related nodes
        fact_ids = []
        if qr.answer_node_id:
            fact_ids.append(qr.answer_node_id)
        fact_ids.extend(qr.related_node_ids)

        # Apply domain filter if provided
        facts: List[MemoryFact] = []
        for nid in fact_ids:
            node = self.graph.get_node(nid)
            if not node:
                continue
            if domain and node.domain != domain:
                continue

            chains = None
            if include_chains:
                chains = self.reasoning_engine.backward_chain(self.graph, nid, max_depth=3)

            facts.append(_node_to_memory_fact(node, chains))

        # Compute overall confidence
        if facts:
            total_weight = sum(f.confidence for f in facts)
            overall_confidence = total_weight / len(facts)
        else:
            overall_confidence = 0.0

        coverage = _compute_coverage(len(facts), overall_confidence)

        # Build summary
        if qr.answer_node_id and qr.answer_text:
            summary = qr.answer_text
        elif facts:
            summary = facts[0].text
        else:
            summary = "No relevant information found."

        return MemoryQueryResult(
            query=question,
            facts=facts,
            summary=summary,
            overall_confidence=round(overall_confidence, 4),
            coverage=coverage,
            total_graph_nodes=total_nodes,
        )

    # ------------------------------------------------------------------
    # Recall (lightweight)
    # ------------------------------------------------------------------

    def recall(
        self,
        topic: str,
        domain: Optional[str] = None,
        k: int = 5,
    ) -> List[MemoryFact]:
        """
        Quick recall: top-k most relevant facts for a topic.

        Lighter weight than ``query()`` — uses embedding search
        directly without reasoning chains.

        Args:
            topic: Topic string to search for.
            domain: Optional domain filter.
            k: Number of results.

        Returns:
            List of MemoryFact instances.
        """
        if len(self.graph) == 0:
            return []

        if self.embedding_engine and self.embedding_engine.index_size > 0:
            results = self.embedding_engine.search(topic, k=k, graph=self.graph)
        else:
            # Fallback to keyword search via reasoning engine
            candidates = self.reasoning_engine._find_candidates(self.graph, topic, self.embedding_engine, k)
            results = [{"node_id": nid, "score": score} for nid, score in candidates]

        facts: List[MemoryFact] = []
        for r in results:
            node = self.graph.get_node(r["node_id"])
            if not node:
                continue
            if domain and node.domain != domain:
                continue
            facts.append(_node_to_memory_fact(node))

        return facts[:k]

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(self, claim: str) -> Dict[str, Any]:
        """
        Verify a claim before the LLM uses it.

        Delegates to VerificationEngine.verify() if available,
        otherwise returns insufficient_evidence.

        Args:
            claim: The factual claim to check.

        Returns:
            Dict with verdict, confidence, explanation, evidence_count.
        """
        if self.verification_engine is None:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "explanation": "No verification engine configured.",
                "evidence_count": 0,
            }

        result = self.verification_engine.verify(claim)
        return {
            "verdict": result.verdict.value,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "evidence_count": (len(result.supporting_evidence) + len(result.contradicting_evidence)),
        }

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        source_title: str = "LLM submission",
        source_quality: float = 0.7,
        domain: Optional[str] = None,
    ) -> MemoryIngestResult:
        """
        LLM submits text for ingestion and verification.

        Delegates to Pipeline.ingest() if available.

        Args:
            text: Text to ingest.
            source_title: Title for source provenance.
            source_quality: Quality score for the source [0, 1].
            domain: Optional domain label.

        Returns:
            MemoryIngestResult with action, node_id, contradictions.
        """
        if self.pipeline is None:
            return MemoryIngestResult(
                action="rejected",
                explanation="No ingestion pipeline configured.",
            )

        source = Source(
            source_type="llm_submission",
            title=source_title,
            quality_score=source_quality,
        )

        result = self.pipeline.ingest(
            text,
            source=source,
            domain=domain or "general",
        )

        # Summarize the result
        if result.nodes_added > 0:
            action = "added"
            # Get the first added node ID
            node_id = None
            confidence = 0.0
            for v in result.verdicts:
                if v.action == "added" and v.node_id:
                    node_id = v.node_id
                    node = self.graph.get_node(v.node_id)
                    confidence = node.confidence if node else 0.0
                    break
        elif result.duplicates_merged > 0:
            action = "duplicate"
            node_id = None
            confidence = 0.0
            for v in result.verdicts:
                if v.action == "duplicate" and v.duplicate_node_id:
                    node_id = v.duplicate_node_id
                    node = self.graph.get_node(v.duplicate_node_id)
                    confidence = node.confidence if node else 0.0
                    break
        elif result.facts_flagged > 0:
            action = "contradicted"
            node_id = None
            confidence = 0.0
        else:
            action = "rejected"
            node_id = None
            confidence = 0.0

        # Collect contradictions
        contradictions = []
        for v in result.verdicts:
            for cid in v.contradiction_node_ids:
                cnode = self.graph.get_node(cid)
                if cnode:
                    contradictions.append(
                        {
                            "node_id": cid,
                            "text": cnode.text,
                            "confidence": cnode.confidence,
                        }
                    )

        return MemoryIngestResult(
            action=action,
            node_id=node_id,
            confidence=confidence,
            contradictions=contradictions,
            explanation=result.summary,
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self, result: MemoryQueryResult) -> Dict[str, Any]:
        """Convert a MemoryQueryResult to a plain dict for JSON serialization."""
        return {
            "query": result.query,
            "facts": [
                {
                    "node_id": f.node_id,
                    "text": f.text,
                    "confidence": f.confidence,
                    "domain": f.domain,
                    "sources": f.sources,
                    "reasoning_chain": f.reasoning_chain,
                }
                for f in result.facts
            ],
            "summary": result.summary,
            "overall_confidence": result.overall_confidence,
            "coverage": result.coverage,
            "total_graph_nodes": result.total_graph_nodes,
        }
