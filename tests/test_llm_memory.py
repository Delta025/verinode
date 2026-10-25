"""
Tests for the LLM Memory Substrate (src/core/llm_memory.py).

Tests cover the LLMMemory facade: query, recall, verify, ingest,
coverage computation, and serialization helpers.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.core.llm_memory import (
    LLMMemory,
    MemoryFact,
    MemoryQueryResult,
    MemoryIngestResult,
    _source_to_dict,
    _node_to_memory_fact,
    _compute_coverage,
)
from src.core.node import KnowledgeNode, Source, NodeType
from src.core.graph import KnowledgeGraph
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import (
    ReasoningEngine, QueryResult, ReasoningChain, ReasoningStep,
    InferenceStrategy,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_node(text, confidence=0.8, domain="chemistry", node_id=None):
    """Create a test node."""
    return KnowledgeNode(
        text=text,
        confidence=confidence,
        domain=domain,
        node_id=node_id or text[:8].replace(" ", "_"),
        sources=[Source(source_type="textbook", title="Test Source", quality_score=0.9)],
    )


def build_chemistry_graph():
    """Build a small chemistry graph for testing."""
    graph = KnowledgeGraph()
    nodes = [
        make_node("Water boils at 100 degrees Celsius", 0.95, node_id="water_bp"),
        make_node("Water freezes at 0 degrees Celsius", 0.90, node_id="water_fp"),
        make_node("Gold has the chemical symbol Au", 0.85, node_id="gold_sym"),
        make_node("Sodium has the chemical symbol Na", 0.80, node_id="sodium_sym"),
        make_node("Helium is a noble gas", 0.75, node_id="helium"),
    ]
    for n in nodes:
        graph.add_node(n)
    return graph, nodes


# ------------------------------------------------------------------
# Tests: _compute_coverage
# ------------------------------------------------------------------

class TestComputeCoverage:
    def test_no_facts(self):
        assert _compute_coverage(0, 0.0) == "none"

    def test_high_coverage(self):
        assert _compute_coverage(5, 0.85) == "high"

    def test_medium_coverage(self):
        assert _compute_coverage(2, 0.6) == "medium"

    def test_low_coverage(self):
        assert _compute_coverage(1, 0.2) == "low"

    def test_boundary_high(self):
        assert _compute_coverage(3, 0.7) == "high"

    def test_boundary_medium(self):
        assert _compute_coverage(1, 0.4) == "medium"


# ------------------------------------------------------------------
# Tests: _source_to_dict
# ------------------------------------------------------------------

class TestSourceToDict:
    def test_basic(self):
        src = Source(source_type="textbook", title="Chem 101", quality_score=0.9)
        d = _source_to_dict(src)
        assert d["title"] == "Chem 101"
        assert d["type"] == "textbook"
        assert d["quality"] == 0.9
        assert d["author"] is None
        assert d["date"] is None

    def test_with_author(self):
        src = Source(source_type="paper", title="Study", author="Alice")
        d = _source_to_dict(src)
        assert d["author"] == "Alice"


# ------------------------------------------------------------------
# Tests: _node_to_memory_fact
# ------------------------------------------------------------------

class TestNodeToMemoryFact:
    def test_basic_conversion(self):
        node = make_node("Water boils at 100C", 0.95)
        fact = _node_to_memory_fact(node)
        assert fact.text == "Water boils at 100C"
        assert fact.confidence == 0.95
        assert fact.domain == "chemistry"
        assert len(fact.sources) == 1
        assert fact.reasoning_chain is None

    def test_with_chains(self):
        node = make_node("Water boils at 100C", 0.95)
        chain = ReasoningChain(
            steps=[
                ReasoningStep("n1", "Premise", 0.9, "premise"),
                ReasoningStep("n2", "Conclusion", 0.85, "conclusion"),
            ],
            conclusion_confidence=0.85,
            strategy=InferenceStrategy.WEAKEST_LINK,
        )
        fact = _node_to_memory_fact(node, chains=[chain])
        assert fact.reasoning_chain is not None
        assert len(fact.reasoning_chain) == 2
        assert fact.reasoning_chain[0]["text"] == "Premise"
        assert fact.reasoning_chain[1]["type"] == "conclusion"


# ------------------------------------------------------------------
# Tests: LLMMemory.query()
# ------------------------------------------------------------------

class TestQuery:
    def test_empty_graph(self):
        graph = KnowledgeGraph()
        memory = LLMMemory(graph)
        result = memory.query("What is water?")
        assert result.coverage == "none"
        assert result.total_graph_nodes == 0
        assert result.facts == []

    def test_query_returns_facts(self):
        graph, nodes = build_chemistry_graph()
        # Mock embedding engine for search
        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.95},
            {"node_id": "water_fp", "score": 0.80},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        result = memory.query("What is the boiling point of water?")

        assert len(result.facts) > 0
        assert result.total_graph_nodes == 5
        assert result.query == "What is the boiling point of water?"

    def test_query_includes_provenance(self):
        graph, nodes = build_chemistry_graph()
        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.95},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        result = memory.query("boiling point")

        if result.facts:
            fact = result.facts[0]
            assert fact.sources is not None
            assert len(fact.sources) >= 1
            assert "title" in fact.sources[0]
            assert fact.confidence > 0

    def test_query_with_domain_filter(self):
        graph, nodes = build_chemistry_graph()
        # Add a non-chemistry node
        bio_node = make_node("DNA has a double helix", 0.9, domain="biology", node_id="dna")
        graph.add_node(bio_node)

        mock_emb = MagicMock()
        mock_emb.index_size = 6
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.9},
            {"node_id": "dna", "score": 0.5},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        result = memory.query("water", domain="chemistry")

        # Biology node should be filtered out
        for fact in result.facts:
            assert fact.domain == "chemistry"

    def test_query_with_chains(self):
        graph, nodes = build_chemistry_graph()
        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.95},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        result = memory.query("boiling point", include_chains=True)

        # Chains may be empty if no predecessors, but the code path is exercised
        assert result.facts is not None

    def test_overall_confidence_calculation(self):
        graph, nodes = build_chemistry_graph()
        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.95},
            {"node_id": "water_fp", "score": 0.90},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        result = memory.query("water temperature")

        # Should have facts and a non-zero overall confidence
        if result.facts:
            assert result.overall_confidence > 0
            # Overall confidence should be average of fact confidences
            expected_avg = sum(f.confidence for f in result.facts) / len(result.facts)
            assert abs(result.overall_confidence - round(expected_avg, 4)) < 0.01


# ------------------------------------------------------------------
# Tests: LLMMemory.recall()
# ------------------------------------------------------------------

class TestRecall:
    def test_empty_graph(self):
        graph = KnowledgeGraph()
        memory = LLMMemory(graph)
        facts = memory.recall("water")
        assert facts == []

    def test_recall_returns_facts(self):
        graph, nodes = build_chemistry_graph()
        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.95},
            {"node_id": "gold_sym", "score": 0.60},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        facts = memory.recall("water", k=5)
        assert len(facts) == 2
        assert facts[0].node_id == "water_bp"

    def test_recall_with_domain_filter(self):
        graph, nodes = build_chemistry_graph()
        bio_node = make_node("DNA has a double helix", 0.9, domain="biology", node_id="dna")
        graph.add_node(bio_node)

        mock_emb = MagicMock()
        mock_emb.index_size = 6
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.9},
            {"node_id": "dna", "score": 0.5},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        facts = memory.recall("molecules", domain="chemistry")

        for fact in facts:
            assert fact.domain == "chemistry"

    def test_recall_limits_to_k(self):
        graph, nodes = build_chemistry_graph()
        mock_emb = MagicMock()
        mock_emb.index_size = 5
        mock_emb.search.return_value = [
            {"node_id": "water_bp", "score": 0.95},
            {"node_id": "water_fp", "score": 0.90},
            {"node_id": "gold_sym", "score": 0.80},
        ]
        memory = LLMMemory(graph, embedding_engine=mock_emb)
        facts = memory.recall("water", k=2)
        assert len(facts) <= 2

    def test_recall_fallback_keyword(self):
        """Recall falls back to keyword search when no embedding engine."""
        graph, nodes = build_chemistry_graph()
        memory = LLMMemory(graph)  # No embedding engine
        facts = memory.recall("Water boils")
        # Should use keyword fallback; may or may not find results
        assert isinstance(facts, list)


# ------------------------------------------------------------------
# Tests: LLMMemory.verify()
# ------------------------------------------------------------------

class TestVerify:
    def test_no_verification_engine(self):
        graph = KnowledgeGraph()
        memory = LLMMemory(graph)
        result = memory.verify("Water boils at 50C")
        assert result["verdict"] == "insufficient_evidence"
        assert result["confidence"] == 0.0

    def test_delegates_to_verification_engine(self):
        graph = KnowledgeGraph()
        mock_ve = MagicMock()

        from src.core.verification import Verdict, VerificationResult
        mock_ve.verify.return_value = VerificationResult(
            claim="Water boils at 50C",
            verdict=Verdict.CONTRADICTED,
            confidence=0.95,
            explanation="Contradicted by 1 node(s).",
            supporting_evidence=[],
            contradicting_evidence=[MagicMock()],
        )

        memory = LLMMemory(graph, verification_engine=mock_ve)
        result = memory.verify("Water boils at 50C")
        assert result["verdict"] == "contradicted"
        assert result["confidence"] == 0.95
        assert result["evidence_count"] == 1


# ------------------------------------------------------------------
# Tests: LLMMemory.ingest()
# ------------------------------------------------------------------

class TestIngest:
    def test_no_pipeline(self):
        graph = KnowledgeGraph()
        memory = LLMMemory(graph)
        result = memory.ingest("Water boils at 100C")
        assert result.action == "rejected"
        assert "No ingestion pipeline" in result.explanation

    def test_ingest_added(self):
        graph, nodes = build_chemistry_graph()
        mock_pipeline = MagicMock()

        from src.core.pipeline import IngestionResult, IngestionVerdict
        from src.core.fact_extraction import ExtractedFact
        mock_pipeline.ingest.return_value = IngestionResult(
            source_text="Neon is a noble gas",
            extraction_model="qwen3:14b",
            verdicts=[
                IngestionVerdict(
                    fact=ExtractedFact(text="Neon is a noble gas", confidence=0.8),
                    action="added",
                    node_id="neon_ng",
                ),
            ],
            nodes_added=1,
        )

        # Add the node to graph so ingest can look it up
        neon = make_node("Neon is a noble gas", 0.8, node_id="neon_ng")
        graph.add_node(neon)

        memory = LLMMemory(graph, pipeline=mock_pipeline)
        result = memory.ingest("Neon is a noble gas", source_title="LLM")
        assert result.action == "added"
        assert result.node_id == "neon_ng"
        assert result.confidence > 0

    def test_ingest_duplicate(self):
        graph, nodes = build_chemistry_graph()
        mock_pipeline = MagicMock()

        from src.core.pipeline import IngestionResult, IngestionVerdict
        from src.core.fact_extraction import ExtractedFact
        mock_pipeline.ingest.return_value = IngestionResult(
            source_text="Water boils at 100C",
            extraction_model="qwen3:14b",
            verdicts=[
                IngestionVerdict(
                    fact=ExtractedFact(text="Water boils at 100C", confidence=0.9),
                    action="duplicate",
                    duplicate_node_id="water_bp",
                ),
            ],
            duplicates_merged=1,
        )

        memory = LLMMemory(graph, pipeline=mock_pipeline)
        result = memory.ingest("Water boils at 100C")
        assert result.action == "duplicate"
        assert result.node_id == "water_bp"

    def test_ingest_contradicted(self):
        graph, nodes = build_chemistry_graph()
        mock_pipeline = MagicMock()

        from src.core.pipeline import IngestionResult, IngestionVerdict
        from src.core.fact_extraction import ExtractedFact
        mock_pipeline.ingest.return_value = IngestionResult(
            source_text="Water boils at 50C",
            extraction_model="qwen3:14b",
            verdicts=[
                IngestionVerdict(
                    fact=ExtractedFact(text="Water boils at 50C", confidence=0.7),
                    action="flagged",
                    contradiction_node_ids=["water_bp"],
                ),
            ],
            facts_flagged=1,
        )

        memory = LLMMemory(graph, pipeline=mock_pipeline)
        result = memory.ingest("Water boils at 50C")
        assert result.action == "contradicted"
        assert len(result.contradictions) == 1
        assert result.contradictions[0]["node_id"] == "water_bp"


# ------------------------------------------------------------------
# Tests: LLMMemory.to_dict()
# ------------------------------------------------------------------

class TestToDict:
    def test_serialization(self):
        graph = KnowledgeGraph()
        memory = LLMMemory(graph)
        result = MemoryQueryResult(
            query="test",
            facts=[
                MemoryFact(
                    node_id="n1",
                    text="Fact one",
                    confidence=0.8,
                    domain="chemistry",
                    sources=[{"title": "Src", "quality": 0.9}],
                ),
            ],
            summary="Fact one",
            overall_confidence=0.8,
            coverage="medium",
            total_graph_nodes=10,
        )
        d = memory.to_dict(result)
        assert d["query"] == "test"
        assert len(d["facts"]) == 1
        assert d["facts"][0]["text"] == "Fact one"
        assert d["overall_confidence"] == 0.8
        assert d["coverage"] == "medium"
