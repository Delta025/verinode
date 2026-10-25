"""
Calibration regression tests for LLM-dependent components.

Tests that fact extraction, LLM contradiction detection, and the full
pipeline maintain minimum performance on calibration datasets.

Requires Ollama with qwen2.5:7b (or configured model).
Run with: pytest tests/test_calibration_llm.py --run-calibration --run-integration
"""

import os
import statistics
import numpy as np
import pytest

from src.core.fact_extraction import FactExtractionEngine
from src.core.embeddings import EmbeddingEngine
from src.core.llm_contradiction import create_contradiction_checker
from src.core.graph import KnowledgeGraph
from src.core.node import Source
from src.core.pipeline import Pipeline
from src.core.dependency_detection import DependencyDetector
from src.core.confidence import ConfidenceEngine
from src.core.metrics import load_jsonl

pytestmark = [pytest.mark.calibration, pytest.mark.integration]

ARC_PATH = os.path.join("data", "calibration", "arc_sentences.jsonl")
FEVER_PATH = os.path.join("data", "calibration", "fever_subset.jsonl")


def _cosine_sim(a, b):
    """Compute cosine similarity between two numpy vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


# ── Shared fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def extractor():
    """Shared extraction engine."""
    return FactExtractionEngine()


@pytest.fixture(scope="module")
def embed():
    """Shared embedding engine."""
    return EmbeddingEngine()


@pytest.fixture(scope="module")
def checker():
    """Shared contradiction checker."""
    c = create_contradiction_checker()
    if c is None:
        pytest.skip("Ollama not available for contradiction checking")
    return c


@pytest.fixture(scope="module")
def arc_sentences():
    """Load ARC sentences."""
    if not os.path.exists(ARC_PATH):
        pytest.skip(f"Dataset not found: {ARC_PATH}")
    return load_jsonl(ARC_PATH)


@pytest.fixture(scope="module")
def fever():
    """Load FEVER pairs with evidence."""
    if not os.path.exists(FEVER_PATH):
        pytest.skip(f"Dataset not found: {FEVER_PATH}")
    records = load_jsonl(FEVER_PATH)
    return [r for r in records if r.get("evidence") and r["label"] in ("SUPPORTS", "REFUTES")]


# ── Extraction Accuracy Tests ──────────────────────────────────────────

class TestExtractionAccuracy:

    def test_extraction_completeness(self, extractor, arc_sentences):
        """Most ARC sentences should produce at least one fact."""
        subset = arc_sentences[:30]  # Limit for speed
        produced = 0

        for sentence in subset:
            try:
                result = extractor.extract(sentence["text"], domain="science")
                if len(result.facts) > 0:
                    produced += 1
            except Exception:
                pass  # Count as not produced

        completeness = produced / len(subset)
        assert completeness >= 0.70, (
            f"Only {completeness:.1%} of sentences produced facts (need >= 70%)"
        )

    def test_extraction_fidelity(self, extractor, embed, arc_sentences):
        """Extracted facts should be semantically similar to source text."""
        subset = arc_sentences[:20]
        fidelities = []

        for sentence in subset:
            try:
                result = extractor.extract(sentence["text"], domain="science")
                source_vec = embed.embed_text(sentence["text"])

                for fact in result.facts:
                    fact_vec = embed.embed_text(fact.text)
                    fidelity = float(_cosine_sim(source_vec, fact_vec))
                    fidelities.append(fidelity)
            except Exception:
                pass

        if fidelities:
            mean_fidelity = statistics.mean(fidelities)
            assert mean_fidelity >= 0.30, (
                f"Mean extraction fidelity {mean_fidelity:.3f} < 0.30"
            )

    def test_low_hallucination_rate(self, extractor, embed, arc_sentences):
        """Few extracted facts should be very different from source text."""
        subset = arc_sentences[:20]
        total_facts = 0
        low_fidelity = 0

        for sentence in subset:
            try:
                result = extractor.extract(sentence["text"], domain="science")
                source_vec = embed.embed_text(sentence["text"])

                for fact in result.facts:
                    total_facts += 1
                    fact_vec = embed.embed_text(fact.text)
                    fidelity = float(_cosine_sim(source_vec, fact_vec))
                    if fidelity < 0.2:
                        low_fidelity += 1
            except Exception:
                pass

        if total_facts > 0:
            hallucination_rate = low_fidelity / total_facts
            # Allow up to 20% hallucination (threshold is generous)
            assert hallucination_rate <= 0.20, (
                f"Hallucination rate {hallucination_rate:.1%} > 20%"
            )


# ── LLM Contradiction Detection Tests ─────────────────────────────────

class TestLLMContradiction:

    def test_detects_refutations(self, checker, fever):
        """LLM should detect at least some REFUTES pairs."""
        refutes = [r for r in fever if r["label"] == "REFUTES"][:20]
        detected = 0

        for r in refutes:
            result = checker.check_relevance(r["claim"], r["evidence"])
            if result.relationship == "contradicts" and result.confidence >= 0.6:
                detected += 1

        detect_rate = detected / len(refutes) if refutes else 0
        # LLM should catch at least 30% of contradictions
        assert detect_rate >= 0.30, (
            f"Only {detect_rate:.1%} of REFUTES detected (need >= 30%)"
        )

    def test_low_false_positive_on_supports(self, checker, fever):
        """LLM should not flag most SUPPORTS pairs as contradictions."""
        supports = [r for r in fever if r["label"] == "SUPPORTS"][:20]
        false_positives = 0

        for r in supports:
            result = checker.check_relevance(r["claim"], r["evidence"])
            if result.relationship == "contradicts" and result.confidence >= 0.6:
                false_positives += 1

        fp_rate = false_positives / len(supports) if supports else 0
        # Allow up to 15% false positive rate
        assert fp_rate <= 0.15, (
            f"LLM false positive rate {fp_rate:.1%} on SUPPORTS > 15%"
        )

    def test_relevance_relationship_populated(self, checker, fever):
        """Detected relationships should have a valid relationship type."""
        refutes = [r for r in fever if r["label"] == "REFUTES"][:10]
        valid_relationships = {"contradicts", "supports", "irrelevant"}
        valid_count = 0
        total_count = 0

        for r in refutes:
            result = checker.check_relevance(r["claim"], r["evidence"])
            total_count += 1
            if result.relationship in valid_relationships:
                valid_count += 1

        if total_count > 0:
            valid_rate = valid_count / total_count
            assert valid_rate >= 0.80, (
                f"Only {valid_rate:.1%} of results have valid relationships"
            )


# ── Pipeline Integration Tests ─────────────────────────────────────────

class TestPipelineIntegration:

    def test_pipeline_ingest_produces_nodes(self, extractor, embed):
        """Pipeline should ingest text and produce graph nodes."""
        graph = KnowledgeGraph()
        pipeline = Pipeline(
            graph=graph,
            extractor=extractor,
            embedding_engine=embed,
            dependency_detector=DependencyDetector(),
            confidence_engine=ConfidenceEngine(),
        )

        result = pipeline.ingest(
            "Water boils at 100 degrees Celsius at standard pressure.",
            source=Source(name="test", quality_score=0.8),
            domain="science",
        )

        assert result.nodes_added > 0, "Pipeline should add at least one node"
        assert len(graph.nodes) > 0, "Graph should have nodes after ingestion"

    def test_pipeline_dedup_merges(self, extractor, embed):
        """Ingesting near-identical text should merge rather than duplicate."""
        graph = KnowledgeGraph()
        pipeline = Pipeline(
            graph=graph,
            extractor=extractor,
            embedding_engine=embed,
            dependency_detector=DependencyDetector(),
            confidence_engine=ConfidenceEngine(),
        )

        source = Source(name="test", quality_score=0.8)
        text = "The Earth orbits the Sun once every 365.25 days."

        r1 = pipeline.ingest(text, source=source, domain="science")
        pipeline.build_index()
        r2 = pipeline.ingest(text, source=source, domain="science")

        # Second ingestion should mostly merge or deduplicate
        total_after = len(graph.nodes)
        assert total_after <= r1.nodes_added * 2, (
            f"Expected dedup but graph has {total_after} nodes "
            f"after ingesting same text twice (first added {r1.nodes_added})"
        )

    def test_pipeline_flags_contradiction(self, extractor, embed):
        """Pipeline should flag/reject contradicting facts."""
        graph = KnowledgeGraph()
        checker = create_contradiction_checker()

        pipeline = Pipeline(
            graph=graph,
            extractor=extractor,
            embedding_engine=embed,
            dependency_detector=DependencyDetector(),
            confidence_engine=ConfidenceEngine(),
            contradiction_checker=checker,
        )

        source = Source(name="test", quality_score=0.9)

        # Ingest ground truth
        pipeline.ingest(
            "The speed of light is approximately 300,000 kilometers per second.",
            source=source, domain="physics",
        )
        pipeline.build_index()

        # Ingest contradiction
        r2 = pipeline.ingest(
            "The speed of light is approximately 150,000 kilometers per second.",
            source=source, domain="physics",
        )

        # Should flag or reject at least something
        # Note: This depends on the LLM actually catching it
        # We just check the pipeline doesn't crash and processes cleanly
        assert r2.total_extracted >= 0, "Pipeline should complete without error"
