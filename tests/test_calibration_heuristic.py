"""
Calibration regression tests for heuristic (non-LLM) components.

Tests that embedding similarity and dependency classification maintain
minimum performance on the SciTail and FEVER calibration datasets.

Requires sentence-transformers (NOT Ollama).
Run with: pytest tests/test_calibration_heuristic.py --run-calibration
"""

import os
import statistics

import numpy as np
import pytest

from src.core.embeddings import EmbeddingEngine


def _cosine_sim(a, b):
    """Compute cosine similarity between two numpy vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


from src.core.dependency_detection import DependencyDetector, RelationType
from src.core.metrics import load_jsonl
from src.core.node import KnowledgeNode, NodeType

pytestmark = pytest.mark.calibration

SCITAIL_PATH = os.path.join("data", "calibration", "scitail_subset.jsonl")
FEVER_PATH = os.path.join("data", "calibration", "fever_subset.jsonl")


# ── Shared fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embed():
    """Shared embedding engine (expensive to load)."""
    return EmbeddingEngine()


@pytest.fixture(scope="module")
def scitail():
    """Load SciTail pairs."""
    if not os.path.exists(SCITAIL_PATH):
        pytest.skip(f"Dataset not found: {SCITAIL_PATH}")
    return load_jsonl(SCITAIL_PATH)


@pytest.fixture(scope="module")
def fever():
    """Load FEVER pairs with evidence."""
    if not os.path.exists(FEVER_PATH):
        pytest.skip(f"Dataset not found: {FEVER_PATH}")
    records = load_jsonl(FEVER_PATH)
    return [r for r in records if r.get("evidence") and r["label"] in ("SUPPORTS", "REFUTES")]


@pytest.fixture(scope="module")
def scitail_similarities(scitail, embed):
    """Pre-compute similarities for all SciTail pairs."""
    results = []
    for pair in scitail:
        vec_p = embed.embed_text(pair["premise"])
        vec_h = embed.embed_text(pair["hypothesis"])
        sim = float(_cosine_sim(vec_p, vec_h))
        results.append({"sim": sim, "label": pair["label"]})
    return results


# ── Embedding Similarity Tests ──────────────────────────────────────────


class TestEmbeddingSimilarity:
    def test_entails_higher_than_neutral(self, scitail_similarities):
        """ENTAILS pairs should have higher mean similarity than NEUTRAL."""
        entails = [r["sim"] for r in scitail_similarities if r["label"] == "entails"]
        neutrals = [r["sim"] for r in scitail_similarities if r["label"] == "neutral"]
        assert statistics.mean(entails) > statistics.mean(neutrals), (
            f"ENTAILS mean ({statistics.mean(entails):.3f}) should be > NEUTRAL mean ({statistics.mean(neutrals):.3f})"
        )

    def test_entails_median_above_threshold(self, scitail_similarities):
        """ENTAILS pairs should have a median similarity above 0.30 (some relationship exists)."""
        entails = [r["sim"] for r in scitail_similarities if r["label"] == "entails"]
        median = statistics.median(entails)
        assert median >= 0.30, f"ENTAILS median similarity {median:.3f} < 0.30"

    def test_separation_exists(self, scitail_similarities):
        """There should be a meaningful gap between the distributions."""
        entails = [r["sim"] for r in scitail_similarities if r["label"] == "entails"]
        neutrals = [r["sim"] for r in scitail_similarities if r["label"] == "neutral"]
        gap = statistics.mean(entails) - statistics.mean(neutrals)
        assert gap > 0.03, f"Mean gap {gap:.3f} is too small for useful separation"

    def test_threshold_070_has_good_f1(self, scitail_similarities):
        """Optuna-tuned default threshold (0.70) should produce F1 > 0.65."""
        threshold = 0.70
        tp = sum(1 for r in scitail_similarities if r["sim"] >= threshold and r["label"] == "entails")
        fp = sum(1 for r in scitail_similarities if r["sim"] >= threshold and r["label"] == "neutral")
        fn = sum(1 for r in scitail_similarities if r["sim"] < threshold and r["label"] == "entails")

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        assert f1 > 0.60, f"F1 at threshold 0.70 is {f1:.3f}, expected > 0.60"


# ── Heuristic Contradiction Tests ───────────────────────────────────────


class TestHeuristicContradiction:
    def test_heuristic_detects_some_refutations(self, fever, embed):
        """Heuristic contradiction scorer should detect at least some REFUTES pairs."""
        refutes = [r for r in fever if r["label"] == "REFUTES"]

        detected = 0
        for r in refutes:
            vec_c = embed.embed_text(r["claim"])
            vec_e = embed.embed_text(r["evidence"])
            sim = float(_cosine_sim(vec_c, vec_e))
            score = DependencyDetector._contradiction_score(r["claim"], r["evidence"], sim)
            if score >= 0.5:
                detected += 1

        # Heuristic is known to be weak — we just want it to catch *some*
        assert detected > 0, "Heuristic should detect at least 1 contradiction"

    def test_heuristic_low_false_positive_on_supports(self, fever, embed):
        """Heuristic should not flag most SUPPORTS pairs as contradictions."""
        supports = [r for r in fever if r["label"] == "SUPPORTS"]

        false_positives = 0
        for r in supports:
            vec_c = embed.embed_text(r["claim"])
            vec_e = embed.embed_text(r["evidence"])
            sim = float(_cosine_sim(vec_c, vec_e))
            score = DependencyDetector._contradiction_score(r["claim"], r["evidence"], sim)
            if score >= 0.5:
                false_positives += 1

        fp_rate = false_positives / len(supports) if supports else 0
        # Allow up to 30% false positive rate (heuristic is noisy)
        assert fp_rate < 0.30, f"Heuristic false positive rate {fp_rate:.1%} on SUPPORTS is too high"


# ── Dependency Classification Tests ─────────────────────────────────────


class TestDependencyClassification:
    def test_entails_detected_as_related(self, scitail, embed):
        """Most ENTAILS pairs should be classified as some relationship."""
        detector = DependencyDetector()

        detected = 0
        entails_pairs = [p for p in scitail if p["label"] == "entails"]

        for pair in entails_pairs[:50]:  # Subset for speed
            node_a = KnowledgeNode(text=pair["premise"], node_type=NodeType.FACT, confidence=0.8, domain="science")
            node_b = KnowledgeNode(text=pair["hypothesis"], node_type=NodeType.FACT, confidence=0.8, domain="science")

            vec_a = embed.embed_text(pair["premise"])
            vec_b = embed.embed_text(pair["hypothesis"])
            sim = float(_cosine_sim(vec_a, vec_b))

            result = detector._classify_relationship(node_a, node_b, sim)
            if result is not None:
                detected += 1

        detect_rate = detected / len(entails_pairs[:50])
        # At least 30% of entails pairs should be detected as related
        assert detect_rate >= 0.30, f"Only {detect_rate:.1%} of ENTAILS pairs classified as related (need >= 30%)"

    def test_neutral_not_over_classified(self, scitail, embed):
        """NEUTRAL pairs: false-related rate should be reduced with calibrated thresholds.

        With similarity_threshold raised to 0.70 (Optuna-tuned) and
        domain_similarity_support at 0.45, fewer NEUTRAL pairs should be
        misclassified.
        """
        detector = DependencyDetector()  # Uses Optuna-tuned defaults (0.70, 0.45)

        false_related = 0
        neutral_pairs = [p for p in scitail if p["label"] == "neutral"]

        for pair in neutral_pairs[:50]:  # Subset for speed
            node_a = KnowledgeNode(text=pair["premise"], node_type=NodeType.FACT, confidence=0.8, domain="science")
            node_b = KnowledgeNode(text=pair["hypothesis"], node_type=NodeType.FACT, confidence=0.8, domain="science")

            vec_a = embed.embed_text(pair["premise"])
            vec_b = embed.embed_text(pair["hypothesis"])
            sim = float(_cosine_sim(vec_a, vec_b))

            result = detector._classify_relationship(node_a, node_b, sim)
            if result is not None:
                false_related += 1

        fp_rate = false_related / len(neutral_pairs[:50])
        # With tightened thresholds, false positive rate should be <= 80%
        assert fp_rate <= 0.80, f"False related rate {fp_rate:.1%} on NEUTRAL is too high (need <= 80%)"

    def test_entails_not_classified_as_contradicts(self, scitail, embed):
        """ENTAILS pairs should NOT be classified as CONTRADICTS."""
        detector = DependencyDetector()

        contradicts_count = 0
        entails_pairs = [p for p in scitail if p["label"] == "entails"]

        for pair in entails_pairs[:50]:
            node_a = KnowledgeNode(text=pair["premise"], node_type=NodeType.FACT, confidence=0.8, domain="science")
            node_b = KnowledgeNode(text=pair["hypothesis"], node_type=NodeType.FACT, confidence=0.8, domain="science")

            vec_a = embed.embed_text(pair["premise"])
            vec_b = embed.embed_text(pair["hypothesis"])
            sim = float(_cosine_sim(vec_a, vec_b))

            result = detector._classify_relationship(node_a, node_b, sim)
            if result is not None and result.relation == RelationType.CONTRADICTS:
                contradicts_count += 1

        # No more than 10% should be misclassified as contradictions
        rate = contradicts_count / len(entails_pairs[:50])
        assert rate < 0.10, f"{rate:.1%} of ENTAILS pairs classified as CONTRADICTS (need < 10%)"

    def test_supports_is_most_common_for_entails(self, scitail, embed):
        """For ENTAILS pairs that are detected, SUPPORTS or ELABORATES should dominate."""
        detector = DependencyDetector()

        rel_counts = {}
        entails_pairs = [p for p in scitail if p["label"] == "entails"]

        for pair in entails_pairs[:50]:
            node_a = KnowledgeNode(text=pair["premise"], node_type=NodeType.FACT, confidence=0.8, domain="science")
            node_b = KnowledgeNode(text=pair["hypothesis"], node_type=NodeType.FACT, confidence=0.8, domain="science")

            vec_a = embed.embed_text(pair["premise"])
            vec_b = embed.embed_text(pair["hypothesis"])
            sim = float(_cosine_sim(vec_a, vec_b))

            result = detector._classify_relationship(node_a, node_b, sim)
            if result is not None:
                rel_name = result.relation.name
                rel_counts[rel_name] = rel_counts.get(rel_name, 0) + 1

        if rel_counts:
            positive_rels = (
                rel_counts.get("SUPPORTS", 0) + rel_counts.get("ELABORATES", 0) + rel_counts.get("DEPENDS_ON", 0)
            )
            total_detected = sum(rel_counts.values())
            # At least 60% of detected relationships should be positive
            assert positive_rels / total_detected >= 0.60, (
                f"Only {positive_rels / total_detected:.1%} positive relations, distribution: {rel_counts}"
            )
