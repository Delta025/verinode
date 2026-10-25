"""
Benchmark: Dependency Classification Accuracy

Measures how well DependencyDetector._classify_relationship() classifies
SciTail ENTAILS pairs as SUPPORTS/DEPENDS_ON and NEUTRAL pairs as None.

Usage:
    python -m benchmarks.bench_dependency_classification
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.node import KnowledgeNode, NodeType
from src.core.dependency_detection import DependencyDetector, RelationType
from src.core.embeddings import EmbeddingEngine


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0
from src.core.metrics import (
    load_jsonl, confusion_matrix, classification_report,
    precision, recall, f1_score,
    BenchmarkResult, save_report,
)


DATA_PATH = os.path.join("data", "calibration", "scitail_subset.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")


def run():
    print("=" * 60)
    print("Benchmark: Dependency Classification")
    print("=" * 60)

    # Load dataset
    pairs = load_jsonl(DATA_PATH)
    print(f"Loaded {len(pairs)} SciTail pairs")

    # Initialize
    detector = DependencyDetector()
    print("Loading embedding model...")
    embed = EmbeddingEngine()

    # Classify each pair
    predictions = []  # "related" or "none"
    actuals = []      # "related" or "none"
    relation_types = []  # Detailed relation type for related pairs
    details = []

    print("Classifying relationships...")
    for i, pair in enumerate(pairs):
        premise = pair["premise"]
        hypothesis = pair["hypothesis"]
        expected_related = pair["label"] == "entails"

        # Create temporary nodes
        node_a = KnowledgeNode(text=premise, node_type=NodeType.FACT, confidence=0.8, domain="science")
        node_b = KnowledgeNode(text=hypothesis, node_type=NodeType.FACT, confidence=0.8, domain="science")

        # Get embedding similarity
        vec_a = embed.embed_text(premise)
        vec_b = embed.embed_text(hypothesis)
        sim = float(_cosine_sim(vec_a, vec_b))

        # Classify
        result = detector._classify_relationship(node_a, node_b, sim)

        if result is not None:
            predicted = "related"
            rel_name = result.relation.name
            confidence = result.confidence
        else:
            predicted = "none"
            rel_name = "none"
            confidence = 0.0

        actual = "related" if expected_related else "none"
        predictions.append(predicted)
        actuals.append(actual)
        relation_types.append(rel_name)

        details.append({
            "premise": premise[:80],
            "hypothesis": hypothesis[:80],
            "expected": pair["label"],
            "predicted_relation": rel_name,
            "confidence": confidence,
            "similarity": sim,
            "correct": predicted == actual,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs processed")

    # Classification report (binary: related vs none)
    report = classification_report(predictions, actuals, labels=["related", "none"])
    print("\n--- Binary Classification (related vs none) ---")
    print(f"Accuracy: {report['accuracy']:.3f}")
    for label in ["related", "none"]:
        c = report["per_class"][label]
        print(f"  {label:>8}: P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} (n={c['support']})")
    print(f"  Macro F1: {report['macro_avg']['f1']:.3f}")

    # Relation type breakdown for entails pairs
    entails_rels = [rt for rt, pair in zip(relation_types, pairs) if pair["label"] == "entails"]
    rel_counts = {}
    for rt in entails_rels:
        rel_counts[rt] = rel_counts.get(rt, 0) + 1

    print("\n--- Relation Types for ENTAILS pairs ---")
    for rt, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rt}: {count} ({count / len(entails_rels) * 100:.1f}%)")

    # Confusion matrix
    cm = confusion_matrix(
        [p == "related" for p in predictions],
        [a == "related" for a in actuals],
    )
    print(f"\n--- Confusion Matrix ---")
    print(f"  TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")

    # Build result
    result = BenchmarkResult(
        name="dependency_classification",
        metrics={
            "accuracy": report["accuracy"],
            "related_precision": report["per_class"]["related"]["precision"],
            "related_recall": report["per_class"]["related"]["recall"],
            "related_f1": report["per_class"]["related"]["f1"],
            "none_precision": report["per_class"]["none"]["precision"],
            "none_recall": report["per_class"]["none"]["recall"],
            "none_f1": report["per_class"]["none"]["f1"],
            "macro_f1": report["macro_avg"]["f1"],
            "relation_type_distribution": rel_counts,
            "total_pairs": len(pairs),
            **cm,
        },
        thresholds={
            "similarity_threshold": 0.35,
            "dependency_confidence_threshold": 0.4,
            "domain_similarity_support": 0.25,
        },
        details=details,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "dependency_classification_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
