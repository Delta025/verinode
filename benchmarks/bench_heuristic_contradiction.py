"""
Benchmark: Heuristic Contradiction Detection

Measures how well the negation-word heuristic in DependencyDetector
separates FEVER REFUTES claims from SUPPORTS claims.

Usage:
    python -m benchmarks.bench_heuristic_contradiction
"""

import os
import sys
import statistics
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dependency_detection import DependencyDetector
from src.core.embeddings import EmbeddingEngine


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0
from src.core.metrics import (
    load_jsonl, threshold_sweep, confusion_matrix,
    precision, recall, f1_score,
    BenchmarkResult, save_report,
)


DATA_PATH = os.path.join("data", "calibration", "fever_subset.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")


def run():
    print("=" * 60)
    print("Benchmark: Heuristic Contradiction Detection")
    print("=" * 60)

    # Load dataset — only pairs with evidence
    records = load_jsonl(DATA_PATH)
    pairs = [r for r in records if r.get("evidence") and r["label"] in ("SUPPORTS", "REFUTES")]
    print(f"Loaded {len(pairs)} FEVER pairs with evidence "
          f"({sum(1 for p in pairs if p['label'] == 'SUPPORTS')} SUPPORTS, "
          f"{sum(1 for p in pairs if p['label'] == 'REFUTES')} REFUTES)")

    # Initialize
    detector = DependencyDetector()
    print("Loading embedding model...")
    embed = EmbeddingEngine()

    # Compute heuristic contradiction scores
    scores = []
    actuals = []  # True = should be contradiction
    details = []

    print("Computing heuristic contradiction scores...")
    for i, pair in enumerate(pairs):
        claim = pair["claim"]
        evidence = pair["evidence"]
        is_refutes = pair["label"] == "REFUTES"

        # Get embedding similarity first (needed by heuristic scorer)
        vec_c = embed.embed_text(claim)
        vec_e = embed.embed_text(evidence)
        sim = float(_cosine_sim(vec_c, vec_e))

        # Use the heuristic scorer
        score = detector._contradiction_score(claim, evidence, sim)

        scores.append(score)
        actuals.append(is_refutes)
        details.append({
            "claim": claim[:80],
            "evidence": evidence[:80],
            "label": pair["label"],
            "heuristic_score": score,
            "similarity": sim,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs processed")

    # Statistics
    refute_scores = [s for s, a in zip(scores, actuals) if a]
    support_scores = [s for s, a in zip(scores, actuals) if not a]

    print("\n--- Heuristic Score Distribution ---")
    if refute_scores:
        print(f"REFUTES ({len(refute_scores)}): "
              f"mean={statistics.mean(refute_scores):.4f}, "
              f"median={statistics.median(refute_scores):.4f}")
    if support_scores:
        print(f"SUPPORTS ({len(support_scores)}): "
              f"mean={statistics.mean(support_scores):.4f}, "
              f"median={statistics.median(support_scores):.4f}")

    # Threshold sweep
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sweep = threshold_sweep(scores, actuals, thresholds)

    print("\n--- Threshold Sweep ---")
    print(f"{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    best = max(sweep, key=lambda r: r["f1"])
    for r in sweep:
        marker = " <-- best" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:>8.2f} {r['precision']:>8.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}{marker}")

    print(f"\nOptimal threshold: {best['threshold']:.2f} (F1={best['f1']:.3f})")
    print(f"Current defaults:  contradiction_detection_cutoff=0.5, "
          f"heuristic_contradiction_threshold=0.5")

    # Build result
    result = BenchmarkResult(
        name="heuristic_contradiction",
        metrics={
            "refutes_mean_score": statistics.mean(refute_scores) if refute_scores else 0,
            "supports_mean_score": statistics.mean(support_scores) if support_scores else 0,
            "optimal_threshold": best["threshold"],
            "optimal_f1": best["f1"],
            "optimal_precision": best["precision"],
            "optimal_recall": best["recall"],
            "total_pairs": len(pairs),
        },
        thresholds={
            "contradiction_detection_cutoff": 0.5,
            "heuristic_contradiction_threshold": 0.5,
        },
        details=details,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "heuristic_contradiction_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
