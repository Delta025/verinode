"""
Benchmark: Embedding Similarity Calibration

Measures cosine similarity distribution for SciTail ENTAILS vs NEUTRAL pairs.
Finds the optimal similarity_threshold for separating related from unrelated pairs.

Usage:
    python -m benchmarks.bench_embeddings
"""

import os
import sys
import statistics
import numpy as np

# Allow running as `python -m benchmarks.bench_embeddings`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embeddings import EmbeddingEngine


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0
from src.core.metrics import (
    load_jsonl, threshold_sweep, BenchmarkResult, save_report,
)


DATA_PATH = os.path.join("data", "calibration", "scitail_subset.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")


def run():
    print("=" * 60)
    print("Benchmark: Embedding Similarity Calibration")
    print("=" * 60)

    # Load dataset
    pairs = load_jsonl(DATA_PATH)
    print(f"Loaded {len(pairs)} SciTail pairs")

    # Initialize embedding engine
    print("Loading embedding model...")
    embed = EmbeddingEngine()

    # Compute cosine similarities
    entails_sims = []
    neutral_sims = []
    all_scores = []
    all_actuals = []

    print("Computing similarities...")
    for i, pair in enumerate(pairs):
        vec_p = embed.embed_text(pair["premise"])
        vec_h = embed.embed_text(pair["hypothesis"])
        sim = float(_cosine_sim(vec_p, vec_h))

        is_entails = pair["label"] == "entails"
        all_scores.append(sim)
        all_actuals.append(is_entails)

        if is_entails:
            entails_sims.append(sim)
        else:
            neutral_sims.append(sim)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs processed")

    # Statistics
    print("\n--- Similarity Distribution ---")
    print(f"ENTAILS ({len(entails_sims)} pairs):")
    print(f"  mean={statistics.mean(entails_sims):.4f}, "
          f"median={statistics.median(entails_sims):.4f}, "
          f"stdev={statistics.stdev(entails_sims):.4f}, "
          f"min={min(entails_sims):.4f}, max={max(entails_sims):.4f}")

    print(f"NEUTRAL ({len(neutral_sims)} pairs):")
    print(f"  mean={statistics.mean(neutral_sims):.4f}, "
          f"median={statistics.median(neutral_sims):.4f}, "
          f"stdev={statistics.stdev(neutral_sims):.4f}, "
          f"min={min(neutral_sims):.4f}, max={max(neutral_sims):.4f}")

    # Threshold sweep
    thresholds = [round(t / 100, 2) for t in range(10, 90, 5)]
    sweep = threshold_sweep(all_scores, all_actuals, thresholds)

    print("\n--- Threshold Sweep ---")
    print(f"{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    best = max(sweep, key=lambda r: r["f1"])
    for r in sweep:
        marker = " <-- best" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:>8.2f} {r['precision']:>8.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}{marker}")

    print(f"\nOptimal threshold: {best['threshold']:.2f} (F1={best['f1']:.3f})")
    print(f"Current default:   0.35")

    # Build result
    result = BenchmarkResult(
        name="embedding_similarity",
        metrics={
            "entails_mean_sim": statistics.mean(entails_sims),
            "entails_median_sim": statistics.median(entails_sims),
            "neutral_mean_sim": statistics.mean(neutral_sims),
            "neutral_median_sim": statistics.median(neutral_sims),
            "optimal_threshold": best["threshold"],
            "optimal_f1": best["f1"],
            "optimal_precision": best["precision"],
            "optimal_recall": best["recall"],
            "current_threshold": 0.35,
        },
        thresholds={"similarity_threshold": 0.35},
        details=sweep,
    )

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "embedding_similarity_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
