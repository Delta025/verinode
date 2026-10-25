"""
Benchmark: LLM Contradiction Detection

Measures ContradictionChecker performance on FEVER pairs:
- Precision/recall/F1 for REFUTES detection
- Breakdown by contradiction type
- False positive rate on SUPPORTS pairs
- Confidence distribution

Requires Ollama with qwen2.5:7b (or configured model).

Usage:
    python -m benchmarks.bench_llm_contradiction
"""

import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_contradiction import ContradictionChecker
from src.core.metrics import (
    load_jsonl, confusion_matrix,
    precision, recall, f1_score,
    threshold_sweep,
    BenchmarkResult, save_report,
)


DATA_PATH = os.path.join("data", "calibration", "fever_subset.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")


def run():
    print("=" * 60)
    print("Benchmark: LLM Contradiction Detection")
    print("=" * 60)

    # Load dataset — only pairs with evidence
    records = load_jsonl(DATA_PATH)
    pairs = [r for r in records if r.get("evidence") and r["label"] in ("SUPPORTS", "REFUTES")]
    supports = [p for p in pairs if p["label"] == "SUPPORTS"]
    refutes = [p for p in pairs if p["label"] == "REFUTES"]
    print(f"Loaded {len(pairs)} FEVER pairs with evidence "
          f"({len(supports)} SUPPORTS, {len(refutes)} REFUTES)")

    # Initialize
    print("Loading ContradictionChecker (Ollama)...")
    checker = ContradictionChecker()

    # Check each pair
    scores = []        # LLM confidence (for threshold sweep)
    actuals = []       # True = should be contradiction
    type_counts = {}   # Breakdown by contradiction type
    details = []

    print("Checking contradiction pairs...")
    for i, pair in enumerate(pairs):
        claim = pair["claim"]
        evidence = pair["evidence"]
        is_refutes = pair["label"] == "REFUTES"

        try:
            result = checker.check_pair(claim, evidence)
            score = result.confidence if result.is_contradiction else 0.0
            ctype = result.contradiction_type
        except Exception as e:
            print(f"  ERROR on pair {i + 1}: {e}")
            score = 0.0
            ctype = "error"
            result = None

        scores.append(score)
        actuals.append(is_refutes)

        if result and result.is_contradiction:
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

        details.append({
            "claim": claim[:80],
            "evidence": evidence[:80],
            "label": pair["label"],
            "llm_is_contradiction": result.is_contradiction if result else False,
            "llm_confidence": result.confidence if result else 0.0,
            "llm_type": ctype,
            "llm_explanation": result.explanation[:100] if result else "",
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs processed")

    # Score distribution
    refute_scores = [s for s, a in zip(scores, actuals) if a]
    support_scores = [s for s, a in zip(scores, actuals) if not a]

    print("\n--- LLM Contradiction Score Distribution ---")
    if refute_scores:
        print(f"REFUTES ({len(refute_scores)}): "
              f"mean={statistics.mean(refute_scores):.4f}, "
              f"median={statistics.median(refute_scores):.4f}")
    if support_scores:
        print(f"SUPPORTS ({len(support_scores)}): "
              f"mean={statistics.mean(support_scores):.4f}, "
              f"median={statistics.median(support_scores):.4f}")

    # Threshold sweep
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = threshold_sweep(scores, actuals, thresholds)

    print("\n--- Threshold Sweep ---")
    print(f"{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
          f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    best = max(sweep, key=lambda r: r["f1"])
    for r in sweep:
        marker = " <-- best" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:>8.2f} {r['precision']:>8.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}{marker}")

    print(f"\nOptimal threshold: {best['threshold']:.2f} (F1={best['f1']:.3f})")
    print(f"Current default:   llm_contradiction_confidence=0.6")

    # Contradiction type breakdown
    if type_counts:
        print("\n--- Contradiction Types Detected ---")
        for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {ctype}: {count}")

    # Confusion matrix at default threshold (0.6)
    at_06 = [r for r in sweep if r["threshold"] == 0.6]
    if at_06:
        r = at_06[0]
        print(f"\n--- At default threshold 0.6 ---")
        print(f"  Precision: {r['precision']:.3f}")
        print(f"  Recall: {r['recall']:.3f}")
        print(f"  F1: {r['f1']:.3f}")

    # Build result
    result = BenchmarkResult(
        name="llm_contradiction",
        metrics={
            "refutes_mean_score": statistics.mean(refute_scores) if refute_scores else 0,
            "supports_mean_score": statistics.mean(support_scores) if support_scores else 0,
            "optimal_threshold": best["threshold"],
            "optimal_f1": best["f1"],
            "optimal_precision": best["precision"],
            "optimal_recall": best["recall"],
            "type_distribution": type_counts,
            "total_pairs": len(pairs),
        },
        thresholds={
            "llm_contradiction_confidence": 0.6,
        },
        details=details,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "llm_contradiction_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
