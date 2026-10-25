"""
Benchmark: Backend Comparison (Ollama vs Gemini)

Runs the FEVER contradiction detection benchmark against one or both
LLM backends, producing per-backend reports and a side-by-side
comparison table.

Measures precision/recall/F1, score distributions, contradiction type
breakdown, and per-call latency on 150 FEVER claim/evidence pairs.

Usage:
    python -m benchmarks.bench_backend_comparison
    python -m benchmarks.bench_backend_comparison --backend gemini
    python -m benchmarks.bench_backend_comparison --backend gemini --limit 10
    python -m benchmarks.bench_backend_comparison --gemini-delay 30  # free tier
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_contradiction import create_contradiction_checker
from src.core.metrics import (
    load_jsonl,
    precision, recall, f1_score,
    threshold_sweep,
    BenchmarkResult, save_report,
)


DATA_PATH = os.path.join("data", "calibration", "fever_subset.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare contradiction detection across LLM backends"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "gemini", "both"],
        default="both",
        help="Which backend(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--gemini-delay",
        type=float,
        default=6.0,
        help="Seconds between Gemini API calls (default: 6.0 for 10 RPM paid tier)",
    )
    parser.add_argument(
        "--ollama-delay",
        type=float,
        default=0.0,
        help="Seconds between Ollama calls (default: 0.0, local has no rate limit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max pairs to process, 0 = all (default: 0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backend creation
# ---------------------------------------------------------------------------

def try_create_checker(backend_name: str):
    """
    Try to create a contradiction checker for *backend_name*.

    Returns (checker, model_name) on success, or (None, error_string)
    on failure.
    """
    try:
        checker = create_contradiction_checker(backend=backend_name)
        if checker is None:
            return None, f"{backend_name} backend not available"
        model_name = getattr(checker, "model", None) or "unknown"
        return checker, model_name
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run_single_backend(backend_name, checker, pairs, inter_call_delay=0.0):
    """
    Run all *pairs* through *checker.check_pair()* sequentially.

    Returns a dict with keys: scores, actuals, type_counts, details,
    latencies, total_time, errors.
    """
    scores = []
    actuals = []
    type_counts = {}
    details = []
    latencies = []
    errors = 0

    t_total = time.perf_counter()

    for i, pair in enumerate(pairs):
        claim = pair["claim"]
        evidence = pair["evidence"]
        is_refutes = pair["label"] == "REFUTES"

        t0 = time.perf_counter()
        try:
            result = checker.check_pair(claim, evidence)
            score = result.confidence if result.is_contradiction else 0.0
            ctype = result.contradiction_type
        except Exception as e:
            print(f"  [{backend_name}] ERROR pair {i + 1}: {e}")
            score = 0.0
            ctype = "error"
            result = None
            errors += 1
        latency = time.perf_counter() - t0

        scores.append(score)
        actuals.append(is_refutes)
        latencies.append(latency)

        if result and result.is_contradiction:
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

        details.append({
            "claim": claim[:80],
            "evidence": evidence[:80],
            "label": pair["label"],
            "is_contradiction": result.is_contradiction if result else False,
            "confidence": result.confidence if result else 0.0,
            "type": ctype,
            "explanation": (result.explanation[:100] if result else ""),
            "latency_s": round(latency, 3),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total
            print(f"  [{backend_name}] {i + 1}/{len(pairs)} processed "
                  f"({elapsed:.0f}s elapsed)")

        # Rate-limit sleep (skip after last pair)
        if inter_call_delay > 0 and i < len(pairs) - 1:
            time.sleep(inter_call_delay)

    total_time = time.perf_counter() - t_total

    return {
        "scores": scores,
        "actuals": actuals,
        "type_counts": type_counts,
        "details": details,
        "latencies": latencies,
        "total_time": total_time,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(raw):
    """Derive threshold sweep, distributions, latency stats from raw results."""
    scores = raw["scores"]
    actuals = raw["actuals"]
    latencies = raw["latencies"]

    refute_scores = [s for s, a in zip(scores, actuals) if a]
    support_scores = [s for s, a in zip(scores, actuals) if not a]

    sweep = threshold_sweep(scores, actuals, THRESHOLDS)
    best = max(sweep, key=lambda r: r["f1"])
    at_06 = next((r for r in sweep if r["threshold"] == 0.6), None)

    return {
        "sweep": sweep,
        "best": best,
        "at_06": at_06,
        "refute_scores": refute_scores,
        "support_scores": support_scores,
        "refute_mean": statistics.mean(refute_scores) if refute_scores else 0,
        "refute_median": statistics.median(refute_scores) if refute_scores else 0,
        "support_mean": statistics.mean(support_scores) if support_scores else 0,
        "support_median": statistics.median(support_scores) if support_scores else 0,
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "median_latency": statistics.median(latencies) if latencies else 0,
    }


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def print_single_results(backend_name, model_name, raw, metrics):
    """Print per-backend results (same format as bench_llm_contradiction)."""
    scores = raw["scores"]
    actuals = raw["actuals"]

    print(f"\n--- {backend_name} ({model_name}) Score Distribution ---")
    print(f"REFUTES  ({len(metrics['refute_scores'])}): "
          f"mean={metrics['refute_mean']:.4f}, "
          f"median={metrics['refute_median']:.4f}")
    print(f"SUPPORTS ({len(metrics['support_scores'])}): "
          f"mean={metrics['support_mean']:.4f}, "
          f"median={metrics['support_median']:.4f}")

    # Threshold sweep
    sweep = metrics["sweep"]
    best = metrics["best"]

    print(f"\n--- {backend_name} Threshold Sweep ---")
    print(f"{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
          f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    for r in sweep:
        marker = " <-- best" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:>8.2f} {r['precision']:>8.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>8.3f} "
              f"{r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}"
              f"{marker}")

    print(f"\nOptimal threshold: {best['threshold']:.2f} "
          f"(F1={best['f1']:.3f})")

    at_06 = metrics["at_06"]
    if at_06:
        print(f"At default 0.6:    P={at_06['precision']:.3f} "
              f"R={at_06['recall']:.3f} F1={at_06['f1']:.3f}")

    # Type breakdown
    if raw["type_counts"]:
        print(f"\n--- {backend_name} Contradiction Types ---")
        for ctype, count in sorted(raw["type_counts"].items(),
                                   key=lambda x: -x[1]):
            print(f"  {ctype}: {count}")

    # Latency
    print(f"\n--- {backend_name} Latency ---")
    print(f"  Mean:  {metrics['mean_latency']:.2f}s")
    print(f"  Median: {metrics['median_latency']:.2f}s")
    print(f"  Total: {raw['total_time']:.1f}s "
          f"({raw['total_time'] / 60:.1f} min)")
    if raw["errors"]:
        print(f"  Errors: {raw['errors']}")


def _fmt_delta(a, b, fmt=".3f", higher_better=True):
    """Format a delta value with +/- sign."""
    delta = b - a
    sign = "+" if delta > 0 else ""
    good = (delta > 0) == higher_better
    indicator = " *" if abs(delta) > 0.01 and good else ""
    return f"{sign}{delta:{fmt}}{indicator}"


def print_comparison_table(o_model, o_raw, o_met, g_model, g_raw, g_met):
    """Print side-by-side comparison of Ollama vs Gemini results."""
    W = 16  # column width

    print(f"\n{'Metric':<28} {'Ollama':>{W}} {'Gemini':>{W}} {'Delta':>{W}}")
    print("-" * (28 + W * 3 + 3))

    # At threshold 0.6
    o6 = o_met["at_06"]
    g6 = g_met["at_06"]
    if o6 and g6:
        print(f"\n  --- At Default Threshold (0.6) ---")
        for label, key, hb in [("Precision", "precision", True),
                                ("Recall", "recall", True),
                                ("F1", "f1", True)]:
            print(f"  {label:<26} {o6[key]:>{W}.3f} {g6[key]:>{W}.3f} "
                  f"{_fmt_delta(o6[key], g6[key]):>{W}}")
        for label, key in [("TP", "tp"), ("FP", "fp"),
                           ("TN", "tn"), ("FN", "fn")]:
            print(f"  {label:<26} {o6[key]:>{W}} {g6[key]:>{W}} "
                  f"{g6[key] - o6[key]:>{W}}")

    # Optimal threshold
    ob = o_met["best"]
    gb = g_met["best"]
    print(f"\n  --- Optimal Threshold ---")
    print(f"  {'Threshold':<26} {ob['threshold']:>{W}.2f} "
          f"{gb['threshold']:>{W}.2f} "
          f"{_fmt_delta(ob['threshold'], gb['threshold'], '.2f'):>{W}}")
    print(f"  {'Best F1':<26} {ob['f1']:>{W}.3f} {gb['f1']:>{W}.3f} "
          f"{_fmt_delta(ob['f1'], gb['f1']):>{W}}")
    print(f"  {'Precision @ Best':<26} {ob['precision']:>{W}.3f} "
          f"{gb['precision']:>{W}.3f} "
          f"{_fmt_delta(ob['precision'], gb['precision']):>{W}}")
    print(f"  {'Recall @ Best':<26} {ob['recall']:>{W}.3f} "
          f"{gb['recall']:>{W}.3f} "
          f"{_fmt_delta(ob['recall'], gb['recall']):>{W}}")

    # Score distributions
    print(f"\n  --- Score Distributions ---")
    for label, ok, gk in [
        ("REFUTES mean", "refute_mean", "refute_mean"),
        ("REFUTES median", "refute_median", "refute_median"),
        ("SUPPORTS mean", "support_mean", "support_mean"),
        ("SUPPORTS median", "support_median", "support_median"),
    ]:
        print(f"  {label:<26} {o_met[ok]:>{W}.4f} {g_met[gk]:>{W}.4f} "
              f"{_fmt_delta(o_met[ok], g_met[gk], '.4f'):>{W}}")

    # Contradiction types
    all_types = sorted(
        set(o_raw["type_counts"].keys()) | set(g_raw["type_counts"].keys())
    )
    if all_types:
        print(f"\n  --- Contradiction Types ---")
        for ctype in all_types:
            oc = o_raw["type_counts"].get(ctype, 0)
            gc = g_raw["type_counts"].get(ctype, 0)
            print(f"  {ctype:<26} {oc:>{W}} {gc:>{W}} "
                  f"{gc - oc:>{W}}")

    # Latency
    print(f"\n  --- Latency ---")
    print(f"  {'Mean (s)':<26} {o_met['mean_latency']:>{W}.2f} "
          f"{g_met['mean_latency']:>{W}.2f} "
          f"{_fmt_delta(o_met['mean_latency'], g_met['mean_latency'], '.2f', False):>{W}}")
    print(f"  {'Median (s)':<26} {o_met['median_latency']:>{W}.2f} "
          f"{g_met['median_latency']:>{W}.2f} "
          f"{_fmt_delta(o_met['median_latency'], g_met['median_latency'], '.2f', False):>{W}}")
    print(f"  {'Total (min)':<26} {o_raw['total_time'] / 60:>{W}.1f} "
          f"{g_raw['total_time'] / 60:>{W}.1f} "
          f"{_fmt_delta(o_raw['total_time'] / 60, g_raw['total_time'] / 60, '.1f', False):>{W}}")
    print(f"  {'Errors':<26} {o_raw['errors']:>{W}} "
          f"{g_raw['errors']:>{W}} "
          f"{g_raw['errors'] - o_raw['errors']:>{W}}")


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def build_backend_result(backend_name, model_name, raw, metrics):
    """Build a BenchmarkResult for a single backend."""
    at_06 = metrics["at_06"] or {}
    best = metrics["best"]

    return BenchmarkResult(
        name=f"llm_contradiction_{backend_name}",
        metrics={
            "backend": backend_name,
            "model": model_name,
            "total_pairs": len(raw["scores"]),
            "refutes_mean_score": metrics["refute_mean"],
            "refutes_median_score": metrics["refute_median"],
            "supports_mean_score": metrics["support_mean"],
            "supports_median_score": metrics["support_median"],
            "optimal_threshold": best["threshold"],
            "optimal_f1": best["f1"],
            "optimal_precision": best["precision"],
            "optimal_recall": best["recall"],
            "at_06_precision": at_06.get("precision", 0),
            "at_06_recall": at_06.get("recall", 0),
            "at_06_f1": at_06.get("f1", 0),
            "type_distribution": raw["type_counts"],
            "mean_latency_s": round(metrics["mean_latency"], 3),
            "median_latency_s": round(metrics["median_latency"], 3),
            "total_runtime_s": round(raw["total_time"], 1),
            "errors": raw["errors"],
        },
        thresholds={"llm_contradiction_confidence": 0.6},
        details=raw["details"],
    )


def build_comparison_result(o_met, g_met, o_raw, g_raw, total_pairs):
    """Build a combined comparison BenchmarkResult."""
    o6 = o_met["at_06"] or {}
    g6 = g_met["at_06"] or {}

    return BenchmarkResult(
        name="backend_comparison",
        metrics={
            "dataset": {
                "total_pairs": total_pairs,
                "supports": sum(1 for a in o_raw["actuals"] if not a),
                "refutes": sum(1 for a in o_raw["actuals"] if a),
            },
            "ollama": {
                "f1_at_06": o6.get("f1", 0),
                "precision_at_06": o6.get("precision", 0),
                "recall_at_06": o6.get("recall", 0),
                "optimal_f1": o_met["best"]["f1"],
                "optimal_threshold": o_met["best"]["threshold"],
                "mean_latency_s": round(o_met["mean_latency"], 3),
                "total_runtime_s": round(o_raw["total_time"], 1),
            },
            "gemini": {
                "f1_at_06": g6.get("f1", 0),
                "precision_at_06": g6.get("precision", 0),
                "recall_at_06": g6.get("recall", 0),
                "optimal_f1": g_met["best"]["f1"],
                "optimal_threshold": g_met["best"]["threshold"],
                "mean_latency_s": round(g_met["mean_latency"], 3),
                "total_runtime_s": round(g_raw["total_time"], 1),
            },
            "delta": {
                "f1_at_06": round(
                    g6.get("f1", 0) - o6.get("f1", 0), 4),
                "precision_at_06": round(
                    g6.get("precision", 0) - o6.get("precision", 0), 4),
                "recall_at_06": round(
                    g6.get("recall", 0) - o6.get("recall", 0), 4),
                "optimal_f1": round(
                    g_met["best"]["f1"] - o_met["best"]["f1"], 4),
                "mean_latency_s": round(
                    g_met["mean_latency"] - o_met["mean_latency"], 3),
                "total_runtime_s": round(
                    g_raw["total_time"] - o_raw["total_time"], 1),
            },
        },
        thresholds={"llm_contradiction_confidence": 0.6},
        details=[],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    args = parse_args()

    print("=" * 60)
    print("Benchmark: Backend Comparison")
    print("=" * 60)

    # Load dataset
    records = load_jsonl(DATA_PATH)
    pairs = [r for r in records
             if r.get("evidence") and r["label"] in ("SUPPORTS", "REFUTES")]
    if args.limit > 0:
        pairs = pairs[:args.limit]

    supports = [p for p in pairs if p["label"] == "SUPPORTS"]
    refutes = [p for p in pairs if p["label"] == "REFUTES"]
    print(f"Loaded {len(pairs)} FEVER pairs "
          f"({len(supports)} SUPPORTS, {len(refutes)} REFUTES)")

    backends_to_run = (
        ["ollama", "gemini"] if args.backend == "both"
        else [args.backend]
    )

    results = {}  # backend_name -> (model_name, raw, metrics)

    for backend_name in backends_to_run:
        checker, info = try_create_checker(backend_name)
        if checker is None:
            print(f"\nWARNING: {backend_name} unavailable -- {info}")
            print(f"Skipping {backend_name} benchmark.\n")
            continue

        model_name = info
        delay = (args.gemini_delay if backend_name == "gemini"
                 else args.ollama_delay)

        print(f"\n{'=' * 60}")
        print(f"Running: {backend_name} ({model_name})")
        if delay > 0:
            print(f"Inter-call delay: {delay}s")
        # Rough time estimate
        per_call = delay + (25.0 if backend_name == "ollama" else 1.5)
        est_min = len(pairs) * per_call / 60
        print(f"Estimated time: ~{est_min:.0f} min for {len(pairs)} pairs")
        print("=" * 60)

        raw = run_single_backend(backend_name, checker, pairs, delay)
        met = compute_metrics(raw)

        print_single_results(backend_name, model_name, raw, met)

        # Save individual report
        report = build_backend_result(backend_name, model_name, raw, met)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR,
                            f"{backend_name}_contradiction_report.json")
        save_report(report, path)
        print(f"\nReport saved to: {path}")

        results[backend_name] = (model_name, raw, met)

    # Side-by-side comparison
    if "ollama" in results and "gemini" in results:
        o_model, o_raw, o_met = results["ollama"]
        g_model, g_raw, g_met = results["gemini"]

        print(f"\n{'=' * 60}")
        print(f"SIDE-BY-SIDE COMPARISON")
        print(f"Ollama ({o_model}) vs Gemini ({g_model})")
        print(f"{'=' * 60}")

        print_comparison_table(o_model, o_raw, o_met, g_model, g_raw, g_met)

        comp = build_comparison_result(
            o_met, g_met, o_raw, g_raw, len(pairs))
        comp_path = os.path.join(RESULTS_DIR,
                                 "backend_comparison_report.json")
        save_report(comp, comp_path)
        print(f"\nComparison report saved to: {comp_path}")

    elif len(results) == 0:
        print("\nERROR: No backends available. Cannot run benchmark.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    run()
