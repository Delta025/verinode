"""
Benchmark: Optuna Bayesian Optimization of Pipeline Thresholds

Jointly optimizes the Pipeline's interacting thresholds using
Bayesian optimization (TPE sampler). Each trial:
  1. Creates a fresh KnowledgeGraph
  2. Seeds with ARC science sentences (trusted source)
  3. Ingests FEVER SUPPORTS claims (should merge/add, not flag)
  4. Ingests FEVER REFUTES claims (should flag/reject)
  5. Score = refutes_flag_rate - supports_false_flag_rate

Higher is better: we want to catch contradictions without
false-flagging correct information.

Requires Ollama with qwen2.5:7b (or configured model).

Usage:
    python benchmarks/bench_optuna_tuning.py
    python benchmarks/bench_optuna_tuning.py --trials 50
"""

import argparse
import json
import os
import sys
import time
import logging

# Force unbuffered output for monitoring progress
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna

from src.core.graph import KnowledgeGraph
from src.core.node import Source
from src.core.fact_extraction import FactExtractionEngine
from src.core.embeddings import EmbeddingEngine
from src.core.dependency_detection import DependencyDetector
from src.core.confidence import ConfidenceEngine
from src.core.llm_contradiction import create_contradiction_checker
from src.core.pipeline import Pipeline
from src.core.metrics import load_jsonl, BenchmarkResult, save_report


ARC_PATH = os.path.join("data", "calibration", "arc_sentences.jsonl")
FEVER_PATH = os.path.join("data", "calibration", "fever_subset.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")

# Subset sizes — keep small for reasonable per-trial time
# 10/5/5 gives ~15min/trial; 20/10/10 gives ~45min/trial
ARC_SEED_COUNT = 10
FEVER_SUPPORTS_COUNT = 5
FEVER_REFUTES_COUNT = 5

# Suppress noisy logs during optimization
logging.getLogger("src.core").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    """Load and prepare datasets (once, shared across trials)."""
    arc = load_jsonl(ARC_PATH)[:ARC_SEED_COUNT]
    fever = load_jsonl(FEVER_PATH)
    supports = [r for r in fever if r.get("evidence") and r["label"] == "SUPPORTS"][:FEVER_SUPPORTS_COUNT]
    refutes = [r for r in fever if r.get("evidence") and r["label"] == "REFUTES"][:FEVER_REFUTES_COUNT]
    return arc, supports, refutes


# Shared heavy objects (created once, reused across trials)
_extractor = None
_checker = None


def get_shared_components():
    """Lazily initialize shared LLM components (expensive to create)."""
    global _extractor, _checker
    if _extractor is None:
        print("Loading shared LLM components (one-time)...")
        _extractor = FactExtractionEngine()
        _checker = create_contradiction_checker()
        if _checker:
            print("  LLM contradiction checker: AVAILABLE")
        else:
            print("  LLM contradiction checker: NOT AVAILABLE")
    return _extractor, _checker


def objective(trial, arc_data, supports_data, refutes_data):
    """Optuna objective function. Returns score to maximize."""

    # Sample thresholds
    duplicate_threshold = trial.suggest_float("duplicate_threshold", 0.85, 0.96)
    contradiction_threshold = trial.suggest_float("contradiction_threshold", 0.5, 0.9)
    min_extraction_confidence = trial.suggest_float("min_extraction_confidence", 0.2, 0.5)
    source_quality_cap = trial.suggest_float("source_quality_cap", 0.3, 0.7)
    similarity_threshold = trial.suggest_float("similarity_threshold", 0.45, 0.80)

    extractor, checker = get_shared_components()

    # Fresh graph and components per trial
    graph = KnowledgeGraph()
    embed = EmbeddingEngine()
    dep_detector = DependencyDetector(similarity_threshold=similarity_threshold)
    conf_engine = ConfidenceEngine()

    pipeline = Pipeline(
        graph=graph,
        extractor=extractor,
        embedding_engine=embed,
        dependency_detector=dep_detector,
        confidence_engine=conf_engine,
        contradiction_checker=checker,
        duplicate_threshold=duplicate_threshold,
        contradiction_threshold=contradiction_threshold,
        min_extraction_confidence=min_extraction_confidence,
        source_quality_cap=source_quality_cap,
    )

    # Phase 1: Seed with ARC
    arc_source = Source(source_type="dataset", title="ARC-Easy", quality_score=0.85)
    for sentence in arc_data:
        try:
            pipeline.ingest(sentence["text"], source=arc_source, domain="science")
        except Exception:
            pass

    pipeline.build_index()

    # Phase 2: Ingest SUPPORTS (should NOT be flagged)
    supports_flagged = 0
    supports_total = 0
    fever_source = Source(source_type="dataset", title="FEVER", quality_score=0.75)

    for pair in supports_data:
        try:
            result = pipeline.ingest(pair["claim"], source=fever_source)
            supports_total += result.nodes_added + result.duplicates_merged + result.facts_flagged + result.facts_rejected
            supports_flagged += result.facts_flagged + result.facts_rejected
        except Exception:
            pass

    # Phase 3: Ingest REFUTES (SHOULD be flagged)
    refutes_flagged = 0
    refutes_total = 0

    for pair in refutes_data:
        # Seed the evidence first
        try:
            pipeline.ingest(pair["evidence"], source=fever_source, domain="general")
        except Exception:
            pass

        try:
            result = pipeline.ingest(pair["claim"], source=fever_source)
            refutes_total += result.nodes_added + result.duplicates_merged + result.facts_flagged + result.facts_rejected
            refutes_flagged += result.facts_flagged + result.facts_rejected
        except Exception:
            pass

    # Calculate rates
    refutes_flag_rate = refutes_flagged / refutes_total if refutes_total > 0 else 0.0
    supports_false_flag_rate = supports_flagged / supports_total if supports_total > 0 else 0.0

    # Score: maximize detection of contradictions, minimize false flags
    score = refutes_flag_rate - supports_false_flag_rate

    # Report intermediate values for monitoring
    trial.set_user_attr("refutes_flag_rate", refutes_flag_rate)
    trial.set_user_attr("supports_false_flag_rate", supports_false_flag_rate)
    trial.set_user_attr("refutes_total", refutes_total)
    trial.set_user_attr("supports_total", supports_total)

    return score


def run(n_trials: int = 30):
    print("=" * 60, flush=True)
    print("Benchmark: Optuna Pipeline Threshold Optimization", flush=True)
    print(f"Trials: {n_trials}", flush=True)
    print("=" * 60, flush=True)

    # Load data once
    arc, supports, refutes = load_data()
    print(f"Data loaded: {len(arc)} ARC, {len(supports)} SUPPORTS, {len(refutes)} REFUTES", flush=True)

    # Warm up shared components
    get_shared_components()
    print("Components ready, starting optimization...", flush=True)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="verinode_pipeline_thresholds",
    )

    t0 = time.perf_counter()

    for i in range(n_trials):
        print(f"  Starting trial {i + 1}/{n_trials}...", flush=True)
        trial_start = time.perf_counter()
        study.optimize(
            lambda trial: objective(trial, arc, supports, refutes),
            n_trials=1,
            show_progress_bar=False,
        )
        trial_time = time.perf_counter() - trial_start
        best = study.best_trial
        print(
            f"  Trial {i + 1}/{n_trials} "
            f"[{trial_time:.1f}s] "
            f"score={study.trials[-1].value:.3f} "
            f"best={best.value:.3f} "
            f"(refutes_flag={best.user_attrs.get('refutes_flag_rate', 0):.1%}, "
            f"supports_fp={best.user_attrs.get('supports_false_flag_rate', 0):.1%})",
            flush=True,
        )

    total_time = time.perf_counter() - t0

    # Results
    best = study.best_trial
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time / n_trials:.1f}s/trial)")
    print(f"Best score: {best.value:.3f}")
    print(f"Best parameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value:.4f}")
    print(f"Best metrics:")
    print(f"  REFUTES flag rate: {best.user_attrs.get('refutes_flag_rate', 0):.1%}")
    print(f"  SUPPORTS false flag rate: {best.user_attrs.get('supports_false_flag_rate', 0):.1%}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save as BenchmarkResult
    result = BenchmarkResult(
        name="optuna_pipeline_tuning",
        metrics={
            "best_score": best.value,
            "best_refutes_flag_rate": best.user_attrs.get("refutes_flag_rate", 0),
            "best_supports_false_flag_rate": best.user_attrs.get("supports_false_flag_rate", 0),
            "n_trials": n_trials,
            "total_time_s": total_time,
            "avg_trial_time_s": total_time / n_trials,
        },
        thresholds=best.params,
        details=[
            {
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "refutes_flag_rate": t.user_attrs.get("refutes_flag_rate", 0),
                "supports_false_flag_rate": t.user_attrs.get("supports_false_flag_rate", 0),
            }
            for t in study.trials
        ],
    )

    report_path = os.path.join(RESULTS_DIR, "optuna_study.json")
    save_report(result, report_path)
    print(f"\nFull study saved to: {report_path}")

    # Also save a compact summary for easy reference
    summary = {
        "best_params": best.params,
        "best_score": best.value,
        "best_refutes_flag_rate": best.user_attrs.get("refutes_flag_rate", 0),
        "best_supports_false_flag_rate": best.user_attrs.get("supports_false_flag_rate", 0),
    }
    summary_path = os.path.join(RESULTS_DIR, "optuna_best_params.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best params saved to: {summary_path}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna pipeline threshold tuning")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()
    run(n_trials=args.trials)
