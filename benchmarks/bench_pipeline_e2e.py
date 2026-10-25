"""
Benchmark: End-to-End Pipeline

Measures full pipeline performance:
1. Seed graph with ARC science sentences
2. Ingest FEVER SUPPORTS pairs → measure dedup accuracy
3. Ingest FEVER REFUTES pairs → measure flagging accuracy
4. Per-stage timing

Requires Ollama with qwen2.5:7b (or configured model).

Usage:
    python -m benchmarks.bench_pipeline_e2e
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Use subsets for reasonable runtime
ARC_SEED_COUNT = 50
FEVER_SUPPORTS_COUNT = 25
FEVER_REFUTES_COUNT = 25


def run():
    print("=" * 60)
    print("Benchmark: End-to-End Pipeline")
    print("=" * 60)

    # Load datasets
    arc_sentences = load_jsonl(ARC_PATH)[:ARC_SEED_COUNT]
    fever_records = load_jsonl(FEVER_PATH)
    fever_supports = [r for r in fever_records
                      if r.get("evidence") and r["label"] == "SUPPORTS"][:FEVER_SUPPORTS_COUNT]
    fever_refutes = [r for r in fever_records
                     if r.get("evidence") and r["label"] == "REFUTES"][:FEVER_REFUTES_COUNT]

    print(f"ARC seed sentences: {len(arc_sentences)}")
    print(f"FEVER SUPPORTS: {len(fever_supports)}")
    print(f"FEVER REFUTES: {len(fever_refutes)}")

    # Initialize pipeline components
    print("\nInitializing pipeline components...")
    t0 = time.perf_counter()

    graph = KnowledgeGraph()
    extractor = FactExtractionEngine()
    embed = EmbeddingEngine()
    dep_detector = DependencyDetector()
    conf_engine = ConfidenceEngine()
    contradiction_checker = create_contradiction_checker()

    pipeline = Pipeline(
        graph=graph,
        extractor=extractor,
        embedding_engine=embed,
        dependency_detector=dep_detector,
        confidence_engine=conf_engine,
        contradiction_checker=contradiction_checker,
    )

    init_time = time.perf_counter() - t0
    print(f"Initialization: {init_time:.2f}s")
    if contradiction_checker:
        print("LLM contradiction checker: AVAILABLE")
    else:
        print("LLM contradiction checker: NOT AVAILABLE (heuristic only)")

    # Phase 1: Seed with ARC sentences
    print(f"\n--- Phase 1: Seeding with {len(arc_sentences)} ARC sentences ---")
    t1 = time.perf_counter()

    seed_facts_total = 0
    seed_errors = 0
    arc_source = Source(source_type="dataset", title="ARC-Easy", quality_score=0.85)

    for i, sentence in enumerate(arc_sentences):
        try:
            result = pipeline.ingest(sentence["text"], source=arc_source, domain="science")
            seed_facts_total += result.nodes_added
        except Exception as e:
            seed_errors += 1
            if (i + 1) <= 5:  # Only print first few errors
                print(f"  ERROR sentence {i + 1}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(arc_sentences)} seeded "
                  f"({seed_facts_total} facts, {seed_errors} errors)")

    seed_time = time.perf_counter() - t1
    print(f"Seed complete: {seed_facts_total} facts from {len(arc_sentences)} sentences "
          f"in {seed_time:.1f}s ({seed_errors} errors)")

    # Build index for dedup/contradiction detection
    pipeline.build_index()
    graph_size_after_seed = len(graph.nodes)
    print(f"Graph size after seed: {graph_size_after_seed} nodes")

    # Phase 2: Ingest SUPPORTS (should mostly merge or add compatible facts)
    print(f"\n--- Phase 2: Ingesting {len(fever_supports)} SUPPORTS claims ---")
    t2 = time.perf_counter()

    supports_added = 0
    supports_merged = 0
    supports_flagged = 0
    supports_rejected = 0
    supports_errors = 0

    fever_source = Source(source_type="dataset", title="FEVER", quality_score=0.75)

    for i, pair in enumerate(fever_supports):
        try:
            result = pipeline.ingest(pair["claim"], source=fever_source)
            supports_added += result.nodes_added
            supports_merged += result.duplicates_merged
            supports_flagged += result.facts_flagged
            supports_rejected += result.facts_rejected
        except Exception as e:
            supports_errors += 1
            if (i + 1) <= 3:
                print(f"  ERROR supports {i + 1}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(fever_supports)} processed")

    supports_time = time.perf_counter() - t2
    print(f"SUPPORTS complete in {supports_time:.1f}s:")
    print(f"  Added: {supports_added}, Merged: {supports_merged}, "
          f"Flagged: {supports_flagged}, Rejected: {supports_rejected}, "
          f"Errors: {supports_errors}")

    # Phase 3: Ingest REFUTES (should mostly be flagged/rejected)
    print(f"\n--- Phase 3: Ingesting {len(fever_refutes)} REFUTES claims ---")
    t3 = time.perf_counter()

    refutes_added = 0
    refutes_merged = 0
    refutes_flagged = 0
    refutes_rejected = 0
    refutes_errors = 0

    for i, pair in enumerate(fever_refutes):
        # Ingest the evidence first (ground truth), then the contradicting claim
        try:
            # First seed the evidence as a fact
            pipeline.ingest(pair["evidence"], source=fever_source, domain="general")
        except Exception:
            pass

        try:
            result = pipeline.ingest(pair["claim"], source=fever_source)
            refutes_added += result.nodes_added
            refutes_merged += result.duplicates_merged
            refutes_flagged += result.facts_flagged
            refutes_rejected += result.facts_rejected
        except Exception as e:
            refutes_errors += 1
            if (i + 1) <= 3:
                print(f"  ERROR refutes {i + 1}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(fever_refutes)} processed")

    refutes_time = time.perf_counter() - t3
    print(f"REFUTES complete in {refutes_time:.1f}s:")
    print(f"  Added: {refutes_added}, Merged: {refutes_merged}, "
          f"Flagged: {refutes_flagged}, Rejected: {refutes_rejected}, "
          f"Errors: {refutes_errors}")

    # Summary
    total_time = time.perf_counter() - t0
    graph_final_size = len(graph.nodes)

    print("\n--- Summary ---")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final graph size: {graph_final_size} nodes")
    print(f"Seed time: {seed_time:.1f}s ({seed_facts_total} facts)")
    print(f"SUPPORTS time: {supports_time:.1f}s")
    print(f"REFUTES time: {refutes_time:.1f}s")

    # Calculate flag rate for REFUTES (ideally high)
    refutes_total = refutes_flagged + refutes_rejected + refutes_added + refutes_merged
    refutes_flag_rate = (refutes_flagged + refutes_rejected) / refutes_total if refutes_total > 0 else 0

    # Calculate false flag rate for SUPPORTS (ideally low)
    supports_total = supports_flagged + supports_rejected + supports_added + supports_merged
    supports_false_flag_rate = (supports_flagged + supports_rejected) / supports_total if supports_total > 0 else 0

    print(f"\nREFUTES flag+reject rate: {refutes_flag_rate:.1%}")
    print(f"SUPPORTS false flag rate: {supports_false_flag_rate:.1%}")

    # Build result
    result = BenchmarkResult(
        name="pipeline_e2e",
        metrics={
            "total_time_s": total_time,
            "init_time_s": init_time,
            "seed_time_s": seed_time,
            "supports_time_s": supports_time,
            "refutes_time_s": refutes_time,
            "seed_facts": seed_facts_total,
            "seed_errors": seed_errors,
            "graph_after_seed": graph_size_after_seed,
            "graph_final_size": graph_final_size,
            "supports_added": supports_added,
            "supports_merged": supports_merged,
            "supports_flagged": supports_flagged,
            "supports_rejected": supports_rejected,
            "supports_false_flag_rate": supports_false_flag_rate,
            "refutes_added": refutes_added,
            "refutes_merged": refutes_merged,
            "refutes_flagged": refutes_flagged,
            "refutes_rejected": refutes_rejected,
            "refutes_flag_rate": refutes_flag_rate,
            "llm_available": contradiction_checker is not None,
        },
        thresholds={
            "duplicate_threshold": pipeline.duplicate_threshold,
            "contradiction_threshold": pipeline.contradiction_threshold,
            "min_extraction_confidence": pipeline.min_extraction_confidence,
            "source_quality_cap": pipeline.source_quality_cap,
            "llm_contradiction_confidence": 0.6,
        },
        details=[],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "pipeline_e2e_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
