#!/usr/bin/env python3
"""
Verification Stress Test — Step 11 Validation

Two modes:
  1. --load <path>  : Load a pre-built graph (e.g. data/stress_test_graph.json)
  2. (default)      : Build a minimal seed graph from scratch (~5-10 min)

Runs 10 false + 10 true claim verifications and measures accuracy.

Requires: Running Ollama server with qwen3:14b
Usage:
    python benchmarks/verify_stress_test.py --load data/stress_test_graph.json
    python benchmarks/verify_stress_test.py   # builds minimal seed graph

Success criteria (Step 11):
  - False claims caught: >= 7/10  (up from 0/10)
  - True claims OK:      10/10   (no regressions)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph import KnowledgeGraph
from src.core.node import Source
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import ReasoningEngine
from src.core.pipeline import Pipeline
from src.core.fact_extraction import FactExtractionEngine
from src.core.embeddings import EmbeddingEngine
from src.core.verification import VerificationEngine
from src.core.llm_contradiction import create_contradiction_checker
from src.core.llm_memory import LLMMemory


# ---------------------------------------------------------------
# Seed facts (used only in default mode)
# ---------------------------------------------------------------

SEED_FACTS = {
    "chemistry": [
        "Gold melts at 1064 degrees Celsius.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Iron has the chemical symbol Fe.",
        "Water has the chemical formula H2O.",
        "Gold has the chemical symbol Au on the periodic table.",
        "Gold is highly resistant to corrosion and oxidation.",
        "Pure gold has a density of 19.3 grams per cubic centimeter.",
        "Iron melts at 1538 degrees Celsius.",
        "Water freezes at 0 degrees Celsius.",
    ],
    "biology": [
        "DNA is a double helix composed of two complementary strands of nucleotides.",
        "The adult human body contains 206 bones.",
        "Mitochondria are the powerhouses of the cell, producing ATP through cellular respiration.",
        "The four DNA bases are adenine, thymine, guanine, and cytosine.",
        "Ribosomes are cellular organelles responsible for protein synthesis.",
        "The human brain contains approximately 86 billion neurons.",
    ],
    "physics": [
        "The speed of light in a vacuum is approximately 299792 kilometers per second.",
        "Absolute zero is minus 273.15 degrees Celsius or zero Kelvin.",
        "Newton's second law states that force equals mass times acceleration.",
        "The Moon orbits the Earth at an average distance of about 384400 kilometers.",
        "Jupiter is the largest planet in our solar system.",
        "Newton's first law states that an object at rest stays at rest unless acted upon by an external force.",
        "Momentum is the product of mass and velocity.",
        "The Earth orbits the Sun at an average distance of about 150 million kilometers.",
    ],
}

FALSE_CLAIMS = [
    ("Gold melts at 500 degrees Celsius.", "chemistry"),
    ("Water boils at 50 degrees Celsius at standard pressure.", "chemistry"),
    ("Iron has the chemical symbol Ir.", "chemistry"),
    ("DNA is a triple helix composed of three strands.", "biology"),
    ("The human body contains 150 bones.", "biology"),
    ("Mitochondria are responsible for protein synthesis.", "biology"),
    ("The speed of light is approximately 150000 kilometers per second.", "physics"),
    ("Absolute zero is minus 100 degrees Celsius.", "physics"),
    ("Newton's second law states that force equals mass times velocity.", "physics"),
    ("The Moon orbits the Earth at a distance of about 38000 kilometers.", "physics"),
]

TRUE_CLAIMS = [
    ("Gold melts at 1064 degrees Celsius.", "chemistry"),
    ("Water has the chemical formula H2O.", "chemistry"),
    ("Iron has the chemical symbol Fe.", "chemistry"),
    ("DNA is a double helix.", "biology"),
    ("The adult human body contains 206 bones.", "biology"),
    ("Mitochondria produce ATP through cellular respiration.", "biology"),
    ("The speed of light is approximately 300000 kilometers per second.", "physics"),
    ("Absolute zero is minus 273 degrees Celsius.", "physics"),
    ("Force equals mass times acceleration.", "physics"),
    ("Jupiter is the largest planet in our solar system.", "physics"),
]


def build_seed_graph():
    """Build a minimal graph from seed facts."""
    graph = KnowledgeGraph()
    confidence_engine = ConfidenceEngine()
    embedding_engine = EmbeddingEngine()

    try:
        extractor = FactExtractionEngine(model="qwen3:14b")
    except Exception as e:
        print(f"\nERROR: Could not initialize Ollama ({e})")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    pipeline = Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
    )

    total_facts = sum(len(facts) for facts in SEED_FACTS.values())
    print(f"  Ingesting {total_facts} seed facts...")

    for domain_name, facts in SEED_FACTS.items():
        source = Source(
            source_type="textbook",
            title=f"{domain_name.title()} Reference",
            quality_score=0.9,
        )
        for i, fact_text in enumerate(facts):
            t0 = time.time()
            result = pipeline.ingest(fact_text, source=source, domain=domain_name)
            elapsed = time.time() - t0
            print(f"    [{domain_name}] [{i+1}/{len(facts)}] "
                  f"+{result.nodes_added} ({elapsed:.1f}s) {fact_text[:50]}")

    pipeline.build_index()
    return graph, embedding_engine, pipeline


def load_saved_graph(path: str):
    """Load a pre-built graph from JSON."""
    graph = KnowledgeGraph.load(path)
    embedding_engine = EmbeddingEngine()

    # Need a pipeline just for build_index
    try:
        extractor = FactExtractionEngine(model="qwen3:14b")
    except Exception:
        extractor = None

    confidence_engine = ConfidenceEngine()
    pipeline = Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
    )

    # Rebuild embedding index from loaded nodes
    print(f"  Rebuilding embedding index for {len(graph)} nodes...")
    pipeline.build_index()

    return graph, embedding_engine, pipeline


def run_verification(graph, embedding_engine, pipeline):
    """Run the 10 false + 10 true claim verification test."""
    confidence_engine = ConfidenceEngine()
    reasoning_engine = ReasoningEngine()

    checker = create_contradiction_checker(model="qwen3:14b")
    if checker is None:
        print("  WARNING: Ollama not available for LLM verification")
        print("  Using heuristic-only mode")

    verification_engine = VerificationEngine(
        graph,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
        reasoning_engine=reasoning_engine,
        contradiction_checker=checker,
    )

    memory = LLMMemory(
        graph,
        embedding_engine=embedding_engine,
        reasoning_engine=reasoning_engine,
        confidence_engine=confidence_engine,
        pipeline=pipeline,
        verification_engine=verification_engine,
    )

    verify_start = time.time()

    # --- False Claims ---
    false_correct = 0
    false_results = []
    print("\n  --- False Claims (target: >= 7/10 CONTRADICTED) ---")
    for claim, domain in FALSE_CLAIMS:
        t0 = time.time()
        v = memory.verify(claim)
        elapsed = time.time() - t0
        is_caught = v["verdict"] == "contradicted"
        if is_caught:
            false_correct += 1
        tag = "\033[92mCAUGHT\033[0m" if is_caught else "\033[91mMISSED\033[0m"
        print(f"  [{tag}] {claim[:60]}")
        print(f"         verdict={v['verdict']} conf={v['confidence']:.2f} ({elapsed:.1f}s)")
        false_results.append({
            "claim": claim,
            "verdict": v["verdict"],
            "confidence": v["confidence"],
            "caught": is_caught,
        })

    # --- True Claims ---
    true_correct = 0
    true_results = []
    print("\n  --- True Claims (target: 10/10 not CONTRADICTED) ---")
    for claim, domain in TRUE_CLAIMS:
        t0 = time.time()
        v = memory.verify(claim)
        elapsed = time.time() - t0
        is_ok = v["verdict"] in ("supported", "uncertain", "insufficient_evidence")
        if is_ok:
            true_correct += 1
        tag = "\033[92mOK\033[0m" if is_ok else "\033[91mFALSE_POS\033[0m"
        print(f"  [{tag}] {claim[:60]}")
        print(f"         verdict={v['verdict']} conf={v['confidence']:.2f} ({elapsed:.1f}s)")
        true_results.append({
            "claim": claim,
            "verdict": v["verdict"],
            "confidence": v["confidence"],
            "ok": is_ok,
        })

    verify_time = time.time() - verify_start

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("VERIFICATION RESULTS")
    print(f"{'=' * 70}")

    false_pct = false_correct / len(FALSE_CLAIMS) * 100
    true_pct = true_correct / len(TRUE_CLAIMS) * 100
    false_pass = false_correct >= 7
    true_pass = true_correct == 10

    status_false = "\033[92mPASS\033[0m" if false_pass else "\033[91mFAIL\033[0m"
    status_true = "\033[92mPASS\033[0m" if true_pass else "\033[91mFAIL\033[0m"

    print(f"  False claims caught: {false_correct}/{len(FALSE_CLAIMS)} ({false_pct:.0f}%)"
          f"  {status_false} (target >= 7/10)")
    print(f"  True claims OK:      {true_correct}/{len(TRUE_CLAIMS)} ({true_pct:.0f}%)"
          f"  {status_true} (target 10/10)")
    print(f"  False positive rate:  {(len(TRUE_CLAIMS)-true_correct)/len(TRUE_CLAIMS)*100:.0f}%")
    print(f"  Time: {verify_time:.1f}s ({verify_time/20:.1f}s/claim avg)")

    if false_correct < 7:
        print(f"\n  MISSED false claims:")
        for r in false_results:
            if not r["caught"]:
                print(f"    - {r['claim'][:60]} (verdict={r['verdict']})")

    if true_correct < 10:
        print(f"\n  FALSE POSITIVE true claims:")
        for r in true_results:
            if not r["ok"]:
                print(f"    - {r['claim'][:60]} (verdict={r['verdict']})")

    overall = false_pass and true_pass
    overall_str = "\033[92mPASS\033[0m" if overall else "\033[91mFAIL\033[0m"
    print(f"\n  OVERALL: {overall_str}")
    print(f"{'=' * 70}")

    return 0 if overall else 1


def main():
    parser = argparse.ArgumentParser(description="Step 11 Verification Stress Test")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to pre-built graph JSON (e.g. data/stress_test_graph.json)")
    args = parser.parse_args()

    print("=" * 70)
    print("Step 11 — Verification Stress Test")
    print("Testing 3-way relevance, weighted evidence, verdict tiebreaker")
    print("=" * 70)

    start = time.time()

    if args.load:
        print(f"\n  Loading graph from {args.load}...")
        graph, embedding_engine, pipeline = load_saved_graph(args.load)
        print(f"  Loaded {len(graph)} nodes.")
    else:
        print(f"\n  Building seed graph from scratch...")
        graph, embedding_engine, pipeline = build_seed_graph()
        print(f"\n  Built graph with {len(graph)} nodes.")

    print(f"\n{'=' * 70}")
    print("CLAIM VERIFICATION (10 false + 10 true)")
    print(f"{'=' * 70}")

    exit_code = run_verification(graph, embedding_engine, pipeline)

    total_time = time.time() - start
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
