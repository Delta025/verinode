#!/usr/bin/env python3
"""
Build a large stress-test graph (150 facts across 3 domains) and save it.

Ingests 50 chemistry, 50 physics, and 50 biology facts through the full
Pipeline (Ollama fact extraction + contradiction detection + embeddings).

Output: data/stress_test_graph.json

Requires: Running Ollama server with qwen3:14b
Usage:    python benchmarks/build_stress_graph.py

Recommended Ollama env vars:
    set OLLAMA_FLASH_ATTENTION=1
    set OLLAMA_KV_CACHE_TYPE=q8_0
    set OLLAMA_NUM_PARALLEL=4
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph import KnowledgeGraph
from src.core.node import Source
from src.core.confidence import ConfidenceEngine
from src.core.pipeline import Pipeline
from src.core.fact_extraction import FactExtractionEngine
from src.core.embeddings import EmbeddingEngine

# Import the fact datasets from stress_test.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_test import CHEMISTRY_FACTS, PHYSICS_FACTS, BIOLOGY_FACTS


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stress_test_graph.json"


def main():
    print("=" * 70)
    print("Building Stress Test Graph (150 facts, 3 domains)")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)

    graph = KnowledgeGraph()
    confidence_engine = ConfidenceEngine()
    embedding_engine = EmbeddingEngine()

    try:
        extractor = FactExtractionEngine(model="qwen3:14b")
    except Exception as e:
        print(f"\nERROR: Could not initialize Ollama ({e})")
        print("Make sure Ollama is running: ollama serve")
        print("And the model is pulled: ollama pull qwen3:14b")
        sys.exit(1)

    pipeline = Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
    )
    print("Pipeline ready.\n")

    domains = [
        ("chemistry", CHEMISTRY_FACTS),
        ("physics", PHYSICS_FACTS),
        ("biology", BIOLOGY_FACTS),
    ]

    total_added = 0
    total_dup = 0
    ingestion_start = time.time()

    for domain_name, facts in domains:
        print(f"--- {domain_name} ({len(facts)} facts) ---")
        source = Source(
            source_type="textbook",
            title=f"{domain_name.title()} Reference",
            quality_score=0.9,
        )

        domain_start = time.time()
        domain_added = 0
        domain_dup = 0

        for i, fact_text in enumerate(facts):
            t0 = time.time()
            result = pipeline.ingest(fact_text, source=source, domain=domain_name)
            elapsed = time.time() - t0

            domain_added += result.nodes_added
            domain_dup += result.duplicates_merged
            total_added += result.nodes_added
            total_dup += result.duplicates_merged

            if (i + 1) % 10 == 0:
                domain_elapsed = time.time() - domain_start
                print(f"  {i+1}/{len(facts)} ingested "
                      f"({domain_elapsed:.1f}s elapsed, "
                      f"last={elapsed:.1f}s, "
                      f"+{domain_added} nodes, "
                      f"{domain_dup} dups)")

        domain_elapsed = time.time() - domain_start
        print(f"  Done: {domain_added} added, {domain_dup} dups "
              f"in {domain_elapsed:.1f}s ({domain_elapsed/len(facts):.1f}s/fact)\n")

    ingestion_time = time.time() - ingestion_start

    # Build embedding index
    print("Building embedding index...")
    pipeline.build_index()

    # Save graph
    print(f"Saving graph to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(OUTPUT_PATH))

    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"  Total nodes: {len(graph)}")
    print(f"  Added: {total_added}, Duplicates: {total_dup}")
    print(f"  Time: {ingestion_time:.1f}s ({ingestion_time/150:.1f}s/fact avg)")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
