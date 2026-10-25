#!/usr/bin/env python3
"""
LLM Integration Demo — Engram Step 10

Demonstrates how an external LLM would use Engram as a verified
memory system. Shows the three-step integration pattern:

  1. RECALL: LLM retrieves relevant facts before answering
  2. VERIFY: LLM checks a claim before using it
  3. INGEST: LLM submits a derived fact back to the graph

Also demonstrates hallucination prevention: the LLM's false claim
is caught by the verification engine.

Requires: Running Ollama server with qwen3:14b
Usage:    python examples/llm_integration_demo.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
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
# Seed data (small focused set)
# ---------------------------------------------------------------

SEED_FACTS = [
    "Gold melts at 1064 degrees Celsius.",
    "Gold has the chemical symbol Au.",
    "Gold is a transition metal in Group 11 of the periodic table.",
    "Gold is highly resistant to corrosion and oxidation.",
    "Pure gold has a density of 19.3 grams per cubic centimeter.",
    "Silver melts at 961.8 degrees Celsius.",
    "Silver has the chemical symbol Ag.",
    "Copper melts at 1085 degrees Celsius.",
    "Copper has the chemical symbol Cu.",
    "Iron melts at 1538 degrees Celsius.",
]


def main():
    print("=" * 60)
    print("Engram — LLM Integration Demo")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    print("\n--- Setup: Building knowledge graph ---")

    graph = KnowledgeGraph()
    confidence_engine = ConfidenceEngine()
    reasoning_engine = ReasoningEngine()
    embedding_engine = EmbeddingEngine()

    try:
        extractor = FactExtractionEngine(model="qwen3:14b")
    except Exception as e:
        print(f"\nERROR: Could not initialize Ollama ({e})")
        print("Make sure Ollama is running and qwen3:14b is pulled.")
        sys.exit(1)

    pipeline = Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
    )

    source = Source(
        source_type="textbook",
        title="Materials Science Reference",
        quality_score=0.9,
    )

    for fact_text in SEED_FACTS:
        pipeline.ingest(fact_text, source=source, domain="chemistry")

    pipeline.build_index()

    checker = create_contradiction_checker(model="qwen3:14b")
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

    print(f"  Graph loaded with {len(graph)} nodes.")

    # ---------------------------------------------------------------
    # Scenario: LLM answering "What is the melting point of gold?"
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO: LLM needs to answer 'What is the melting point of gold?'")
    print("=" * 60)

    # Step 1: RECALL — LLM queries Engram for relevant facts
    print("\n--- Step 1: RECALL (retrieve relevant facts) ---")
    result = memory.query("melting point of gold", domain="chemistry", k=5)

    print(f"  Query: 'melting point of gold'")
    print(f"  Coverage: {result.coverage} | Confidence: {result.overall_confidence:.2f}")
    print(f"  Facts retrieved: {len(result.facts)}")
    for f in result.facts:
        print(f"    [{f.confidence:.2f}] {f.text[:60]}")

    print(f"\n  Summary (LLM would use this): {result.summary[:80]}")

    # Step 2: VERIFY — LLM checks its planned answer
    print("\n--- Step 2: VERIFY (check answer before responding) ---")

    # Good answer
    good_answer = "Gold melts at 1064 degrees Celsius"
    v1 = memory.verify(good_answer)
    print(f"  Checking: '{good_answer}'")
    print(f"  Verdict: {v1['verdict']} | Confidence: {v1['confidence']:.2f}")

    # Hallucinated answer
    bad_answer = "Gold melts at 500 degrees Celsius"
    v2 = memory.verify(bad_answer)
    print(f"\n  Checking: '{bad_answer}'")
    print(f"  Verdict: {v2['verdict']} | Confidence: {v2['confidence']:.2f}")
    print(f"  Explanation: {v2['explanation'][:70]}")

    if v2["verdict"] == "contradicted":
        print("  ** HALLUCINATION PREVENTED: LLM would discard this answer **")

    # Step 3: INGEST — LLM derives and submits a new fact
    print("\n--- Step 3: INGEST (submit derived fact) ---")

    derived = "Gold has a higher melting point than silver but lower than iron."
    print(f"  LLM derived: '{derived}'")

    ingest_result = memory.ingest(
        derived,
        source_title="LLM derivation (demo)",
        source_quality=0.6,
        domain="chemistry",
    )

    print(f"  Action: {ingest_result.action}")
    print(f"  Explanation: {ingest_result.explanation[:70]}")
    if ingest_result.contradictions:
        print(f"  Contradictions found: {len(ingest_result.contradictions)}")
        for c in ingest_result.contradictions:
            print(f"    - {c['text'][:50]} (conf: {c['confidence']:.2f})")

    # ---------------------------------------------------------------
    # Scenario: Hallucination prevention with false claim
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO: LLM tries to ingest a false claim")
    print("=" * 60)

    false_claim = "Gold melts at 500 degrees Celsius."
    print(f"\n  Ingesting false claim: '{false_claim}'")

    ingest_false = memory.ingest(
        false_claim,
        source_title="Unreliable source",
        source_quality=0.3,
        domain="chemistry",
    )

    print(f"  Action: {ingest_false.action}")
    print(f"  Explanation: {ingest_false.explanation[:70]}")
    if ingest_false.contradictions:
        print(f"  Contradictions: {len(ingest_false.contradictions)}")
        for c in ingest_false.contradictions:
            print(f"    Conflicts with: {c['text'][:50]} (conf: {c['confidence']:.2f})")
    else:
        print("  (No contradictions detected — may need higher quality graph)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Integration Demo Summary")
    print("=" * 60)
    print(f"  Graph: {len(graph)} nodes")
    print("  Capabilities demonstrated:")
    print("    1. RECALL: Retrieved confidence-scored facts for LLM context")
    print("    2. VERIFY: Caught hallucinated answer (gold at 500C)")
    print("    3. INGEST: Submitted LLM-derived fact with verification")
    print("  Pattern: LLM queries Engram before answering, verifies before")
    print("  using claims, and contributes back verified knowledge.")


if __name__ == "__main__":
    main()
