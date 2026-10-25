"""
End-to-end pipeline demo with live Ollama extraction.

This example demonstrates the full ingestion pipeline:
1. Extracting facts from natural-language text via Ollama
2. Source quality gating (low-quality sources produce PROVISIONAL nodes)
3. Deduplication via convergent evidence (noisy-OR boost)
4. Contradiction detection and flagging
5. Dependency detection and confidence propagation
6. Querying and reasoning over the resulting graph

Prerequisites:
    1. Ollama installed and running: ollama serve
    2. A model pulled: ollama pull qwen2.5:7b
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import (
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    NodeStatus,
    Source,
    ConfidenceEngine,
    EmbeddingEngine,
    FactExtractionEngine,
    DependencyDetector,
    ReasoningEngine,
    Pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_verdict(v) -> None:
    """Pretty-print a single IngestionVerdict."""
    icon = {
        "added": "+",
        "duplicate": "~",
        "flagged": "!",
        "rejected": "X",
    }.get(v.action, "?")

    print(f"  [{icon}] {v.action.upper()}: \"{v.fact.text[:70]}...\"" if len(v.fact.text) > 70
          else f"  [{icon}] {v.action.upper()}: \"{v.fact.text}\"")

    if v.action == "duplicate":
        print(f"      Confidence: {v.confidence_before:.3f} -> {v.confidence_after:.3f}")
    if v.reason:
        print(f"      Reason: {v.reason}")


def print_result(result, label: str = "Ingestion") -> None:
    """Pretty-print an IngestionResult."""
    print(f"\n--- {label} ---")
    print(f"  Model: {result.extraction_model}")
    print(f"  Summary: {result.summary}")
    for v in result.verdicts:
        print_verdict(v)
    print()


def print_graph_state(graph) -> None:
    """Show current graph statistics."""
    stats = graph.get_statistics()
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
    print(f"  Domains: {dict(stats['nodes_by_domain'])}")
    print(f"  Statuses: {dict(stats['nodes_by_status'])}")
    print()


# ---------------------------------------------------------------------------
# Demo Steps
# ---------------------------------------------------------------------------

def step1_seed_with_vetted_data(pipeline) -> None:
    """
    Step 1: Seed the graph with high-quality, vetted information.

    This is the 'trust bootstrapping' phase. We use a high-quality
    source so these facts become the foundation that later facts are
    checked against.
    """
    banner("STEP 1: Seed the graph with vetted data")

    textbook = Source(
        source_type="textbook",
        title="General Chemistry, 10th Edition",
        author="Ebbing & Gammon",
        quality_score=0.95,
    )

    passages = [
        "Water is a chemical compound with the formula H2O. Each molecule "
        "consists of two hydrogen atoms covalently bonded to one oxygen atom. "
        "Water is essential for all known forms of life.",

        "At standard atmospheric pressure (1 atm), pure water boils at 100 "
        "degrees Celsius (212 degrees Fahrenheit) and freezes at 0 degrees "
        "Celsius (32 degrees Fahrenheit).",

        "Gold is a chemical element with the symbol Au and atomic number 79. "
        "It is a dense, soft, malleable metal. Gold is one of the least "
        "reactive chemical elements.",
    ]

    print("Ingesting 3 passages from a high-quality textbook...\n")

    batch = pipeline.ingest_batch(passages, source=textbook, domain="chemistry")

    for i, r in enumerate(batch.results):
        print_result(r, label=f"Passage {i + 1}")

    print(f"Batch totals: {batch.total_added} added, "
          f"{batch.total_duplicates} merged, "
          f"{batch.total_flagged} flagged, "
          f"{batch.total_rejected} rejected")

    print("\nGraph state after seeding:")
    print_graph_state(pipeline.graph)


def step2_convergent_evidence(pipeline) -> None:
    """
    Step 2: Demonstrate convergent evidence (deduplication).

    When a second source confirms an existing fact, the pipeline
    detects the duplicate and boosts confidence via noisy-OR instead
    of creating a new node.
    """
    banner("STEP 2: Convergent evidence (deduplication)")

    encyclopedia = Source(
        source_type="encyclopedia",
        title="Encyclopedia Britannica",
        quality_score=0.9,
    )

    print("Ingesting a passage that overlaps with existing facts...\n")

    result = pipeline.ingest(
        "Water (H2O) boils at 100 degrees Celsius under standard "
        "atmospheric pressure. It freezes at 0 degrees Celsius.",
        source=encyclopedia,
        domain="chemistry",
    )

    print_result(result, label="Overlapping passage")
    print("Graph state after convergent evidence:")
    print_graph_state(pipeline.graph)


def step3_low_quality_source(pipeline) -> None:
    """
    Step 3: Demonstrate source quality gating.

    Facts from a low-quality source get capped confidence, keeping
    them PROVISIONAL until corroborated by a better source.
    """
    banner("STEP 3: Source quality gating")

    blog = Source(
        source_type="social_media",
        title="RandomChemFacts.blog",
        quality_score=0.25,
    )

    print("Ingesting a passage from a low-quality blog...\n")

    result = pipeline.ingest(
        "Silver has the chemical symbol Ag and atomic number 47. "
        "Silver is used in electronics and jewelry.",
        source=blog,
        domain="chemistry",
    )

    print_result(result, label="Low-quality source")

    # Show that the added nodes have capped confidence
    for v in result.verdicts:
        if v.action == "added" and v.node_id:
            node = pipeline.graph.get_node(v.node_id)
            print(f"  Node \"{node.text[:60]}...\"")
            print(f"    Confidence: {node.confidence:.3f} (capped by source quality 0.25)")
            print(f"    Status: {node.status.value}")

    print("\nGraph state after low-quality ingestion:")
    print_graph_state(pipeline.graph)


def step4_query_and_reason(pipeline) -> None:
    """
    Step 4: Query the graph and demonstrate reasoning.

    Use the ReasoningEngine to query the graph, test hypotheses,
    and explain nodes.
    """
    banner("STEP 4: Query and Reasoning")

    reasoning = ReasoningEngine()

    # Rebuild the embedding index to include all nodes
    pipeline.build_index()

    # Query
    print("Query: 'What is the boiling point of water?'\n")
    q_result = reasoning.query(
        pipeline.graph,
        "What is the boiling point of water?",
        embedding_engine=pipeline.embedding_engine,
    )

    print(f"  Answer: \"{q_result.answer_text}\"")
    print(f"  Confidence: {q_result.confidence:.3f}")
    if q_result.supporting_chains:
        print(f"  Supporting chains: {len(q_result.supporting_chains)}")
        for chain in q_result.supporting_chains[:2]:
            if chain.steps:
                print(f"    Chain: {' -> '.join(s.node_id[:8] + '...' for s in chain.steps)}")
                print(f"    Chain confidence: {chain.conclusion_confidence:.3f}")
    print()

    # Hypothesis testing
    print("Hypothesis: 'Gold is a reactive metal'\n")
    h_result = reasoning.test_hypothesis(
        pipeline.graph,
        "Gold is a reactive metal",
        embedding_engine=pipeline.embedding_engine,
    )

    print(f"  Supported: {h_result.supported}")
    print(f"  Confidence: {h_result.confidence:.3f}")
    print(f"  Supporting evidence: {len(h_result.supporting_evidence)}")
    print(f"  Contradicting evidence: {len(h_result.contradicting_evidence)}")
    if h_result.explanation:
        print(f"  Explanation: {h_result.explanation[:120]}...")
    print()


def step5_anomaly_detection(pipeline) -> None:
    """
    Step 5: Demonstrate anomaly detection.

    The confidence engine flags nodes whose confidence is inconsistent
    with their sources and predecessors.
    """
    banner("STEP 5: Anomaly Detection")

    conf_engine = pipeline.confidence_engine

    anomalies = conf_engine.detect_anomalies(pipeline.graph)

    if anomalies:
        print(f"Found {len(anomalies)} anomaly(ies):\n")
        for a in anomalies:
            print(f"  Node: \"{a['node_text'][:60]}...\"")
            print(f"  Type: {a['anomaly_type']}")
            print(f"  Expected: {a['expected_confidence']:.3f}, "
                  f"Actual: {a['actual_confidence']:.3f}")
            print(f"  Severity: {a['severity']:.3f}")
            print()
    else:
        print("No anomalies detected. The graph is internally consistent.")
    print()


def step6_save_graph(pipeline) -> None:
    """Step 6: Save the graph to disk."""
    banner("STEP 6: Save Graph")

    output_path = Path(__file__).parent.parent / "data" / "pipeline_demo.json"
    output_path.parent.mkdir(exist_ok=True)
    pipeline.graph.save(str(output_path))
    print(f"Graph saved to: {output_path}")
    print(f"Total nodes: {len(pipeline.graph)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner("KNOWLEDGE GRAPH AI - PIPELINE DEMO")

    # Check Ollama connectivity
    print("Checking Ollama server...")
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        if not models.models:
            print("ERROR: No models pulled. Run: ollama pull qwen2.5:7b")
            return
        model_name = models.models[0].model
        print(f"  Connected. Using model: {model_name}\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    # Create pipeline
    graph = KnowledgeGraph()
    pipeline = Pipeline(
        graph,
        extractor=FactExtractionEngine(model=model_name),
        embedding_engine=EmbeddingEngine(),
        dependency_detector=DependencyDetector(),
        confidence_engine=ConfidenceEngine(),
        min_extraction_confidence=0.2,
        reject_contradictions=False,  # Flag contradictions, don't reject
    )

    start = time.time()

    # Run demo steps
    step1_seed_with_vetted_data(pipeline)
    step2_convergent_evidence(pipeline)
    step3_low_quality_source(pipeline)
    step4_query_and_reason(pipeline)
    step5_anomaly_detection(pipeline)
    step6_save_graph(pipeline)

    elapsed = time.time() - start

    banner("DEMO COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Final graph: {len(graph)} nodes")
    stats = graph.get_statistics()
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")
    print(f"  Domains: {dict(stats['nodes_by_domain'])}")
    print()


if __name__ == "__main__":
    main()
