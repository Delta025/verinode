"""
Benchmark: Adversarial Testing

Stress-tests the pipeline with deliberately problematic inputs:
A. Contradictory facts (TruthfulQA correct+false answers)
B. Low-quality flood (ARC + low quality source)
C. Circular dependencies (synthetic A→B→C→A)
D. Near-duplicate flooding (same fact paraphrased 20x)
E. Malformed inputs (empty, whitespace, 10K chars, special chars)

Scenarios A & B require Ollama. Scenarios C, D, E are purely programmatic.

Usage:
    python -m benchmarks.bench_adversarial
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, Source
from src.core.embeddings import EmbeddingEngine
from src.core.dependency_detection import DependencyDetector, RelationType
from src.core.confidence import ConfidenceEngine
from src.core.metrics import load_jsonl, BenchmarkResult, save_report

TRUTHFULQA_PATH = os.path.join("data", "calibration", "truthfulqa_subset.jsonl")
ARC_PATH = os.path.join("data", "calibration", "arc_sentences.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")


def scenario_c_circular_deps():
    """Scenario C: Circular dependencies (no LLM needed)."""
    print("\n--- Scenario C: Circular Dependencies ---")
    graph = KnowledgeGraph()
    dep_detector = DependencyDetector()

    # Create a cycle: A → B → C → A
    node_a = KnowledgeNode(text="Fact A depends on fact C", node_type=NodeType.FACT,
                            confidence=0.8, domain="test")
    node_b = KnowledgeNode(text="Fact B depends on fact A", node_type=NodeType.FACT,
                            confidence=0.8, domain="test")
    node_c = KnowledgeNode(text="Fact C depends on fact B", node_type=NodeType.FACT,
                            confidence=0.8, domain="test")

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    # Wire circular edges via predecessor/successor
    node_b.add_predecessor(node_a.id)
    node_a.add_successor(node_b.id)
    node_c.add_predecessor(node_b.id)
    node_b.add_successor(node_c.id)
    node_a.add_predecessor(node_c.id)
    node_c.add_successor(node_a.id)

    # Test confidence propagation doesn't infinite loop
    conf_engine = ConfidenceEngine()
    try:
        t0 = time.perf_counter()
        node_a.confidence = 0.5
        affected = conf_engine.propagate_confidence_update(graph, node_a.id)
        elapsed = time.perf_counter() - t0
        terminated = True
        print(f"  Propagation terminated in {elapsed:.3f}s, {len(affected)} nodes affected")
    except RecursionError:
        terminated = False
        elapsed = -1
        print("  FAILED: RecursionError (infinite loop)")

    return {
        "terminated": terminated,
        "elapsed_s": elapsed,
        "nodes_affected": len(affected) if terminated else -1,
    }


def scenario_d_duplicate_flood():
    """Scenario D: Near-duplicate flooding (no LLM needed)."""
    print("\n--- Scenario D: Near-Duplicate Flooding ---")
    graph = KnowledgeGraph()
    embed = EmbeddingEngine()

    # Create 20 paraphrases of the same fact
    base_fact = "Water boils at 100 degrees Celsius at sea level"
    paraphrases = [
        base_fact,
        "At sea level, water reaches its boiling point at 100°C",
        "The boiling temperature of water is 100 degrees Celsius at standard pressure",
        "Water starts to boil when heated to 100 degrees C at sea level",
        "100 degrees Celsius is the boiling point of water at sea level",
        "H2O boils at a temperature of 100 Celsius at sea level pressure",
        "At standard atmospheric pressure, water boils at 100 degrees Celsius",
        "The temperature at which water boils at sea level is 100°C",
        "Water will begin boiling at one hundred degrees Celsius at sea level",
        "Sea level boiling point of water equals 100 degrees Celsius",
        "At normal pressure, water reaches boiling at 100 C",
        "The boiling point of H2O is 100 deg C at 1 atm",
        "Water transitions to steam at 100 celsius at sea level",
        "Liquid water becomes vapor at 100 degrees C at sea level",
        "100C is when water boils under standard conditions at sea level",
        "Under sea level conditions, water boils at the temperature of 100°C",
        "At sea-level atmospheric pressure, boiling occurs at 100 Celsius for water",
        "Pure water at sea level has a boiling point of 100 degrees C",
        "Water heated to 100 degrees Celsius at sea level will boil",
        "The boiling temperature of pure water at sea level is 100 degrees C",
    ]

    # Add first, then check how many others would be detected as duplicates
    first_node = KnowledgeNode(text=paraphrases[0], node_type=NodeType.FACT,
                                confidence=0.8, domain="science")
    graph.add_node(first_node)
    embed.index_graph(graph)

    duplicates_detected = 0
    unique_added = 1

    for text in paraphrases[1:]:
        vec = embed.embed_text(text)
        results = embed.search(text, k=1, threshold=0.92)

        if results:
            duplicates_detected += 1
        else:
            # Would be added as a new node
            node = KnowledgeNode(text=text, node_type=NodeType.FACT,
                                  confidence=0.8, domain="science")
            graph.add_node(node)
            embed.add_to_index(node)
            unique_added += 1

    print(f"  {len(paraphrases)} paraphrases, {duplicates_detected} duplicates detected, "
          f"{unique_added} unique nodes")
    print(f"  Dedup rate: {duplicates_detected / (len(paraphrases) - 1):.1%}")

    return {
        "total_paraphrases": len(paraphrases),
        "duplicates_detected": duplicates_detected,
        "unique_nodes": unique_added,
        "dedup_rate": duplicates_detected / (len(paraphrases) - 1),
    }


def scenario_e_malformed():
    """Scenario E: Malformed inputs (no LLM needed)."""
    print("\n--- Scenario E: Malformed Inputs ---")
    graph = KnowledgeGraph()
    embed = EmbeddingEngine()

    malformed_inputs = [
        ("empty_string", ""),
        ("whitespace_only", "   \n\t  "),
        ("single_char", "x"),
        ("special_chars", "!@#$%^&*(){}[]|\\/<>"),
        ("unicode_emoji", "🔥🎉🚀💡✨🌍"),
        ("very_long", "a " * 5000),  # 10K chars
        ("null_bytes", "hello\x00world\x00test"),
        ("html_tags", "<script>alert('xss')</script><b>bold</b>"),
        ("sql_injection", "'; DROP TABLE nodes; --"),
        ("newlines_only", "\n\n\n\n\n"),
    ]

    results = {}
    for name, text in malformed_inputs:
        try:
            node = KnowledgeNode(text=text, node_type=NodeType.FACT,
                                  confidence=0.5, domain="test")
            graph.add_node(node)

            # Try embedding (may fail gracefully)
            try:
                vec = embed.embed_text(text)
                embedded = vec is not None and len(vec) > 0
            except Exception:
                embedded = False

            results[name] = {"crashed": False, "embedded": embedded}
            print(f"  {name}: OK (embedded={embedded})")
        except Exception as e:
            results[name] = {"crashed": True, "error": str(e)[:100]}
            print(f"  {name}: CRASHED ({type(e).__name__}: {str(e)[:60]})")

    crashes = sum(1 for r in results.values() if r.get("crashed"))
    print(f"\n  {crashes}/{len(malformed_inputs)} inputs caused crashes")

    return results


def scenario_a_truthfulqa(extractor, embed):
    """Scenario A: TruthfulQA adversarial (requires Ollama)."""
    print("\n--- Scenario A: TruthfulQA Adversarial ---")

    from src.core.pipeline import Pipeline

    questions = load_jsonl(TRUTHFULQA_PATH)[:20]  # Limit for speed
    graph = KnowledgeGraph()

    pipeline = Pipeline(
        graph=graph,
        extractor=extractor,
        embedding_engine=embed,
        dependency_detector=DependencyDetector(),
        confidence_engine=ConfidenceEngine(),
    )

    source_correct = Source(source_type="dataset", title="TruthfulQA-correct", quality_score=0.9)
    source_incorrect = Source(source_type="dataset", title="TruthfulQA-incorrect", quality_score=0.5)

    correct_ingested = 0
    incorrect_flagged = 0
    incorrect_total = 0

    for q in questions:
        # Seed with correct answer
        correct_text = f"{q['question']} {q['best_answer']}"
        try:
            pipeline.ingest(correct_text, source=source_correct, domain="general")
            correct_ingested += 1
        except Exception:
            pass

    pipeline.build_index()

    # Now try incorrect answers
    for q in questions:
        for incorrect in q.get("incorrect_answers", [])[:2]:  # Limit per question
            incorrect_text = f"{q['question']} {incorrect}"
            incorrect_total += 1
            try:
                result = pipeline.ingest(incorrect_text, source=source_incorrect)
                if result.facts_flagged > 0 or result.facts_rejected > 0:
                    incorrect_flagged += 1
            except Exception:
                pass

    flag_rate = incorrect_flagged / incorrect_total if incorrect_total > 0 else 0
    print(f"  Correct answers ingested: {correct_ingested}")
    print(f"  Incorrect answers tested: {incorrect_total}")
    print(f"  Incorrect flagged/rejected: {incorrect_flagged} ({flag_rate:.1%})")

    return {
        "correct_ingested": correct_ingested,
        "incorrect_total": incorrect_total,
        "incorrect_flagged": incorrect_flagged,
        "flag_rate": flag_rate,
    }


def scenario_b_low_quality_flood(extractor, embed):
    """Scenario B: Low-quality source flood (requires Ollama)."""
    print("\n--- Scenario B: Low-Quality Flood ---")

    from src.core.pipeline import Pipeline

    arc_sentences = load_jsonl(ARC_PATH)[:20]
    graph = KnowledgeGraph()

    pipeline = Pipeline(
        graph=graph,
        extractor=extractor,
        embedding_engine=embed,
        dependency_detector=DependencyDetector(),
        confidence_engine=ConfidenceEngine(),
    )

    # Ingest with very low quality source
    low_source = Source(source_type="blog", title="unreliable-blog", quality_score=0.1)
    capped_count = 0
    total_nodes = 0

    for sentence in arc_sentences:
        try:
            result = pipeline.ingest(sentence["text"], source=low_source, domain="science")
            total_nodes += result.nodes_added
        except Exception:
            pass

    # Check that confidences are capped
    for node in graph.nodes.values():
        if node.confidence <= 0.1 + 0.01:  # Small epsilon
            capped_count += 1

    cap_rate = capped_count / total_nodes if total_nodes > 0 else 0
    print(f"  Nodes added: {total_nodes}")
    print(f"  Nodes with capped confidence: {capped_count} ({cap_rate:.1%})")

    return {
        "total_nodes": total_nodes,
        "capped_count": capped_count,
        "cap_rate": cap_rate,
    }


def run():
    print("=" * 60)
    print("Benchmark: Adversarial Testing")
    print("=" * 60)

    all_results = {}

    # Programmatic scenarios (no LLM needed)
    all_results["circular_deps"] = scenario_c_circular_deps()
    all_results["duplicate_flood"] = scenario_d_duplicate_flood()
    all_results["malformed_inputs"] = scenario_e_malformed()

    # LLM scenarios (try, but don't fail if Ollama unavailable)
    try:
        from src.core.fact_extraction import FactExtractionEngine

        print("\nLoading LLM components...")
        extractor = FactExtractionEngine()
        embed_engine = EmbeddingEngine()

        all_results["truthfulqa"] = scenario_a_truthfulqa(extractor, embed_engine)
        all_results["low_quality_flood"] = scenario_b_low_quality_flood(extractor, embed_engine)
    except Exception as e:
        print(f"\nSkipping LLM scenarios (Ollama unavailable): {e}")
        all_results["truthfulqa"] = {"skipped": True, "reason": str(e)}
        all_results["low_quality_flood"] = {"skipped": True, "reason": str(e)}

    # Build result
    result = BenchmarkResult(
        name="adversarial",
        metrics={
            "circular_terminated": all_results["circular_deps"].get("terminated", False),
            "dedup_rate": all_results["duplicate_flood"].get("dedup_rate", 0),
            "malformed_crashes": sum(
                1 for r in all_results["malformed_inputs"].values()
                if isinstance(r, dict) and r.get("crashed")
            ),
        },
        thresholds={},
        details=[all_results],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "adversarial_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
