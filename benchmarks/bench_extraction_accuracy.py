"""
Benchmark: Fact Extraction Accuracy

Measures FactExtractionEngine performance on ARC science sentences:
- Completeness: % of sentences producing ≥1 fact
- Fidelity: embedding similarity between extracted text and source
- Hallucination rate: % of extracted facts with <0.3 similarity to source
- Confidence calibration: LLM confidence vs actual fidelity

Requires Ollama with qwen2.5:7b (or configured model).

Usage:
    python -m benchmarks.bench_extraction_accuracy
"""

import os
import sys
import statistics
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fact_extraction import FactExtractionEngine
from src.core.embeddings import EmbeddingEngine


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0
from src.core.metrics import load_jsonl, BenchmarkResult, save_report


DATA_PATH = os.path.join("data", "calibration", "arc_sentences.jsonl")
RESULTS_DIR = os.path.join("data", "calibration", "results")


def run():
    print("=" * 60)
    print("Benchmark: Fact Extraction Accuracy")
    print("=" * 60)

    # Load dataset
    sentences = load_jsonl(DATA_PATH)
    print(f"Loaded {len(sentences)} ARC science sentences")

    # Initialize engines
    print("Loading extraction engine (Ollama)...")
    extractor = FactExtractionEngine()
    print("Loading embedding engine...")
    embed = EmbeddingEngine()

    # Process each sentence
    completeness_count = 0  # Sentences producing ≥1 fact
    all_fidelities = []
    low_fidelity_count = 0  # Facts with similarity < 0.3
    total_facts = 0
    confidence_fidelity_pairs = []  # (confidence, fidelity) for calibration
    details = []

    print("Extracting facts from ARC sentences...")
    for i, sentence in enumerate(sentences):
        text = sentence["text"]
        try:
            result = extractor.extract(text, domain="science")
            facts = result.facts
        except Exception as e:
            facts = []
            print(f"  ERROR on sentence {i + 1}: {e}")

        if len(facts) > 0:
            completeness_count += 1

        source_vec = embed.embed_text(text)

        for fact in facts:
            total_facts += 1
            fact_vec = embed.embed_text(fact.text)
            fidelity = float(_cosine_sim(source_vec, fact_vec))

            all_fidelities.append(fidelity)
            confidence_fidelity_pairs.append((fact.confidence, fidelity))

            if fidelity < 0.3:
                low_fidelity_count += 1

        details.append({
            "source": text[:100],
            "num_facts": len(facts),
            "fact_texts": [f.text[:80] for f in facts[:5]],
            "fact_confidences": [f.confidence for f in facts[:5]],
            "fidelities": [float(_cosine_sim(source_vec, embed.embed_text(f.text)))
                          for f in facts[:5]],
        })

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sentences)} sentences processed "
                  f"({total_facts} facts so far)")

    # Compute metrics
    completeness = completeness_count / len(sentences) if sentences else 0
    avg_fidelity = statistics.mean(all_fidelities) if all_fidelities else 0
    median_fidelity = statistics.median(all_fidelities) if all_fidelities else 0
    hallucination_rate = low_fidelity_count / total_facts if total_facts else 0
    avg_facts_per_sentence = total_facts / len(sentences) if sentences else 0

    # Confidence calibration: correlation between confidence and fidelity
    if len(confidence_fidelity_pairs) > 10:
        confs = [p[0] for p in confidence_fidelity_pairs]
        fidels = [p[1] for p in confidence_fidelity_pairs]
        # Simple Pearson correlation
        mean_c = statistics.mean(confs)
        mean_f = statistics.mean(fidels)
        cov = sum((c - mean_c) * (f - mean_f) for c, f in zip(confs, fidels)) / len(confs)
        std_c = statistics.stdev(confs) if len(confs) > 1 else 1
        std_f = statistics.stdev(fidels) if len(fidels) > 1 else 1
        correlation = cov / (std_c * std_f) if std_c > 0 and std_f > 0 else 0
    else:
        correlation = 0

    print("\n--- Results ---")
    print(f"Sentences processed: {len(sentences)}")
    print(f"Total facts extracted: {total_facts}")
    print(f"Avg facts/sentence: {avg_facts_per_sentence:.1f}")
    print(f"Completeness (>=1 fact): {completeness:.1%}")
    print(f"Mean fidelity: {avg_fidelity:.4f}")
    print(f"Median fidelity: {median_fidelity:.4f}")
    print(f"Hallucination rate (<0.3 sim): {hallucination_rate:.1%}")
    print(f"Confidence-fidelity correlation: {correlation:.4f}")

    # Build result
    result = BenchmarkResult(
        name="extraction_accuracy",
        metrics={
            "total_sentences": len(sentences),
            "total_facts": total_facts,
            "avg_facts_per_sentence": avg_facts_per_sentence,
            "completeness": completeness,
            "mean_fidelity": avg_fidelity,
            "median_fidelity": median_fidelity,
            "hallucination_rate": hallucination_rate,
            "confidence_fidelity_correlation": correlation,
        },
        thresholds={
            "min_extraction_confidence": 0.3,
            "hallucination_fidelity_cutoff": 0.3,
        },
        details=details,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "extraction_accuracy_report.json")
    save_report(result, report_path)
    print(f"\nReport saved to: {report_path}")

    return result


if __name__ == "__main__":
    run()
