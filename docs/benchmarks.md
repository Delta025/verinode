# Verinode — Benchmarks and Performance

All performance data in one place. Individual results are linked from [ROADMAP.md](../ROADMAP.md), [architecture.md](architecture.md), and [research.md](research.md).

---

## Vulnerability Resistance Benchmark (Feb 2026)

Side-by-side comparison of bare qwen3:14b vs qwen3:14b + Verinode on 75 test cases across 5 vulnerability categories. Uses the 215-node stress test graph (`data/stress_test_graph.json`) as the knowledge base.

**Scripts**: `examples/benchmark_vulnerability.py`, `examples/benchmark_test_cases.py`
**Full results**: `data/benchmark_results.json`

### Summary Table

| Category | Without Verinode | With Verinode | Delta |
|---|:---:|:---:|:---:|
| A. Factual Accuracy (15) | 100% (15/15) | 100% (15/15) | +0pp |
| B. False Claim Detection (20) | 100% (20/20) | 95% (19/20) | -5pp |
| C. Poisoned Context Resistance (15) | **20%** (3/15) | **100%** (15/15) | **+80pp** |
| D. Confidence Calibration (10) | Brier 0.001 / Sep 0.044 | Brier 0.029 / Sep 0.082 | Sep +86% |
| E. Subtle Numerical Errors (15) | **20%** (3/15) | **80%** (12/15) | **+60pp** |

Total runtime: 30.4 minutes on a single machine (qwen3:14b, 14.8B params, Q4_K_M quantization).

### Category Breakdown

#### A. Factual Accuracy — 15 questions, 3 domains

Asks the LLM factual questions (numerical, categorical, descriptive) and compares answers to ground truth.

| Mode | Accuracy | Avg Time/Case |
|------|:--------:|:-------------:|
| Without Verinode | 100% (15/15) | 4.8s |
| With Verinode (recall + LLM) | 100% (15/15) | 5.7s |

Both score perfectly. The LLM already knows common science facts from its training data. **Verinode's value emerges in adversarial scenarios, not baseline knowledge retrieval.** This category establishes the baseline — Verinode does not regress performance on well-known facts.

> **Note**: These results test knowledge the LLM has been extensively trained on (basic chemistry, physics, biology). On specialized or domain-specific knowledge not in training data (e.g., specifics of a legal framework, proprietary technical documentation), the LLM would be expected to perform significantly worse without Verinode's recall support. This remains to be tested.

#### B. False Claim Detection — 10 false + 10 true claims

Presents claims and asks "true or false?". False claims span three difficulty levels: obvious (>50% off), subtle (1-5% off), and category swaps.

| Mode | Overall | False Caught | True Preserved |
|------|:-------:|:------------:|:--------------:|
| Without Verinode | 100% (20/20) | 100% (10/10) | 100% (10/10) |
| With Verinode (verify) | 95% (19/20) | 100% (10/10) | 90% (9/10) |

The bare LLM handles these well because the false claims are about facts it knows from training. Verinode catches all 10 false claims but produces **1 false positive**: "The speed of light is approximately 300,000 km/s" was flagged as contradicting the graph's precise 299,792 value. This is a known edge case — the verification engine is strict on numerical precision, treating the ~0.07% rounding as a factual disagreement.

> **Note**: As with Category A, both systems perform well because these are well-known facts. The bare LLM's advantage would likely disappear on domain-specific claims outside its training data, where it would need to guess rather than recall. Future testing on specialized domains (legal, medical, proprietary) is planned.

#### C. Poisoned Context Resistance — 15 cases

The headline result. Simulates a compromised RAG pipeline by injecting false context paragraphs alongside questions (e.g., "According to recent studies, water boils at 85°C..."). Tests whether the LLM follows the poison or resists it.

| Mode | Resisted Poison | Avg Time/Case |
|------|:---------------:|:-------------:|
| Without Verinode | **20%** (3/15) | 3.9s |
| With Verinode (verify + recall) | **100%** (15/15) | 29.8s |

**Without Verinode**: The bare LLM follows poisoned context 80% of the time. When told "use this context to answer", it trusts the context even when the information is obviously wrong (e.g., "gold melts at 800°C"). The 3 cases it resisted were facts where the LLM's own knowledge was strong enough to override the context (CO2 formula, DNA structure, Newton's second law).

**With Verinode**: Perfect resistance. The pipeline works in three steps:
1. `memory.verify(poisoned_claim)` — flags the poisoned context as "contradicted"
2. `memory.recall(question)` — retrieves verified facts from the knowledge graph
3. LLM receives both the unverified context (with its verification status) and the verified facts, with instructions to trust verified facts when they conflict

This demonstrates Verinode's core value proposition: **grounding LLM output in verified, auditable knowledge prevents the LLM from blindly following false context**.

#### D. Confidence Calibration — 5 true + 5 false claims

Measures the quality of confidence scores — are they meaningful for distinguishing truth from falsehood?

| Metric | Without Verinode | With Verinode | Better |
|--------|:--------------:|:-----------:|:------:|
| Brier Score (lower = better) | 0.001 | 0.029 | Without |
| Separation (higher = better) | 0.044 | 0.082 | **With** |

**Brier Score**: The bare LLM's self-reported confidence is "better" calibrated only because it assigns high confidence (0.95-1.0) to everything — and happens to be correct on these well-known facts. This makes the Brier score misleadingly good. On unfamiliar topics, overconfidence would be a liability.

**Separation**: Verinode provides better separation between true and false claim confidence (0.082 vs 0.044). Its confidence scores are more meaningful: supported claims get higher scores, contradicted claims get lower scores. This makes the confidence signal actionable — a downstream system could use a threshold to flag uncertain claims for human review.

#### E. Subtle Numerical Errors — 15 claims with 0.1-17% perturbations

Tests the ability to catch small numerical errors — a known LLM weakness. All claims are false (e.g., "Gold melts at 1060°C" when the correct value is 1064°C).

| Mode | Caught | Avg Time/Case |
|------|:------:|:-------------:|
| Without Verinode | **20%** (3/15) | 2.8s |
| With Verinode (verify) | **80%** (12/15) | 18.5s |

**Without Verinode**: The bare LLM almost never catches small numerical errors. It accepted "Iron melts at 1540°C" (actual: 1538), "Gold melts at 1060°C" (actual: 1064), and "205 bones" (actual: 206) as correct. LLMs are notoriously imprecise with numbers — they learn distributions of values, not exact figures.

**With Verinode**: Caught 12/15, a 4× improvement. The 3 misses were cases where the graph also used rounded values or lacked the specific fact:
- "Ethanol boils at 78°C" (actual: 78.37) — graph has rounded value
- "Speed of sound is 340 m/s" (actual: 343) — graph has "approximately 343"
- "CMB radiation is 2.7K" (actual: 2.725) — graph lacks this specific fact

**By error magnitude**:

| Error Range | Without | With |
|-------------|:-------:|:----:|
| ≤ 1% | 20% | 70% |
| ≤ 5% | 18% | 73% |
| All (0.1-17%) | 20% | 80% |

### Interpretation and Caveats

**Where Verinode excels**: Categories C and E — adversarial scenarios where the LLM's parametric knowledge is insufficient. Poisoned context resistance (+80pp) and subtle numerical error detection (+60pp) demonstrate that a verified knowledge graph provides a fundamentally different kind of capability than parametric memory.

**Where the LLM already does well**: Categories A and B — well-known science facts that the LLM has been trained on extensively. These categories would likely show larger differences on domain-specific knowledge (legal frameworks, proprietary documentation, niche technical standards) where the LLM has less training coverage.

**Known edge case**: Verinode's strict numerical verification occasionally flags rounded values as contradictions (300,000 vs 299,792 km/s). This is a precision/recall tradeoff — strictness catches subtle errors but creates occasional false positives on approximate values.

**Latency tradeoff**: Verinode-augmented responses take longer (5-30s vs 3-5s per case) because each verification involves embedding search, entity overlap retrieval, and multiple LLM calls for 3-way relevance classification. For offline verification or batch processing this is acceptable; for real-time applications, caching and index optimizations would be needed.

---

## Chemistry Slab Benchmark (Step 14 — Feb 2026)

Graph size and source quality impact on verification accuracy. Tests 35 claims (10 true, 10 false, 8 planted errors, 7 subtle numerical) across 5 composable chemistry graph slabs of different sizes and quality tiers.

**Script**: `examples/benchmark_chemistry_slabs.py`
**Full results**: `data/slab_benchmark_combined.json`

### Summary Table

| Slab | Nodes | Tiers | True Claims | False Claims | Planted Errors | Subtle Numerical |
|------|:-----:|:-----:|:-----------:|:-----------:|:--------------:|:----------------:|
| T1 Textbook | 52 | 1 | 100% (10/10) | 80% (8/10) | 0% (0/8) | 0% (0/7) |
| T3+4 Lecture+Student | 74 | 3,4 | 80% (8/10) | 10% (1/10) | 88% (7/8)* | 14% (1/7) |
| T1+T2 High-quality | 109 | 1,2 | **100%** (10/10) | **100%** (10/10) | 12% (1/8) | **86%** (6/7) |
| T1+T3+4 Mixed | 126 | 1,3,4 | 90% (9/10) | 90% (9/10) | 12% (1/8) | 14% (1/7) |
| All Tiers Full | 183 | 1,2,3,4 | **100%** (10/10) | **100%** (10/10) | 25% (2/8) | **86%** (6/7) |

*T3+4 planted errors: 88% = correctly NOT contradicted (errors are ground truth in this slab, with no high-quality reference to contradict them).

Total runtime: ~2 hours across all slabs (qwen3:14b, ~50s/claim on large graphs).

### Key Findings

**1. Source quality matters more than graph size.**
- T1+T2 (109 nodes) matches the full graph (183 nodes) on all categories. Tier 2's precise measurements are the key ingredient for catching subtle errors.
- The mixed slab (T1+T3+4, 126 nodes) — which has MORE nodes than T1+T2 but LOWER average quality — drops to 14% on subtle numerical detection.
- **Conclusion**: Adding low-quality data doesn't hurt accuracy WHEN high-quality data is also present, but it doesn't help catch numerical errors either. T2's precise values are essential.

**2. Low-quality data self-validates when unchecked.**
- The T3+4 slab (errors-only) shows planted errors being "supported" — the graph validates its own mistakes because no contradicting high-quality facts exist.
- In the full graph (with T1+T2 present), some planted errors shift from "supported" to "contradicted" — but only 2/8 (25%). Most planted errors still evade detection because the graph doesn't contain the specific counter-fact.

**3. The retrieval ceiling limits planted error detection.**
- Planted error detection (12-25%) is low across all merged slabs, not because of quality issues but because the embedding retrieval doesn't surface the right contradicting facts.
- Example: "Carbon has 8 protons" is planted in T4, and T1 contains "Carbon has atomic number 6" — but the retrieval engine doesn't consistently connect these as contradictions.
- This confirms the retrieval ceiling finding from Step 7.5 and motivates a cross-encoder re-ranker.

**4. Verification latency scales with graph size.**
- T1 (52 nodes): ~14s/claim
- T1+T2 (109 nodes): ~49s/claim
- Full (183 nodes): ~54s/claim
- Larger graphs have more candidate nodes to score, increasing per-claim verification time. Optimization (caching, index pruning) would be needed at 1000+ nodes.

**5. Ethanol boiling point is a persistent edge case.**
- "Ethanol boils at 78°C" (actual: 78.37°C) was accepted as correct on every slab — the 0.5% error is within rounding tolerance. This matches the same class of edge case seen in the vulnerability benchmark (300,000 vs 299,792 km/s).

---

## Calibration Benchmarks (Step 7 — Feb 2026)

Model upgrade from qwen2.5:7b to qwen3:14b, followed by systematic code improvements and threshold optimization.

### Before/After Comparison

| Benchmark | Key Metric | Before | After | Change |
|-----------|-----------|--------|-------|--------|
| LLM Contradiction | F1 | 0.355 | **0.842** | **+137%** |
| LLM Contradiction | Recall | 22% | **80%** | **+264%** |
| Extraction Accuracy | Mean Fidelity | 0.742 | **0.800** | **+7.9%** |
| Extraction Accuracy | Hallucination Rate | 0.2% | **0.0%** | **-100%** |
| E2E Pipeline | REFUTES Flag Rate | 21.4% | **48.3%** | **+126%** |
| E2E Pipeline | SUPPORTS False Flag Rate | 0.0% | **0.0%** | Maintained |
| Embedding Similarity | Optimal F1 | — | **0.841** | Baseline |
| Heuristic Contradiction | F1 | 0 | **0.172** | New capability |
| Dependency Classification | Accuracy | — | **0.595** | Baseline |

### Optimized Thresholds

Via Optuna Bayesian optimization (10 trials on interacting thresholds):

| Threshold | Before | After | Source |
|-----------|--------|-------|--------|
| similarity_threshold | 0.35 | **0.698** | Optuna + embedding benchmark |
| contradiction_threshold | 0.50 | **0.523** | Optuna |
| duplicate_threshold | 0.90 | **0.867** | Optuna |
| min_extraction_confidence | 0.30 | **0.460** | Optuna |
| source_quality_cap | 0.50 | **0.540** | Optuna |
| llm_contradiction_confidence | 0.70 | **0.60** | Manual tuning |

Full calibration data: `data/calibration/results/final_comparison_report.json`

---

## Retrieval Benchmarks (Step 7.5)

Three E2E runs on 50 ARC seed + 50 FEVER test sentences, qwen3:14b:

| Run | Strategy | REFUTES Flag Rate | False Positives | Dedup Stolen | Time |
|-----|----------|:-----------------:|:---------------:|:------------:|:----:|
| 1 | Baseline (Step 7) | 48.3% | 0% | 3 | 3.3h |
| 2 | + Neighbor expansion | 41.4% ❌ | 0% | 3 | 2.8h |
| 3 | + Entity retrieval + contradiction-first | 48.3% | 0% | 1 | 3.3h |

### Retrieval Ceiling Analysis

The remaining 51.7% gap between E2E pipeline flag rate (48.3%) and standalone LLM contradiction detection recall (80%) breaks down as:
- **LLM quality limits**: ~6 cases — the LLM fails to detect contradictions even when candidates are correctly retrieved
- **Extraction loss**: ~5 cases — relevant facts are extracted with altered wording that reduces semantic similarity
- **Entity overlap gaps**: ~4 cases — entities share few surface-level content words, and the facts are too semantically different for embedding similarity alone

### BM25 Hybrid Retrieval

Added BM25 scoring with IDF weighting to EntityIndex, combined with embedding search via Reciprocal Rank Fusion (RRF, k=60). Calibration tests on a 112-node solar system graph:

| Test | Result |
|------|--------|
| BM25 finds Earth distance fact (149.6 million km) | Top-5 |
| BM25 ranks Neptune facts highly (rare entity) | 2+ Neptune facts in top-5 |
| BM25 finds Mars moon facts | Top-5 |
| BM25 finds Venus classification | Top-5 |
| BM25 finds Sun composition (hydrogen) | Top-5 |
| RRF merges embedding + BM25 results | Candidates from both sources |
| RRF recall >= embedding-only recall | 3/3 queries matched |
| True claims not falsely contradicted (heuristic) | <= 1/5 false positives |
| False claims get lower confidence than true claims | avg_true > avg_false |
| BM25 ranks rare terms higher than entity overlap | Neptune in top result |
| BM25 retrieves numerical candidates | 3/3 queries found |

Key improvement: BM25's IDF weighting ranks rare entities (Neptune, Mercury) higher than common terms (planet, Sun), which raw entity overlap count could not. RRF fusion provides equal or better recall than either method alone.

---

## Verification Stress Tests (Steps 10-11)

### 150-Fact Ingestion (215 nodes)

| Phase | Result | Key Metric |
|-------|--------|------------|
| Ingestion | 218 nodes from 150 facts | 39.2s/fact avg, 6 duplicates detected |
| Retrieval | 10/10 queries HIGH coverage | 0.01s query latency, 0.87 avg confidence |
| Verification (post-Step 11) | 9/10 false caught, 9/10 true OK | 13.1s/claim avg |
| Export/Import | 218/218 round-trip | 0.004s export, 0.006s import |
| Graph Stats | 218 nodes, 105 edges | 194 facts, 22 definitions, 1 rule, 1 inference |

### Per-Domain Ingestion

| Domain | Facts | Nodes | Duplicates | Time | Rate |
|--------|-------|-------|------------|------|------|
| Chemistry | 50 | 66 | 1 | 31.4 min | 37.7s/fact |
| Physics | 50 | 65 | 0 | 29.5 min | 35.4s/fact |
| Biology | 50 | 87 | 5 | 37.2 min | 44.6s/fact |

### Claim Verification (10 false + 10 true)

| Graph Size | False Caught | True OK | Avg Time/Claim |
|:----------:|:-----------:|:-------:|:--------------:|
| Seed (34 nodes) | 9/10 (90%) | 9/10 (90%) | 25.0s |
| Full (215 nodes) | 9/10 (90%) | 9/10 (90%) | 13.1s |

The larger graph is faster due to more precise embedding matches that reduce unnecessary LLM processing.

### Known Edge Cases

- **Rounding precision**: "Absolute zero is -273°C" contradicted against graph's "-273.15°C" — the LLM treats the 0.05% rounding as a factual disagreement
- **Large-number comparison**: "Moon orbits at 38000 km" vs 384400 km returned uncertain — the 10× difference wasn't confidently detected by the LLM
- Both are LLM judgment limitations, not verification engine defects

---

## Module-Level Performance

### FactExtractionEngine
- Mean fidelity: 0.800 (on 200 ARC science sentences)
- Hallucination rate: 0.0%
- Average facts per sentence: 1.87
- Model: qwen3:14b with JSON mode, thinking disabled

### ContradictionChecker
- F1: 0.842, Precision: 88.9%, Recall: 80% (on 150 FEVER pairs)
- Detects 4 contradiction types: semantic (29), numerical (7), negation (9), logical
- Thinking mode disabled for 2-5× speedup

### Pipeline (E2E)
- REFUTES flag rate: 48.3% (0% false positives on true statements)
- Entity-overlap retrieval + contradiction-first ordering
- Batch parallelism (ThreadPoolExecutor, max_workers=4)

---

## Scale Testing (Step 15 -- Feb 2026)

End-to-end performance and accuracy testing on large programmatically generated graphs (1000--3000 nodes) and LLM-imported content. Tests all subsystems: loading, embedding index, bi-encoder retrieval, cross-encoder re-ranking, neuroplasticity maintenance, and LLM-based verification.

**Full results**: `data/scale_test_1000_full.json`, `data/scale_test_3000_full.json`, `data/quick_verify_1000.json`, `data/quick_verify_104.json`

### Infrastructure Performance

| Metric | 1000 nodes | 3000 nodes | Scaling |
|--------|:----------:|:----------:|:-------:|
| Graph load time | 0.58--0.76s | 1.73--2.09s | Linear |
| File size | 12.4 MB | 37.2 MB | Linear |
| Peak memory | 37 MB | 112 MB | Linear |
| Embedding index build | 0.07--0.20s | 0.01--0.10s | Fast |
| Bi-encoder retrieval (p50) | 9--11ms | 7--9ms | Constant |
| Bi-encoder cold-start (first query) | 9--12.5s | 1.3--8.0s | Model load |
| Reranker retrieval (p50) | 94--187ms | 97--106ms | Constant |
| Neuroplasticity maintenance | 6--17ms | 16--17ms | Linear |
| Avg confidence | 0.826 | 0.820 | Stable |

All infrastructure operations scale linearly or better to 3000 nodes. The bi-encoder first-query latency is model cold-start overhead, not graph-size dependent. Re-ranking adds ~100ms per query. Neuroplasticity maintenance stays sub-20ms even at 3000 nodes.

### Verification Accuracy at Scale (With Cross-Encoder Reranker)

5 true + 5 false claims across 6 domains (chemistry, physics, biology, earth science, mathematics, computer science). Verification uses k=10 retrieval + cross-encoder re-ranking + 3-way LLM relevance classification with max 20 candidates.

| Claim | 1000 nodes | 3000 nodes |
|-------|:----------:|:----------:|
| Gold 1064C (true) | supported (0.10) | supported (0.39) |
| Oxygen #8 (true) | insuff. evidence | supported (0.17) |
| Speed of light (true) | insuff. evidence | insuff. evidence |
| DNA double helix (true) | supported (0.94) | supported (0.53) |
| Binary search O(log n) (true) | supported (0.35) | supported (0.35) |
| **True accuracy** | **60%** | **80%** |
| Gold 500C (false) | contradicted (0.09) | contradicted (0.38) |
| Oxygen #12 (false) | insuff. evidence | contradicted (0.16) |
| Speed of light wrong (false) | insuff. evidence | insuff. evidence |
| DNA triple helix (false) | contradicted (0.86) | contradicted (0.48) |
| Binary search O(n) (false) | contradicted (0.35) | contradicted (0.74) |
| **False detection** | **60%** | **80%** |
| Avg time/true claim | 97.3s | 60.7s |
| Avg time/false claim | 49.1s | 60.6s |
| Avg candidates checked | 15.5 | 17.9 |

### Key Findings

**1. Larger graphs improve accuracy.** The 3000-node graph scores 80% on both true and false claims vs 60% on 1000 nodes. The improvement comes from more diverse paraphrased content that the LLM can recognize as semantically relevant.

**2. Speed of light is a persistent retrieval gap.** Both graph sizes return insufficient evidence for speed-of-light claims because the graph stores physics facts using different terminology (planetary speeds, wave equations) that embedding search does not connect to "speed of light in vacuum".

**3. Confidence scores are meaningful but variable.** High-confidence verdicts (DNA double helix 0.94, DNA triple helix 0.86) come from claims with direct textual matches. Low-confidence verdicts (Gold 0.09--0.39) arise when the LLM finds relevant but not precisely matching candidates.

**4. Verification without reranker is 3x faster.** Quick test (no reranker, max 10 candidates): 36.8s/claim avg vs 73.2s/claim with reranker and 20 candidates. Accuracy was 2/3 for both true and false claims on a 3-claim subset. The reranker improves recall at the cost of latency.

### Textbook Import (LLM Pipeline)

Testing real-world content ingestion from a 20KB chemistry reference text (20 paragraphs).

| Metric | Value |
|--------|-------|
| Input | 20 paragraphs (20KB chemistry reference text) |
| Output | 104 nodes |
| Facts per passage | 5.2 avg (range: 3--8) |
| Time per passage | 149s avg (range: 62--257s) |
| Total import time | 49m 44s |
| Checkpoint frequency | Every 10 passages |

Verification on the 104-node textbook graph correctly handles domain-specific claims (Gold 1064C: supported, Gold 500C: contradicted) but returns insufficient evidence for out-of-domain claims (Water pH, DNA bases), which is expected behavior for a chemistry-only graph.

### Programmatic vs LLM-Extracted Graphs

| Aspect | Programmatic (`build_scale_graph.py`) | LLM-Extracted (`import_text.py`) |
|--------|:-------------------------------------:|:--------------------------------:|
| Build speed | ~1s per 1000 nodes | ~149s per 5 nodes |
| Coverage | Broad (6 domains), formulaic | Deep, narrow (1 domain) |
| Fact quality | Template-based, precise values | Natural language, rich context |
| Verification accuracy | 60--80% (multi-domain claims) | 33% (but correct within domain) |
| Best use case | Scale testing, benchmarks | Real knowledge ingestion |

---

## Measured Scaling Data

Empirical measurements replacing earlier estimates. Per-node overhead: ~37 KB at 1000 nodes, ~37 KB at 3000 nodes (consistent, dominated by embeddings in JSON).

| Graph Size | File Size | RAM Usage | Load Time | Retrieval p50 | Maintenance | Verification |
|:----------:|:---------:|:---------:|:---------:|:-------------:|:-----------:|:------------:|
| 104 nodes | ~1.2 MB | ~4 MB | 0.03s | <1ms | <1ms | 6.1s/claim |
| 215 nodes | ~2.8 MB | ~8 MB | ~0.1s | ~1ms | ~1ms | 13.1s/claim |
| 1000 nodes | 12.4 MB | 37 MB | 0.58s | 9--11ms | 6ms | 73.2s/claim |
| 3000 nodes | 37.2 MB | 112 MB | 1.73s | 7--9ms | 17ms | 60.7s/claim |

Note: verification time per claim is dominated by LLM calls (qwen3:14b), not graph operations. The 3000-node graph is faster per claim than 1000-node despite more candidates because the LLM model was already warm.

### Scaling Bottlenecks
1. **Entity retrieval**: Now uses `EntityIndex` inverted index with BM25 scoring — O(k) per query. Combined with embedding search via RRF fusion.
2. **FAISS IndexFlatIP**: O(N) brute-force. Switch to IndexIVFFlat or IndexHNSW at ~100K nodes.
3. **Triple embedding storage**: Vectors stored 3x in memory. Unify to single reference at ~50K nodes.
4. **SqliteGraph startup**: Loads all nodes on init. Lazy loading needed at ~100K+ nodes.
5. **LLM verification latency**: 60--97s/claim at scale with 20 candidates. Primary bottleneck is Ollama throughput on constrained GPU (10GB VRAM). Reducing max_candidates or using a smaller model improves latency at the cost of accuracy.

---

## Test Suite

797 passed + 106 skipped (integration/calibration requiring live servers or datasets) across 34 files.

All unit tests use mocked Ollama/Gemini responses — no running server needed. Integration tests require a live Ollama server with qwen3:14b (skipped by default). Calibration tests use real datasets but no LLM.

| Test File | Count | Coverage |
|-----------|:-----:|---------|
| test_pipeline.py | 69 | Pipeline ingestion, BM25/RRF hybrid retrieval, entity retrieval, batch parallelism, tiered ingestion, quote verification |
| test_storage.py | 55 | SQLite storage, round-trips, conversions, edge labels |
| test_graph.py | 55 | Graph operations, cycle detection, statistics |
| test_confidence.py | 47 | Confidence propagation, bounded hops |
| test_neuroplasticity.py | 45 | Staleness decay, pathway strengthening, doubt flagging, corroboration decay |
| test_node.py | 44 | Node construction, source tiers, edge labels |
| test_export.py | 43 | Sub-graph export/import, graph merge, conflict resolution |
| test_cli.py | 43 | CLI commands, formatting, state management |
| test_verification.py | 43 | Claim verification, retraction, relevance filtering, source quality weighting |
| test_unit_normalizer.py | 40 | Unit normalization, number formatting |
| test_metrics.py | 38 | Precision, recall, F1, threshold sweep, classification report |
| test_entity_index.py | 32 | Entity index, BM25 search, IDF weighting, content word extraction |
| test_llm_contradiction.py | 30 | 3-way relevance detection, caching (mocked Ollama + Gemini) |
| test_llm_memory.py | 28 | LLM memory facade |
| test_reranker.py | 27 | Cross-encoder re-ranking, score normalization, integration |
| test_reasoning.py | 27 | Reasoning chains, queries, hypotheses |
| test_rate_limiter.py | 26 | Rate limiting, sliding window, backoff, budget exhaustion |
| test_fact_extraction.py | 25 | LLM extraction (mocked Ollama) |
| test_dependency_detection.py | 25 | Relationship detection, LLM integration |
| test_loader.py | 23 | Web content loading, chunking, sentence splitting |
| test_embeddings.py | 17 | Embedding, indexing, search, deferred FAISS rebuild |
| test_basic.py | 7 | Graph, confidence, serialization |
| + integration & calibration | 115 | Live Ollama, real datasets, BM25 retrieval calibration |
