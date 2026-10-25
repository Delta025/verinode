# Engram -- Development Roadmap

## Current Status

**All 15 development steps complete.** Core system fully built, tested (797 unit tests, 106 integration/calibration across 34 files), and stress-tested on scale graphs (1K--3K nodes across 6 domains). 214K lines of code across 20 core modules, 21 CLI commands, 30 API endpoints, and 8 web UI pages.

**Recent additions:**
- **Gemini backend** -- Alternative LLM backend via `ENGRAM_CHECKER_BACKEND=gemini`. F1=0.900 (vs Ollama 0.842), 0.95s/call (vs ~25s).
- **Multi-pair batching** -- Packs 10 contradiction/relevance checks into one Gemini call. 72% fewer API calls, 91% cheaper for bulk ingestion.
- **Web ingestion pipeline** -- `scripts/ingest_sources.py` fetches URLs, chunks, and ingests with rate-limited Gemini calls. Resume support, cost tracking.
- **False positive fix** -- Pipeline now uses 3-way relevance classification (`check_relevance_batch`) instead of binary contradiction checks. Confidence threshold raised from 0.5 to 0.7. Few-shot negative examples added to prompts. `rebuild_contradictions()` method for re-checking existing graphs.
- **Content pre-filters** -- Content-word overlap gate, Jaccard similarity threshold, unit normalization (`unit_normalizer.py`), and NLI cross-encoder pre-filter (`nli_filter.py`). Reduced false positive conflicts by 80.6% (1,106 to 215 unique pairs) and cut API calls by 49%.

---

## Completed: False Positive Contradiction Rate Fix

**Problem:** Analysis of a 2,247-node ingested graph revealed a 92.4% false positive rate in contradiction detection -- only 42/553 conflict pairs were genuine contradictions. Categories: complementary facts (46.8%), different subjects (33.1%), temporal (6.9%), numerical rounding (5.6%), genuine (7.6%).

**Changes implemented:**

| Change | Description | Status |
|--------|-------------|--------|
| D. Prompt improvements | Few-shot negative examples in CONTRADICTION_SYSTEM_PROMPT and RELEVANCE_SYSTEM_PROMPT | Done |
| A. Pipeline rewire | `_find_contradictions()` uses `check_relevance_batch` with `hasattr` fallback | Done |
| B. Threshold raise | `llm_contradiction_confidence` 0.5 -> 0.7 | Done |
| E. Rebuild utility | `Pipeline.rebuild_contradictions()` -- re-check existing graph without re-extraction | Done |
| F. Content overlap gate | `min_content_overlap=1` filters candidates with zero shared content words | Done |
| G. Jaccard threshold | `min_jaccard_threshold=0.05` filters very-low-overlap pairs | Done |
| H. Unit normalization | `unit_normalizer.py` canonicalizes measurements (miles->km, °F->°C, etc.) before comparison | Done |
| I. NLI pre-filter | `nli_filter.py` uses local cross-encoder to classify pairs before LLM | Done |

**Results on 2,247-node solar system graph:**

| Stage | Unique Conflict Pairs | API Calls | Time | Cost |
|-------|----------------------|-----------|------|------|
| Original (binary check) | 553 | -- | -- | -- |
| + 3-way relevance + prompts (A,B,D) | 444 | 3,980 | 3.9h | $0.10 |
| + Content pre-filters (F,G,H) | 215 | 2,036 | 1.2h | $0.05 |
| **Total reduction** | **-61.1%** | **-49%** | **-69%** | **-50%** |

**Remaining:** ~30-50% estimated false positive rate in remaining 215 pairs.

---

## Roadmap

### Phase 1: Productize (Packaging & Deployment)

Everything works -- the gap is packaging, not features. This phase turns a research project into something others can install and run.

**1a. Python packaging**
Add `pyproject.toml` with proper metadata, dependencies, and entry points (`engram` for the CLI, `engram-server` for the API). This enables `pip install engram` and publishing to PyPI. Include a `py.typed` marker for downstream type checking. Set initial version to `0.1.0` with semantic versioning going forward.

**1b. Docker containerization**
Create a `Dockerfile` and `docker-compose.yml` that bundles the API server, pre-built web frontend, and an Ollama sidecar container. A single `docker compose up` should give users a fully working system with no manual setup. Include volume mounts for graph persistence and an `.env.example` with all configuration options documented.

**1c. CI/CD pipeline**
Add GitHub Actions workflows: run the 797 unit tests on every push, lint with ruff, type-check with mypy. Add a release workflow that builds and publishes to PyPI on tagged commits. This catches regressions early and automates the release process.

**1d. Startup validation & observability**
Add configuration validation on startup (check Ollama reachability, validate env vars, verify graph file permissions). Add structured logging (replace print statements with Python logging module). Add a `/health` endpoint that reports component status. Add a `CHANGELOG.md` and maintain it with each release.

### Phase 2: Close the Retrieval Ceiling

The E2E pipeline flags 48.3% of contradictions vs 80% standalone LLM recall. This 31.7pp gap is the single biggest accuracy lever -- closing it would make Engram's verification credible at production scale.

**2a. Extraction prompt optimization**
The 7.9% extraction fidelity improvement from Optuna tuning suggests more gains are available. Systematically test extraction prompt variants across domains, measure fact completeness and hallucination rate, and tune with Optuna. Target: extraction fidelity > 0.85 (currently 0.80). Key focus areas: numerical precision, relationship extraction, and entity disambiguation.

**2b. Multi-pass retrieval**
Run verification in two passes with different strategies. First pass: embedding-based retrieval (catches semantic similarity). Second pass: BM25 term retrieval (catches exact entity matches the embeddings miss). Union the candidate sets before LLM verification. Current RRF fusion helps but a two-pass approach with separate LLM calls per pass could catch cases where the merged candidate list drowns relevant results.

**2c. Chunk-level indexing**
Currently facts are indexed as whole extracted statements. Index at sub-fact granularity so that a fact about "Mars orbital period" retrieves related facts about "Mars distance from Sun" that share the entity but not the exact claim. This requires splitting extracted facts into entity-claim pairs and indexing both the full fact and its constituent claims.

**2d. Domain-adapted embeddings**
Fine-tune the bi-encoder on actual graph data using contrastive learning. Use confirmed contradictions as positive pairs and confirmed irrelevant pairs as hard negatives. Even light fine-tuning (1-2 epochs on graph-derived pairs) typically improves domain-specific retrieval by 10-20%. Evaluate with retrieval recall@k metrics on held-out contradiction pairs.

### Phase 3: LLM Verification Middleware

This is the killer use case. Everyone building with LLMs worries about hallucination. A drop-in verification layer that demonstrably catches 90% of false claims has real value.

**3a. Claim extraction from LLM output**
Build a lightweight claim extractor that identifies verifiable factual statements in free-text LLM responses. Not every sentence is a claim -- filter for statements that make falsifiable assertions (numbers, dates, named entities, causal relationships). Reuse the existing `FactExtractionEngine` with a simplified prompt tuned for claim detection rather than full knowledge extraction.

**3b. Real-time verification API**
Add a `/api/verify-response` endpoint that accepts an LLM response string, extracts claims, verifies each against the graph, and returns an annotated response with per-claim confidence scores and flags. Target latency: < 5s per response using Gemini backend (vs 60-97s with Ollama). Batch claim verification to minimize LLM round-trips.

**3c. Middleware integration patterns**
Build integration examples for common LLM frameworks: OpenAI-compatible API proxy (sits between client and LLM, annotates responses), LangChain/LlamaIndex tool (LLM calls Engram to self-verify), and a simple Python decorator (`@engram_verified`) that wraps any function returning LLM text. Ship these as examples in an `integrations/` directory.

**3d. Streaming verification**
For long-form LLM output, verify claims as they arrive rather than waiting for the full response. Buffer sentences, extract claims from complete sentences, verify in parallel with continued generation, and surface flags in near-real-time. This requires async architecture and careful batching to balance latency vs throughput.

### Phase 4: Domain Knowledge Packs

The "novel domain = no protection" problem is the system's biggest fundamental limitation. The chemistry slab benchmark proved that 109 high-quality nodes matched 183 mixed-quality nodes -- source quality matters more than quantity.

**4a. Curated source configs**
Create JSON ingestion configs for high-value domains: medicine (PubMed, WHO guidelines), law (case law databases, statutes), finance (SEC filings, central bank data), and general science (Wikipedia featured articles, textbook sources). Each config includes URLs, quality scores, and domain tags. Users run `engram ingest --config medicine.json` to bootstrap a domain.

**4b. Pre-built graph distribution**
Build and publish pre-ingested graph files (SQLite `.db` format) for common domains. Host on GitHub Releases or a dedicated CDN. Users download a graph and start verifying immediately with zero LLM cost. Include metadata: node count, domain coverage, source list, build date, verification stats.

**4c. Domain-specific extraction prompts**
The generic extraction prompt works across domains but misses domain-specific patterns (chemical formulas, legal citations, financial figures). Create per-domain prompt templates that improve extraction fidelity for specialized content. Package these with the domain configs so the right prompt is used automatically.

**4d. Coverage metrics and gap reporting**
Add tooling to measure what a graph covers and where it has gaps. Given a domain taxonomy (e.g., medical specialties), report which areas have dense coverage vs sparse. This helps users understand what their graph can and can't verify, and feeds into curiosity-driven acquisition (Phase 6).

### Phase 5: Curiosity-Driven Knowledge Acquisition

A knowledge graph that identifies its own blind spots and fills them moves from a static reference to an adaptive system.

**5a. Gap detection**
Track verification queries that return low-confidence or zero results. Cluster these by topic to identify systematic knowledge gaps rather than one-off misses. Expose gaps via the API and CLI (`engram gaps --domain science`). Rank gaps by frequency and impact (how many verification requests would benefit from filling this gap).

**5b. Research query generation**
Given a knowledge gap, generate targeted search queries to fill it. Use the LLM to formulate queries based on what the graph knows nearby (e.g., if we know about Mars's atmosphere but not its magnetic field, generate "Mars magnetic field properties"). Filter for queries likely to yield high-quality, verifiable facts.

**5c. Semi-automated ingestion**
Fetch candidate sources for generated queries, extract facts, and present them for human review before adding to the graph. The human-in-the-loop step is critical -- fully automated ingestion risks importing false information. Provide a review UI showing extracted facts, sources, confidence estimates, and potential conflicts with existing knowledge.

**5d. Source integration APIs**
Build connectors for structured knowledge sources: PubMed (biomedical literature), arXiv (scientific preprints), Wikidata (structured facts), and Wikipedia (general knowledge). These provide higher-quality starting points than web scraping and come with built-in provenance metadata.

### Phase 6: Federated Trust Networks (Research)

Multiple Engram instances cross-validating each other addresses the "coherent falsehood" problem -- if one graph is poisoned, others can catch it.

**6a. Cross-instance verification protocol**
Define a protocol where one Engram instance can submit a claim to another for verification without sharing raw graph data. The responding instance returns a verdict (supported/contradicted/unknown) with a confidence score. This preserves data privacy while enabling collaborative truth-checking.

**6b. Trust scoring between instances**
Track agreement history between instances. Instances that consistently agree on verified facts build mutual trust; instances that frequently disagree on well-established facts lose trust. Use trust scores to weight cross-instance verdicts. This creates a reputation system for knowledge sources.

**6c. Conflict resolution**
When instances disagree, surface the conflict with evidence from both sides. Let humans or a meta-verification process resolve disputes. Track resolution outcomes to improve future trust scores. Handle edge cases: instances with different temporal knowledge, domain-specific disagreements, and deliberately adversarial instances.

**6d. Adversarial robustness**
Defense against coordinated false information campaigns targeting the federated network. Detect patterns like multiple new instances simultaneously contradicting established facts, Sybil attacks (one actor creating many instances), and gradual drift attacks that slowly shift consensus. Borrow from Byzantine fault tolerance literature.

### Ongoing: Visualization & UX Improvements

**Graph visualizer enhancements**
The current hub-based LOD system works for graphs up to ~2K nodes. Improvements: smoother zoom-based LOD transitions, semantic clustering (group by domain/topic rather than just degree), expandable cluster nodes with animation, WebGL rendering via `cytoscape-webgl` for 10K+ node graphs, search/filter within the visualization, and minimap navigation.

**Temporal knowledge**
Track how facts change over time, handle superseded knowledge. Add timestamps to nodes, support "valid from/until" ranges, and visualize knowledge evolution.

**Explainable AI for high-stakes domains**
Format reasoning chains into structured reports for legal/medical/scientific review. Include confidence bounds, source provenance, and alternative interpretations.

---

## Key Results

| Metric | Value |
|--------|-------|
| Verification accuracy | 90% false caught, 90% true correct (245-node graph) |
| Poisoned context resistance | +80pp over bare LLM |
| Subtle numerical errors | +60pp over bare LLM |
| Scale | 60% accuracy at 1K nodes, 80% at 3K nodes |
| Gemini batch ingestion (4 sources) | 2,247 nodes, 3,719 API calls, $0.09, 3.9h |
| False positive reduction | 92.4% -> ~30-50% FP rate, 553 -> 215 conflict pairs (-61%) |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed session data.

---

## Development History

| Step | Description | Status |
|------|-------------|--------|
| Phase 1--3 | Foundation, intelligence, Ollama migration | Done |
| Step 2--6 | Ingestion pipeline, contradiction detection, CLI, SQLite storage, calibration framework | Done |
| Step 7 | Calibration tuning (qwen3:14b, Optuna, F1 0.355->0.842) | Done |
| Step 7.5 | Retrieval improvement (entity overlap, contradiction-first ordering, batch parallelism) | Done |
| Step 8 | Verification API (claim verification + retraction simulation) | Done |
| Step 9 | Web UI (FastAPI + React/Cytoscape.js, 8 pages, 32 routes) | Done |
| Step 10 | LLM Memory Substrate (structured memory API, export/import) | Done |
| Step 11 | Fix verification engine (3-way relevance, weighted evidence, tiebreaker) | Done |
| Step 12 | Demo experience (benchmark, presentations, AI safety primer) | Done |
| Step 13 | Graph merge + large-scale testing infrastructure | Done |
| Step 14+ | Cross-encoder re-ranker, source quality weighting, neuroplasticity engine | Done |
| Step 15 | Scale testing (1K--3K nodes), text import pipeline | Done |
| Post-15 | Gemini backend, multi-pair batching, web ingestion, false positive fix, content pre-filters | Done |
| Post-15b | Web UI improvements: graph visualizer LOD system (fcose + hub-based clustering + zoom LOD), file browser, anomaly review, node CRUD endpoints | Done |

---

## Architecture Reference

### Scaling Considerations

| Bottleneck | Complexity | Triggers At | Fix |
|------------|:----------:|:-----------:|-----|
| Entity retrieval | O(k) via inverted index | Done | `EntityIndex` |
| FAISS search | O(N) brute-force | ~100K nodes | IndexIVFFlat or IndexHNSW |
| FAISS rebuild | O(N) deferred to batch end | Done | `flush_index()` |
| Graph startup | O(N) from SQLite | ~100K nodes | Lazy loading |
| LLM calls | 10-30s/call (Ollama) | Always | Gemini API or vLLM |

### Truth Verification Strategy

1. **Seed** with vetted, high-quality sources (high source quality scores)
2. **Gate** new inputs via 5 verification layers: source quality gating, LLM contradiction detection, ingestion gating (dedup + flagging), convergent evidence, anomaly detection
3. **Grow**: More facts create more contradiction surfaces, improving detection

### Known Limitations
- Novel domains: no facts -> no verification ability
- Subtle errors: near-correct values may pass checks
- Coherent false systems: internally-consistent false claims pass all checks
- LLM fallibility: the LLM itself may fail to detect contradictions
