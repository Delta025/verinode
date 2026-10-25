# Verinode — System Architecture

This document describes the implemented architecture of Verinode. All class names, method signatures, and behaviors described here reflect the current codebase.

## High-Level Data Flow

```
Text Input + Source metadata
    |
    v
Pipeline.ingest()  (orchestrator — end-to-end with verification)
    |
    | 1. Extract
    v
FactExtractionEngine  (LLM: Ollama, local, JSON mode)
    |  Extracts structured facts as JSON
    v
List[ExtractedFact]
    |
    | 2. Quality filter (drop low-confidence extractions)
    |
    | 3. Per-fact verification:
    |    a. Dedup check (embedding cosine similarity >= 0.92)
    |       → If duplicate: noisy-OR confidence boost on existing node
    |    b. Contradiction check (hybrid retrieval + multi-stage filtering)
    |       → Phase 0: Embedding search (FAISS/numpy cosine similarity)
    |       → Phase 0.5: BM25 search (IDF-weighted term matching via EntityIndex)
    |       → Phase 0.5a: Reciprocal Rank Fusion (RRF, k=60) merges both lists
    |       → Phase 0.75: Cross-encoder reranker (optional, top_n candidates)
    |       → Phase 0.9: Content pre-filter (word overlap)
    |       → Phase 1: Heuristic pre-filter (negation words, numerical mismatches)
    |       → Phase 2: LLM verification via qwen3:14b or Gemini 2.0 Flash
    |       → If contradiction: flag (CONTESTED) or reject
    |    c. Source quality gating (cap confidence if source quality < 0.5)
    |       → If low-quality source: PROVISIONAL node
    v
KnowledgeGraph.add_node()  (DAG container)
    |
    | 4. Dependency detection for new nodes
    v
DependencyDetector.detect_for_node()  (heuristic + LLM relationship classification)
    |  Classifies: DEPENDS_ON, SUPPORTS, CONTRADICTS, ELABORATES
    v
DependencyDetector.apply_dependencies()  (wires edges into graph)
    |
    | 5. Confidence propagation
    v
ConfidenceEngine.propagate_confidence_update()  (updates confidence scores)
    |
    v
IngestionResult  (summary of what happened)
    |
    v
ReasoningEngine.query() / forward_chain() / backward_chain()
    |  Multi-hop inference with confidence tracking
    v
QueryResult / ReasoningChain / HypothesisResult
```

The `Pipeline` class orchestrates all of these steps in a single `ingest()` call. Individual modules can still be used independently for more control.

---

## Module Details

### node.py (KnowledgeNode)

The fundamental data unit. Each node represents one atomic piece of knowledge.

```python
class KnowledgeNode:
    id: str                     # UUID, auto-generated
    text: str                   # Human-readable statement
    node_type: NodeType         # FACT, RULE, OBSERVATION, INFERENCE, DEFINITION
    confidence: float           # [0.0, 1.0]
    domain: str                 # e.g. "chemistry", "physics", "general"
    status: NodeStatus          # ESTABLISHED, WELL_SUPPORTED, PROVISIONAL, CONTESTED, SUPERSEDED, FALSIFIED
    sources: List[Source]       # Provenance records
    derivation: Derivation      # How this node was obtained (method, inference_depth, derivation_type)
    predecessor_nodes: List[str]  # IDs of nodes this depends on
    successor_nodes: List[str]    # IDs of nodes that depend on this
    conflicts_with: List[str]     # IDs of contradicting nodes
    metadata: Dict[str, Any]      # Arbitrary key-value pairs
    created_at: datetime
    last_validated: datetime
    access_count: int
```

Supporting data classes:
- `Source`: source_type, title, author, date, url, quality_score [0,1], metadata
- `Derivation`: method (str), inference_depth (int), derivation_type (str), source_nodes (list), notes (str)

Key methods:
- `add_source(source)`, `add_predecessor(node_id)`, `add_successor(node_id)`
- `to_dict()` / `from_dict()` for serialization
- `update_status()` - auto-updates status based on confidence thresholds

### graph.py (KnowledgeGraph)

DAG container with fast lookup indexes.

```python
class KnowledgeGraph:
    nodes: Dict[str, KnowledgeNode]   # id -> node
    _domain_index: Dict[str, Set[str]]
    _type_index: Dict[str, Set[str]]
    _status_index: Dict[str, Set[str]]
    _contradictions: Set[FrozenSet[str]]
```

Key methods:
- `add_node(node)` / `remove_node(node_id)` - maintains all indexes
- `get_node(node_id)` - returns KnowledgeNode or None
- `get_nodes_by_domain(domain)` / `get_nodes_by_type(node_type)` / `get_nodes_by_status(status)`
- `add_contradiction(id_a, id_b)` / `get_contradictions()` / `find_potential_contradictions(node)`
- `update_node_confidence(node_id, new_confidence)` - updates confidence and auto-updates status
- `save(filepath)` / `KnowledgeGraph.load(filepath)` - JSON serialization
- `get_statistics()` - returns dict with `total_nodes`, `nodes_by_domain`, `nodes_by_type`, `nodes_by_status`, `avg_confidence`, `nodes_with_conflicts`

### confidence.py (ConfidenceEngine)

Handles confidence propagation through the dependency graph.

```python
class ConfidenceEngine:
    def __init__(self, config: dict = None)
    # config keys: inference_strength (0.9), depth_decay (0.95),
    #   convergence_boost (1.2), min_confidence (0.1), anomaly_threshold (0.15)
```

Key methods:
- `calculate_node_confidence(node, graph)` - computes confidence from predecessors and sources
- `propagate_confidence_update(graph, node_id)` - BFS propagation to all successors
- `detect_anomalies(graph)` - finds nodes where actual confidence deviates from expected
- `trace_confidence_source(graph, node_id)` - traces reasoning path back to root sources
- `estimate_impact(graph, node_id)` - estimates how many nodes would be affected by a change

Confidence formulas:
- **Chain reasoning**: `confidence <= min(predecessor confidences) * inference_strength`
- **Convergent evidence**: `confidence = 1 - product(1 - predecessor_i)` (noisy-OR)
- **Depth decay**: `decay = depth_decay ^ inference_depth`
- **Source-based**: weighted average of source quality scores

### embeddings.py (EmbeddingEngine)

Semantic vector search using sentence-transformers.

```python
class EmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2")
    # Model loaded lazily on first use
    # FAISS optional: if not installed, uses numpy brute-force cosine similarity
```

Key methods:
- `embed_text(text)` / `embed_texts(texts)` - returns numpy arrays (384 dimensions)
- `embed_node(node)` - embeds a KnowledgeNode's text
- `index_graph(graph)` - builds vector index from all nodes in a graph
- `add_to_index(node)` / `remove_from_index(node_id)` - incremental index updates
- `search(query_text, k=5, threshold=0.0)` - semantic search by text string
- `find_similar(node_id, k=5, threshold=0.0)` - find nodes similar to a given node
- `cosine_similarity(vec_a, vec_b)` - static utility method

Internal state:
- `_embeddings: np.ndarray` - matrix of all indexed embeddings
- `_id_to_position: Dict[str, int]` - maps node IDs to matrix rows
- `_position_to_id: Dict[int, str]` - reverse mapping
- `_faiss_index` - optional FAISS index (rebuilt on changes)
- `index_size: int` - property returning number of indexed vectors

### fact_extraction.py (FactExtractionEngine)

Extracts structured facts from text using a local Ollama LLM.

```python
class FactExtractionEngine:
    DEFAULT_MODEL = "qwen3:14b"
    def __init__(self, model=None, host=None, temperature=0.0, num_ctx=4096)
    # Uses Ollama with JSON mode for structured output
    # Requires a running Ollama server with a pulled model
    # Qwen3 thinking mode explicitly disabled (think=False) for 2-5x speedup on structured tasks
```

**Calibrated performance** (Feb 2026, 200 ARC science sentences): Mean fidelity 0.800, 0% hallucination rate, 1.87 facts/sentence.

Key methods:
- `extract(text, domain=None)` - sends text to LLM, returns ExtractionResult
- `extract_to_nodes(text, domain=None, source=None)` - extract + convert to KnowledgeNode list
- `facts_to_nodes(result, source=None)` - convert ExtractionResult to nodes with predecessor wiring

Data classes:
- `ExtractedFact`: text, fact_type, confidence, domain, related_facts (list of int indices), source_quote, metadata
- `ExtractionResult`: facts (list), source_text, model, domain, raw_response

The LLM is prompted to return a JSON array where each element has: text, fact_type (fact/rule/observation/inference/definition), confidence (0.0-1.0), domain, related_facts (indices into same batch), source_quote.

### llm_contradiction.py (ContradictionChecker)

LLM-assisted contradiction detection using Ollama. Provides semantic contradiction checking that goes beyond the heuristic negation-word approach.

```python
class ContradictionChecker:
    DEFAULT_MODEL = "qwen3:14b"
    def __init__(self, model=None, host=None, temperature=0.0, num_ctx=1024)
    # Uses Ollama with JSON mode for structured output
    # Qwen3 thinking mode disabled (think=False) for 2-5x speedup on structured tasks
    # Enhanced system prompt with explicit contradiction types, examples, and anti-false-positive rules
```

Key methods:
- `check_relevance(claim, knowledge)` - 3-way classification of (claim, knowledge) relationship. Returns `RelevanceResult` with relationship = "contradicts"/"supports"/"irrelevant". Used by Pipeline and VerificationEngine.
- `check_relevance_batch(pairs, max_workers=2)` - batch 3-way classification. Ollama uses ThreadPoolExecutor parallelism; Gemini uses sequential calls.
- `clear_cache()` - clears the `_relevance_cache`.

**Calibrated performance** (Feb 2026, 150 FEVER pairs): F1=0.842, Precision=88.9%, Recall=80%. Detects semantic (29), numerical (7), and negation (9) contradiction types.

Factory function:
- `create_contradiction_checker(backend="ollama"|"gemini", gemini_api_key=None)` - returns checker or `None` if backend unavailable. Selects backend via `VERINODE_CHECKER_BACKEND` env var.

Data class:
- `RelevanceResult`: `claim`, `knowledge`, `relationship` ("contradicts"/"supports"/"irrelevant"), `confidence` (0.0-1.0), `explanation` (str). Used by Pipeline and VerificationEngine for 3-way evidence classification.

**Gemini backend**: `GeminiContradictionChecker` is a drop-in replacement using `google-genai` SDK. Same `check_relevance`/`check_relevance_batch` interface. Sequential batch processing (rate-limited).

Contradiction types detected:
1. **Negation**: One statement negates the other ("X is true" vs "X is not true")
2. **Numerical**: Incompatible numerical values for the same property ("freezes at 0°C" vs "freezes at -100°C")
3. **Semantic**: Semantically incompatible claims without negation words ("safe to drink" vs "toxic")
4. **Logical**: Logically invalid inferences ("boils at 100C" does NOT imply "cannot freeze unless below -100C")

Anti-false-positive rules built into the system prompt:
- Different properties of the same subject are NOT contradictions
- A more specific statement does not contradict a more general one
- Only flags contradictions where two statements CANNOT both be true simultaneously

### dependency_detection.py (DependencyDetector)

Detects relationships between nodes using linguistic heuristics, optionally enhanced with LLM-assisted contradiction detection.

```python
class DependencyDetector:
    def __init__(self, similarity_threshold=0.698,
                 dependency_confidence_threshold=0.4,
                 embedding_engine=None)
    # similarity_threshold calibrated via Optuna (was 0.35, now 0.698)
```

Key methods:
- `detect_all(graph, embedding_engine=None, contradiction_checker=None)` - detect dependencies across all node pairs
- `detect_for_node(node, graph, embedding_engine=None, k=10, contradiction_checker=None)` - detect for one node
- `apply_dependencies(dependencies, graph)` - wire DetectedDependency objects into graph edges

Relationship types (RelationType enum):
- `DEPENDS_ON` - B logically depends on A (A is prerequisite for B)
- `SUPPORTS` - A provides evidence for B
- `CONTRADICTS` - A and B make conflicting claims
- `ELABORATES` - B adds detail to A

Classification pipeline (`_classify_relationship`):
1. Contradiction check:
   - Heuristic: negation mismatch on similar content (score > 0.5)
   - LLM override (when `contradiction_checker` provided and similarity > 0.3): LLM verdict overrides heuristic — catches numerical/semantic/logical contradictions the heuristic misses and removes heuristic false positives
2. Definitional dependency: source is a definition, target references its concept
3. Causal dependency: causal language ("because", "therefore", etc.) in similar texts
4. Entity overlap: 2+ shared non-stop-words, classified by relative text length
5. Same-domain support: moderate similarity (> 0.25) in the same domain

Candidate selection:
- With embeddings: k-NN search via EmbeddingEngine
- Without embeddings: brute-force Jaccard word overlap

Known limitations:
- Jaccard similarity is fragile with short or dissimilar texts
- Heuristic contradiction detection catches negation words and numerical mismatches; LLM fallback covers the rest when available
- Stage 4 (entity overlap) can produce false positives in densely related domains

### reasoning.py (ReasoningEngine)

Multi-hop inference over the knowledge graph.

```python
class ReasoningEngine:
    def __init__(self, max_depth=10, min_confidence=0.1,
                 inference_strategy=InferenceStrategy.WEAKEST_LINK)
```

Key methods:
- `forward_chain(graph, premise_id, max_depth=None)` - follow successors from a premise, returns ReasoningChain
- `backward_chain(graph, conclusion_id, max_depth=None)` - follow predecessors from a conclusion, returns ReasoningChain
- `query(graph, question, embedding_engine=None, k=10)` - find relevant nodes and build reasoning chains, returns QueryResult
- `test_hypothesis(graph, hypothesis_text, embedding_engine=None, k=10)` - aggregate evidence for/against, returns HypothesisResult
- `explain_node(graph, node_id)` - generate human-readable explanation of a node's confidence and provenance

Inference strategies (InferenceStrategy enum):
- `WEAKEST_LINK` - chain confidence = min(step confidences) * strength
- `PRODUCT` - chain confidence = product(step confidences)
- `AVERAGE` - chain confidence = mean(step confidences)

Data classes:
- `ReasoningStep`: node_id, text, confidence, step_type (premise/intermediate/conclusion)
- `ReasoningChain`: steps (list), overall_confidence, strategy_used
- `QueryResult`: question, chains (list), best_answer (str or None), confidence, relevant_nodes
- `HypothesisResult`: hypothesis, supporting_chains, contradicting_chains, confidence, verdict (supported/contradicted/uncertain/insufficient_evidence)

### pipeline.py (Pipeline)

End-to-end ingestion with truth verification (all 5 Layers).

```python
class Pipeline:
    def __init__(self, graph, extractor=None, embedding_engine=None,
                 dependency_detector=None, confidence_engine=None,
                 contradiction_checker=None, reranker=None, entity_index=None, *,
                 duplicate_threshold=0.87, contradiction_threshold=0.52,
                 min_extraction_confidence=0.46, source_quality_cap=0.54,
                 reject_contradictions=False, use_llm_contradiction=True,
                 llm_contradiction_confidence=0.7,
                 verify_quotes=True, quote_penalty=0.7, min_quote_overlap=0.6,
                 use_tiered_ingestion=False, detect_dependencies=True,
                 min_content_overlap=1, use_unit_normalization=True, use_bm25=True)
    # All sub-engines default to standard instances if not provided
    # contradiction_checker auto-created via factory if use_llm_contradiction=True
    # Threshold defaults are Optuna-optimized (10-trial Bayesian optimization, Feb 2026)
```

Key methods:
- `ingest(text, source=None, domain=None)` - Ingest one passage: extract → verify → add → wire → propagate. Returns `IngestionResult`.
- `ingest_batch(texts, source=None, domain=None)` - Ingest multiple passages sequentially (later benefits from earlier). Returns `BatchIngestionResult`.
- `build_index()` - (Re)build embedding index from all graph nodes. Call after loading a saved graph.

Data classes:
- `IngestionVerdict`: Per-fact result with `action` ("added", "duplicate", "flagged", "rejected"), `node_id`, `reason`, `contradiction_node_ids`, `confidence_before`/`confidence_after` (for duplicates).
- `IngestionResult`: Per-passage result with `verdicts` list, `nodes_added`, `duplicates_merged`, `facts_flagged`, `facts_rejected`, `dependencies_added`, `nodes_propagated`, `summary` property.
- `BatchIngestionResult`: Multi-passage result with `results` list, `total_added`, `total_duplicates`, `total_flagged`, `total_rejected` properties.

Internal per-fact pipeline (`_process_fact`):
1. **Contradiction check** (runs first to prevent dedup from "stealing" contradiction cases):
   - Phase 0: Embedding search (top-8 by cosine similarity, threshold 0.45)
   - Phase 0.5a: Entity-overlap retrieval — finds nodes sharing >=2 non-stopword content words (`_find_entity_matches()`). Catches contradictions where embedding similarity is low.
   - Phase 0.5b: (Optional) BM25 + RRF hybrid retrieval when `use_bm25=True`.
   - Phase 0.75: (Optional) Cross-encoder reranker.
   - Phase 0.9: Content pre-filter — word overlap gate (`min_content_overlap`).
   - Phase 1: Heuristic pre-filter — negation word mismatch (fast).
   - Phase 2: LLM 3-way relevance via `check_relevance_batch` — only `relationship="contradicts"` triggers conflict handling.
   - If contradiction found, flag or reject. Verdict reason includes LLM explanation.
2. **Dedup**: Search for near-identical existing node (cosine similarity >= `duplicate_threshold`). If found, merge via noisy-OR boost. Only runs if no contradiction was detected.
3. **Normal add**: Create node. If source quality < `source_quality_cap`, cap confidence.

Configuration (defaults are Optuna-optimized):
- `duplicate_threshold` (default 0.87): Cosine similarity above which two facts are considered duplicates
- `contradiction_threshold` (default 0.52): Confidence above which contradicting existing nodes trigger flagging
- `min_extraction_confidence` (default 0.46): Minimum LLM-assigned confidence to accept a fact
- `source_quality_cap` (default 0.54): Below this quality, source quality caps node confidence
- `reject_contradictions` (default False): If True, contradictions are rejected; if False, flagged as CONTESTED
- `use_llm_contradiction` (default True): If True and no `contradiction_checker` provided, auto-creates one via factory
- `llm_contradiction_confidence` (default 0.7): Minimum confidence to accept LLM contradiction verdict
- `verify_quotes` (default True): Verify source_quote against input text
- `use_tiered_ingestion` (default False): Enable source tier confidence caps
- `min_content_overlap` (default 1): Minimum content-word overlap for contradiction candidates
- `use_unit_normalization` (default True): Normalize units before contradiction checking
- `use_bm25` (default True): Enable BM25 + RRF hybrid retrieval
- `max_contradiction_candidates` (default 20): Total cap on contradiction candidates

**Calibrated E2E performance** (Feb 2026, 50 ARC seed + 50 FEVER test): 48.3% REFUTES flag rate, 0% SUPPORTS false flag rate.

**Retrieval strategies**: Multiple retrieval strategies are combined to maximize contradiction candidate recall:
- **Embedding search**: Top-8 by cosine similarity (threshold 0.45). Fast but misses contradictions with different wording.
- **Entity-overlap retrieval** (`_find_entity_matches()`): Via `EntityIndex` inverted index for O(k) lookup of nodes sharing non-stopword content words.
- **BM25 hybrid** (when `use_bm25=True`): BM25 search combined with embedding search via Reciprocal Rank Fusion (RRF, k=60).
- **Cross-encoder reranker** (optional): Reranks candidates for higher precision.
- **Content pre-filter**: Gates candidates by content-word overlap count.
- **Heuristic pre-filter**: Fast negation-word and numerical mismatch detection.
- **LLM 3-way verification**: All candidates checked via `check_relevance_batch` — contradicts/supports/irrelevant classification.

**Speed optimizations**: Qwen3 thinking mode disabled (`think=False`) for both extraction and contradiction checking (2-5x speedup per call). Contradiction `num_ctx` reduced from 2048 to 1024. Batch parallelism for concurrent Ollama inference.

### verification.py (VerificationEngine)

Read-only claim verification and retraction simulation against the knowledge graph.

**`verify(claim, k=10)` algorithm**:
1. **Candidate gathering**: Embedding search (top-k by cosine similarity) + entity-overlap retrieval (nodes sharing >=2 non-stopword content words). Capped at `max_candidates=20`.
2. **3-way evidence classification** (`_classify_evidence`): Routes to one of two paths:
   - **LLM 3-way** (`_llm_classify_3way`): Uses `check_relevance_batch()` to classify each candidate as contradicts/supports/irrelevant. Irrelevant candidates excluded from verdict. Evidence confidence weighted by similarity: `node.confidence * sim_score`.
   - **Heuristic** (`_heuristic_classify_with_relevance`): No LLM path. Similarity threshold (0.65) filters irrelevant candidates. Remaining candidates scored via `DependencyDetector._contradiction_score()`.
3. **Backward chains**: Build provenance chains for each evidence node via `ReasoningEngine.backward_chain()`.
4. **Verdict computation** (`_compute_verdict`): Compares max supporting vs max contradicting confidence with 0.05 margin. When within margin, count-based tiebreaker (side with more evidence wins). Returns `Verdict` enum (SUPPORTED/CONTRADICTED/UNCERTAIN/INSUFFICIENT_EVIDENCE) with confidence, explanation, coverage.

**`simulate_retraction(node_id)` algorithm**:
1. BFS from target node to find all descendants with hop distances.
2. Save original confidences for target + all descendants.
3. Set target node confidence to 0.
4. Recalculate descendant confidences in topological order via `ConfidenceEngine.calculate_node_confidence()`.
5. Record impact (delta per node), detect orphaned nodes.
6. **Restore** all original confidences — graph is left unchanged.

**Key data structures**: `Verdict` enum, `SourceEvidence`, `VerificationResult`, `RetractedNodeImpact`, `RetractionSimulation`.

**CLI**: `verify "claim" [--show-evidence]`, `what-if <node_id>`.
**API**: `POST /api/verify`, `POST /api/simulate-retraction`.
**Tests**: 43 unit tests covering verify, retraction simulation, relevance filtering, weighted evidence, verdict tiebreaker.

### llm_memory.py (LLMMemory)

Thin facade providing LLM-optimized access to the knowledge graph. All methods delegate to existing engines and reformat results into flat, JSON-serializable structures suitable for LLM consumption.

```python
class LLMMemory:
    def __init__(self, graph, embedding_engine, reasoning_engine,
                 confidence_engine, pipeline=None, verification_engine=None)
    # Wraps existing engines — no new logic, pure reformatting
```

Key methods:
- `query(question, domain=None, k=10, include_chains=False)` → `MemoryQueryResult` — queries graph via `ReasoningEngine.query()`, reformats into flat fact list with provenance. Includes coverage level (high/medium/low/none) and weighted overall confidence.
- `recall(topic, domain=None, k=5)` → `List[MemoryFact]` — lightweight retrieval via `EmbeddingEngine.search()` with keyword fallback. For LLM context injection without reasoning chains.
- `verify(claim)` → `Dict` — delegates to `VerificationEngine.verify()`, returns `{verdict, confidence, explanation, evidence_count}`.
- `ingest(text, source_title, source_quality=0.7, domain=None)` → `MemoryIngestResult` — submits text via `Pipeline.ingest()`, returns structured result with action (added/duplicate/contradicted/rejected), contradictions, and explanation.
- `to_dict()` → `Dict` — graph stats and capabilities summary for LLM context.

Data classes:
- `MemoryFact`: node_id, text, confidence, domain, sources (list of dicts), reasoning_chain (optional list of dicts)
- `MemoryQueryResult`: query, facts, summary, overall_confidence, coverage, total_graph_nodes
- `MemoryIngestResult`: action, node_id, confidence, contradictions, explanation

Coverage levels:
- `high`: ≥3 facts with average confidence ≥0.7
- `medium`: ≥1 fact with average confidence ≥0.4
- `low`: ≥1 fact with average confidence <0.4
- `none`: 0 facts

**Tests**: 28 unit tests covering query, recall, verify, ingest, coverage computation, and serialization.

### export.py (Sub-graph Export/Import)

Portable domain slices for sharing knowledge between Verinode instances.

```python
EXPORT_SCHEMA_VERSION = 1

def export_subgraph(graph, domain, confidence_threshold=0.5) -> SubGraphExport
def export_to_json(export) -> str
def import_from_json(json_str) -> SubGraphExport
def import_subgraph(graph, export, conflict_strategy="keep_higher_confidence") -> Dict
```

Key functions:
- `export_subgraph()` — filters nodes by domain and confidence threshold, preserves internal edges (predecessor/successor and conflict edges where both endpoints are in the export set). Returns `SubGraphExport` with schema version, metadata, and serialized nodes/edges.
- `export_to_json()` / `import_from_json()` — JSON serialization with schema version validation.
- `import_subgraph()` — imports into a target graph with conflict resolution. Exact text+domain dedup. Returns `{added, merged, skipped, conflicts}`.

Conflict strategies:
- `keep_higher_confidence`: higher confidence wins (default)
- `keep_existing`: existing node always wins
- `keep_imported`: imported node always wins

Data class:
- `SubGraphExport`: schema_version, domain, exported_at, source_graph, confidence_threshold, nodes, edges, node_count, edge_count

**Tests**: 23 unit tests covering export filtering, JSON round-trip, import conflict resolution, and edge rebuilding.

---

## Web UI (src/api/ + src/web/)

### Architecture Overview

```
Browser (React + Cytoscape.js)
    |
    | HTTP (fetch)
    v
FastAPI Backend (src/api/)
    |
    | Python function calls
    v
Core Modules (src/core/)
    |
    v
SQLite / JSON storage
```

The Web UI wraps the same core modules used by the CLI into a REST API, with a React frontend for interactive exploration.

### Backend: FastAPI (src/api/)

**`main.py`** — Application factory with:
- CORS middleware for Vite dev server (localhost:5173)
- Lifespan context manager: loads graph on startup from `VERINODE_GRAPH` env var, closes SQLite on shutdown
- API route registration at `/api/*` prefix
- Static asset serving at `/assets/*` for production builds
- SPA catch-all route: any non-API, non-file path serves `index.html` for React Router client-side routing

**`state.py`** — `AppState` class (adapted from `src/cli/state.py`):
- Lightweight components init eagerly: `KnowledgeGraph`, `ConfidenceEngine`, `ReasoningEngine`
- Heavyweight components lazy on first use: `EmbeddingEngine` (loads sentence-transformers), `Pipeline` (creates `FactExtractionEngine` + connects to Ollama), `ContradictionChecker` (Ollama, None if unavailable), `VerificationEngine`, `LLMMemory`
- Auto-detects `.db` vs `.json` graph files (same pattern as CLI)
- Module-level singleton set during app lifespan

**`schemas.py`** — 51 Pydantic models for request/response validation and automatic OpenAPI documentation.

**API Routes** (30 endpoints + health check):

| Method | Route | Maps to | Notes |
|--------|-------|---------|-------|
| GET | `/api/health` | — | Returns status, node count, model name |
| GET | `/api/stats` | `graph.get_statistics()` | Dashboard data |
| GET | `/api/graph/data` | Node/edge iteration | Cytoscape.js-format elements |
| GET | `/api/anomalies` | `confidence.detect_anomalies()` | Optional `threshold` query param |
| GET | `/api/history` | `graph.get_changelog()` | SQLite graphs only |
| GET | `/api/nodes` | Graph node iteration | Query params: domain, type, status, sort, limit, offset |
| GET | `/api/nodes/{id}` | `graph.get_node(id)` | Full node detail with sources |
| PATCH | `/api/nodes/{id}` | `graph.update_node()` | Body: confidence, domain, status, text (all optional) |
| DELETE | `/api/nodes/{id}` | `graph.remove_node()` | Deletes node and all associated edges |
| GET | `/api/nodes/{id}/explain` | `reasoning.explain_node()` | Confidence explanation |
| GET | `/api/nodes/{id}/trace` | `confidence.trace_confidence_source()` | Source trace |
| GET | `/api/nodes/{id}/neighbors` | `get_predecessors()` + `get_successors()` | For graph expansion |
| POST | `/api/ingest` | `pipeline.ingest()` | Body: text, source metadata, domain |
| POST | `/api/query` | `reasoning.query()` | Body: question, k |
| POST | `/api/hypothesis` | `reasoning.test_hypothesis()` | Body: hypothesis, k |
| POST | `/api/verify` | `verification.verify()` | Body: claim, k |
| POST | `/api/simulate-retraction` | `verification.simulate_retraction()` | Body: node_id |
| POST | `/api/memory/query` | `llm_memory.query()` | Body: question, domain, k, include_chains |
| POST | `/api/memory/recall` | `llm_memory.recall()` | Body: topic, domain, k |
| POST | `/api/memory/verify` | `llm_memory.verify()` | Body: claim |
| POST | `/api/memory/ingest` | `llm_memory.ingest()` | Body: text, source_title, source_quality, domain |
| GET | `/api/memory/export/{domain}` | `export_subgraph()` | Query: confidence_threshold |
| POST | `/api/memory/import` | `import_subgraph()` | Body: JSON export + conflict_strategy |
| POST | `/api/load-graph` | `KnowledgeGraph.load()` | Body: path; reloads graph at runtime |
| POST | `/api/save-graph` | `graph.save()` | Body: path; saves current graph |
| POST | `/api/merge-graph` | `merge_graphs()` | Body: path; merges another graph into current |
| GET | `/api/files/browse` | Filesystem listing | Query: dir, filter (graph/all); for file browser UI |
| POST | `/api/maintenance/run` | `neuroplasticity.run_cycle()` | Body: optional `now` override |
| GET | `/api/maintenance/doubt` | `neuroplasticity.flag_unsupported()` | Returns flagged nodes |
| POST | `/api/maintenance/strengthen` | `neuroplasticity.strengthen_pathway()` | Body: node_id |
| GET | `/api/maintenance/status` | Graph maintenance stats | Node counts by status, avg confidence |

### Frontend: React + TypeScript (src/web/)

**Technology**: React 19 + TypeScript + Cytoscape.js (via `react-cytoscapejs` + `cytoscape-fcose`) + Vite

**Components** (10 pages):

| Component | Purpose |
|-----------|---------|
| `Layout.tsx` | App shell with sidebar navigation, outlet for child routes |
| `Dashboard.tsx` | Stats overview: node counts, domain distribution, confidence histogram, graph health |
| `GraphView.tsx` | Interactive graph visualization with hub-based LOD, dynamic zoom clustering, and budget slider |
| `NodeList.tsx` | Filterable/sortable table of nodes with detail panel (explain, trace, metadata) |
| `IngestForm.tsx` | Text input + source metadata → POST /api/ingest with live result display |
| `QueryInterface.tsx` | Natural language query → reasoning chains display |
| `HypothesisTest.tsx` | Claim testing → verdict, confidence, supporting/contradicting chains |
| `AnomalyList.tsx` | Anomaly detection with threshold slider + change history tab |
| `AnomalyReview.tsx` | Anomaly review and resolution workflow with node editing |
| `FileBrowser.tsx` | File browser for loading, saving, and merging graph files |

**Graph Visualization** (`GraphView.tsx`):

The graph visualizer uses a hub-based Level-of-Detail (LOD) system to handle graphs from tens to thousands of nodes efficiently.

- **Layout engine**: `cytoscape-fcose` (Fast Compound Spring Embedder) — significantly faster than built-in COSE, with quality tiers:
  - `quality: "proof"` (≤80 nodes): Full iterative layout, highest quality
  - `quality: "default"` (80–200 nodes): Balanced speed/quality
  - `quality: "draft"` (200+ nodes): Spectral-only placement, near-instant for large graphs
  - `animate: false` for 100+ nodes to avoid blocking the main thread
- **Hub-based LOD**: Nodes are ranked by degree (connection count). A budget slider controls how many top-ranked nodes are visible. Hidden nodes are "absorbed" into their nearest visible hub neighbor, creating implicit clusters.
- **Edge rerouting**: Edges between absorbed (hidden) nodes are rerouted to their hub representatives. Deduplication uses separate keys for conflict edges (`|C`) and structural edges (`|S`) so both types remain visible between the same pair of nodes.
- **Dynamic zoom LOD**: When enabled ("Zoom LOD" toggle), the budget automatically adjusts based on the zoom level — zooming in increases the budget to show more detail, zooming out decreases it. Protected by feedback-loop guards (`layoutInProgressRef`, `zoomTriggeredRef`, 500ms cooldown, 15% hysteresis).
- **Node colors**: green (confidence ≥0.8), yellow (≥0.5), red (<0.5) — matches CLI `formatting.py`
- **Edge styles**: blue arrows for dependency edges, red dashed lines for contradiction edges
- **Node size**: scales with connection count (degree); cluster nodes show absorbed count
- **Performance optimizations**: `curve-style: 'straight'` (2× faster than bezier), `opacity: 1` (avoids compositing overhead), `pixelRatio={1}` (avoids 4× HiDPI overhead), 300ms debounce on budget slider
- **Interaction**: Click node to inspect in sidebar detail panel with full provenance; toolbar with Fit, Re-layout, All (show all nodes), Auto (enable zoom LOD) buttons
- **Topology pre-computation**: Connected components detected via BFS; edges filtered to only include valid endpoints to prevent crashes on stale graph data

**Development workflow**:
- `npm run dev` (Vite, port 5173) with proxy config forwarding `/api/*` to FastAPI (port 8000)
- `npm run build` creates `dist/` directory served by FastAPI in production
- Hot module replacement for frontend changes; `--reload` flag for backend changes

### Design Decisions

9. **FastAPI for Web UI**: Wraps the same core modules as the CLI. Route handlers are thin adapters that call core functions and convert results to Pydantic schemas. No business logic in the API layer.

10. **Lazy initialization for heavyweight components**: Same pattern as CLI — `EmbeddingEngine` and `Pipeline` are only created when a route that needs them is first called. Read-only endpoints (stats, node list, graph data) respond instantly without loading sentence-transformers or connecting to Ollama.

11. **SPA routing**: FastAPI serves the built React app in production. API routes at `/api/*` take priority; all other paths fall through to `index.html` for React Router client-side routing. Static assets (JS/CSS bundles) are mounted at `/assets/*`.

12. **Cytoscape.js for graph rendering**: Chosen over D3 for its built-in graph layout algorithms, node/edge styling DSL, and event handling. `react-cytoscapejs` wrapper manages the lifecycle. `cytoscape-fcose` provides fast force-directed positioning with quality tiers (draft/default/proof) for graphs ranging from tens to thousands of nodes. Hub-based LOD with dynamic zoom clustering handles large-scale visualization without blocking the main thread.

---

## Stress Test and Benchmark Results

A comprehensive suite of benchmarks validates system behavior at scale. Detailed results, methodology, and analysis for all benchmarks are consolidated in [benchmarks.md](benchmarks.md).

**Headline results**:
- **150-fact stress test**: 218 nodes from 150 facts, 39.2s/fact avg ingestion, 10/10 retrieval coverage, 218/218 export/import round-trip
- **Claim verification** (post-Step 11): 9/10 false claims caught (90%), 9/10 true claims correct, 13.1s/claim avg on 215-node graph
- **Vulnerability resistance benchmark** (75 tests, bare LLM vs LLM + Verinode): Poisoned context resistance +80pp (20%→100%), subtle numerical errors +60pp (20%→80%)
- **Calibration**: Contradiction F1=0.842, extraction fidelity=0.800, 0% hallucination, E2E flag rate 48.3% with 0% false positives

## Truth Verification: Current Capabilities and Gaps

A critical analysis of how each module contributes to (or fails to contribute to) preventing false information from entering the graph.

### What exists and works

| Mechanism | Module | What it does | Limitation |
|-----------|--------|-------------|------------|
| Source quality scores | `node.py` + `pipeline.py` | Every node has a `Source` with `quality_score` [0,1]. Pipeline enforces this as an active gate. | Low-quality sources cap confidence; doesn't verify factual accuracy |
| Confidence propagation | `confidence.py` | Scores flow through predecessor chains; low-confidence predecessors limit successors | Only constrains *derived* nodes; a false root fact with high confidence is unchecked |
| Anomaly detection | `confidence.py` | `detect_anomalies()` flags nodes where actual confidence deviates from expected | Only works *after* ingestion; cannot prevent bad data from entering |
| Heuristic contradiction detection | `dependency_detection.py` | `CONTRADICTS` relation detected via negation word mismatch | Only catches explicit negation words; use with LLM for full coverage |
| LLM 3-way relevance | `llm_contradiction.py` + `pipeline.py` | `ContradictionChecker` classifies fact pairs as contradicts/supports/irrelevant via Ollama or Gemini. Detects numerical, semantic, logical, and negation-based contradictions. | Requires LLM server; adds latency per comparison |
| Hypothesis testing | `reasoning.py` | `test_hypothesis()` aggregates supporting and contradicting evidence | Uses heuristic + optional LLM for contradiction identification |
| Convergent evidence | `confidence.py` | Noisy-OR combination boosts confidence when multiple independent sources agree | Works correctly, but only helps if the seed data is accurate |

### Trust bootstrapping strategy

See ROADMAP.md for the full truth verification strategy. In summary:
1. Seed the graph with vetted information from trusted sources
2. New inputs are checked against the seed graph via all 5 verification layers
3. Untrusted inputs enter as PROVISIONAL until corroborated
4. The system's reliability grows with the size and quality of its seed data

### storage.py (SqliteGraph)

SQLite-backed persistent storage that extends KnowledgeGraph via subclassing. Every mutation is automatically persisted to SQLite, while the in-memory state stays in sync.

```python
class SqliteGraph(KnowledgeGraph):
    def __init__(self, db_path: str)
    # Opens or creates a SQLite database at db_path
    # On construction: creates schema if needed, loads all data into memory
    # Uses WAL mode for concurrent read performance
```

Key overridden methods (each calls `super()` first, then persists to SQLite):
- `add_node(node)` — INSERT node + sources + edges + embedding + changelog entry
- `remove_node(node_id)` — DELETE with CASCADE + changelog entry
- `update_node_confidence(node_id, new_confidence)` — UPDATE confidence/status + changelog entry
- `add_contradiction(node_id1, node_id2)` — INSERT conflict + changelog entry

Additional methods:
- `close()` — close the database connection
- `get_changelog(limit=50)` — return recent change history from the changelog table
- `save(filepath)` — inherited JSON export (works for `.json` files); for `.db` files, no-op (already persisted)

Class methods:
- `SqliteGraph.open(db_path)` — open an existing SQLite graph
- `SqliteGraph.from_graph(graph, db_path)` — bulk-convert an in-memory KnowledgeGraph to SQLite

Conversion functions:
- `json_to_sqlite(json_path, db_path)` — convert a JSON graph file to a SQLite database
- `sqlite_to_json(db_path, json_path)` — export a SQLite graph to JSON format

SQLite schema (7 tables):
- `nodes` — core node data (id, text, type, confidence, domain, status, derivation, timestamps)
- `sources` — one-to-many sources per node (type, title, author, date, url, quality_score)
- `edges` — predecessor → successor relationships
- `conflicts` — bidirectional contradiction markers
- `embeddings` — numpy vectors stored as BLOBs
- `changelog` — version history with timestamp, change_type, node_id, details (JSON)
- `schema_version` — for future migrations

Design notes:
- **Zero changes to existing code**: Pipeline, DependencyDetector, ReasoningEngine all work unchanged via polymorphism
- **No new dependencies**: Uses Python's built-in `sqlite3`
- **Foreign key handling**: Successor edges check for existence before insert; `from_graph` disables FK checks during bulk insert to avoid ordering issues
- **WAL mode**: `PRAGMA journal_mode=WAL` for better concurrent read performance

---

## CLI (src/cli/)

The CLI wraps all core operations into 21 click subcommands with rich-formatted terminal output. Architecture:

### Modules
- **`main.py`**: Click command group with global `--graph` and `--model` options. 21 subcommands: `ingest`, `query`, `hypothesis`, `verify`, `what-if`, `memory-query`, `export`, `import-graph`, `merge-graphs`, `stats`, `list`, `inspect`, `explain`, `trace`, `anomalies`, `save`, `load`, `history`, `maintain`, `doubt`, `strengthen`.
- **`state.py`**: `EngineState` class managing shared state across commands. Auto-detects `.db` vs `.json` graph files — `.db` files use `SqliteGraph` with automatic persistence, `.json` files use `KnowledgeGraph` with explicit save/load. Heavy components (`EmbeddingEngine`, `Pipeline`, `ContradictionChecker`, `VerificationEngine`, `LLMMemory`) are lazily initialized on first access — read-only commands start instantly without loading Ollama or sentence-transformers.
- **`formatting.py`**: Rich-based formatters for all output types: node tables, statistics panels, anomaly tables, ingestion results, query/hypothesis results, verification results, retraction simulation panels, node detail panels, confidence trace panels, memory query results, export/import summaries. Color-coded confidence values (green ≥ 0.8, yellow ≥ 0.5, red < 0.5).
- **`__main__.py`**: Entry point for `python -m src.cli`.

### Design
- **Lazy initialization**: `EngineState.pipeline` and `EngineState.embedding_engine` are `@property` methods that create the heavyweight objects on first access. Read-only commands (`stats`, `list`, `inspect`, etc.) never trigger this.
- **Node ID prefix matching**: `resolve_node_id()` accepts full UUIDs or unique prefixes (first 8 chars). Used by `inspect`, `explain`, `trace`.
- **Auto-load/save**: Graph is loaded from `--graph` path on startup (if exists). After `ingest`, graph is auto-saved. For SQLite graphs, auto-save is a no-op (changes are already persisted).
- **SQLite support**: CLI auto-detects `.db` file extension and uses `SqliteGraph`. The `save` command supports `--format` option (json/sqlite) for format conversion. The `history` command shows recent changelog entries (SQLite graphs only).

---

## Memory Model and Scaling

### In-Memory Architecture
The entire knowledge graph is loaded into RAM at startup. There is no lazy loading, paging, or eviction mechanism:

- **KnowledgeGraph**: All nodes stored in a `Dict[str, KnowledgeNode]` with three auxiliary indexes (`_domain_index`, `_type_index`, `_status_index`).
- **SqliteGraph**: Subclass of KnowledgeGraph. On construction, loads *all* rows from SQLite into the in-memory dict. Mutations are write-through (call `super()` to update in-memory state, then persist to SQLite).
- **EmbeddingEngine**: Maintains `_embeddings` (numpy matrix of all vectors), `_id_to_position`/`_position_to_id` mappings, and optionally a FAISS index (which stores its own normalized copy of the vectors).

### Per-Node Memory Cost
Approximate per-node memory usage: **~6.6KB**
- KnowledgeNode object: ~1.2KB (text, metadata, source list, edge lists, UUID)
- Embedding vector (384 × float32): ~1.5KB
- EmbeddingEngine._embeddings row: ~1.5KB (second copy)
- FAISS normalized copy: ~1.5KB (third copy, when FAISS is installed)
- Index entries (domain, type, status, id_to_position): ~0.4KB

### Scaling Estimates
| Graph Size | RAM Usage | Entity Retrieval | Notes |
|:----------:|:---------:|:----------------:|-------|
| 1K nodes | ~7 MB | Instant | Current demo/test range |
| 10K nodes | ~65 MB | Fast | Comfortable for single-user |
| 100K nodes | ~650 MB | ~1s per scan | Consider inverted index |
| 1M nodes | ~6.5 GB | ~10s per scan | Requires architectural changes |

### Scaling Bottlenecks
1. **Entity retrieval**: Uses `EntityIndex` inverted index with BM25 scoring (O(k) where k is the number of matching terms). The original O(N) full scan has been replaced. FAISS handles vector search; BM25 handles term-based retrieval. Reciprocal Rank Fusion (RRF) combines both result lists.
2. **FAISS IndexFlatIP**: Brute-force inner product search, O(N) per query. Switch to IndexIVFFlat or IndexHNSW at ~100K nodes for sub-linear search.
3. **Triple embedding storage**: Vectors stored 3× in memory (node.embedding, EmbeddingEngine._embeddings, FAISS). Could be unified to a single reference at ~50K nodes.
4. **SqliteGraph startup**: Loads all nodes into memory on init. At ~100K+ nodes, consider lazy loading or memory-mapped storage.

---

## Key Design Decisions

1. **DAG, not general graph**: The knowledge graph is a directed acyclic graph. Cycles are prevented by `graph.would_create_cycle()` (BFS ancestor check) which guards `add_node()` and `apply_dependencies()`.

2. **Confidence is [0, 1]**: All confidence scores are clamped to the unit interval. The system uses three combination strategies (weakest-link, noisy-OR, depth decay) depending on the relationship type.

3. **Embeddings are optional**: The system works without sentence-transformers or FAISS (falls back to text-based Jaccard similarity and brute-force search), but semantic search quality degrades significantly.

4. **FAISS is optional**: When FAISS is not installed, EmbeddingEngine uses numpy cosine similarity. This is slower for large indexes but functionally identical.

5. **LLM used for extraction and contradiction detection**: The LLM (Ollama qwen3:14b or Gemini 2.0 Flash) is used in two modules: `fact_extraction.py` for structured fact extraction and `llm_contradiction.py` for 3-way relevance classification (contradicts/supports/irrelevant). Qwen3's thinking mode is explicitly disabled (`think=False`) for 2-5x speedup. The reasoning engine and confidence engine remain fully rule-based/heuristic. The dependency detector uses heuristics by default but can optionally use the LLM for contradiction verification.

6. **Tests mock the LLM**: All unit tests use mocked Ollama/Gemini responses across 34 test files. No running server is needed for the unit test suite. Integration tests require a running Ollama server with qwen3:14b and are skipped by default. Calibration tests use real datasets but no LLM.

7. **CLI uses click + rich**: Click provides clean subcommand routing with type validation and help generation. Rich provides formatted tables, panels, trees, and colored output. Lazy initialization ensures read-only commands don't load heavyweight components.

8. **Subclassing for SQLite persistence**: `SqliteGraph(KnowledgeGraph)` overrides 4 mutation methods to auto-persist changes to SQLite while keeping in-memory state correct via `super()` calls. This requires zero changes to existing code (Pipeline, DependencyDetector, ReasoningEngine all work unchanged). Uses built-in `sqlite3` — no new dependencies. WAL mode for concurrent read performance. Foreign keys with CASCADE deletes for referential integrity.

---

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | Yes | Embedding math, cosine similarity |
| sentence-transformers | Yes (for embeddings) | Text-to-vector encoding |
| faiss-cpu | Optional | Fast approximate nearest-neighbor search |
| ollama | Yes (for fact extraction + contradiction detection) | Local LLM via Ollama server |
| click | Yes (for CLI) | Command-line interface framework |
| rich | Yes (for CLI) | Formatted terminal output |
| fastapi | Yes (for Web UI) | REST API framework with automatic OpenAPI docs |
| uvicorn | Yes (for Web UI) | ASGI server for FastAPI |
| pydantic | Yes (for Web UI) | Request/response validation (installed with FastAPI) |
| pytest | Dev | Test runner |

Frontend dependencies (managed via npm in `src/web/`):
- `react`, `react-dom` — UI framework
- `react-router-dom` — Client-side routing
- `react-cytoscapejs`, `cytoscape`, `cytoscape-fcose` — Graph visualization with fast force-directed layout
- `typescript`, `vite` — Build tooling

All core modules (node, graph, confidence, dependency_detection, reasoning, storage) have no external dependencies beyond the Python standard library (storage uses built-in `sqlite3`).
