# Verinode — Quick Start Guide

## Installation

```bash
# Clone the project
cd verinode

# Install core dependencies
pip install numpy sentence-transformers ollama click rich

# Optional: fast vector search
pip install faiss-cpu

# Install Ollama from https://ollama.com/download
# Then pull a model (qwen3:14b recommended for RTX 3080 / 10GB VRAM):
ollama pull qwen3:14b

# Run unit tests (595 tests, no Ollama server needed)
pytest tests/

# Run integration tests (requires running Ollama server + pulled model)
pytest tests/ --run-integration

# Run the chemistry demo (no Ollama needed)
python examples/chemistry_demo.py

# Run the pipeline demo (requires Ollama)
python examples/pipeline_demo.py
```

## CLI Usage

The CLI wraps all core operations. Run via `python -m src.cli`.

```bash
# Global options
python -m src.cli --graph my_graph.json --model qwen3:14b COMMAND

# Ingest text (requires Ollama)
python -m src.cli ingest "Water boils at 100 degrees Celsius." --domain chemistry
python -m src.cli ingest paper.txt --file --source-type paper --source-title "Chemistry 101" --source-quality 0.95
python -m src.cli ingest textbook.txt --file --batch  # each paragraph as separate passage

# Read-only commands (no Ollama needed)
python -m src.cli stats                              # graph statistics
python -m src.cli list --domain chemistry --limit 10 # list nodes (filterable)
python -m src.cli list --type fact --sort text        # sort by text
python -m src.cli inspect a1b2c3d4                   # node detail (prefix match)
python -m src.cli explain a1b2c3d4                   # confidence explanation
python -m src.cli trace a1b2c3d4                     # trace to root sources
python -m src.cli anomalies --threshold 0.15         # detect anomalies

# Query and hypothesis (requires Ollama for embedding)
python -m src.cli query "What is the boiling point of water?" --show-chains
python -m src.cli hypothesis "Gold is a reactive metal" --show-evidence

# Graph persistence
python -m src.cli save backup.json
python -m src.cli load backup.json

# SQLite persistent storage (auto-detected from .db extension)
python -m src.cli --graph my_graph.db ingest "Water boils at 100C." --domain chemistry
python -m src.cli --graph my_graph.db stats          # works with SQLite
python -m src.cli --graph my_graph.db history         # view change log
python -m src.cli --graph old.json save new.db        # convert JSON → SQLite
```

## Module-by-Module Usage

### 1. Knowledge Nodes and Graph

Create nodes and build a graph. No external dependencies required.

```python
from src.core import KnowledgeGraph, KnowledgeNode, NodeType, Source, Derivation

graph = KnowledgeGraph()

# Create a base fact with provenance
water = KnowledgeNode(
    text="Water boils at 100 degrees Celsius at sea level",
    node_type=NodeType.FACT,
    confidence=0.95,
    domain="chemistry",
    sources=[Source(
        source_type="textbook",
        title="Chemistry 101",
        quality_score=0.95
    )]
)
graph.add_node(water)

# Create a derived fact that depends on the base
steam = KnowledgeNode(
    text="Steam is produced when water reaches its boiling point",
    node_type=NodeType.INFERENCE,
    confidence=0.8,
    domain="chemistry",
    predecessor_nodes=[water.id],
    derivation=Derivation(
        method="inferred",
        inference_depth=1,
        derivation_type="chain"
    )
)
graph.add_node(steam)

# Query the graph
chem_nodes = graph.get_nodes_by_domain("chemistry")
stats = graph.get_statistics()

# Save and load
graph.save("my_graph.json")
loaded = KnowledgeGraph.load("my_graph.json")
```

### 2. Confidence Engine

Propagate confidence scores through the dependency graph and detect anomalies.

```python
from src.core import ConfidenceEngine

engine = ConfidenceEngine(config={
    'inference_strength': 0.9,
    'depth_decay': 0.95,
    'anomaly_threshold': 0.15
})

# Calculate what a node's confidence should be based on its predecessors
expected = engine.calculate_node_confidence(steam, graph)

# Propagate a confidence change through all successors
affected = engine.propagate_confidence_update(graph, water.id)

# Detect nodes whose confidence doesn't match their evidence
anomalies = engine.detect_anomalies(graph)
for a in anomalies:
    print(f"Anomaly: {a['node_text']}")
    print(f"  Expected: {a['expected_confidence']:.2f}, Actual: {a['actual_confidence']:.2f}")

# Trace a node's confidence back to its root sources
trace = engine.trace_confidence_source(graph, steam.id)

# Estimate how many nodes would be affected by changing this node
impact = engine.estimate_impact(graph, water.id)
```

### 3. Embedding Engine

Semantic vector search over the graph. Requires sentence-transformers.

```python
from src.core import EmbeddingEngine

embed = EmbeddingEngine()  # model loaded lazily on first use

# Index all nodes in the graph
embed.index_graph(graph)

# Search by text
results = embed.search("boiling point of water", k=5)
for r in results:
    print(f"  {r['node_id']}: score={r['score']:.3f}")

# Find nodes similar to a specific node
similar = embed.find_similar(water.id, k=5, threshold=0.3)

# Add a new node to the index incrementally
new_node = KnowledgeNode(text="Ice melts at 0 degrees Celsius", domain="chemistry")
graph.add_node(new_node)
embed.add_to_index(new_node)

# Direct embedding
vec = embed.embed_text("some query text")  # numpy array, shape (384,)
```

### 4. Dependency Detection

Automatically detect relationships between nodes.

```python
from src.core import DependencyDetector, RelationType

detector = DependencyDetector(
    similarity_threshold=0.35,
    dependency_confidence_threshold=0.4
)

# Detect all dependencies in the graph (optionally using embeddings for candidate selection)
deps = detector.detect_all(graph, embedding_engine=embed)
for dep in deps:
    print(f"  {dep.source_id} --{dep.relation.value}--> {dep.target_id}")
    print(f"    confidence={dep.confidence:.2f}, evidence: {dep.evidence}")

# Detect dependencies for a single newly added node
new_deps = detector.detect_for_node(new_node, graph, embedding_engine=embed)

# Wire detected dependencies into the graph as edges
added = detector.apply_dependencies(deps, graph)
print(f"Added {added} new edges")
```

Relationship types:
- `DEPENDS_ON` - target logically depends on source
- `SUPPORTS` - source provides evidence for target
- `CONTRADICTS` - source and target conflict
- `ELABORATES` - target adds detail to source

### 5. Fact Extraction (Ollama)

Extract structured facts from text using a local Ollama LLM.

```python
from src.core import FactExtractionEngine, Source

# Requires a running Ollama server with a pulled model
# Default model: qwen3:14b
extractor = FactExtractionEngine()
# Or specify model/host:
# extractor = FactExtractionEngine(model="qwen3:14b", host="http://localhost:11434")

# Extract facts from text
result = extractor.extract(
    "Water is composed of hydrogen and oxygen. It boils at 100 degrees "
    "Celsius at standard atmospheric pressure.",
    domain="chemistry"
)
for fact in result.facts:
    print(f"  [{fact.fact_type}] {fact.text} (confidence={fact.confidence:.2f})")

# Extract directly to KnowledgeNode objects
nodes = extractor.extract_to_nodes(
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    domain="biology",
    source=Source(source_type="textbook", title="Biology 101", quality_score=0.9)
)
for node in nodes:
    graph.add_node(node)
```

### 6. LLM Contradiction Detection (Ollama)

Check whether two factual statements contradict each other using a local LLM.

```python
from src.core import ContradictionChecker, create_contradiction_checker

# Requires a running Ollama server with a pulled model
# Default model: qwen3:14b
checker = ContradictionChecker()
# Or use the factory (returns None if Ollama unavailable):
# checker = create_contradiction_checker()

# Check a single pair
result = checker.check_pair(
    "Water freezes at 0 degrees Celsius",
    "Water freezes at -100 degrees Celsius",
)
if result.is_contradiction:
    print(f"Contradiction detected! ({result.contradiction_type})")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation}")

# Check multiple pairs
pairs = [
    ("Water is safe to drink", "Water is toxic"),
    ("Iron is a metal", "Iron is used in steel production"),
]
results = checker.check_pairs_batch(pairs)
for r in results:
    status = "CONTRADICTS" if r.is_contradiction else "compatible"
    print(f"  '{r.text_a}' vs '{r.text_b}': {status}")
```

Contradiction types detected:
- `"negation"` — One statement negates the other
- `"numerical"` — Incompatible numerical values for the same property
- `"semantic"` — Semantically incompatible claims without negation words
- `"logical"` — Logically invalid inferences

The `DependencyDetector` and `Pipeline` both accept an optional `contradiction_checker` parameter. When provided, it enhances contradiction detection beyond the heuristic negation-word approach.

#### 3-Way Relevance Classification (used by VerificationEngine)

```python
# Classify the relationship between a claim and a knowledge statement
result = checker.check_relevance(
    "Gold melts at 500 degrees Celsius",
    "Gold melts at 1064 degrees Celsius"
)
print(result.relationship)  # "contradicts"
print(result.confidence)    # 0.95
print(result.explanation)   # "Numerical contradiction: 500C vs 1064C"

# Batch classification (used internally by VerificationEngine)
pairs = [
    ("Gold melts at 500C", "Gold melts at 1064C"),
    ("Gold melts at 500C", "Gold is a chemical element"),
]
results = checker.check_relevance_batch(pairs)
# results[0].relationship = "contradicts"
# results[1].relationship = "irrelevant" (different property: element vs melting point)
```

Relationship types:
- `"contradicts"` — Same property/predicate, incompatible values
- `"supports"` — Same property/predicate, consistent values
- `"irrelevant"` — Same subject but different property (excluded from verdicts)

### 7. Reasoning Engine

Multi-hop inference, query answering, and hypothesis testing.

```python
from src.core import ReasoningEngine, InferenceStrategy

reasoner = ReasoningEngine(
    max_depth=10,
    min_confidence=0.1,
    inference_strategy=InferenceStrategy.WEAKEST_LINK
)

# Forward chain: what can we conclude from a given premise?
chain = reasoner.forward_chain(graph, water.id, max_depth=3)
print(f"Forward chain: {len(chain.steps)} steps, confidence={chain.overall_confidence:.2f}")
for step in chain.steps:
    print(f"  [{step.step_type}] {step.text} ({step.confidence:.2f})")

# Backward chain: what evidence supports a conclusion?
evidence = reasoner.backward_chain(graph, steam.id)
print(f"Evidence chain: {len(evidence.steps)} steps")

# Query: find relevant facts and build reasoning chains
result = reasoner.query(graph, "What happens when water boils?", embedding_engine=embed)
print(f"Best answer: {result.best_answer} (confidence={result.confidence:.2f})")
for chain in result.chains:
    print(f"  Chain ({chain.overall_confidence:.2f}):")
    for step in chain.steps:
        print(f"    {step.text}")

# Hypothesis testing: is a claim supported by the graph?
hyp = reasoner.test_hypothesis(graph, "Water produces steam when heated", embedding_engine=embed)
print(f"Verdict: {hyp.verdict} (confidence={hyp.confidence:.2f})")
print(f"  Supporting chains: {len(hyp.supporting_chains)}")
print(f"  Contradicting chains: {len(hyp.contradicting_chains)}")

# Explain a node: why does it have this confidence?
explanation = reasoner.explain_node(graph, steam.id)
print(explanation)
```

### 8. Pipeline (End-to-End Ingestion)

The Pipeline class connects all modules into a single workflow with built-in truth verification (all 5 layers).

```python
from src.core import Pipeline, KnowledgeGraph, Source

# Create pipeline (requires running Ollama server)
# Automatically creates a ContradictionChecker if Ollama is available
graph = KnowledgeGraph()
pipeline = Pipeline(graph)

# Or with custom configuration:
pipeline = Pipeline(
    graph,
    duplicate_threshold=0.92,        # Cosine similarity for dedup
    contradiction_threshold=0.7,     # Min confidence to flag contradictions
    min_extraction_confidence=0.3,   # Drop low-confidence extractions
    source_quality_cap=0.5,          # Below this, source quality caps confidence
    reject_contradictions=False,     # False = flag, True = reject outright
    use_llm_contradiction=True,      # Enable LLM-assisted contradiction detection
)

# Ingest a single passage
result = pipeline.ingest(
    "Water boils at 100 degrees Celsius at sea level.",
    source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
    domain="chemistry",
)
print(result.summary)
# "Extracted 2 fact(s), added 2, merged 0 duplicate(s), wired 3 dependenc(ies)"

# Check what happened to each fact
for verdict in result.verdicts:
    print(f"  [{verdict.action}] {verdict.fact.text}")
    if verdict.action == "duplicate":
        print(f"    Boosted: {verdict.confidence_before:.3f} -> {verdict.confidence_after:.3f}")
    if verdict.reason:
        print(f"    Reason: {verdict.reason}")

# Batch ingest multiple passages
batch = pipeline.ingest_batch(
    ["Passage 1 text...", "Passage 2 text...", "Passage 3 text..."],
    source=Source(source_type="textbook", title="Textbook", quality_score=0.9),
    domain="chemistry",
)
print(f"Total: {batch.total_added} added, {batch.total_duplicates} merged, "
      f"{batch.total_flagged} flagged, {batch.total_rejected} rejected")

# Rebuild embedding index (e.g. after loading a saved graph)
pipeline.build_index()
```

Pipeline ingestion steps (per fact):
1. **Extract** facts from text via Ollama (JSON mode)
2. **Quality filter**: Drop facts below `min_extraction_confidence`
3. **Deduplication**: If cosine similarity >= `duplicate_threshold` with existing node, merge via noisy-OR boost
4. **Contradiction check** (two-phase when LLM available):
   - Phase 1: Heuristic pre-filter (negation word mismatch) — fast
   - Phase 2: LLM verification — verifies heuristic hits (removes false positives) and checks remaining candidates (catches numerical, semantic, logical contradictions)
   - If contradiction found: flag or reject. Verdict reason includes LLM explanation.
5. **Source quality gating**: If source quality < `source_quality_cap`, cap node confidence at source quality
6. **Add to graph**: Create KnowledgeNode, add to graph and embedding index
7. **Dependency detection**: Detect relationships with existing nodes (with LLM-assisted contradiction detection)
8. **Confidence propagation**: Propagate confidence updates through the graph

### 9. SQLite Persistent Storage

`SqliteGraph` is a drop-in replacement for `KnowledgeGraph` that auto-persists every mutation to SQLite. No new dependencies (uses Python's built-in `sqlite3`).

```python
from src.core import SqliteGraph, json_to_sqlite, sqlite_to_json

# Create a new SQLite-backed graph
graph = SqliteGraph("my_graph.db")
graph.add_node(node)  # persisted immediately to SQLite

# Reopen an existing database — all data loaded automatically
graph = SqliteGraph("my_graph.db")

# Use with Pipeline (drop-in replacement)
from src.core import Pipeline
pipeline = Pipeline(graph)
pipeline.ingest(text, source=my_source, domain="chemistry")
# Changes are auto-persisted — no save() needed

# View change history
entries = graph.get_changelog(limit=20)
for entry in entries:
    print(f"{entry['timestamp']} | {entry['change_type']} | {entry['node_id'][:8]}")

# Convert JSON → SQLite
sg = json_to_sqlite("old_graph.json", "new_graph.db")

# Convert SQLite → JSON
sqlite_to_json("graph.db", "export.json")

# Convert in-memory graph to SQLite
from src.core import KnowledgeGraph
mem_graph = KnowledgeGraph()
# ... add nodes ...
sg = SqliteGraph.from_graph(mem_graph, "persistent.db")

graph.close()  # Close when done
```

### 10. Claim Verification

Verify claims against the knowledge graph and simulate the impact of retracting facts.

```python
from src.core import VerificationEngine

verifier = VerificationEngine(graph, embedding_engine=embed,
                               reasoning_engine=reasoner,
                               confidence_engine=conf_engine)

# Verify a claim against the graph
result = verifier.verify("Water boils at 100 degrees Celsius")
print(f"Verdict: {result.verdict}")       # SUPPORTED / CONTRADICTED / UNCERTAIN
print(f"Confidence: {result.confidence:.2f}")
print(f"Evidence: {len(result.supporting)} supporting, {len(result.contradicting)} contradicting")

# Simulate retraction: what would happen if a fact were removed?
sim = verifier.simulate_retraction(water.id)
print(f"Affected nodes: {sim.total_affected}")
print(f"Orphaned nodes: {len(sim.orphaned_nodes)}")
for impact in sim.impacts:
    print(f"  {impact.node_id[:8]}: {impact.before:.2f} -> {impact.after:.2f}")
```

### 11. LLM Memory Substrate

Use Verinode as verified long-term memory for LLMs.

```python
from src.core import LLMMemory

memory = LLMMemory(graph, embed, reasoner, conf_engine,
                    pipeline=pipeline, verification_engine=verifier)

# Query memory with reasoning chains
result = memory.query("What is the boiling point of water?")
print(f"Coverage: {result.coverage}")  # high/medium/low/none
for fact in result.facts:
    print(f"  [{fact.confidence:.2f}] {fact.text}")

# Lightweight retrieval for context injection
facts = memory.recall("water properties", k=5)

# Verify a claim before trusting it
verification = memory.verify("Gold melts at 500 degrees Celsius")
print(f"Verdict: {verification['verdict']}")

# Export a domain for sharing
from src.core.export import export_subgraph, import_subgraph
export = export_subgraph(graph, domain="chemistry", confidence_threshold=0.7)
```

## Common Patterns

### Pattern: Ingest text and build a graph (using Pipeline)

```python
# The Pipeline handles all steps automatically:
result = pipeline.ingest(text, source=my_source, domain="chemistry")
# That's it! Extraction, verification, wiring, and propagation all happen internally.
```

### Pattern: Ingest text and build a graph (manual, without Pipeline)

```python
# 1. Extract facts from text
nodes = extractor.extract_to_nodes(text, domain="chemistry", source=my_source)

# 2. Add to graph
for node in nodes:
    graph.add_node(node)

# 3. Index for semantic search
embed.index_graph(graph)

# 4. Detect and wire dependencies
deps = detector.detect_all(graph, embedding_engine=embed)
detector.apply_dependencies(deps, graph)

# 5. Propagate confidence
conf_engine = ConfidenceEngine()
for node in nodes:
    conf_engine.propagate_confidence_update(graph, node.id)

# 6. Check for anomalies
anomalies = conf_engine.detect_anomalies(graph)
```

### Pattern: Add a single fact and integrate it

```python
# Create and add the node
node = KnowledgeNode(text="New fact", confidence=0.8, domain="chemistry")
graph.add_node(node)

# Add to embedding index
embed.add_to_index(node)

# Find and wire dependencies for just this node
deps = detector.detect_for_node(node, graph, embedding_engine=embed)
detector.apply_dependencies(deps, graph)

# Propagate confidence from this node
conf_engine.propagate_confidence_update(graph, node.id)
```

### Pattern: Audit the knowledge base

```python
# Detect anomalies
anomalies = conf_engine.detect_anomalies(graph)

# Check for contradictions
contradictions = graph.get_contradictions()

# Get graph statistics
stats = graph.get_statistics()

# Explain suspicious nodes
for anomaly in anomalies:
    explanation = reasoner.explain_node(graph, anomaly['node_id'])
    trace = conf_engine.trace_confidence_source(graph, anomaly['node_id'])
```

## Running Tests

```bash
# All 595 unit tests (mocked, no Ollama needed)
pytest tests/

# Verbose output
pytest tests/ -v

# Include integration tests (requires running Ollama + pulled model)
pytest tests/ -v --run-integration

# Include calibration tests (uses real datasets, no Ollama needed)
pytest tests/ -v --run-calibration

# Single module
pytest tests/test_cli.py -v
pytest tests/test_reasoning.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_llm_contradiction.py -v
```

No Ollama server or API keys needed for unit tests. All LLM calls are mocked.
Integration tests (49 tests) require `--run-integration` flag and a running Ollama server.
Calibration tests (46 tests) require `--run-calibration` flag and use real datasets.
