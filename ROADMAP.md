# Roadmap

Verinode is not under active development. These are the directions that looked most promising when work paused, roughly in order of impact. Contributions and forks are welcome.

## Close the retrieval gap

The largest accuracy lever. The full pipeline flags 48% of contradictions, but the LLM catches 80% when it is shown the right pair directly, so most misses happen before the LLM is asked.

- Multi-pass retrieval: run embedding and BM25 retrieval as separate LLM passes instead of only fusing their rankings.
- Index sub-facts (entity–claim pairs) so related facts about the same entity are retrieved together.
- Fine-tune the bi-encoder on confirmed contradiction and irrelevant pairs from real graphs.
- Keep tuning extraction prompts; extraction fidelity is currently 0.80.

## LLM verification middleware

The most practical use case: check an LLM's answer before showing it to a user.

- Extract verifiable claims (numbers, dates, entities, causal statements) from free-form LLM output.
- A `/api/verify-response` endpoint that returns the response annotated with per-claim verdicts.
- Integration examples: an OpenAI-compatible proxy, a LangChain/LlamaIndex tool and a Python decorator.
- Streaming verification for long responses.

## Domain knowledge packs

A graph can only verify what it knows. Claims in empty domains return *insufficient evidence*.

- Curated ingestion configs (like `configs/solar_system.json`) for medicine, law, finance and general science.
- Downloadable prebuilt graphs, so users can verify immediately without paying for ingestion.
- Domain-specific extraction prompts, and coverage reports that show where a graph is thin.

## Self-directed knowledge acquisition

- Track verification requests that find no evidence and cluster them into knowledge gaps.
- Generate search queries for each gap, fetch candidate sources, and queue extracted facts for human review.
- Connectors for structured sources such as Wikidata, PubMed and arXiv.

## Federated trust (research)

Several independent graphs could cross-check each other to catch a poisoned instance. That needs a privacy-preserving verification protocol between instances, trust scores built from agreement history, and defences against Sybil and slow-drift attacks.

## Engineering

- Authentication and multi-user support for the API.
- Structured logging, a health endpoint and startup configuration checks.
- Publish to PyPI under a proper package name (the code currently imports from `src.*`).
- Approximate nearest-neighbour search (FAISS IVF/HNSW) and lazy graph loading beyond about 100K nodes.
- Graph view: WebGL rendering, semantic clustering and in-graph search for large graphs.
