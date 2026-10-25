# Changelog

All notable changes to Engram AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-24

Initial pre-release. All 15 development steps complete.

### Core System
- Knowledge graph with DAG structure, typed edges, cycle detection, and SQLite/JSON persistence
- 5-layer ingestion pipeline: deduplication, contradiction detection, source quality gating, dependency detection, confidence propagation
- LLM-based fact extraction and 3-way relevance classification (contradicts/supports/irrelevant)
- Hybrid retrieval: sentence-transformer embeddings + BM25 term search with reciprocal rank fusion
- Cross-encoder re-ranking for improved retrieval precision
- Neuroplasticity engine: staleness decay, pathway strengthening, doubt flagging with corroboration-adjusted half-lives
- Claim verification with evidence aggregation and retraction simulation
- Multi-hop reasoning with forward/backward chaining
- Source provenance with quality tiers (AUTHORITATIVE/STANDARD/UNVERIFIED) and confidence caps
- Unit normalization for consistent numerical comparisons
- Content pre-filters reducing false positive conflicts by 80.6%

### LLM Backends
- Ollama backend (default) with batch parallelism via ThreadPoolExecutor
- Gemini 2.0 Flash backend with multi-pair batching (72% fewer API calls, 91% cost reduction)
- Factory pattern for backend selection via environment variables
- Shared rate limiter with RPM/RPD sliding windows and graceful daily limit handling

### Interfaces
- CLI with 21 Click commands and Rich formatting
- FastAPI backend with 30 endpoints and 51 Pydantic schemas
- React 19 + TypeScript web UI with Cytoscape.js graph visualization (8 pages)
- Web content loader with trafilatura extraction and text chunking
- Bulk ingestion script with resume support and cost tracking

### Testing
- 797 unit tests (all mocked, no external dependencies)
- 106 integration and calibration tests (optional, require Ollama)
- Benchmarks: 90% false claims caught, 90% true claims correct, +80pp poisoned context resistance

### Packaging
- `pyproject.toml` with setuptools build system
- `engram` CLI and `engram-server` entry points
- Docker support with Ollama sidecar
- GitHub Actions CI/CD pipeline
