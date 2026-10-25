# Changelog

All notable changes to Verinode are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-25

First public release of the research prototype, developed privately from February to March 2026 under the working name "Engram".

### Features
- Knowledge graph with typed dependency edges, cycle detection, provenance and JSON/SQLite persistence
- Ingestion pipeline: LLM fact extraction, source-quote verification, source-quality tiers, deduplication, contradiction detection, dependency linking and confidence propagation
- Claim verification (supported / contradicted / insufficient evidence) with evidence aggregation and retraction simulation
- Hybrid retrieval: sentence-transformer embeddings and BM25 with reciprocal rank fusion, plus cross-encoder re-ranking
- Confidence maintenance: staleness decay, strengthening on use and doubt flagging, with corroboration-adjusted half-lives
- Ollama and Gemini backends, chosen with environment variables
- `verinode` CLI (21 commands), FastAPI server and a React + Cytoscape.js web UI
- Web source loader and a bulk ingestion script with resume support

### Changed since the private prototype
- Renamed from Engram to Verinode. Environment variables are now `VERINODE_*`.
- The default Gemini model is now `gemini-3.8-flash`, because `gemini-2.0-flash` was shut down on 2026-06-01. It can be overridden with `VERINODE_GEMINI_MODEL`.
- Benchmark scripts moved to `benchmarks/`. A small demo graph and the stress-test graph are bundled in `data/`.
- Lint and formatting are clean under ruff, which is now pinned in CI.

### Fixed
- The CLI ignored `VERINODE_EXTRACTOR_BACKEND` and passed the Ollama model name to the Gemini backend, so every Gemini call failed.
