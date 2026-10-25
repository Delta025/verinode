"""
Overnight data ingestion from web sources into knowledge graphs.

Reads a JSON config listing URLs with quality scores, fetches and
chunks each source, and ingests into the pipeline with Gemini rate
limiting.

Features:
- Rate-limited Gemini calls (10 RPM / 1500 RPD paid tier)
- Resumable: tracks completed URLs in a progress file
- Progress logging for morning review
- Graceful shutdown on daily limit or Ctrl+C

Usage:
    python scripts/ingest_sources.py --config configs/solar_system.json
    python scripts/ingest_sources.py --config configs/solar_system.json --resume
    python scripts/ingest_sources.py --config configs/solar_system.json --dry-run
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Disable tqdm progress bars to avoid OSError on Windows when stderr
# is redirected (transformers/sentence-transformers use tqdm internally)
os.environ["TQDM_DISABLE"] = "1"

# Project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.graph import KnowledgeGraph
from src.core.node import Source
from src.core.embeddings import EmbeddingEngine
from src.core.confidence import ConfidenceEngine
from src.core.entity_index import EntityIndex
from src.core.pipeline import Pipeline
from src.core.fact_extraction import create_fact_extractor
from src.core.llm_contradiction import create_contradiction_checker
from src.core.rate_limiter import GeminiRateLimiter, DailyLimitReached, BudgetExhausted
from src.core.loader import fetch_and_extract, chunk_text


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load ingestion config from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress(progress_path: Path) -> set:
    """Load set of completed URLs from progress file."""
    if not progress_path.exists():
        return set()
    with open(progress_path, "r", encoding="utf-8") as f:
        return set(json.load(f).get("completed_urls", []))


def save_progress(progress_path: Path, completed: set, stats: dict):
    """Save progress for resume capability."""
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_urls": sorted(completed),
                "last_updated": datetime.now().isoformat(),
                "stats": stats,
            },
            f,
            indent=2,
        )


def setup_logging(log_path: Path):
    """Configure logging to file and console.

    Uses the root logger but avoids capturing stderr directly,
    which would break tqdm progress bars inside sentence-transformers.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)  # stdout, not stderr
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest web sources into knowledge graph"
    )
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from progress file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and chunk but don't ingest",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = Path(args.config).stem

    # Paths
    output_path = Path(config.get("graph_output", f"data/{config_name}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.json")
    log_path = output_path.with_suffix(".log")

    setup_logging(log_path)
    logger = logging.getLogger("ingest_sources")

    logger.info("=" * 60)
    logger.info("Verinode - Source Ingestion")
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", output_path)
    logger.info("Dry run: %s", args.dry_run)
    logger.info("=" * 60)

    # Resume support
    completed_urls = load_progress(progress_path) if args.resume else set()
    if completed_urls:
        logger.info("Resuming: %d URLs already completed", len(completed_urls))

    # Rate limiter (shared between extractor and checker)
    gemini_tier = config.get("gemini_tier", "paid")
    rpm = config.get("rpm", 60 if gemini_tier == "paid" else 2)
    rpd = config.get("rpd", 0)  # 0 = unlimited (Gemini paid has no daily cap)
    max_cost = config.get("max_cost_usd", 1.0)  # default $1.00 safety cap
    rate_limiter = GeminiRateLimiter(
        rpm=rpm,
        rpd=rpd,
        max_cost_usd=max_cost,
    )
    logger.info(
        "Rate limiter: %d RPM, RPD=%s, cost budget=$%.2f (%s tier)",
        rpm,
        str(rpd) if rpd > 0 else "unlimited",
        max_cost,
        gemini_tier,
    )

    # Create pipeline components
    extractor_backend = config.get("extractor_backend", "gemini")
    checker_backend = config.get("checker_backend", "gemini")

    extractor = create_fact_extractor(
        backend=extractor_backend,
        rate_limiter=rate_limiter,
    )
    if extractor is None:
        logger.error("Failed to create %s fact extractor. Exiting.", extractor_backend)
        sys.exit(1)
    logger.info("Extractor: %s (%s)", type(extractor).__name__, extractor_backend)

    checker = create_contradiction_checker(
        backend=checker_backend,
        rate_limiter=rate_limiter,
    )
    if checker:
        logger.info("Checker: %s (%s)", type(checker).__name__, checker_backend)
    else:
        logger.warning("No contradiction checker available (will skip)")

    # Load or create graph
    graph = KnowledgeGraph()
    if output_path.exists() and args.resume:
        graph = KnowledgeGraph.load(str(output_path))
        logger.info("Resumed graph: %d existing nodes", len(graph))

    embedding_engine = EmbeddingEngine()
    if len(graph) > 0:
        embedding_engine.index_graph(graph)

    entity_index = EntityIndex()
    if graph.nodes:
        entity_index.index_graph(graph)

    if not args.dry_run:
        pipeline = Pipeline(
            graph,
            extractor=extractor,
            embedding_engine=embedding_engine,
            confidence_engine=ConfidenceEngine(),
            contradiction_checker=checker,
            entity_index=entity_index,
            use_tiered_ingestion=True,
            verify_quotes=True,
            detect_dependencies=False,  # Skip during bulk ingestion for speed
        )

    # Graceful Ctrl+C handling
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupt received. Finishing current source...")

    signal.signal(signal.SIGINT, signal_handler)

    # Process sources
    sources = config.get("sources", [])
    total_added = 0
    total_flagged = 0
    total_chunks_processed = 0

    for i, src_cfg in enumerate(sources):
        if interrupted:
            logger.info("Interrupted. Saving progress and exiting.")
            break

        url = src_cfg["url"]
        if url in completed_urls:
            logger.info("[%d/%d] SKIP (already done): %s", i + 1, len(sources), url)
            continue

        logger.info("[%d/%d] Fetching: %s", i + 1, len(sources), url)

        # Fetch and extract
        content = fetch_and_extract(url)
        if not content.success:
            logger.error("  Failed: %s", content.error)
            continue

        logger.info(
            "  Extracted: %d chars, title='%s'",
            len(content.text),
            (content.title or "")[:60],
        )

        # Chunk
        chunks = chunk_text(content.text, max_tokens=1500)
        logger.info("  Chunks: %d", len(chunks))

        if args.dry_run:
            for j, chunk in enumerate(chunks):
                logger.info(
                    "  Chunk %d: %d chars (~%d tokens)",
                    j,
                    len(chunk),
                    len(chunk) // 4,
                )
            completed_urls.add(url)
            continue

        # Create Source object
        source = Source(
            source_type=src_cfg.get("source_type", "web"),
            title=src_cfg.get("title", content.title or url),
            author=src_cfg.get("author", content.author) or None,
            url=url,
            quality_score=src_cfg.get("quality_score", 0.5),
        )

        # Ingest chunks
        try:
            batch_result = pipeline.ingest_batch(
                chunks,
                source=source,
                domain=src_cfg.get("domain", config.get("domain", "general")),
            )
            total_added += batch_result.total_added
            total_flagged += batch_result.total_flagged
            total_chunks_processed += len(chunks)

            logger.info(
                "  Ingested: +%d added, %d flagged, %d dupes",
                batch_result.total_added,
                batch_result.total_flagged,
                batch_result.total_duplicates,
            )

        except (DailyLimitReached, BudgetExhausted) as budget_err:
            logger.warning(
                "%s: Saving progress and stopping.",
                type(budget_err).__name__,
            )
            graph.save(str(output_path))
            save_progress(progress_path, completed_urls, {
                "total_added": total_added,
                "total_flagged": total_flagged,
                "total_chunks": total_chunks_processed,
                "stopped_reason": "daily_limit",
                "rate_limiter": rate_limiter.log_progress(),
            })
            logger.info("Graph saved: %d nodes -> %s", len(graph), output_path)
            logger.info(rate_limiter.log_progress())
            return

        except Exception as e:
            logger.error("  Error ingesting %s: %s", url, e, exc_info=True)
            # Save what we have and continue to next source
            graph.save(str(output_path))
            save_progress(progress_path, completed_urls, {
                "total_added": total_added,
                "total_flagged": total_flagged,
                "total_chunks": total_chunks_processed,
                "last_error": str(e),
                "rate_limiter": rate_limiter.log_progress(),
            })
            continue

        # Mark done and save progress
        completed_urls.add(url)
        graph.save(str(output_path))
        save_progress(progress_path, completed_urls, {
            "total_added": total_added,
            "total_flagged": total_flagged,
            "total_chunks": total_chunks_processed,
            "rate_limiter": rate_limiter.log_progress(),
        })
        logger.info(
            "  Saved. Graph: %d nodes. %s",
            len(graph),
            rate_limiter.log_progress(),
        )

    # Final summary
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("  Graph: %d total nodes", len(graph))
    logger.info("  Added: %d, Flagged: %d", total_added, total_flagged)
    logger.info("  Chunks processed: %d", total_chunks_processed)
    logger.info("  Sources completed: %d/%d", len(completed_urls), len(sources))
    logger.info("  %s", rate_limiter.log_progress())
    logger.info("  Output: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
