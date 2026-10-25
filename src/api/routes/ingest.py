"""Ingestion route: add text to the knowledge graph."""

import logging
from fastapi import APIRouter, HTTPException

import src.api.state as state_mod
from src.core.node import Source
from src.api.schemas import IngestionRequest, IngestionResponse, VerdictResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestionResponse)
def ingest_text(req: IngestionRequest):
    """
    Ingest text into the knowledge graph.

    Extracts facts via LLM, verifies against existing graph
    (dedup, contradiction check), adds new nodes, detects
    dependencies, and propagates confidence.

    Requires a running Ollama server with qwen3:14b.
    """
    st = state_mod.app_state

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Build source metadata
    source = Source(
        source_type=req.source_type,
        title=req.source_title,
        quality_score=req.source_quality,
    )

    try:
        result = st.pipeline.ingest(
            req.text,
            source=source,
            domain=req.domain,
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ingestion failed (is Ollama running?): {str(e)}",
        )

    # Auto-save for JSON graphs
    if not hasattr(st.graph, "get_changelog") and st.graph_path:
        try:
            st.graph.save(st.graph_path)
        except Exception:
            logger.warning("Failed to save graph after ingestion", exc_info=True)

    verdicts = []
    for v in result.verdicts:
        verdicts.append(VerdictResponse(
            fact_text=v.fact.text,
            action=v.action,
            node_id=v.node_id,
            reason=v.reason,
            contradiction_node_ids=v.contradiction_node_ids,
        ))

    return IngestionResponse(
        total_extracted=result.total_extracted,
        nodes_added=result.nodes_added,
        duplicates_merged=result.duplicates_merged,
        facts_flagged=result.facts_flagged,
        facts_rejected=result.facts_rejected,
        dependencies_added=result.dependencies_added,
        verdicts=verdicts,
        summary=result.summary,
    )
