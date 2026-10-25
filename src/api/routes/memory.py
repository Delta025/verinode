"""
LLM Memory API routes: structured memory queries, recall, verification,
ingestion, and sub-graph export/import.

POST /api/memory/query     — Query graph for facts relevant to a question
POST /api/memory/recall    — Quick top-k fact recall
POST /api/memory/verify    — Verify a claim before using it
POST /api/memory/ingest    — Submit text for ingestion + verification
GET  /api/memory/export/{domain} — Export a domain slice as JSON
POST /api/memory/import    — Import a sub-graph export
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

import src.api.state as state_mod

from ..schemas import (
    MemoryFactItem,
    MemoryImportRequest,
    MemoryImportResponse,
    MemoryIngestRequest,
    MemoryIngestResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryRecallRequest,
    MemoryVerifyRequest,
    MemoryVerifyResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/query", response_model=MemoryQueryResponse)
def memory_query(req: MemoryQueryRequest):
    """Query the knowledge graph for facts relevant to a question."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    st.ensure_embeddings_indexed()
    result = st.llm_memory.query(
        req.question,
        domain=req.domain,
        k=req.k,
        include_chains=req.include_chains,
    )

    return MemoryQueryResponse(
        query=result.query,
        facts=[
            MemoryFactItem(
                node_id=f.node_id,
                text=f.text,
                confidence=f.confidence,
                domain=f.domain,
                sources=f.sources,
                reasoning_chain=f.reasoning_chain,
            )
            for f in result.facts
        ],
        summary=result.summary,
        overall_confidence=result.overall_confidence,
        coverage=result.coverage,
        total_graph_nodes=result.total_graph_nodes,
    )


@router.post("/recall", response_model=list[MemoryFactItem])
def memory_recall(req: MemoryRecallRequest):
    """Quick top-k fact recall for a topic."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    st.ensure_embeddings_indexed()
    facts = st.llm_memory.recall(req.topic, domain=req.domain, k=req.k)

    return [
        MemoryFactItem(
            node_id=f.node_id,
            text=f.text,
            confidence=f.confidence,
            domain=f.domain,
            sources=f.sources,
            reasoning_chain=f.reasoning_chain,
        )
        for f in facts
    ]


@router.post("/verify", response_model=MemoryVerifyResponse)
def memory_verify(req: MemoryVerifyRequest):
    """Verify a claim against the knowledge graph."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    st.ensure_embeddings_indexed()
    result = st.llm_memory.verify(req.claim)

    return MemoryVerifyResponse(**result)


@router.post("/ingest", response_model=MemoryIngestResponse)
def memory_ingest(req: MemoryIngestRequest):
    """Submit text for ingestion and verification."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    result = st.llm_memory.ingest(
        req.text,
        source_title=req.source_title,
        source_quality=req.source_quality,
        domain=req.domain,
    )

    return MemoryIngestResponse(
        action=result.action,
        node_id=result.node_id,
        confidence=result.confidence,
        contradictions=result.contradictions,
        explanation=result.explanation,
    )


@router.get("/export/{domain}")
def memory_export(domain: str, confidence_threshold: float = Query(0.5, ge=0.0, le=1.0)):
    """Export a domain slice as a portable JSON file."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    import json

    from src.core.export import export_subgraph, export_to_json

    export = export_subgraph(st.graph, domain, confidence_threshold=confidence_threshold)
    return JSONResponse(content=json.loads(export_to_json(export)))


@router.post("/import", response_model=MemoryImportResponse)
def memory_import(req: MemoryImportRequest):
    """Import a sub-graph export into the current graph."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    from src.core.export import import_from_json, import_subgraph

    try:
        export = import_from_json(req.export_json)
    except (ValueError, Exception) as e:
        raise HTTPException(status_code=400, detail=str(e))

    stats = import_subgraph(
        st.graph,
        export,
        conflict_strategy=req.conflict_strategy,
    )

    return MemoryImportResponse(**stats)
