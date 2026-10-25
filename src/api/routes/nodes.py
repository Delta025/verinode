"""Node routes: list, detail, explain, trace, neighbors, update."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

import src.api.state as state_mod
from src.api.schemas import NodeDetail, NodeNeighbors, NodeSummary, NodeUpdateRequest, NodeUpdateResponse, SourceInfo

router = APIRouter(tags=["nodes"])


def _node_to_summary(node) -> NodeSummary:
    """Convert a KnowledgeNode to a NodeSummary."""
    return NodeSummary(
        id=node.id,
        text=node.text,
        confidence=node.confidence,
        domain=node.domain or "general",
        node_type=node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
        status=node.status.value if hasattr(node.status, "value") else str(node.status),
    )


def _node_to_detail(node) -> NodeDetail:
    """Convert a KnowledgeNode to a NodeDetail."""
    sources = []
    for s in node.sources:
        sources.append(
            SourceInfo(
                source_type=s.source_type or "",
                title=s.title or "",
                author=s.author or "",
                date=str(s.date) if s.date else None,
                url=s.url or "",
                quality_score=s.quality_score,
            )
        )

    return NodeDetail(
        id=node.id,
        text=node.text,
        node_type=node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
        confidence=node.confidence,
        confidence_bounds=list(node.confidence_bounds)
        if hasattr(node, "confidence_bounds") and node.confidence_bounds
        else None,
        domain=node.domain or "general",
        status=node.status.value if hasattr(node.status, "value") else str(node.status),
        sources=sources,
        predecessor_nodes=list(node.predecessor_nodes),
        successor_nodes=list(node.successor_nodes),
        conflicts_with=list(node.conflicts_with),
        created_at=node.created_at.isoformat() if node.created_at else None,
        last_validated=node.last_validated.isoformat() if node.last_validated else None,
        access_count=node.access_count,
        metadata=node.metadata,
    )


@router.get("/nodes", response_model=List[NodeSummary])
def list_nodes(
    domain: Optional[str] = None,
    node_type: Optional[str] = None,
    status: Optional[str] = None,
    sort: str = Query("confidence", pattern="^(confidence|created|text)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List nodes with optional filters, sorting, and pagination."""
    graph = state_mod.app_state.graph
    nodes = list(graph.nodes.values())

    # Filter
    if domain:
        nodes = [n for n in nodes if n.domain == domain]
    if node_type:
        nodes = [
            n for n in nodes if (n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type)) == node_type
        ]
    if status:
        nodes = [n for n in nodes if (n.status.value if hasattr(n.status, "value") else str(n.status)) == status]

    # Sort
    if sort == "confidence":
        nodes.sort(key=lambda n: n.confidence, reverse=True)
    elif sort == "created":
        nodes.sort(key=lambda n: n.created_at or "", reverse=True)
    elif sort == "text":
        nodes.sort(key=lambda n: n.text.lower())

    # Paginate
    nodes = nodes[offset : offset + limit]

    return [_node_to_summary(n) for n in nodes]


@router.get("/nodes/{node_id}", response_model=NodeDetail)
def get_node(node_id: str):
    """Get full details for a single node."""
    graph = state_mod.app_state.graph

    # Support prefix matching (like CLI)
    node = graph.get_node(node_id)
    if not node:
        # Try prefix match
        matches = [nid for nid in graph.nodes if nid.startswith(node_id)]
        if len(matches) == 1:
            node = graph.get_node(matches[0])

    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    return _node_to_detail(node)


@router.get("/nodes/{node_id}/explain")
def explain_node(node_id: str):
    """Explain why a node has its current confidence."""
    st = state_mod.app_state
    node = st.graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    explanation = st.reasoning_engine.explain_node(st.graph, node_id)
    return {"node_id": node_id, "explanation": explanation}


@router.get("/nodes/{node_id}/trace")
def trace_node(node_id: str):
    """Trace confidence back to root sources."""
    st = state_mod.app_state
    node = st.graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    trace = st.confidence_engine.trace_confidence_source(st.graph, node_id)
    return {"node_id": node_id, "trace": trace}


@router.get("/nodes/{node_id}/neighbors", response_model=NodeNeighbors)
def get_neighbors(node_id: str):
    """Get predecessor, successor, and conflict neighbors."""
    graph = state_mod.app_state.graph
    node = graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    predecessors = [_node_to_summary(n) for n in graph.get_predecessors(node_id)]
    successors = [_node_to_summary(n) for n in graph.get_successors(node_id)]
    conflicts = []
    for cid in node.conflicts_with:
        cnode = graph.get_node(cid)
        if cnode:
            conflicts.append(_node_to_summary(cnode))

    return NodeNeighbors(
        predecessors=predecessors,
        successors=successors,
        conflicts=conflicts,
    )


@router.patch("/nodes/{node_id}", response_model=NodeUpdateResponse)
def update_node(node_id: str, req: NodeUpdateRequest):
    """Update editable fields on a node (confidence, domain, status, text)."""
    graph = state_mod.app_state.graph
    node = graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    updated = []

    if req.confidence is not None:
        node.confidence = req.confidence
        updated.append("confidence")

    if req.domain is not None:
        node.domain = req.domain
        updated.append("domain")

    if req.status is not None:
        from src.core.node import NodeStatus

        try:
            node.status = NodeStatus(req.status)
        except ValueError:
            valid = [s.value for s in NodeStatus]
            raise HTTPException(status_code=400, detail=f"Invalid status '{req.status}'. Valid: {valid}")
        updated.append("status")

    if req.text is not None:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        node.text = req.text.strip()
        updated.append("text")

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    return NodeUpdateResponse(node_id=node_id, updated_fields=updated)


@router.delete("/nodes/{node_id}")
def delete_node(node_id: str):
    """Remove a node from the graph."""
    graph = state_mod.app_state.graph
    node = graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    graph.remove_node(node_id)
    return {"deleted": node_id}
