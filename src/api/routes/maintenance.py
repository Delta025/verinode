"""
Maintenance API routes: neuroplasticity engine operations.

POST /api/maintenance/run        -- Run full maintenance cycle
GET  /api/maintenance/doubt      -- List doubtful nodes
POST /api/maintenance/strengthen -- Strengthen a specific node
GET  /api/maintenance/status     -- Quick maintenance status overview
"""

import copy
from datetime import datetime

from fastapi import APIRouter, HTTPException

import src.api.state as state_mod
from ..schemas import (
    MaintenanceRequest,
    MaintenanceResponse,
    DecayResultItem,
    StrengthResultItem,
    DoubtFlagItem,
    StrengthenRequest,
    StrengthenResponse,
    MaintenanceStatusResponse,
)

router = APIRouter(tags=["maintenance"])


def _copy_graph_for_dry_run(graph):
    """Create a plain KnowledgeGraph copy of any graph (JSON or SQLite).

    SqliteGraph holds a sqlite3.Connection which cannot be deep-copied,
    so we build a fresh KnowledgeGraph and deep-copy each node into it.
    """
    from src.core.graph import KnowledgeGraph
    graph_copy = KnowledgeGraph()
    for node in graph.nodes.values():
        graph_copy.add_node(copy.deepcopy(node))
    return graph_copy


def _report_to_response(report, dry_run: bool = False) -> MaintenanceResponse:
    """Convert a MaintenanceReport dataclass to a Pydantic response."""
    return MaintenanceResponse(
        timestamp=report.timestamp.isoformat(),
        nodes_examined=report.nodes_examined,
        nodes_decayed=report.nodes_decayed,
        total_decay_amount=report.total_decay_amount,
        nodes_strengthened=report.nodes_strengthened,
        total_strength_boost=report.total_strength_boost,
        nodes_flagged=report.nodes_flagged,
        flags=[
            DoubtFlagItem(
                node_id=f.node_id,
                node_text=f.node_text,
                confidence=f.confidence,
                reason=f.reason,
                severity=f.severity,
            )
            for f in report.flags
        ],
        decay_details=[
            DecayResultItem(
                node_id=d.node_id,
                node_text=d.node_text,
                original_confidence=d.original_confidence,
                new_confidence=d.new_confidence,
                days_since_validated=d.days_since_validated,
                decay_applied=d.decay_applied,
            )
            for d in report.decay_details
        ],
        strength_details=[
            StrengthResultItem(
                node_id=s.node_id,
                node_text=s.node_text,
                original_confidence=s.original_confidence,
                new_confidence=s.new_confidence,
                access_count=s.access_count,
                boost_applied=s.boost_applied,
            )
            for s in report.strength_details
        ],
        dry_run=dry_run,
    )


@router.post("/maintenance/run", response_model=MaintenanceResponse)
def run_maintenance(req: MaintenanceRequest):
    """Run a full neuroplasticity maintenance cycle (decay + strengthen + flag)."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")
    if len(st.graph) == 0:
        raise HTTPException(status_code=400, detail="Graph is empty")

    if req.dry_run:
        graph_copy = _copy_graph_for_dry_run(st.graph)
        report = st.neuroplasticity_engine.run_maintenance(graph_copy)
    else:
        report = st.neuroplasticity_engine.run_maintenance(st.graph)

    return _report_to_response(report, dry_run=req.dry_run)


@router.get("/maintenance/doubt")
def get_doubt_flags():
    """List all currently doubtful nodes (read-only diagnostic)."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    flags = st.neuroplasticity_engine.flag_doubtful_nodes(st.graph)
    return [
        DoubtFlagItem(
            node_id=f.node_id,
            node_text=f.node_text,
            confidence=f.confidence,
            reason=f.reason,
            severity=f.severity,
        ).model_dump()
        for f in flags
    ]


@router.post("/maintenance/strengthen", response_model=StrengthenResponse)
def strengthen_node(req: StrengthenRequest):
    """Strengthen a specific node (record access + optional validation)."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    node = st.graph.get_node(req.node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {req.node_id} not found")

    result = st.neuroplasticity_engine.strengthen_pathway(
        st.graph, req.node_id, is_validation=req.is_validation,
    )

    if result is not None:
        return StrengthenResponse(
            node_id=result.node_id,
            node_text=result.node_text,
            original_confidence=result.original_confidence,
            new_confidence=result.new_confidence,
            access_count=result.access_count,
            boost_applied=result.boost_applied,
            was_boosted=True,
        )
    else:
        # Access was recorded but no boost (below threshold)
        node = st.graph.get_node(req.node_id)
        return StrengthenResponse(
            node_id=req.node_id,
            node_text=node.text,
            original_confidence=node.confidence,
            new_confidence=node.confidence,
            access_count=node.access_count,
            boost_applied=0.0,
            was_boosted=False,
        )


@router.get("/maintenance/status", response_model=MaintenanceStatusResponse)
def maintenance_status():
    """Quick maintenance status: nodes needing review, avg staleness, etc."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    if len(st.graph) == 0:
        return MaintenanceStatusResponse(
            total_nodes=0,
            nodes_needing_review=0,
            avg_confidence=0.0,
            flags_by_severity={},
        )

    flags = st.neuroplasticity_engine.flag_doubtful_nodes(st.graph)
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for f in flags:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    stats = st.graph.get_statistics()

    # Find oldest unvalidated node
    now = datetime.now()
    oldest_days = None
    for node in st.graph.nodes.values():
        try:
            delta = now - node.last_validated
            days = delta.total_seconds() / 86400.0
            if oldest_days is None or days > oldest_days:
                oldest_days = days
        except (TypeError, AttributeError):
            pass

    return MaintenanceStatusResponse(
        total_nodes=len(st.graph),
        nodes_needing_review=len(flags),
        avg_confidence=stats["avg_confidence"],
        flags_by_severity=severity_counts,
        oldest_unvalidated_days=oldest_days,
    )
