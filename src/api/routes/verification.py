"""
Verification API routes: claim verification and retraction simulation.

POST /api/verify              — Verify a claim against the graph (read-only)
POST /api/simulate-retraction — What-if retraction analysis (read-only)
"""

from fastapi import APIRouter, HTTPException

import src.api.state as state_mod
from ..schemas import (
    VerifyRequest,
    VerifyResponse,
    EvidenceItem,
    RetractRequest,
    RetractResponse,
    ImpactItem,
)

router = APIRouter(tags=["verification"])


@router.post("/verify", response_model=VerifyResponse)
def verify_claim(req: VerifyRequest):
    """Verify a claim against the knowledge graph."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    st.ensure_embeddings_indexed()
    result = st.verification_engine.verify(req.claim, k=req.k)

    return VerifyResponse(
        claim=result.claim,
        verdict=result.verdict.value,
        confidence=result.confidence,
        explanation=result.explanation,
        supporting_evidence=[
            EvidenceItem(
                node_id=ev.node_id,
                node_text=ev.node_text,
                confidence=ev.confidence,
                relationship=ev.relationship,
                explanation=ev.explanation,
                num_chains=len(ev.chains),
            )
            for ev in result.supporting_evidence
        ],
        contradicting_evidence=[
            EvidenceItem(
                node_id=ev.node_id,
                node_text=ev.node_text,
                confidence=ev.confidence,
                relationship=ev.relationship,
                explanation=ev.explanation,
                num_chains=len(ev.chains),
            )
            for ev in result.contradicting_evidence
        ],
        coverage=result.coverage,
        num_candidates_checked=result.num_candidates_checked,
    )


@router.post("/simulate-retraction", response_model=RetractResponse)
def simulate_retraction(req: RetractRequest):
    """Simulate retracting a node and see the confidence cascade impact."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    node = st.graph.get_node(req.node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {req.node_id} not found")

    sim = st.verification_engine.simulate_retraction(req.node_id)

    return RetractResponse(
        retracted_node_id=sim.retracted_node_id,
        retracted_node_text=sim.retracted_node_text,
        original_confidence=sim.original_confidence,
        affected_nodes=[
            ImpactItem(
                node_id=impact.node_id,
                node_text=impact.node_text,
                original_confidence=impact.original_confidence,
                new_confidence=impact.new_confidence,
                change=impact.change,
                hops_from_retracted=impact.hops_from_retracted,
            )
            for impact in sim.affected_nodes
        ],
        total_affected=sim.total_affected,
        max_confidence_drop=sim.max_confidence_drop,
        avg_confidence_drop=sim.avg_confidence_drop,
        orphaned_nodes=sim.orphaned_nodes,
    )
