"""Query and hypothesis routes."""

from fastapi import APIRouter, HTTPException

import src.api.state as state_mod
from src.api.schemas import (
    HypothesisRequest,
    HypothesisResponse,
    QueryRequest,
    QueryResponse,
    ReasoningChainResponse,
    ReasoningStepResponse,
)

router = APIRouter(tags=["query"])


def _chain_to_response(chain) -> ReasoningChainResponse:
    """Convert a ReasoningChain to response model."""
    steps = []
    for step in chain.steps:
        steps.append(
            ReasoningStepResponse(
                node_id=step.node_id,
                text=step.node_text,
                confidence=step.confidence,
                step_type=step.step_type,
            )
        )
    return ReasoningChainResponse(
        steps=steps,
        overall_confidence=chain.conclusion_confidence,
        explanation=getattr(chain, "explanation", ""),
    )


@router.post("/query", response_model=QueryResponse)
def query_graph(req: QueryRequest):
    """
    Query the knowledge graph with a natural language question.

    Returns the best answer, confidence, and supporting reasoning chains.
    """
    st = state_mod.app_state

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Ensure embeddings are indexed for search
    st.ensure_embeddings_indexed()

    try:
        result = st.reasoning_engine.query(
            st.graph,
            req.question,
            embedding_engine=st.embedding_engine,
            k=req.k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    chains = [_chain_to_response(c) for c in result.supporting_chains]

    return QueryResponse(
        question=result.query,
        answer=result.answer_text,
        confidence=result.confidence,
        chains=chains,
        relevant_node_ids=result.related_node_ids,
    )


@router.post("/hypothesis", response_model=HypothesisResponse)
def test_hypothesis(req: HypothesisRequest):
    """
    Test a hypothesis against the knowledge graph.

    Returns a verdict (supported/contradicted/uncertain/insufficient_evidence),
    confidence, and supporting/contradicting evidence chains.
    """
    st = state_mod.app_state

    if not req.hypothesis.strip():
        raise HTTPException(status_code=400, detail="Hypothesis cannot be empty")

    st.ensure_embeddings_indexed()

    try:
        result = st.reasoning_engine.test_hypothesis(
            st.graph,
            req.hypothesis,
            embedding_engine=st.embedding_engine,
            k=req.k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hypothesis test failed: {str(e)}")

    # Convert bool + confidence to verdict string
    if result.confidence < 0.2:
        verdict = "insufficient_evidence"
    elif result.supported:
        verdict = "supported"
    elif not result.supported and result.confidence >= 0.5:
        verdict = "contradicted"
    else:
        verdict = "uncertain"

    return HypothesisResponse(
        hypothesis=result.hypothesis,
        verdict=verdict,
        confidence=result.confidence,
        supporting_chains=[_chain_to_response(c) for c in result.supporting_evidence],
        contradicting_chains=[_chain_to_response(c) for c in result.contradicting_evidence],
    )
