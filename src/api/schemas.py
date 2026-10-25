"""
Pydantic response models for the Verinode API.

These provide automatic JSON serialization, validation, and OpenAPI
documentation. Shapes match the dicts returned by src/core modules.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# --- Graph & Stats ---

class GraphStats(BaseModel):
    total_nodes: int
    nodes_by_domain: Dict[str, int]
    nodes_by_type: Dict[str, int]
    nodes_by_status: Dict[str, int]
    avg_confidence: float
    nodes_with_conflicts: int


# --- Nodes ---

class NodeSummary(BaseModel):
    id: str
    text: str
    confidence: float
    domain: str
    node_type: str
    status: str


class SourceInfo(BaseModel):
    source_type: str = ""
    title: str = ""
    author: str = ""
    date: Optional[str] = None
    url: str = ""
    quality_score: float = 0.0


class NodeDetail(BaseModel):
    id: str
    text: str
    node_type: str
    confidence: float
    confidence_bounds: Optional[List[float]] = None
    domain: str
    status: str
    sources: List[SourceInfo] = []
    predecessor_nodes: List[str] = []
    successor_nodes: List[str] = []
    conflicts_with: List[str] = []
    created_at: Optional[str] = None
    last_validated: Optional[str] = None
    access_count: int = 0
    metadata: Dict[str, Any] = {}


class NodeNeighbors(BaseModel):
    predecessors: List[NodeSummary]
    successors: List[NodeSummary]
    conflicts: List[NodeSummary]


# --- Graph Visualization (Cytoscape.js) ---

class CytoscapeNodeData(BaseModel):
    id: str
    label: str
    confidence: float
    domain: str
    node_type: str
    status: str


class CytoscapeNode(BaseModel):
    data: CytoscapeNodeData


class CytoscapeEdgeData(BaseModel):
    id: str
    source: str
    target: str
    type: str  # "depends_on", "supports", "contradicts", "elaborates"


class CytoscapeEdge(BaseModel):
    data: CytoscapeEdgeData


class GraphData(BaseModel):
    nodes: List[CytoscapeNode]
    edges: List[CytoscapeEdge]


# --- Anomalies ---

class AnomalyItem(BaseModel):
    node_id: str
    node_text: str
    anomaly_type: str
    expected_confidence: float
    actual_confidence: float
    deviation: float
    severity: float


# --- Ingestion ---

class IngestionRequest(BaseModel):
    text: str
    domain: Optional[str] = None
    source_type: str = "user_input"
    source_title: str = ""
    source_quality: float = Field(0.7, ge=0.0, le=1.0)


class VerdictResponse(BaseModel):
    fact_text: str
    action: str  # "added", "duplicate", "flagged", "rejected"
    node_id: Optional[str] = None
    reason: str = ""
    contradiction_node_ids: List[str] = []


class IngestionResponse(BaseModel):
    total_extracted: int
    nodes_added: int
    duplicates_merged: int
    facts_flagged: int
    facts_rejected: int
    dependencies_added: int
    verdicts: List[VerdictResponse]
    summary: str


# --- Query ---

class QueryRequest(BaseModel):
    question: str
    k: int = 10


class ReasoningStepResponse(BaseModel):
    node_id: str
    text: str
    confidence: float
    step_type: str


class ReasoningChainResponse(BaseModel):
    steps: List[ReasoningStepResponse]
    overall_confidence: float
    explanation: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    confidence: float = 0.0
    chains: List[ReasoningChainResponse] = []
    relevant_node_ids: List[str] = []


# --- Hypothesis ---

class HypothesisRequest(BaseModel):
    hypothesis: str
    k: int = 10


class HypothesisResponse(BaseModel):
    hypothesis: str
    verdict: str  # "supported", "contradicted", "uncertain", "insufficient_evidence"
    confidence: float
    supporting_chains: List[ReasoningChainResponse] = []
    contradicting_chains: List[ReasoningChainResponse] = []


# --- Verification ---

class VerifyRequest(BaseModel):
    claim: str
    k: int = 10


class EvidenceItem(BaseModel):
    node_id: str
    node_text: str
    confidence: float
    relationship: str  # "supports" or "contradicts"
    explanation: str = ""
    num_chains: int = 0


class VerifyResponse(BaseModel):
    claim: str
    verdict: str  # "supported", "contradicted", "uncertain", "insufficient_evidence"
    confidence: float
    explanation: str
    supporting_evidence: List[EvidenceItem] = []
    contradicting_evidence: List[EvidenceItem] = []
    coverage: float = 0.0
    num_candidates_checked: int = 0


class RetractRequest(BaseModel):
    node_id: str


class ImpactItem(BaseModel):
    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    change: float
    hops_from_retracted: int


class RetractResponse(BaseModel):
    retracted_node_id: str
    retracted_node_text: str
    original_confidence: float
    affected_nodes: List[ImpactItem] = []
    total_affected: int = 0
    max_confidence_drop: float = 0.0
    avg_confidence_drop: float = 0.0
    orphaned_nodes: int = 0


# --- Graph Loading ---

class LoadGraphRequest(BaseModel):
    path: str


class LoadGraphResponse(BaseModel):
    path: str
    total_nodes: int
    avg_confidence: float
    domains: int


class SaveGraphRequest(BaseModel):
    path: str


class SaveGraphResponse(BaseModel):
    path: str
    total_nodes: int


class MergeGraphRequest(BaseModel):
    path: str
    strategy: str = "keep_higher_confidence"
    merge_sources: bool = True


class MergeGraphResponse(BaseModel):
    total_nodes: int
    nodes_from_a: int
    nodes_from_b_added: int
    nodes_from_b_merged: int
    nodes_from_b_skipped: int
    edges_added: int
    conflicts_added: int
    sources_merged: int


# --- LLM Memory ---

class MemoryQueryRequest(BaseModel):
    question: str
    domain: Optional[str] = None
    k: int = 10
    include_chains: bool = False


class MemoryFactItem(BaseModel):
    node_id: str
    text: str
    confidence: float
    domain: str
    sources: List[Dict[str, Any]] = []
    reasoning_chain: Optional[List[Dict[str, Any]]] = None


class MemoryQueryResponse(BaseModel):
    query: str
    facts: List[MemoryFactItem] = []
    summary: str = ""
    overall_confidence: float = 0.0
    coverage: str = "none"
    total_graph_nodes: int = 0


class MemoryRecallRequest(BaseModel):
    topic: str
    domain: Optional[str] = None
    k: int = 5


class MemoryVerifyRequest(BaseModel):
    claim: str


class MemoryVerifyResponse(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    evidence_count: int = 0


class MemoryIngestRequest(BaseModel):
    text: str
    source_title: str = "LLM submission"
    source_quality: float = Field(0.7, ge=0.0, le=1.0)
    domain: Optional[str] = None


class MemoryIngestResponse(BaseModel):
    action: str
    node_id: Optional[str] = None
    confidence: float = 0.0
    contradictions: List[Dict[str, Any]] = []
    explanation: str = ""


class MemoryImportRequest(BaseModel):
    export_json: str
    conflict_strategy: str = "keep_higher_confidence"


class MemoryImportResponse(BaseModel):
    added: int = 0
    merged: int = 0
    skipped: int = 0
    conflicts: int = 0


# --- History ---

class ChangelogEntry(BaseModel):
    id: int
    timestamp: str
    change_type: str
    node_id: Optional[str] = None
    details: Dict[str, Any] = {}


# --- Maintenance / Neuroplasticity ---

class MaintenanceRequest(BaseModel):
    dry_run: bool = False


class DecayResultItem(BaseModel):
    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    days_since_validated: float
    decay_applied: float


class StrengthResultItem(BaseModel):
    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    access_count: int
    boost_applied: float


class DoubtFlagItem(BaseModel):
    node_id: str
    node_text: str
    confidence: float
    reason: str
    severity: str


class MaintenanceResponse(BaseModel):
    timestamp: str
    nodes_examined: int
    nodes_decayed: int
    total_decay_amount: float
    nodes_strengthened: int
    total_strength_boost: float
    nodes_flagged: int
    flags: List[DoubtFlagItem] = []
    decay_details: List[DecayResultItem] = []
    strength_details: List[StrengthResultItem] = []
    dry_run: bool = False


class StrengthenRequest(BaseModel):
    node_id: str
    is_validation: bool = False


class StrengthenResponse(BaseModel):
    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    access_count: int
    boost_applied: float
    was_boosted: bool


class MaintenanceStatusResponse(BaseModel):
    total_nodes: int
    nodes_needing_review: int
    avg_confidence: float
    flags_by_severity: Dict[str, int] = {}
    oldest_unvalidated_days: Optional[float] = None


# --- Node Update ---

class NodeUpdateRequest(BaseModel):
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    domain: Optional[str] = None
    status: Optional[str] = None
    text: Optional[str] = None


class NodeUpdateResponse(BaseModel):
    node_id: str
    updated_fields: List[str]


# --- File Browser ---

class FileEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int = 0
    modified: Optional[str] = None


class FileBrowseResponse(BaseModel):
    current_dir: str
    parent_dir: Optional[str] = None
    entries: List[FileEntry]
