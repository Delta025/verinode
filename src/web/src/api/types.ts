/** TypeScript interfaces matching the FastAPI Pydantic schemas. */

// --- Graph & Stats ---

export interface GraphStats {
  total_nodes: number;
  nodes_by_domain: Record<string, number>;
  nodes_by_type: Record<string, number>;
  nodes_by_status: Record<string, number>;
  avg_confidence: number;
  nodes_with_conflicts: number;
}

// --- Nodes ---

export interface NodeSummary {
  id: string;
  text: string;
  confidence: number;
  domain: string;
  node_type: string;
  status: string;
}

export interface SourceInfo {
  source_type: string;
  title: string;
  author: string;
  date: string | null;
  url: string;
  quality_score: number;
}

export interface NodeDetail {
  id: string;
  text: string;
  node_type: string;
  confidence: number;
  confidence_bounds: number[] | null;
  domain: string;
  status: string;
  sources: SourceInfo[];
  predecessor_nodes: string[];
  successor_nodes: string[];
  conflicts_with: string[];
  created_at: string | null;
  last_validated: string | null;
  access_count: number;
  metadata: Record<string, unknown>;
}

export interface NodeNeighbors {
  predecessors: NodeSummary[];
  successors: NodeSummary[];
  conflicts: NodeSummary[];
}

// --- Graph Visualization ---

export interface CytoscapeNodeData {
  id: string;
  label: string;
  confidence: number;
  domain: string;
  node_type: string;
  status: string;
}

export interface CytoscapeNode {
  data: CytoscapeNodeData;
}

export interface CytoscapeEdgeData {
  id: string;
  source: string;
  target: string;
  type: string;
}

export interface CytoscapeEdge {
  data: CytoscapeEdgeData;
}

export interface GraphData {
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
}

// --- Anomalies ---

export interface AnomalyItem {
  node_id: string;
  node_text: string;
  anomaly_type: string;
  expected_confidence: number;
  actual_confidence: number;
  deviation: number;
  severity: number;
}

// --- Ingestion ---

export interface IngestionRequest {
  text: string;
  domain?: string;
  source_type?: string;
  source_title?: string;
  source_quality?: number;
}

export interface VerdictResponse {
  fact_text: string;
  action: string;
  node_id: string | null;
  reason: string;
  contradiction_node_ids: string[];
}

export interface IngestionResponse {
  total_extracted: number;
  nodes_added: number;
  duplicates_merged: number;
  facts_flagged: number;
  facts_rejected: number;
  dependencies_added: number;
  verdicts: VerdictResponse[];
  summary: string;
}

// --- Query ---

export interface QueryRequest {
  question: string;
  k?: number;
}

export interface ReasoningStep {
  node_id: string;
  text: string;
  confidence: number;
  step_type: string;
}

export interface ReasoningChain {
  steps: ReasoningStep[];
  overall_confidence: number;
  explanation: string;
}

export interface QueryResponse {
  question: string;
  answer: string | null;
  confidence: number;
  chains: ReasoningChain[];
  relevant_node_ids: string[];
}

// --- Hypothesis ---

export interface HypothesisRequest {
  hypothesis: string;
  k?: number;
}

export interface HypothesisResponse {
  hypothesis: string;
  verdict: string;
  confidence: number;
  supporting_chains: ReasoningChain[];
  contradicting_chains: ReasoningChain[];
}

// --- History ---

export interface ChangelogEntry {
  id: number;
  timestamp: string;
  change_type: string;
  node_id: string | null;
  details: Record<string, unknown>;
}

// --- Explain / Trace ---

export interface ExplainResponse {
  node_id: string;
  explanation: string;
}

export interface TraceResponse {
  node_id: string;
  trace: unknown;
}

// --- Verification ---

export interface VerifyRequest {
  claim: string;
  k?: number;
}

export interface EvidenceItem {
  node_id: string;
  node_text: string;
  confidence: number;
  relationship: string;
  explanation: string;
  num_chains: number;
}

export interface VerifyResponse {
  claim: string;
  verdict: string;
  confidence: number;
  explanation: string;
  supporting_evidence: EvidenceItem[];
  contradicting_evidence: EvidenceItem[];
  coverage: number;
  num_candidates_checked: number;
}

// --- Graph Loading ---

export interface LoadGraphRequest {
  path: string;
}

export interface LoadGraphResponse {
  path: string;
  total_nodes: number;
  avg_confidence: number;
  domains: number;
}

// --- Save Graph ---

export interface SaveGraphRequest {
  path: string;
}

export interface SaveGraphResponse {
  path: string;
  total_nodes: number;
}

// --- Merge Graph ---

export interface MergeGraphRequest {
  path: string;
  strategy?: string;
  merge_sources?: boolean;
}

export interface MergeGraphResponse {
  total_nodes: number;
  nodes_from_a: number;
  nodes_from_b_added: number;
  nodes_from_b_merged: number;
  nodes_from_b_skipped: number;
  edges_added: number;
  conflicts_added: number;
  sources_merged: number;
}

// --- Node Update ---

export interface NodeUpdateRequest {
  confidence?: number;
  domain?: string;
  status?: string;
  text?: string;
}

export interface NodeUpdateResponse {
  node_id: string;
  updated_fields: string[];
}

// --- File Browser ---

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  modified: string | null;
}

export interface FileBrowseResponse {
  current_dir: string;
  parent_dir: string | null;
  entries: FileEntry[];
}

// --- Health ---

export interface HealthResponse {
  status: string;
  graph_nodes: number;
  model: string | null;
}
