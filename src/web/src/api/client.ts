/**
 * API client for the Verinode FastAPI backend.
 *
 * All routes are prefixed with /api. In development, Vite proxies
 * these to http://127.0.0.1:8000.
 */

import type {
  GraphStats,
  NodeSummary,
  NodeDetail,
  NodeNeighbors,
  GraphData,
  AnomalyItem,
  IngestionRequest,
  IngestionResponse,
  QueryRequest,
  QueryResponse,
  HypothesisRequest,
  HypothesisResponse,
  ChangelogEntry,
  ExplainResponse,
  TraceResponse,
  VerifyRequest,
  VerifyResponse,
  LoadGraphRequest,
  LoadGraphResponse,
  SaveGraphRequest,
  SaveGraphResponse,
  MergeGraphRequest,
  MergeGraphResponse,
  NodeUpdateRequest,
  NodeUpdateResponse,
  FileBrowseResponse,
  HealthResponse,
} from './types';

const BASE = '/api';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// --- Graph & Stats ---

export function getStats(): Promise<GraphStats> {
  return request<GraphStats>('/stats');
}

export function getGraphData(): Promise<GraphData> {
  return request<GraphData>('/graph/data');
}

export function getAnomalies(threshold?: number): Promise<AnomalyItem[]> {
  const q = threshold != null ? `?threshold=${threshold}` : '';
  return request<AnomalyItem[]>(`/anomalies${q}`);
}

export function getHistory(limit = 50): Promise<ChangelogEntry[]> {
  return request<ChangelogEntry[]>(`/history?limit=${limit}`);
}

// --- Nodes ---

export interface NodeListParams {
  domain?: string;
  node_type?: string;
  status?: string;
  sort?: string;
  limit?: number;
  offset?: number;
}

export function getNodes(params: NodeListParams = {}): Promise<NodeSummary[]> {
  const q = new URLSearchParams();
  if (params.domain) q.set('domain', params.domain);
  if (params.node_type) q.set('node_type', params.node_type);
  if (params.status) q.set('status', params.status);
  if (params.sort) q.set('sort', params.sort);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.offset != null) q.set('offset', String(params.offset));
  const qs = q.toString();
  return request<NodeSummary[]>(`/nodes${qs ? '?' + qs : ''}`);
}

export function getNode(id: string): Promise<NodeDetail> {
  return request<NodeDetail>(`/nodes/${encodeURIComponent(id)}`);
}

export function explainNode(id: string): Promise<ExplainResponse> {
  return request<ExplainResponse>(`/nodes/${encodeURIComponent(id)}/explain`);
}

export function traceNode(id: string): Promise<TraceResponse> {
  return request<TraceResponse>(`/nodes/${encodeURIComponent(id)}/trace`);
}

export function getNeighbors(id: string): Promise<NodeNeighbors> {
  return request<NodeNeighbors>(`/nodes/${encodeURIComponent(id)}/neighbors`);
}

// --- Ingestion ---

export function ingest(data: IngestionRequest): Promise<IngestionResponse> {
  return request<IngestionResponse>('/ingest', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Query ---

export function queryGraph(data: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>('/query', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Hypothesis ---

export function testHypothesis(data: HypothesisRequest): Promise<HypothesisResponse> {
  return request<HypothesisResponse>('/hypothesis', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Verification ---

export function verifyClaim(data: VerifyRequest): Promise<VerifyResponse> {
  return request<VerifyResponse>('/verify', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Graph Loading ---

export function loadGraph(data: LoadGraphRequest): Promise<LoadGraphResponse> {
  return request<LoadGraphResponse>('/load-graph', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Save Graph ---

export function saveGraph(data: SaveGraphRequest): Promise<SaveGraphResponse> {
  return request<SaveGraphResponse>('/save-graph', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Merge Graph ---

export function mergeGraph(data: MergeGraphRequest): Promise<MergeGraphResponse> {
  return request<MergeGraphResponse>('/merge-graph', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// --- Node Update ---

export function updateNode(id: string, data: NodeUpdateRequest): Promise<NodeUpdateResponse> {
  return request<NodeUpdateResponse>(`/nodes/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export function deleteNode(id: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/nodes/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

// --- File Browser ---

export function browseFiles(dir?: string, filter: string = 'graph'): Promise<FileBrowseResponse> {
  const q = new URLSearchParams();
  if (dir) q.set('dir', dir);
  q.set('filter', filter);
  return request<FileBrowseResponse>(`/files/browse?${q.toString()}`);
}

// --- Health ---

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}
