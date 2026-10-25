import { useEffect, useState } from 'react'
import { getNodes, getNode, explainNode, traceNode } from '../api/client'
import type { NodeSummary, NodeDetail, ExplainResponse, TraceResponse } from '../api/types'

function confidenceBadge(c: number) {
  const cls = c >= 0.8 ? 'badge-green' : c >= 0.5 ? 'badge-yellow' : 'badge-red'
  return <span className={`badge ${cls}`}>{(c * 100).toFixed(0)}%</span>
}

export default function NodeList() {
  const [nodes, setNodes] = useState<NodeSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [domain, setDomain] = useState('')
  const [sort, setSort] = useState('confidence')
  const [search, setSearch] = useState('')

  // Detail panel
  const [selected, setSelected] = useState<NodeDetail | null>(null)
  const [explain, setExplain] = useState<ExplainResponse | null>(null)
  const [trace, setTrace] = useState<TraceResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    getNodes({ domain: domain || undefined, sort, limit: 500 })
      .then(setNodes)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [domain, sort])

  const filtered = search
    ? nodes.filter((n) => n.text.toLowerCase().includes(search.toLowerCase()) || n.id.startsWith(search))
    : nodes

  const domains = [...new Set(nodes.map((n) => n.domain))].sort()

  async function selectNode(id: string) {
    setDetailLoading(true)
    setExplain(null)
    setTrace(null)
    try {
      const detail = await getNode(id)
      setSelected(detail)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setDetailLoading(false)
    }
  }

  async function loadExplain(id: string) {
    try {
      const result = await explainNode(id)
      setExplain(result)
    } catch (e) {
      setExplain({ node_id: id, explanation: `Error: ${(e as Error).message}` })
    }
  }

  async function loadTrace(id: string) {
    try {
      const result = await traceNode(id)
      setTrace(result)
    } catch (e) {
      setTrace({ node_id: id, trace: `Error: ${(e as Error).message}` })
    }
  }

  if (error) return <div className="page-content"><div className="error-message">Error: {error}</div></div>

  return (
    <div className="page-full">
      {/* Node table */}
      <div style={styles.tablePanel}>
        <div style={styles.header}>
          <h1 style={styles.title}>Nodes</h1>
          <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>{filtered.length} nodes</span>
        </div>

        {/* Filters */}
        <div style={styles.filters}>
          <input
            type="text"
            placeholder="Search by text or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={styles.searchInput}
          />
          <select value={domain} onChange={(e) => setDomain(e.target.value)} style={styles.select}>
            <option value="">All Domains</option>
            {domains.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value)} style={styles.select}>
            <option value="confidence">Sort: Confidence</option>
            <option value="created">Sort: Created</option>
            <option value="text">Sort: Text</option>
          </select>
        </div>

        {/* Table */}
        {loading ? (
          <div className="loading-overlay"><div className="spinner" /> Loading nodes...</div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">No nodes found</div>
        ) : (
          <div style={styles.tableContainer}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Text</th>
                  <th style={{ ...styles.th, width: 80, textAlign: 'center' }}>Confidence</th>
                  <th style={{ ...styles.th, width: 100 }}>Domain</th>
                  <th style={{ ...styles.th, width: 80 }}>Type</th>
                  <th style={{ ...styles.th, width: 80 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((n) => (
                  <tr
                    key={n.id}
                    style={{
                      ...styles.tr,
                      background: selected?.id === n.id ? 'var(--bg-hover)' : undefined,
                    }}
                    onClick={() => selectNode(n.id)}
                  >
                    <td style={styles.td}>
                      <div style={styles.nodeText}>{n.text}</div>
                      <div style={styles.nodeId}>{n.id.slice(0, 8)}</div>
                    </td>
                    <td style={{ ...styles.td, textAlign: 'center' }}>
                      {confidenceBadge(n.confidence)}
                    </td>
                    <td style={styles.td}>
                      <span className="badge badge-blue">{n.domain}</span>
                    </td>
                    <td style={{ ...styles.td, color: 'var(--text-secondary)', fontSize: 12 }}>
                      {n.node_type}
                    </td>
                    <td style={{ ...styles.td, color: 'var(--text-secondary)', fontSize: 12 }}>
                      {n.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {(selected || detailLoading) && (
        <div style={styles.detailPanel}>
          {detailLoading ? (
            <div className="loading-overlay"><div className="spinner" /></div>
          ) : selected ? (
            <>
              <div style={styles.detailHeader}>
                <h2 style={{ fontSize: 16, fontWeight: 600 }}>Node Details</h2>
                <button style={styles.closeBtn} onClick={() => setSelected(null)}>x</button>
              </div>

              <div style={styles.detailBody}>
                <div style={styles.detailText}>{selected.text}</div>

                <div style={styles.detailMeta}>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>ID</span>
                    <code style={styles.metaValue}>{selected.id}</code>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Confidence</span>
                    <span>{confidenceBadge(selected.confidence)}</span>
                  </div>
                  {selected.confidence_bounds && (
                    <div style={styles.metaRow}>
                      <span style={styles.metaLabel}>Bounds</span>
                      <span style={styles.metaValue}>
                        [{selected.confidence_bounds.map((b) => b.toFixed(3)).join(', ')}]
                      </span>
                    </div>
                  )}
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Domain</span>
                    <span className="badge badge-blue">{selected.domain}</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Type</span>
                    <span style={styles.metaValue}>{selected.node_type}</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Status</span>
                    <span style={styles.metaValue}>{selected.status}</span>
                  </div>
                  {selected.created_at && (
                    <div style={styles.metaRow}>
                      <span style={styles.metaLabel}>Created</span>
                      <span style={styles.metaValue}>{new Date(selected.created_at).toLocaleString()}</span>
                    </div>
                  )}
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Predecessors</span>
                    <span style={styles.metaValue}>{selected.predecessor_nodes.length}</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Successors</span>
                    <span style={styles.metaValue}>{selected.successor_nodes.length}</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Conflicts</span>
                    <span style={styles.metaValue}>{selected.conflicts_with.length}</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Access Count</span>
                    <span style={styles.metaValue}>{selected.access_count}</span>
                  </div>
                </div>

                {/* Sources */}
                {selected.sources.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={styles.sectionTitle}>Sources</div>
                    {selected.sources.map((s, i) => (
                      <div key={i} style={styles.sourceItem}>
                        <div style={{ fontWeight: 500, fontSize: 13 }}>{s.title || s.source_type}</div>
                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                          Quality: {(s.quality_score * 100).toFixed(0)}%
                          {s.author && ` | ${s.author}`}
                          {s.date && ` | ${s.date}`}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Debug buttons */}
                <div style={styles.debugButtons}>
                  <button className="btn" onClick={() => loadExplain(selected.id)}>
                    Explain
                  </button>
                  <button className="btn" onClick={() => loadTrace(selected.id)}>
                    Trace
                  </button>
                </div>

                {/* Explain output */}
                {explain && (
                  <div style={{ marginTop: 12 }}>
                    <div style={styles.sectionTitle}>Explanation</div>
                    <pre style={styles.debugOutput}>
                      {typeof explain.explanation === 'string'
                        ? explain.explanation
                        : JSON.stringify(explain.explanation, null, 2)}
                    </pre>
                  </div>
                )}

                {/* Trace output */}
                {trace && (
                  <div style={{ marginTop: 12 }}>
                    <div style={styles.sectionTitle}>Confidence Trace</div>
                    <pre style={styles.debugOutput}>
                      {JSON.stringify(trace.trace, null, 2)}
                    </pre>
                  </div>
                )}

                {/* Metadata */}
                {Object.keys(selected.metadata).length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={styles.sectionTitle}>Metadata</div>
                    <pre style={styles.debugOutput}>
                      {JSON.stringify(selected.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </>
          ) : null}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  tablePanel: {
    flex: 1, display: 'flex', flexDirection: 'column',
    minWidth: 0, minHeight: 0, padding: '16px 24px',
  },
  header: { display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 16, flexShrink: 0 },
  title: { fontSize: 24, fontWeight: 600 },
  filters: { display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', flexShrink: 0 },
  searchInput: {
    flex: 1, minWidth: 200, padding: '8px 12px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg-card)',
    color: 'var(--text-primary)', outline: 'none',
  },
  select: {
    padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border)',
    background: 'var(--bg-card)', color: 'var(--text-primary)', outline: 'none',
  },
  tableContainer: { flex: 1, overflowY: 'auto', minHeight: 0 },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: {
    textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid var(--border)',
    color: 'var(--text-muted)', fontSize: 12, fontWeight: 600, textTransform: 'uppercase',
    letterSpacing: '0.5px', position: 'sticky', top: 0, background: 'var(--bg-primary)',
    zIndex: 1,
  },
  tr: { cursor: 'pointer', transition: 'background 0.1s' },
  td: { padding: '10px 12px', borderBottom: '1px solid var(--border)', verticalAlign: 'top' },
  nodeText: { fontSize: 13, lineHeight: 1.4, maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  nodeId: { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 },
  detailPanel: {
    width: 360, minWidth: 280, maxWidth: '40vw',
    borderLeft: '1px solid var(--border)',
    background: 'var(--bg-secondary)', overflowY: 'auto',
    flexShrink: 0,
  },
  detailHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '12px 16px', borderBottom: '1px solid var(--border)',
  },
  closeBtn: {
    background: 'none', border: 'none', color: 'var(--text-muted)',
    fontSize: 16, cursor: 'pointer', padding: '4px 8px',
  },
  detailBody: { padding: 16 },
  detailText: {
    fontSize: 14, lineHeight: 1.6, color: 'var(--text-primary)',
    padding: '12px', background: 'var(--bg-card)', borderRadius: 6,
    marginBottom: 16, border: '1px solid var(--border)',
  },
  detailMeta: { display: 'flex', flexDirection: 'column', gap: 8 },
  metaRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 },
  metaLabel: { color: 'var(--text-muted)' },
  metaValue: { color: 'var(--text-secondary)', fontSize: 12 },
  sectionTitle: {
    fontSize: 12, fontWeight: 600, color: 'var(--text-muted)',
    textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8,
  },
  sourceItem: {
    padding: '8px 12px', background: 'var(--bg-card)', borderRadius: 6,
    marginBottom: 4, border: '1px solid var(--border)',
  },
  debugButtons: { display: 'flex', gap: 8, marginTop: 16 },
  debugOutput: {
    padding: '12px', background: 'var(--bg-primary)', borderRadius: 6,
    fontSize: 12, fontFamily: 'monospace', color: 'var(--text-secondary)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 300,
    overflowY: 'auto', border: '1px solid var(--border)',
  },
}
