import { useEffect, useState } from 'react'
import { getStats, getAnomalies, loadGraph, saveGraph, mergeGraph } from '../api/client'
import type { GraphStats, AnomalyItem, LoadGraphResponse, SaveGraphResponse, MergeGraphResponse } from '../api/types'
import FileBrowser from './FileBrowser'

function confidenceColor(c: number): string {
  if (c >= 0.8) return 'var(--conf-high)'
  if (c >= 0.5) return 'var(--conf-medium)'
  return 'var(--conf-low)'
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="card" style={styles.statCard}>
      <div style={styles.statLabel}>{label}</div>
      <div style={styles.statValue}>{value}</div>
      {sub && <div style={styles.statSub}>{sub}</div>}
    </div>
  )
}

type BrowseTarget = 'load' | 'save' | 'merge' | null

export default function Dashboard() {
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [anomalies, setAnomalies] = useState<AnomalyItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Load Graph state
  const [graphPath, setGraphPath] = useState('engram_graph.json')
  const [graphLoading, setGraphLoading] = useState(false)
  const [graphResult, setGraphResult] = useState<LoadGraphResponse | null>(null)
  const [graphError, setGraphError] = useState<string | null>(null)

  // Save Graph state
  const [savePath, setSavePath] = useState('data/my_graph.json')
  const [saveLoading, setSaveLoading] = useState(false)
  const [saveResult, setSaveResult] = useState<SaveGraphResponse | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)

  // Merge Graph state
  const [mergePath, setMergePath] = useState('')
  const [mergeLoading, setMergeLoading] = useState(false)
  const [mergeResult, setMergeResult] = useState<MergeGraphResponse | null>(null)
  const [mergeError, setMergeError] = useState<string | null>(null)

  // File browser state
  const [browseTarget, setBrowseTarget] = useState<BrowseTarget>(null)

  function refreshStats() {
    Promise.all([getStats(), getAnomalies()])
      .then(([s, a]) => { setStats(s); setAnomalies(a) })
      .catch(() => { /* ignore refresh errors */ })
  }

  useEffect(() => {
    Promise.all([getStats(), getAnomalies()])
      .then(([s, a]) => { setStats(s); setAnomalies(a) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  async function handleLoadGraph(e: React.FormEvent) {
    e.preventDefault()
    if (!graphPath.trim()) return
    setGraphLoading(true)
    setGraphError(null)
    setGraphResult(null)
    try {
      const res = await loadGraph({ path: graphPath.trim() })
      setGraphResult(res)
      refreshStats()
    } catch (e) {
      setGraphError((e as Error).message)
    } finally {
      setGraphLoading(false)
    }
  }

  async function handleSaveGraph(e: React.FormEvent) {
    e.preventDefault()
    if (!savePath.trim()) return
    setSaveLoading(true)
    setSaveError(null)
    setSaveResult(null)
    try {
      const res = await saveGraph({ path: savePath.trim() })
      setSaveResult(res)
    } catch (e) {
      setSaveError((e as Error).message)
    } finally {
      setSaveLoading(false)
    }
  }

  async function handleMergeGraph(e: React.FormEvent) {
    e.preventDefault()
    if (!mergePath.trim()) return
    setMergeLoading(true)
    setMergeError(null)
    setMergeResult(null)
    try {
      const res = await mergeGraph({ path: mergePath.trim() })
      setMergeResult(res)
      refreshStats()
    } catch (e) {
      setMergeError((e as Error).message)
    } finally {
      setMergeLoading(false)
    }
  }

  function handleFileBrowseSelect(path: string) {
    if (browseTarget === 'load') setGraphPath(path)
    else if (browseTarget === 'save') setSavePath(path)
    else if (browseTarget === 'merge') setMergePath(path)
    setBrowseTarget(null)
  }

  if (loading) return <div className="page-content"><div className="loading-overlay"><div className="spinner" /> Loading dashboard...</div></div>
  if (error) return <div className="page-content"><div className="error-message">Error: {error}</div></div>
  if (!stats) return <div className="page-content"><div className="empty-state">No data available</div></div>

  const domains = Object.entries(stats.nodes_by_domain)
  const types = Object.entries(stats.nodes_by_type)

  return (
    <div className="page-content">
      <h1 style={styles.title}>Dashboard</h1>

      {/* Graph Operations */}
      <div className="card" style={{ marginBottom: 16, padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {/* Load */}
        <form onSubmit={handleLoadGraph} style={styles.graphOpRow}>
          <span style={styles.graphOpLabel}>Load</span>
          <input type="text" value={graphPath} onChange={(e) => setGraphPath(e.target.value)}
            placeholder="engram_graph.json" style={styles.graphOpInput} />
          <button type="button" className="btn" onClick={() => setBrowseTarget('load')}
            style={styles.browseBtn} title="Browse files">...</button>
          <button type="submit" className="btn btn-primary" disabled={graphLoading || !graphPath.trim()}
            style={styles.graphOpBtn}>{graphLoading ? 'Loading...' : 'Load'}</button>
          {graphResult && <span style={styles.graphOpOk}>{graphResult.total_nodes} nodes, {graphResult.domains} domains</span>}
          {graphError && <span style={styles.graphOpErr}>{graphError.length > 80 ? graphError.slice(0, 80) + '...' : graphError}</span>}
        </form>
        {/* Save */}
        <form onSubmit={handleSaveGraph} style={styles.graphOpRow}>
          <span style={styles.graphOpLabel}>Save</span>
          <input type="text" value={savePath} onChange={(e) => setSavePath(e.target.value)}
            placeholder="data/my_graph.json" style={styles.graphOpInput} />
          <button type="button" className="btn" onClick={() => setBrowseTarget('save')}
            style={styles.browseBtn} title="Browse files">...</button>
          <button type="submit" className="btn btn-primary" disabled={saveLoading || !savePath.trim()}
            style={styles.graphOpBtn}>{saveLoading ? 'Saving...' : 'Save'}</button>
          {saveResult && <span style={styles.graphOpOk}>Saved {saveResult.total_nodes} nodes</span>}
          {saveError && <span style={styles.graphOpErr}>{saveError.length > 80 ? saveError.slice(0, 80) + '...' : saveError}</span>}
        </form>
        {/* Merge */}
        <form onSubmit={handleMergeGraph} style={styles.graphOpRow}>
          <span style={styles.graphOpLabel}>Merge</span>
          <input type="text" value={mergePath} onChange={(e) => setMergePath(e.target.value)}
            placeholder="path/to/other_graph.json" style={styles.graphOpInput} />
          <button type="button" className="btn" onClick={() => setBrowseTarget('merge')}
            style={styles.browseBtn} title="Browse files">...</button>
          <button type="submit" className="btn btn-primary" disabled={mergeLoading || !mergePath.trim()}
            style={styles.graphOpBtn}>{mergeLoading ? 'Merging...' : 'Merge'}</button>
          {mergeResult && <span style={styles.graphOpOk}>+{mergeResult.nodes_from_b_added} new, {mergeResult.nodes_from_b_merged} merged = {mergeResult.total_nodes} total</span>}
          {mergeError && <span style={styles.graphOpErr}>{mergeError.length > 80 ? mergeError.slice(0, 80) + '...' : mergeError}</span>}
        </form>
      </div>

      {/* Stats Grid */}
      <div style={styles.statsGrid}>
        <StatCard label="Total Nodes" value={stats.total_nodes} />
        <StatCard
          label="Avg Confidence"
          value={`${(stats.avg_confidence * 100).toFixed(1)}%`}
          sub={stats.avg_confidence >= 0.8 ? 'High' : stats.avg_confidence >= 0.5 ? 'Medium' : 'Low'}
        />
        <StatCard label="Domains" value={domains.length} />
        <StatCard
          label="Anomalies"
          value={anomalies.length}
          sub={anomalies.length > 0 ? 'Needs review' : 'All clear'}
        />
        <StatCard label="Conflicts" value={stats.nodes_with_conflicts} />
      </div>

      {/* Distribution panels */}
      <div style={styles.panelRow}>
        {/* Domains */}
        <div className="card" style={styles.panel}>
          <h3 style={styles.panelTitle}>Domains</h3>
          {domains.length === 0 ? (
            <div style={styles.emptyPanel}>No domains yet</div>
          ) : (
            <div style={styles.barList}>
              {domains.sort((a, b) => b[1] - a[1]).map(([domain, count]) => (
                <div key={domain} style={styles.barItem}>
                  <div style={styles.barLabel}>
                    <span>{domain}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{count}</span>
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${(count / stats.total_nodes) * 100}%`,
                        background: 'var(--accent)',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Node Types */}
        <div className="card" style={styles.panel}>
          <h3 style={styles.panelTitle}>Node Types</h3>
          {types.length === 0 ? (
            <div style={styles.emptyPanel}>No nodes yet</div>
          ) : (
            <div style={styles.barList}>
              {types.sort((a, b) => b[1] - a[1]).map(([type, count]) => (
                <div key={type} style={styles.barItem}>
                  <div style={styles.barLabel}>
                    <span>{type}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{count}</span>
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${(count / stats.total_nodes) * 100}%`,
                        background: 'var(--status-info)',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Confidence distribution */}
        <div className="card" style={styles.panel}>
          <h3 style={styles.panelTitle}>Confidence</h3>
          <div style={styles.barList}>
            <div style={styles.barItem}>
              <div style={styles.barLabel}>
                <span style={{ color: confidenceColor(0.9) }}>High (0.8+)</span>
                <span style={{ color: 'var(--text-muted)' }}>
                  {Object.values(stats.nodes_by_status).reduce((a, b) => a + b, 0) > 0 ? '' : '0'}
                </span>
              </div>
              <div className="confidence-bar">
                <div className="confidence-fill" style={{ width: `${stats.avg_confidence >= 0.8 ? 80 : stats.avg_confidence * 100}%`, background: confidenceColor(0.9) }} />
              </div>
            </div>
            <div style={styles.barItem}>
              <div style={styles.barLabel}>
                <span>Average</span>
                <span style={{ color: confidenceColor(stats.avg_confidence) }}>
                  {(stats.avg_confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="confidence-bar">
                <div className="confidence-fill" style={{ width: `${stats.avg_confidence * 100}%`, background: confidenceColor(stats.avg_confidence) }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Anomalies */}
      {anomalies.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <h3 style={styles.panelTitle}>Active Anomalies</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
            {anomalies.slice(0, 5).map((a) => (
              <div key={a.node_id} style={styles.anomalyRow}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 500 }}>{a.node_text}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                    {a.anomaly_type} | Expected: {(a.expected_confidence * 100).toFixed(0)}%, Actual: {(a.actual_confidence * 100).toFixed(0)}%
                  </div>
                </div>
                <span className={`badge ${a.severity > 0.5 ? 'badge-red' : 'badge-yellow'}`}>
                  {(a.severity * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* File Browser Modal */}
      {browseTarget && (
        <FileBrowser
          mode={browseTarget === 'save' ? 'save' : 'open'}
          title={browseTarget === 'load' ? 'Open Graph' : browseTarget === 'save' ? 'Save Graph' : 'Select Graph to Merge'}
          onSelect={handleFileBrowseSelect}
          onCancel={() => setBrowseTarget(null)}
        />
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  title: {
    fontSize: 24,
    fontWeight: 600,
    marginBottom: 20,
    color: 'var(--text-primary)',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
    gap: 12,
    marginBottom: 20,
  },
  statCard: {
    textAlign: 'center',
  },
  statLabel: {
    fontSize: 12,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 28,
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  statSub: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    marginTop: 2,
  },
  panelRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 12,
  },
  panel: {
    display: 'flex',
    flexDirection: 'column',
  },
  panelTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: 'var(--text-primary)',
    marginBottom: 12,
  },
  emptyPanel: {
    color: 'var(--text-muted)',
    fontSize: 13,
    padding: '12px 0',
  },
  barList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  barItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  barLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 13,
    color: 'var(--text-secondary)',
  },
  anomalyRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '8px 12px',
    borderRadius: 6,
    background: 'var(--bg-primary)',
  },
  graphOpRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  } as React.CSSProperties,
  graphOpLabel: {
    fontSize: 13,
    color: 'var(--text-secondary)',
    fontWeight: 500,
    whiteSpace: 'nowrap',
    width: 42,
  } as React.CSSProperties,
  graphOpInput: {
    flex: 1,
    padding: '6px 10px',
    borderRadius: 6,
    fontSize: 13,
    border: '1px solid var(--border)',
    background: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    outline: 'none',
    fontFamily: 'monospace',
  } as React.CSSProperties,
  graphOpBtn: {
    padding: '6px 16px',
    fontSize: 13,
  } as React.CSSProperties,
  browseBtn: {
    padding: '6px 10px',
    fontSize: 13,
    fontFamily: 'monospace',
    fontWeight: 600,
  } as React.CSSProperties,
  graphOpOk: {
    fontSize: 12,
    color: 'var(--conf-high)',
    whiteSpace: 'nowrap',
  } as React.CSSProperties,
  graphOpErr: {
    fontSize: 12,
    color: 'var(--conf-low)',
    whiteSpace: 'nowrap',
  } as React.CSSProperties,
}
