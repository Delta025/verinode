import { useEffect, useState } from 'react'
import { getAnomalies, getNode, updateNode, deleteNode } from '../api/client'
import type { AnomalyItem, NodeDetail } from '../api/types'

const STATUS_OPTIONS = [
  'established', 'well_supported', 'provisional', 'contested', 'superseded', 'falsified',
]

function severityColor(s: number): string {
  if (s >= 0.7) return 'var(--conf-low)'
  if (s >= 0.4) return 'var(--conf-medium)'
  return 'var(--conf-high)'
}

function confColor(c: number): string {
  if (c >= 0.8) return 'var(--conf-high)'
  if (c >= 0.5) return 'var(--conf-medium)'
  return 'var(--conf-low)'
}

interface EditState {
  confidence: string
  domain: string
  status: string
  text: string
}

export default function AnomalyReview() {
  const [anomalies, setAnomalies] = useState<AnomalyItem[]>([])
  const [threshold, setThreshold] = useState(0.15)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Review panel
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyItem | null>(null)
  const [nodeDetail, setNodeDetail] = useState<NodeDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // Edit state
  const [editing, setEditing] = useState(false)
  const [editState, setEditState] = useState<EditState>({ confidence: '', domain: '', status: '', text: '' })
  const [saveLoading, setSaveLoading] = useState(false)
  const [saveResult, setSaveResult] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)

  // Delete
  const [confirmDelete, setConfirmDelete] = useState(false)

  function loadAnomalies() {
    setLoading(true)
    setError(null)
    getAnomalies(threshold)
      .then(setAnomalies)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadAnomalies() }, [threshold])

  async function selectAnomaly(a: AnomalyItem) {
    setSelectedAnomaly(a)
    setEditing(false)
    setSaveResult(null)
    setSaveError(null)
    setConfirmDelete(false)
    setDetailLoading(true)
    try {
      const detail = await getNode(a.node_id)
      setNodeDetail(detail)
      setEditState({
        confidence: (detail.confidence * 100).toFixed(1),
        domain: detail.domain,
        status: detail.status,
        text: detail.text,
      })
    } catch (e) {
      setNodeDetail(null)
      setSaveError(`Failed to load node: ${(e as Error).message}`)
    } finally {
      setDetailLoading(false)
    }
  }

  async function handleSave() {
    if (!nodeDetail) return
    setSaveLoading(true)
    setSaveResult(null)
    setSaveError(null)

    const updates: Record<string, unknown> = {}
    const confVal = parseFloat(editState.confidence) / 100
    if (!isNaN(confVal) && Math.abs(confVal - nodeDetail.confidence) > 0.001) {
      updates.confidence = Math.max(0, Math.min(1, confVal))
    }
    if (editState.domain !== nodeDetail.domain) updates.domain = editState.domain
    if (editState.status !== nodeDetail.status) updates.status = editState.status
    if (editState.text.trim() !== nodeDetail.text) updates.text = editState.text.trim()

    if (Object.keys(updates).length === 0) {
      setSaveError('No changes to save')
      setSaveLoading(false)
      return
    }

    try {
      const res = await updateNode(nodeDetail.id, updates)
      setSaveResult(`Updated: ${res.updated_fields.join(', ')}`)
      // Refresh the detail
      const refreshed = await getNode(nodeDetail.id)
      setNodeDetail(refreshed)
      setEditing(false)
      // Refresh anomaly list
      loadAnomalies()
    } catch (e) {
      setSaveError((e as Error).message)
    } finally {
      setSaveLoading(false)
    }
  }

  async function handleDelete() {
    if (!nodeDetail) return
    setSaveLoading(true)
    try {
      await deleteNode(nodeDetail.id)
      setSelectedAnomaly(null)
      setNodeDetail(null)
      setConfirmDelete(false)
      loadAnomalies()
    } catch (e) {
      setSaveError(`Delete failed: ${(e as Error).message}`)
    } finally {
      setSaveLoading(false)
    }
  }

  function dismiss() {
    // Remove from local list without deleting from graph
    if (selectedAnomaly) {
      setAnomalies((prev) => prev.filter((a) => a.node_id !== selectedAnomaly.node_id))
      setSelectedAnomaly(null)
      setNodeDetail(null)
    }
  }

  return (
    <div className="page-content" style={{ display: 'flex', gap: 16, maxWidth: 1200 }}>
      {/* Left: anomaly list */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <h1 style={styles.title}>Anomaly Review</h1>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
          Review flagged anomalies, edit node properties, or dismiss false positives.
        </p>

        {/* Controls */}
        <div style={styles.controls}>
          <label style={styles.controlLabel}>
            Threshold: {threshold.toFixed(2)}
          </label>
          <input
            type="range" min={0.05} max={1.0} step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            style={{ width: 180 }}
          />
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {anomalies.length} anomalies
          </span>
          <button className="btn" onClick={loadAnomalies} style={{ marginLeft: 'auto', fontSize: 12 }}>
            Refresh
          </button>
        </div>

        {/* Anomaly list */}
        {loading ? (
          <div className="loading-overlay"><div className="spinner" /> Detecting anomalies...</div>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : anomalies.length === 0 ? (
          <div className="empty-state" style={{ marginTop: 40 }}>
            <div style={{ fontSize: 24 }}>OK</div>
            <div>No anomalies at threshold {threshold.toFixed(2)}</div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {anomalies.map((a) => (
              <div
                key={a.node_id}
                style={{
                  ...styles.anomalyRow,
                  background: selectedAnomaly?.node_id === a.node_id ? 'var(--bg-hover)' : 'var(--bg-card)',
                  borderColor: selectedAnomaly?.node_id === a.node_id ? 'var(--accent)' : 'var(--border)',
                }}
                onClick={() => selectAnomaly(a)}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {a.node_text}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, display: 'flex', gap: 12 }}>
                    <span>{a.anomaly_type}</span>
                    <span>expected {(a.expected_confidence * 100).toFixed(0)}%</span>
                    <span>actual {(a.actual_confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <span style={{ fontWeight: 600, fontSize: 13, color: severityColor(a.severity), whiteSpace: 'nowrap' }}>
                  {(a.severity * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Right: review panel */}
      {selectedAnomaly && (
        <div style={styles.reviewPanel}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Review Node</h3>

          {detailLoading ? (
            <div className="loading-overlay"><div className="spinner" /> Loading node...</div>
          ) : !nodeDetail ? (
            <div className="error-message">Node not found</div>
          ) : (
            <>
              {/* Anomaly summary */}
              <div style={styles.anomalySummary}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                  Anomaly Info
                </div>
                <div style={styles.summaryRow}>
                  <span>Type</span>
                  <span className="badge badge-yellow">{selectedAnomaly.anomaly_type}</span>
                </div>
                <div style={styles.summaryRow}>
                  <span>Expected</span>
                  <span style={{ color: confColor(selectedAnomaly.expected_confidence) }}>
                    {(selectedAnomaly.expected_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={styles.summaryRow}>
                  <span>Actual</span>
                  <span style={{ color: confColor(selectedAnomaly.actual_confidence) }}>
                    {(selectedAnomaly.actual_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={styles.summaryRow}>
                  <span>Deviation</span>
                  <span>{selectedAnomaly.deviation >= 0 ? '+' : ''}{(selectedAnomaly.deviation * 100).toFixed(1)}%</span>
                </div>
                <div style={styles.summaryRow}>
                  <span>Severity</span>
                  <span style={{ fontWeight: 600, color: severityColor(selectedAnomaly.severity) }}>
                    {(selectedAnomaly.severity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Node content */}
              {!editing ? (
                <>
                  <div style={styles.nodeContent}>
                    <div style={{ fontSize: 13, lineHeight: 1.6 }}>{nodeDetail.text}</div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 12 }}>
                    <div style={styles.summaryRow}><span>Confidence</span><span style={{ color: confColor(nodeDetail.confidence) }}>{(nodeDetail.confidence * 100).toFixed(1)}%</span></div>
                    <div style={styles.summaryRow}><span>Domain</span><span className="badge badge-blue">{nodeDetail.domain}</span></div>
                    <div style={styles.summaryRow}><span>Status</span><span>{nodeDetail.status}</span></div>
                    <div style={styles.summaryRow}><span>Type</span><span>{nodeDetail.node_type}</span></div>
                    <div style={styles.summaryRow}><span>Sources</span><span>{nodeDetail.sources.length}</span></div>
                    <div style={styles.summaryRow}><span>Conflicts</span><span style={{ color: nodeDetail.conflicts_with.length > 0 ? 'var(--conf-low)' : 'var(--text-muted)' }}>{nodeDetail.conflicts_with.length}</span></div>
                  </div>
                </>
              ) : (
                <>
                  {/* Editable fields */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 12 }}>
                    <div>
                      <label style={styles.editLabel}>Text</label>
                      <textarea
                        value={editState.text}
                        onChange={(e) => setEditState((s) => ({ ...s, text: e.target.value }))}
                        style={styles.editTextarea}
                        rows={3}
                      />
                    </div>
                    <div style={{ display: 'flex', gap: 10 }}>
                      <div style={{ flex: 1 }}>
                        <label style={styles.editLabel}>Confidence (%)</label>
                        <input
                          type="number" min={0} max={100} step={0.1}
                          value={editState.confidence}
                          onChange={(e) => setEditState((s) => ({ ...s, confidence: e.target.value }))}
                          style={styles.editInput}
                        />
                      </div>
                      <div style={{ flex: 1 }}>
                        <label style={styles.editLabel}>Domain</label>
                        <input
                          type="text"
                          value={editState.domain}
                          onChange={(e) => setEditState((s) => ({ ...s, domain: e.target.value }))}
                          style={styles.editInput}
                        />
                      </div>
                    </div>
                    <div>
                      <label style={styles.editLabel}>Status</label>
                      <select
                        value={editState.status}
                        onChange={(e) => setEditState((s) => ({ ...s, status: e.target.value }))}
                        style={styles.editInput}
                      >
                        {STATUS_OPTIONS.map((s) => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </>
              )}

              {/* Result messages */}
              {saveResult && <div style={{ fontSize: 12, color: 'var(--conf-high)', marginBottom: 8 }}>{saveResult}</div>}
              {saveError && <div style={{ fontSize: 12, color: 'var(--conf-low)', marginBottom: 8 }}>{saveError}</div>}

              {/* Action buttons */}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {!editing ? (
                  <>
                    <button className="btn btn-primary" onClick={() => setEditing(true)}>
                      Edit
                    </button>
                    <button className="btn" onClick={dismiss}>
                      Dismiss
                    </button>
                    <button
                      className="btn"
                      style={{ color: 'var(--conf-low)', borderColor: 'var(--conf-low)' }}
                      onClick={() => setConfirmDelete(true)}
                    >
                      Delete
                    </button>
                  </>
                ) : (
                  <>
                    <button className="btn btn-primary" onClick={handleSave} disabled={saveLoading}>
                      {saveLoading ? 'Saving...' : 'Save'}
                    </button>
                    <button className="btn" onClick={() => {
                      setEditing(false)
                      setSaveError(null)
                      setSaveResult(null)
                      if (nodeDetail) {
                        setEditState({
                          confidence: (nodeDetail.confidence * 100).toFixed(1),
                          domain: nodeDetail.domain,
                          status: nodeDetail.status,
                          text: nodeDetail.text,
                        })
                      }
                    }}>
                      Cancel
                    </button>
                  </>
                )}
              </div>

              {/* Delete confirmation */}
              {confirmDelete && (
                <div style={styles.deleteConfirm}>
                  <div style={{ fontSize: 12, color: 'var(--conf-low)', marginBottom: 8 }}>
                    Permanently remove this node? This cannot be undone.
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="btn"
                      style={{ background: 'var(--conf-low)', color: 'white', borderColor: 'var(--conf-low)' }}
                      onClick={handleDelete}
                      disabled={saveLoading}
                    >
                      Confirm Delete
                    </button>
                    <button className="btn" onClick={() => setConfirmDelete(false)}>Cancel</button>
                  </div>
                </div>
              )}

              {/* Node ID */}
              <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                ID: {nodeDetail.id}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  title: { fontSize: 24, fontWeight: 600, marginBottom: 4 },
  controls: {
    display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12,
    padding: '8px 0',
  },
  controlLabel: {
    fontSize: 13, color: 'var(--text-secondary)', minWidth: 100,
  },
  anomalyRow: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '10px 12px', borderRadius: 6,
    border: '1px solid var(--border)', cursor: 'pointer',
    transition: 'background 0.1s, border-color 0.1s',
  },
  reviewPanel: {
    width: 380, minWidth: 320, maxWidth: '40vw',
    padding: 16, background: 'var(--bg-secondary)',
    borderRadius: 8, border: '1px solid var(--border)',
    alignSelf: 'flex-start', position: 'sticky', top: 16,
  },
  anomalySummary: {
    padding: '10px 12px', background: 'var(--bg-primary)',
    borderRadius: 6, border: '1px solid var(--border)',
    marginBottom: 12,
  },
  summaryRow: {
    display: 'flex', justifyContent: 'space-between',
    fontSize: 12, color: 'var(--text-secondary)', padding: '2px 0',
  },
  nodeContent: {
    padding: '10px 12px', background: 'var(--bg-card)',
    borderRadius: 6, border: '1px solid var(--border)',
    marginBottom: 12,
  },
  editLabel: {
    display: 'block', fontSize: 11, fontWeight: 600,
    color: 'var(--text-muted)', textTransform: 'uppercase',
    letterSpacing: '0.5px', marginBottom: 4,
  },
  editInput: {
    width: '100%', padding: '6px 10px', borderRadius: 6,
    fontSize: 13, border: '1px solid var(--border)',
    background: 'var(--bg-primary)', color: 'var(--text-primary)',
    outline: 'none',
  },
  editTextarea: {
    width: '100%', padding: '8px 10px', borderRadius: 6,
    fontSize: 13, border: '1px solid var(--border)',
    background: 'var(--bg-primary)', color: 'var(--text-primary)',
    outline: 'none', resize: 'vertical', fontFamily: 'inherit',
    lineHeight: 1.5,
  },
  deleteConfirm: {
    marginTop: 12, padding: '10px 12px',
    background: 'rgba(248,113,113,0.1)',
    borderRadius: 6, border: '1px solid rgba(248,113,113,0.3)',
  },
}
