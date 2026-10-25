import { useEffect, useState } from 'react'
import { getAnomalies, getHistory } from '../api/client'
import type { AnomalyItem, ChangelogEntry } from '../api/types'

function severityColor(s: number): string {
  if (s >= 0.7) return 'var(--conf-low)'
  if (s >= 0.4) return 'var(--conf-medium)'
  return 'var(--conf-high)'
}

export default function AnomalyList() {
  const [anomalies, setAnomalies] = useState<AnomalyItem[]>([])
  const [history, setHistory] = useState<ChangelogEntry[]>([])
  const [threshold, setThreshold] = useState(0.3)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'anomalies' | 'history'>('anomalies')

  useEffect(() => {
    setLoading(true)
    Promise.all([getAnomalies(threshold), getHistory(100)])
      .then(([a, h]) => { setAnomalies(a); setHistory(h) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [threshold])

  if (error) return <div className="page-content"><div className="error-message">Error: {error}</div></div>

  return (
    <div className="page-content" style={styles.container}>
      <h1 style={styles.title}>Diagnostics</h1>

      {/* Tab switcher */}
      <div style={styles.tabs}>
        <button
          className="btn"
          style={tab === 'anomalies' ? styles.tabActive : undefined}
          onClick={() => setTab('anomalies')}
        >
          Anomalies ({anomalies.length})
        </button>
        <button
          className="btn"
          style={tab === 'history' ? styles.tabActive : undefined}
          onClick={() => setTab('history')}
        >
          Change History ({history.length})
        </button>
      </div>

      {/* Anomalies tab */}
      {tab === 'anomalies' && (
        <div>
          <div style={styles.controls}>
            <label style={styles.controlLabel}>
              Threshold: {threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min={0.05}
              max={1.0}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: 200 }}
            />
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Lower = more sensitive
            </span>
          </div>

          {loading ? (
            <div className="loading-overlay"><div className="spinner" /> Detecting anomalies...</div>
          ) : anomalies.length === 0 ? (
            <div className="empty-state" style={{ marginTop: 40 }}>
              <div style={{ fontSize: 32 }}>OK</div>
              <div>No anomalies detected at threshold {threshold.toFixed(2)}</div>
              <div style={{ fontSize: 12 }}>Try lowering the threshold for more sensitive detection.</div>
            </div>
          ) : (
            <div style={styles.anomalyList}>
              <div style={styles.tableHeader}>
                <span style={{ flex: 1 }}>Node</span>
                <span style={{ width: 120, textAlign: 'center' }}>Type</span>
                <span style={{ width: 80, textAlign: 'center' }}>Expected</span>
                <span style={{ width: 80, textAlign: 'center' }}>Actual</span>
                <span style={{ width: 80, textAlign: 'center' }}>Deviation</span>
                <span style={{ width: 80, textAlign: 'center' }}>Severity</span>
              </div>

              {anomalies.map((a) => (
                <div key={a.node_id} style={styles.anomalyRow}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={styles.nodeText}>{a.node_text}</div>
                    <code style={styles.nodeId}>{a.node_id.slice(0, 8)}</code>
                  </div>
                  <div style={{ width: 120, textAlign: 'center' }}>
                    <span className="badge badge-yellow">{a.anomaly_type}</span>
                  </div>
                  <div style={{ width: 80, textAlign: 'center', fontSize: 13 }}>
                    {(a.expected_confidence * 100).toFixed(0)}%
                  </div>
                  <div style={{ width: 80, textAlign: 'center', fontSize: 13 }}>
                    {(a.actual_confidence * 100).toFixed(0)}%
                  </div>
                  <div style={{ width: 80, textAlign: 'center', fontSize: 13 }}>
                    {a.deviation >= 0 ? '+' : ''}{(a.deviation * 100).toFixed(0)}%
                  </div>
                  <div style={{ width: 80, textAlign: 'center' }}>
                    <span style={{
                      fontWeight: 600, fontSize: 13,
                      color: severityColor(a.severity),
                    }}>
                      {(a.severity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* History tab */}
      {tab === 'history' && (
        <div>
          {loading ? (
            <div className="loading-overlay"><div className="spinner" /> Loading history...</div>
          ) : history.length === 0 ? (
            <div className="empty-state" style={{ marginTop: 40 }}>
              <div>No change history available</div>
              <div style={{ fontSize: 12 }}>History is only tracked when using SQLite storage.</div>
            </div>
          ) : (
            <div style={styles.historyList}>
              {history.map((entry) => (
                <div key={entry.id} style={styles.historyRow}>
                  <div style={styles.historyTime}>
                    {new Date(entry.timestamp).toLocaleString()}
                  </div>
                  <div style={styles.historyContent}>
                    <span className={`badge ${
                      entry.change_type === 'add' ? 'badge-green' :
                      entry.change_type === 'update' ? 'badge-blue' :
                      entry.change_type === 'remove' ? 'badge-red' : 'badge-yellow'
                    }`}>
                      {entry.change_type}
                    </span>
                    {entry.node_id && (
                      <code style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {entry.node_id.slice(0, 8)}
                      </code>
                    )}
                    {Object.keys(entry.details).length > 0 && (
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {JSON.stringify(entry.details).slice(0, 80)}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 1000 },
  title: { fontSize: 24, fontWeight: 600, marginBottom: 16 },
  tabs: { display: 'flex', gap: 8, marginBottom: 16 },
  tabActive: {
    background: 'var(--accent)', borderColor: 'var(--accent)', color: 'white',
  },
  controls: {
    display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16,
    padding: '8px 0',
  },
  controlLabel: {
    fontSize: 13, color: 'var(--text-secondary)', minWidth: 100,
  },
  anomalyList: {
    display: 'flex', flexDirection: 'column',
  },
  tableHeader: {
    display: 'flex', alignItems: 'center', padding: '8px 12px',
    fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
    textTransform: 'uppercase', letterSpacing: '0.5px',
    borderBottom: '1px solid var(--border)',
  },
  anomalyRow: {
    display: 'flex', alignItems: 'center', padding: '10px 12px',
    borderBottom: '1px solid var(--border)',
    transition: 'background 0.1s',
  },
  nodeText: {
    fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  nodeId: {
    fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace',
  },
  historyList: {
    display: 'flex', flexDirection: 'column', gap: 4,
  },
  historyRow: {
    display: 'flex', gap: 16, padding: '8px 12px',
    borderBottom: '1px solid var(--border)',
  },
  historyTime: {
    fontSize: 12, color: 'var(--text-muted)', minWidth: 160,
    fontFamily: 'monospace',
  },
  historyContent: {
    display: 'flex', gap: 8, alignItems: 'center', flex: 1,
  },
}
