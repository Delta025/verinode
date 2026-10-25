import { useState } from 'react'
import { ingest } from '../api/client'
import type { IngestionResponse } from '../api/types'

export default function IngestForm() {
  const [text, setText] = useState('')
  const [domain, setDomain] = useState('')
  const [sourceType, setSourceType] = useState('user_input')
  const [sourceTitle, setSourceTitle] = useState('')
  const [sourceQuality, setSourceQuality] = useState(0.7)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<IngestionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!text.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await ingest({
        text: text.trim(),
        domain: domain || undefined,
        source_type: sourceType,
        source_title: sourceTitle || undefined,
        source_quality: sourceQuality,
      })
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  function actionColor(action: string) {
    switch (action) {
      case 'added': return 'var(--conf-high)'
      case 'duplicate': return 'var(--status-info)'
      case 'flagged': return 'var(--conf-medium)'
      case 'rejected': return 'var(--conf-low)'
      default: return 'var(--text-secondary)'
    }
  }

  return (
    <div className="page-content" style={styles.container}>
      <h1 style={styles.title}>Ingest Text</h1>
      <p style={styles.subtitle}>
        Extract facts from text using the LLM pipeline. Requires a running Ollama server.
      </p>

      <form onSubmit={handleSubmit} style={styles.form}>
        {/* Text input */}
        <div style={styles.field}>
          <label style={styles.label}>Text to Ingest</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text containing facts to extract...&#10;&#10;Example: Water boils at 100 degrees Celsius at standard atmospheric pressure. It freezes at 0 degrees Celsius."
            style={styles.textarea}
            rows={6}
          />
        </div>

        {/* Metadata row */}
        <div style={styles.metaRow}>
          <div style={styles.field}>
            <label style={styles.label}>Domain</label>
            <input
              type="text"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              placeholder="e.g., chemistry, physics"
              style={styles.input}
            />
          </div>

          <div style={styles.field}>
            <label style={styles.label}>Source Type</label>
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)} style={styles.input}>
              <option value="user_input">User Input</option>
              <option value="textbook">Textbook</option>
              <option value="journal">Journal</option>
              <option value="web">Web</option>
              <option value="expert">Expert</option>
            </select>
          </div>

          <div style={styles.field}>
            <label style={styles.label}>Source Title</label>
            <input
              type="text"
              value={sourceTitle}
              onChange={(e) => setSourceTitle(e.target.value)}
              placeholder="Optional title"
              style={styles.input}
            />
          </div>

          <div style={styles.field}>
            <label style={styles.label}>Quality ({(sourceQuality * 100).toFixed(0)}%)</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={sourceQuality}
              onChange={(e) => setSourceQuality(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || !text.trim()}
          style={{ marginTop: 8 }}
        >
          {loading ? <><span className="spinner" /> Ingesting...</> : 'Ingest'}
        </button>
      </form>

      {/* Error */}
      {error && <div className="error-message" style={{ marginTop: 16 }}>{error}</div>}

      {/* Results */}
      {result && (
        <div className="card" style={{ marginTop: 20 }}>
          <h3 style={styles.resultTitle}>Ingestion Results</h3>

          {/* Summary */}
          <div style={styles.summaryBar}>
            <div style={styles.summaryItem}>
              <span style={styles.summaryLabel}>Extracted</span>
              <span style={styles.summaryValue}>{result.total_extracted}</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryLabel}>Added</span>
              <span style={{ ...styles.summaryValue, color: 'var(--conf-high)' }}>{result.nodes_added}</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryLabel}>Duplicates</span>
              <span style={{ ...styles.summaryValue, color: 'var(--status-info)' }}>{result.duplicates_merged}</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryLabel}>Flagged</span>
              <span style={{ ...styles.summaryValue, color: 'var(--conf-medium)' }}>{result.facts_flagged}</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryLabel}>Rejected</span>
              <span style={{ ...styles.summaryValue, color: 'var(--conf-low)' }}>{result.facts_rejected}</span>
            </div>
            <div style={styles.summaryItem}>
              <span style={styles.summaryLabel}>Dependencies</span>
              <span style={styles.summaryValue}>{result.dependencies_added}</span>
            </div>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 8 }}>
            {result.summary}
          </div>

          {/* Verdicts */}
          {result.verdicts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h4 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>
                Fact Verdicts
              </h4>
              {result.verdicts.map((v, i) => (
                <div key={i} style={styles.verdictItem}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                      background: actionColor(v.action),
                    }} />
                    <span style={{ fontSize: 13 }}>{v.fact_text}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4, paddingLeft: 16 }}>
                    {v.action.toUpperCase()}
                    {v.reason && ` — ${v.reason}`}
                    {v.node_id && (
                      <span style={{ fontFamily: 'monospace', marginLeft: 8 }}>
                        [{v.node_id.slice(0, 8)}]
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
  container: { maxWidth: 900 },
  title: { fontSize: 24, fontWeight: 600, marginBottom: 4 },
  subtitle: { fontSize: 14, color: 'var(--text-secondary)', marginBottom: 20 },
  form: { display: 'flex', flexDirection: 'column', gap: 12 },
  field: { display: 'flex', flexDirection: 'column', gap: 4, flex: 1 },
  label: { fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' },
  textarea: {
    padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border)',
    background: 'var(--bg-card)', color: 'var(--text-primary)', resize: 'vertical',
    outline: 'none', fontFamily: 'inherit', lineHeight: 1.5,
  },
  input: {
    padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border)',
    background: 'var(--bg-card)', color: 'var(--text-primary)', outline: 'none',
  },
  metaRow: {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: 12,
  },
  resultTitle: { fontSize: 16, fontWeight: 600, marginBottom: 12 },
  summaryBar: {
    display: 'flex', gap: 20, flexWrap: 'wrap', padding: '12px 0',
    borderBottom: '1px solid var(--border)',
  },
  summaryItem: { display: 'flex', flexDirection: 'column', alignItems: 'center' },
  summaryLabel: { fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' },
  summaryValue: { fontSize: 22, fontWeight: 700, color: 'var(--text-primary)' },
  verdictItem: {
    padding: '8px 12px', background: 'var(--bg-primary)', borderRadius: 6,
    marginBottom: 6, border: '1px solid var(--border)',
  },
}
