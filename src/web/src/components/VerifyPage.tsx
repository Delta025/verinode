import { useState } from 'react'
import { verifyClaim } from '../api/client'
import type { VerifyResponse, EvidenceItem } from '../api/types'

function confidenceColor(c: number): string {
  if (c >= 0.8) return 'var(--conf-high)'
  if (c >= 0.5) return 'var(--conf-medium)'
  return 'var(--conf-low)'
}

function verdictStyle(verdict: string): { bg: string; color: string; label: string } {
  switch (verdict) {
    case 'supported':
      return { bg: 'rgba(74,222,128,0.12)', color: 'var(--conf-high)', label: 'SUPPORTED' }
    case 'contradicted':
      return { bg: 'rgba(248,113,113,0.12)', color: 'var(--conf-low)', label: 'CONTRADICTED' }
    case 'uncertain':
      return { bg: 'rgba(250,204,21,0.12)', color: 'var(--conf-medium)', label: 'UNCERTAIN' }
    case 'insufficient_evidence':
      return { bg: 'rgba(156,163,184,0.12)', color: 'var(--text-muted)', label: 'INSUFFICIENT EVIDENCE' }
    default:
      return { bg: 'rgba(156,163,184,0.08)', color: 'var(--text-secondary)', label: verdict.toUpperCase() }
  }
}

function EvidenceList({ items, type }: { items: EvidenceItem[]; type: 'supporting' | 'contradicting' }) {
  if (items.length === 0) return null

  const titleColor = type === 'supporting' ? 'var(--conf-high)' : 'var(--conf-low)'
  const title = type === 'supporting' ? 'Supporting Evidence' : 'Contradicting Evidence'

  return (
    <div style={{ marginBottom: 16 }}>
      <h3 style={{ fontSize: 14, fontWeight: 600, color: titleColor, marginBottom: 10 }}>
        {title} ({items.length})
      </h3>
      {items.map((ev, i) => (
        <div key={i} className="card" style={{ marginBottom: 8, padding: '12px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 14, lineHeight: 1.5, marginBottom: 6 }}>{ev.node_text}</div>
              {ev.explanation && (
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                  {ev.explanation}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flexShrink: 0 }}>
              <span style={{ color: confidenceColor(ev.confidence), fontWeight: 600, fontSize: 14 }}>
                {(ev.confidence * 100).toFixed(1)}%
              </span>
              <code style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {ev.node_id.slice(0, 8)}
              </code>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function VerifyPage() {
  const [claim, setClaim] = useState('')
  const [k, setK] = useState(10)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<VerifyResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!claim.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await verifyClaim({ claim: claim.trim(), k })
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const vs = result ? verdictStyle(result.verdict) : null

  return (
    <div className="page-content" style={styles.container}>
      <h1 style={styles.title}>Verify Claim</h1>
      <p style={styles.subtitle}>
        Check a factual claim against the verified knowledge graph. The verification engine finds
        supporting and contradicting evidence using 3-way relevance classification.
      </p>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.inputRow}>
          <input
            type="text"
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            placeholder='e.g. "Gold melts at 1064 degrees Celsius"'
            style={styles.claimInput}
          />
          <div style={styles.kField}>
            <label style={styles.kLabel}>k</label>
            <input
              type="number"
              min={1}
              max={50}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value) || 10)}
              style={styles.kInput}
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading || !claim.trim()}>
            {loading ? <><span className="spinner" /> Verifying...</> : 'Verify'}
          </button>
        </div>
      </form>

      {error && <div className="error-message" style={{ marginTop: 16 }}>{error}</div>}

      {result && vs && (
        <div style={{ marginTop: 24 }}>
          {/* Verdict card — large, prominent */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Claim</div>
              <div style={{ fontSize: 16, fontWeight: 500, lineHeight: 1.5 }}>{result.claim}</div>
            </div>

            {/* Verdict badge — centered, big */}
            <div style={{
              display: 'flex', justifyContent: 'center', padding: '20px 0',
              borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)',
            }}>
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                padding: '16px 32px', borderRadius: 12,
                background: vs.bg,
              }}>
                <span style={{ fontSize: 28, fontWeight: 700, color: vs.color, letterSpacing: '1px' }}>
                  {vs.label}
                </span>
                <span style={{ fontSize: 16, color: confidenceColor(result.confidence) }}>
                  {(result.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
            </div>

            {/* Explanation + meta */}
            <div style={{ marginTop: 16 }}>
              {result.explanation && (
                <div style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
                  {result.explanation}
                </div>
              )}
              <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-muted)' }}>
                <span>Coverage: {(result.coverage * 100).toFixed(0)}%</span>
                <span>Candidates checked: {result.num_candidates_checked}</span>
                <span>Supporting: {result.supporting_evidence.length}</span>
                <span>Contradicting: {result.contradicting_evidence.length}</span>
              </div>
            </div>
          </div>

          {/* Evidence lists */}
          <EvidenceList items={result.supporting_evidence} type="supporting" />
          <EvidenceList items={result.contradicting_evidence} type="contradicting" />

          {/* Raw response (collapsible) */}
          <details style={{ marginTop: 16 }}>
            <summary style={{ cursor: 'pointer', fontSize: 12, color: 'var(--text-muted)' }}>
              Raw response
            </summary>
            <pre style={styles.rawOutput}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 900 },
  title: { fontSize: 24, fontWeight: 600, marginBottom: 4 },
  subtitle: { fontSize: 14, color: 'var(--text-secondary)', marginBottom: 20, lineHeight: 1.5 },
  form: {},
  inputRow: { display: 'flex', gap: 8, alignItems: 'flex-end' },
  claimInput: {
    flex: 1, padding: '10px 14px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg-card)',
    color: 'var(--text-primary)', outline: 'none', fontSize: 14,
  },
  kField: { display: 'flex', flexDirection: 'column', gap: 2 },
  kLabel: { fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' },
  kInput: {
    width: 50, padding: '8px', borderRadius: 6, textAlign: 'center',
    border: '1px solid var(--border)', background: 'var(--bg-card)',
    color: 'var(--text-primary)', outline: 'none',
  },
  rawOutput: {
    padding: '12px', background: 'var(--bg-card)', borderRadius: 6,
    fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 300,
    overflowY: 'auto', border: '1px solid var(--border)', marginTop: 8,
  },
}
