import { useState } from 'react'
import { testHypothesis } from '../api/client'
import type { HypothesisResponse } from '../api/types'

function confidenceColor(c: number): string {
  if (c >= 0.8) return 'var(--conf-high)'
  if (c >= 0.5) return 'var(--conf-medium)'
  return 'var(--conf-low)'
}

function verdictStyle(verdict: string): { color: string; label: string } {
  switch (verdict) {
    case 'supported':
      return { color: 'var(--conf-high)', label: 'SUPPORTED' }
    case 'contradicted':
      return { color: 'var(--conf-low)', label: 'CONTRADICTED' }
    case 'uncertain':
      return { color: 'var(--conf-medium)', label: 'UNCERTAIN' }
    case 'insufficient_evidence':
      return { color: 'var(--text-muted)', label: 'INSUFFICIENT EVIDENCE' }
    default:
      return { color: 'var(--text-secondary)', label: verdict.toUpperCase() }
  }
}

export default function HypothesisTest() {
  const [hypothesis, setHypothesis] = useState('')
  const [k, setK] = useState(10)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<HypothesisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!hypothesis.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await testHypothesis({ hypothesis: hypothesis.trim(), k })
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-content" style={styles.container}>
      <h1 style={styles.title}>Hypothesis Testing</h1>
      <p style={styles.subtitle}>
        Test a claim against the knowledge graph. The reasoning engine finds supporting and contradicting evidence.
      </p>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.inputRow}>
          <input
            type="text"
            value={hypothesis}
            onChange={(e) => setHypothesis(e.target.value)}
            placeholder="Gold is a reactive metal"
            style={styles.hypothesisInput}
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
          <button type="submit" className="btn btn-primary" disabled={loading || !hypothesis.trim()}>
            {loading ? <><span className="spinner" /> Testing...</> : 'Test'}
          </button>
        </div>
      </form>

      {error && <div className="error-message" style={{ marginTop: 16 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 20 }}>
          {/* Verdict card */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={styles.verdictHeader}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
                  Hypothesis
                </div>
                <div style={{ fontSize: 16, fontWeight: 500 }}>{result.hypothesis}</div>
              </div>
            </div>

            <div style={styles.verdictBody}>
              <div style={styles.verdictBadge}>
                <span style={{
                  fontSize: 22, fontWeight: 700,
                  color: verdictStyle(result.verdict).color,
                }}>
                  {verdictStyle(result.verdict).label}
                </span>
                <span style={{
                  fontSize: 14,
                  color: confidenceColor(result.confidence),
                }}>
                  {(result.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
            </div>
          </div>

          {/* Supporting chains */}
          {result.supporting_chains.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <h3 style={{ ...styles.sectionTitle, color: 'var(--conf-high)' }}>
                Supporting Evidence ({result.supporting_chains.length})
              </h3>
              {result.supporting_chains.map((chain, ci) => (
                <div key={ci} className="card" style={{ marginBottom: 8 }}>
                  <div style={styles.chainHeader}>
                    <span>Chain {ci + 1}</span>
                    <span style={{ color: confidenceColor(chain.overall_confidence) }}>
                      {(chain.overall_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {chain.explanation && (
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 6 }}>
                      {chain.explanation}
                    </div>
                  )}
                  {chain.steps.map((step, si) => (
                    <div key={si} style={styles.stepItem}>
                      <span style={{ color: confidenceColor(step.confidence), fontSize: 12, minWidth: 40 }}>
                        {(step.confidence * 100).toFixed(0)}%
                      </span>
                      <span style={{ fontSize: 13 }}>{step.text}</span>
                      <code style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        {step.node_id.slice(0, 8)}
                      </code>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Contradicting chains */}
          {result.contradicting_chains.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <h3 style={{ ...styles.sectionTitle, color: 'var(--conf-low)' }}>
                Contradicting Evidence ({result.contradicting_chains.length})
              </h3>
              {result.contradicting_chains.map((chain, ci) => (
                <div key={ci} className="card" style={{ marginBottom: 8 }}>
                  <div style={styles.chainHeader}>
                    <span>Chain {ci + 1}</span>
                    <span style={{ color: confidenceColor(chain.overall_confidence) }}>
                      {(chain.overall_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {chain.explanation && (
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 6 }}>
                      {chain.explanation}
                    </div>
                  )}
                  {chain.steps.map((step, si) => (
                    <div key={si} style={styles.stepItem}>
                      <span style={{ color: confidenceColor(step.confidence), fontSize: 12, minWidth: 40 }}>
                        {(step.confidence * 100).toFixed(0)}%
                      </span>
                      <span style={{ fontSize: 13 }}>{step.text}</span>
                      <code style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        {step.node_id.slice(0, 8)}
                      </code>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Debug: raw response */}
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
  subtitle: { fontSize: 14, color: 'var(--text-secondary)', marginBottom: 20 },
  form: {},
  inputRow: { display: 'flex', gap: 8, alignItems: 'flex-end' },
  hypothesisInput: {
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
  verdictHeader: { marginBottom: 16 },
  verdictBody: { display: 'flex', justifyContent: 'center', padding: '16px 0' },
  verdictBadge: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
  },
  sectionTitle: { fontSize: 14, fontWeight: 600, marginBottom: 8 },
  chainHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: 8, fontSize: 13, fontWeight: 500,
  },
  stepItem: {
    display: 'flex', gap: 8, alignItems: 'baseline', padding: '4px 0',
    borderTop: '1px solid var(--border)',
  },
  rawOutput: {
    padding: '12px', background: 'var(--bg-card)', borderRadius: 6,
    fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 300,
    overflowY: 'auto', border: '1px solid var(--border)', marginTop: 8,
  },
}
