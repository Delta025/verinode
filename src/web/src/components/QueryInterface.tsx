import { useState } from 'react'
import { queryGraph } from '../api/client'
import type { QueryResponse } from '../api/types'

function confidenceColor(c: number): string {
  if (c >= 0.8) return 'var(--conf-high)'
  if (c >= 0.5) return 'var(--conf-medium)'
  return 'var(--conf-low)'
}

export default function QueryInterface() {
  const [question, setQuestion] = useState('')
  const [k, setK] = useState(10)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await queryGraph({ question: question.trim(), k })
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-content" style={styles.container}>
      <h1 style={styles.title}>Query</h1>
      <p style={styles.subtitle}>
        Ask questions about the knowledge graph. The reasoning engine finds relevant facts and builds answer chains.
      </p>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.inputRow}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What is the boiling point of water?"
            style={styles.questionInput}
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
          <button type="submit" className="btn btn-primary" disabled={loading || !question.trim()}>
            {loading ? <><span className="spinner" /> Querying...</> : 'Query'}
          </button>
        </div>
      </form>

      {error && <div className="error-message" style={{ marginTop: 16 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 20 }}>
          {/* Answer */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={styles.answerHeader}>
              <h3 style={{ fontSize: 14, fontWeight: 600 }}>Answer</h3>
              <div style={{
                fontSize: 14, fontWeight: 600,
                color: confidenceColor(result.confidence),
              }}>
                {(result.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>

            {result.answer ? (
              <div style={styles.answerText}>{result.answer}</div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', marginTop: 8 }}>
                No direct answer found. See reasoning chains below.
              </div>
            )}

            {result.relevant_node_ids.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-muted)' }}>
                {result.relevant_node_ids.length} relevant node(s) found
              </div>
            )}
          </div>

          {/* Reasoning Chains */}
          {result.chains.length > 0 && (
            <div>
              <h3 style={styles.sectionTitle}>Reasoning Chains</h3>
              {result.chains.map((chain, ci) => (
                <div key={ci} className="card" style={{ marginBottom: 12 }}>
                  <div style={styles.chainHeader}>
                    <span style={{ fontWeight: 500 }}>Chain {ci + 1}</span>
                    <span style={{
                      color: confidenceColor(chain.overall_confidence),
                      fontWeight: 600, fontSize: 13,
                    }}>
                      {(chain.overall_confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  {chain.explanation && (
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 8, marginBottom: 8 }}>
                      {chain.explanation}
                    </div>
                  )}

                  <div style={styles.stepsContainer}>
                    {chain.steps.map((step, si) => (
                      <div key={si} style={styles.stepItem}>
                        <div style={styles.stepConnector}>
                          <div style={stepDotStyle(step.confidence)} />
                          {si < chain.steps.length - 1 && <div style={styles.stepLine} />}
                        </div>
                        <div style={styles.stepContent}>
                          <div style={{ fontSize: 13 }}>{step.text}</div>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                            <span style={{ color: confidenceColor(step.confidence) }}>
                              {(step.confidence * 100).toFixed(0)}%
                            </span>
                            {' | '}{step.step_type}
                            {' | '}<code style={{ fontSize: 10 }}>{step.node_id.slice(0, 8)}</code>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
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

function stepDotStyle(confidence: number): React.CSSProperties {
  return {
    width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
    background: confidence >= 0.8 ? 'var(--conf-high)' : confidence >= 0.5 ? 'var(--conf-medium)' : 'var(--conf-low)',
    marginTop: 4,
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 900 },
  title: { fontSize: 24, fontWeight: 600, marginBottom: 4 },
  subtitle: { fontSize: 14, color: 'var(--text-secondary)', marginBottom: 20 },
  form: {},
  inputRow: { display: 'flex', gap: 8, alignItems: 'flex-end' },
  questionInput: {
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
  answerHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  },
  answerText: {
    marginTop: 12, padding: '12px', background: 'var(--bg-primary)',
    borderRadius: 6, fontSize: 14, lineHeight: 1.6,
    border: '1px solid var(--border)',
  },
  sectionTitle: {
    fontSize: 14, fontWeight: 600, marginBottom: 12,
  },
  chainHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  },
  stepsContainer: {
    marginTop: 12, display: 'flex', flexDirection: 'column',
  },
  stepItem: {
    display: 'flex', gap: 12,
  },
  stepConnector: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', width: 16,
  },
  stepLine: {
    width: 2, flex: 1, background: 'var(--border)', marginTop: 2,
  },
  stepContent: {
    paddingBottom: 16, flex: 1,
  },
  rawOutput: {
    padding: '12px', background: 'var(--bg-card)', borderRadius: 6,
    fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 300,
    overflowY: 'auto', border: '1px solid var(--border)', marginTop: 8,
  },
}
