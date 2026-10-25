import { useState, useEffect } from 'react'
import { browseFiles } from '../api/client'
import type { FileEntry, FileBrowseResponse } from '../api/types'

interface FileBrowserProps {
  onSelect: (path: string) => void
  onCancel: () => void
  mode: 'open' | 'save'
  title?: string
}

function formatSize(bytes: number): string {
  if (bytes === 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function FileBrowser({ onSelect, onCancel, mode, title }: FileBrowserProps) {
  const [data, setData] = useState<FileBrowseResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filename, setFilename] = useState('')

  function loadDir(dir?: string) {
    setLoading(true)
    setError(null)
    browseFiles(dir, 'graph')
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadDir() }, [])

  function handleEntryClick(entry: FileEntry) {
    if (entry.is_dir) {
      loadDir(entry.path)
    } else {
      onSelect(entry.path)
    }
  }

  function handleSave() {
    if (!filename.trim() || !data) return
    // Combine current dir with filename
    const sep = data.current_dir.includes('/') ? '/' : '\\'
    const path = data.current_dir + sep + filename.trim()
    onSelect(path)
  }

  return (
    <div style={styles.overlay} onClick={onCancel}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h3 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>
            {title || (mode === 'open' ? 'Open Graph' : 'Save Graph')}
          </h3>
          <button style={styles.closeBtn} onClick={onCancel}>x</button>
        </div>

        {/* Current path breadcrumb */}
        {data && (
          <div style={styles.breadcrumb}>
            {data.parent_dir && (
              <button
                style={styles.upBtn}
                onClick={() => loadDir(data.parent_dir!)}
                title="Go up"
              >
                ..
              </button>
            )}
            <code style={{ fontSize: 12, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {data.current_dir}
            </code>
          </div>
        )}

        {/* File list */}
        <div style={styles.fileList}>
          {loading ? (
            <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>Loading...</div>
          ) : error ? (
            <div style={{ padding: 20, textAlign: 'center', color: 'var(--conf-low)' }}>{error}</div>
          ) : data && data.entries.length === 0 ? (
            <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>No graph files found</div>
          ) : data && data.entries.map((entry) => (
            <div
              key={entry.path}
              style={styles.fileRow}
              onClick={() => handleEntryClick(entry)}
            >
              <span style={{ ...styles.fileIcon, color: entry.is_dir ? '#60a5fa' : '#9ca3b8' }}>
                {entry.is_dir ? '/' : '#'}
              </span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {entry.name}
              </span>
              {!entry.is_dir && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                  {formatSize(entry.size)}
                </span>
              )}
            </div>
          ))}
        </div>

        {/* Save mode: filename input */}
        {mode === 'save' && (
          <div style={styles.saveRow}>
            <input
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="my_graph.json"
              style={styles.saveInput}
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            />
            <button
              className="btn btn-primary"
              onClick={handleSave}
              disabled={!filename.trim()}
              style={{ padding: '6px 16px', fontSize: 13 }}
            >
              Save
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    background: 'var(--bg-secondary)', borderRadius: 8,
    border: '1px solid var(--border)', width: 520, maxWidth: '90vw',
    maxHeight: '70vh', display: 'flex', flexDirection: 'column',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '12px 16px', borderBottom: '1px solid var(--border)',
  },
  closeBtn: {
    background: 'none', border: 'none', color: 'var(--text-muted)',
    fontSize: 16, cursor: 'pointer',
  },
  breadcrumb: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 16px', borderBottom: '1px solid var(--border)',
    background: 'var(--bg-primary)',
  },
  upBtn: {
    padding: '2px 8px', fontSize: 13, fontWeight: 600, fontFamily: 'monospace',
    background: 'var(--bg-card)', border: '1px solid var(--border)',
    borderRadius: 4, color: 'var(--text-secondary)', cursor: 'pointer',
  },
  fileList: {
    flex: 1, overflowY: 'auto', minHeight: 200, maxHeight: 400,
  },
  fileRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 16px', cursor: 'pointer', fontSize: 13,
    color: 'var(--text-primary)', borderBottom: '1px solid var(--border)',
    transition: 'background 0.1s',
  },
  fileIcon: {
    fontSize: 14, fontWeight: 600, fontFamily: 'monospace',
    width: 20, textAlign: 'center',
  },
  saveRow: {
    display: 'flex', gap: 8, padding: '10px 16px',
    borderTop: '1px solid var(--border)',
  },
  saveInput: {
    flex: 1, padding: '6px 10px', borderRadius: 6,
    fontSize: 13, border: '1px solid var(--border)',
    background: 'var(--bg-primary)', color: 'var(--text-primary)',
    outline: 'none', fontFamily: 'monospace',
  },
}
