import { NavLink, Outlet } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { getHealth } from '../api/client'
import type { HealthResponse } from '../api/types'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: '{}' },
  { to: '/verify', label: 'Verify', icon: '\u2713' },
  { to: '/nodes', label: 'Nodes', icon: '#' },
  { to: '/graph', label: 'Graph', icon: '~' },
  { to: '/ingest', label: 'Ingest', icon: '+' },
  { to: '/query', label: 'Query', icon: '?' },
  { to: '/hypothesis', label: 'Hypothesis', icon: '!' },
  { to: '/anomalies', label: 'Anomalies', icon: '^' },
  { to: '/review', label: 'Review', icon: '*' },
]

export default function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [connected, setConnected] = useState<boolean | null>(null)

  useEffect(() => {
    getHealth()
      .then((h) => { setHealth(h); setConnected(true) })
      .catch(() => setConnected(false))

    const interval = setInterval(() => {
      getHealth()
        .then((h) => { setHealth(h); setConnected(true) })
        .catch(() => setConnected(false))
    }, 15000)
    return () => clearInterval(interval)
  }, [])

  return (
    <>
      {/* Sidebar */}
      <nav style={styles.sidebar}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>V</span>
          <span style={styles.logoText}>Verinode</span>
        </div>

        <div style={styles.nav}>
          {NAV_ITEMS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              style={({ isActive }) => ({
                ...styles.navLink,
                ...(isActive ? styles.navLinkActive : {}),
              })}
            >
              <span style={styles.navIcon}>{icon}</span>
              {label}
            </NavLink>
          ))}
        </div>

        {/* Status bar */}
        <div style={styles.statusBar}>
          <div style={statusDotStyle(connected)} />
          <div style={styles.statusInfo}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              {connected === null ? 'Connecting...' : connected ? 'Connected' : 'Disconnected'}
            </div>
            {health && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {health.graph_nodes} nodes | {health.model || 'no model'}
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main style={styles.main}>
        <Outlet />
      </main>
    </>
  )
}

function statusDotStyle(connected: boolean | null): React.CSSProperties {
  return {
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
    background: connected === null
      ? 'var(--status-warning)'
      : connected
        ? 'var(--status-success)'
        : 'var(--status-error)',
  }
}

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 220,
    minWidth: 220,
    background: 'var(--bg-secondary)',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '20px 16px',
    borderBottom: '1px solid var(--border)',
  },
  logoIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    background: 'var(--accent)',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
    fontSize: 18,
  },
  logoText: {
    fontSize: 16,
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  nav: {
    flex: 1,
    padding: '12px 8px',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    overflowY: 'auto',
  },
  navLink: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '8px 12px',
    borderRadius: 6,
    color: 'var(--text-secondary)',
    fontSize: 14,
    textDecoration: 'none',
    transition: 'background 0.15s, color 0.15s',
  },
  navLinkActive: {
    background: 'var(--bg-hover)',
    color: 'var(--accent)',
  },
  navIcon: {
    width: 24,
    textAlign: 'center' as const,
    fontSize: 14,
    fontWeight: 600,
    fontFamily: 'monospace',
    color: 'inherit',
  },
  statusBar: {
    padding: '12px 16px',
    borderTop: '1px solid var(--border)',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  statusInfo: {
    display: 'flex',
    flexDirection: 'column' as const,
  },
  main: {
    flex: 1,
    minWidth: 0,
    minHeight: 0,
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
    overflowX: 'hidden',
  },
}
