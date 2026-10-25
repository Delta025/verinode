import { useEffect, useState, useRef, useCallback, useMemo } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import type { Core, LayoutOptions, NodeSingular } from 'cytoscape'
// @ts-expect-error no types for cytoscape-fcose
import fcose from 'cytoscape-fcose'
import { getGraphData, getNode, explainNode, traceNode } from '../api/client'
import type { ElementDefinition } from 'cytoscape'
import type { GraphData, NodeDetail, ExplainResponse, TraceResponse, CytoscapeEdge } from '../api/types'

// Register fcose layout once
try { cytoscape.use(fcose) } catch { /* already registered */ }

/**
 * Graph view with dynamic connection-based Level-of-Detail.
 *
 * Performance strategy (based on Cytoscape.js perf research):
 * - fcose layout with quality="draft" for large graphs (spectral only, no iterative)
 * - animate:false on initial layout to avoid blocking the main thread
 * - Straight edges instead of bezier (2x faster rendering)
 * - Full opacity on edges (opaque + arrows is 2x faster than semi-transparent)
 * - pixelRatio:1 to avoid 4x overhead on HiDPI displays
 * - min-zoomed-font-size to skip label rendering when zoomed out
 * - Budget slider with debounce to avoid spamming layout reruns
 * - Zoom-based dynamic LOD: budget adjusts as you zoom in/out
 */

// ============================================================
//  Types & topology
// ============================================================

interface NodeRank {
  id: string
  degree: number
  label: string
  confidence: number
  domain: string
  nodeType: string
  status: string
  neighbors: string[]
  component: number
}

interface GraphTopology {
  ranked: NodeRank[]
  byId: Map<string, NodeRank>
  edges: CytoscapeEdge[]
  nodeCount: number
  componentCount: number
}

function buildTopology(graphData: GraphData): GraphTopology {
  const adjacency = new Map<string, Set<string>>()
  const nodeIds = new Set<string>()

  for (const n of graphData.nodes) {
    adjacency.set(n.data.id, new Set())
    nodeIds.add(n.data.id)
  }
  // Only add edges where both endpoints exist in the node list
  for (const e of graphData.edges) {
    if (!nodeIds.has(e.data.source) || !nodeIds.has(e.data.target)) continue
    adjacency.get(e.data.source)!.add(e.data.target)
    adjacency.get(e.data.target)!.add(e.data.source)
  }

  // Connected components via BFS
  const componentOf = new Map<string, number>()
  let compId = 0
  for (const n of graphData.nodes) {
    if (componentOf.has(n.data.id)) continue
    const queue = [n.data.id]
    componentOf.set(n.data.id, compId)
    while (queue.length > 0) {
      const cur = queue.shift()!
      for (const nb of adjacency.get(cur) ?? []) {
        if (!componentOf.has(nb)) {
          componentOf.set(nb, compId)
          queue.push(nb)
        }
      }
    }
    compId++
  }

  const ranked: NodeRank[] = []
  const byId = new Map<string, NodeRank>()

  for (const n of graphData.nodes) {
    const neighbors = adjacency.get(n.data.id) ?? new Set()
    const rank: NodeRank = {
      id: n.data.id,
      degree: neighbors.size,
      label: n.data.label,
      confidence: n.data.confidence,
      domain: n.data.domain,
      nodeType: n.data.node_type,
      status: n.data.status,
      neighbors: [...neighbors],
      component: componentOf.get(n.data.id) ?? 0,
    }
    ranked.push(rank)
    byId.set(n.data.id, rank)
  }

  ranked.sort((a, b) => b.degree - a.degree || b.confidence - a.confidence)

  return { ranked, byId, edges: graphData.edges, nodeCount: graphData.nodes.length, componentCount: compId }
}

// ============================================================
//  Dynamic element builder
// ============================================================

interface BuildResult {
  elements: ElementDefinition[]
  visibleCount: number
  hubCount: number
  orphanClusterCount: number
}

function buildElementsForBudget(topology: GraphTopology, budget: number): BuildResult {
  const { ranked, byId, edges } = topology
  if (ranked.length === 0) return { elements: [], visibleCount: 0, hubCount: 0, orphanClusterCount: 0 }

  const effectiveBudget = Math.max(3, Math.min(budget, ranked.length))

  // Phase 1: visible set = top-N by degree
  const visibleSet = new Set<string>()
  for (let i = 0; i < Math.min(effectiveBudget, ranked.length); i++) {
    visibleSet.add(ranked[i].id)
  }

  // Phase 2: absorb remaining into nearest visible hub
  const absorbedInto = new Map<string, string>()
  const absorbedCounts = new Map<string, number>()

  for (let i = effectiveBudget; i < ranked.length; i++) {
    const r = ranked[i]
    let bestHub: string | null = null
    let bestDegree = -1

    for (const nid of r.neighbors) {
      if (visibleSet.has(nid)) {
        const nbRank = byId.get(nid)
        if (nbRank && nbRank.degree > bestDegree) {
          bestDegree = nbRank.degree
          bestHub = nid
        }
      }
    }

    if (!bestHub) {
      for (const nid of r.neighbors) {
        const parentHub = absorbedInto.get(nid)
        if (parentHub && visibleSet.has(parentHub)) {
          const nbRank = byId.get(parentHub)
          if (nbRank && nbRank.degree > bestDegree) {
            bestDegree = nbRank.degree
            bestHub = parentHub
          }
        }
      }
    }

    if (!bestHub) {
      const orphanId = `__orphan__${r.domain}`
      if (!visibleSet.has(orphanId)) visibleSet.add(orphanId)
      bestHub = orphanId
    }

    absorbedInto.set(r.id, bestHub)
    absorbedCounts.set(bestHub, (absorbedCounts.get(bestHub) ?? 0) + 1)
  }

  // Phase 3: node elements
  const elements: ElementDefinition[] = []
  let hubCount = 0
  let orphanClusterCount = 0

  for (const id of visibleSet) {
    const isOrphan = id.startsWith('__orphan__')
    const absorbed = absorbedCounts.get(id) ?? 0

    if (isOrphan) {
      const domain = id.replace('__orphan__', '')
      orphanClusterCount++
      elements.push({
        group: 'nodes',
        data: {
          id,
          label: domain,
          shortLabel: `${domain} (+${absorbed})`,
          fullLabel: `${domain} -- ${absorbed} unclustered nodes`,
          confidence: 0.5,
          domain,
          node_type: 'cluster',
          status: 'provisional',
          absorbedCount: absorbed,
          originalDegree: 0,
          nodeSize: Math.min(60, 28 + absorbed * 1.0),
        },
        classes: 'orphan-node',
      })
      continue
    }

    const r = byId.get(id)!
    const isHub = r.degree >= 2 || absorbed > 0
    if (isHub) hubCount++

    const shortLabel = r.label.length > 35 ? r.label.slice(0, 33) + '\u2026' : r.label
    const displayLabel = absorbed > 0 ? `${shortLabel}\n(+${absorbed})` : shortLabel

    const baseSize = 18
    const degreeSize = Math.min(16, r.degree * 2)
    const absorbedSize = Math.min(18, absorbed * 0.6)
    const nodeSize = baseSize + degreeSize + absorbedSize

    elements.push({
      group: 'nodes',
      data: {
        id,
        label: displayLabel,
        shortLabel: displayLabel,
        fullLabel: r.label,
        confidence: r.confidence,
        domain: r.domain,
        node_type: r.nodeType,
        status: r.status,
        absorbedCount: absorbed,
        originalDegree: r.degree,
        nodeSize,
      },
      classes: isHub ? 'hub-node' : 'detail-node',
    })
  }

  // Phase 4: edges (reroute, deduplicate)
  // Key includes edge type category to keep contradicts edges separate from dependency edges
  const edgeMap = new Map<string, { count: number; type: string }>()

  for (const e of edges) {
    let src = e.data.source
    let tgt = e.data.target
    if (!visibleSet.has(src)) src = absorbedInto.get(src) ?? src
    if (!visibleSet.has(tgt)) tgt = absorbedInto.get(tgt) ?? tgt
    if (src === tgt) continue
    if (!visibleSet.has(src) || !visibleSet.has(tgt)) continue

    const pairKey = src < tgt ? `${src}|${tgt}` : `${tgt}|${src}`
    // Separate conflict edges from structural edges so they don't overwrite each other
    const isConflict = e.data.type === 'contradicts'
    const key = isConflict ? `${pairKey}|C` : `${pairKey}|S`

    if (!edgeMap.has(key)) {
      edgeMap.set(key, { count: 0, type: e.data.type })
    }
    const entry = edgeMap.get(key)!
    entry.count++
    // For structural edges, pick the most "informative" type
    if (!isConflict) {
      // supports > elaborates > depends_on (for display priority)
      if (e.data.type === 'supports' && entry.type !== 'supports') entry.type = 'supports'
      else if (e.data.type === 'elaborates' && entry.type === 'depends_on') entry.type = 'elaborates'
    }
  }

  let edgeIdx = 0
  for (const [key, info] of edgeMap) {
    const keyParts = key.split('|')
    const src = keyParts[0]
    const tgt = keyParts[1]

    elements.push({
      group: 'edges',
      data: {
        id: `e${edgeIdx++}`,
        source: src,
        target: tgt,
        type: info.type,
        weight: info.count,
        label: info.count > 1 ? `${info.count}` : '',
      },
      classes: info.count > 1 ? 'multi-edge' : 'single-edge',
    })
  }

  return { elements, visibleCount: visibleSet.size, hubCount, orphanClusterCount }
}

// ============================================================
//  Layout — performance-tuned per node count
// ============================================================

function getLayoutOptions(nodeCount: number, animate: boolean): LayoutOptions {
  let quality: string
  let iters: number

  if (nodeCount > 200) {
    quality = 'draft'
    iters = 100
  } else if (nodeCount > 80) {
    quality = 'default'
    iters = 300
  } else {
    quality = 'proof'
    iters = 600
  }

  // More generous spacing to reduce overlaps
  const repulsion = nodeCount > 200 ? 8000 : nodeCount > 80 ? 14000 : 22000
  const idealEdge = nodeCount > 200 ? 100 : nodeCount > 80 ? 140 : 180
  const gravity = nodeCount > 200 ? 0.3 : nodeCount > 80 ? 0.2 : 0.1
  const separation = nodeCount > 200 ? 60 : nodeCount > 80 ? 80 : 100

  return {
    name: 'fcose',
    animate,
    animationDuration: animate ? 350 : 0,
    quality,
    randomize: true,
    nodeRepulsion: () => repulsion,
    idealEdgeLength: () => idealEdge,
    edgeElasticity: () => 0.45,
    numIter: iters,
    gravity,
    gravityRange: 3.8,
    padding: 40,
    fit: true,                    // caller can override to false for zoom-triggered
    nodeSeparation: separation,
    uniformNodeDimensions: true,
    packComponents: true,
    tile: true,
    tilingPaddingVertical: 25,
    tilingPaddingHorizontal: 25,
  } as LayoutOptions
}

// ============================================================
//  Stylesheet — optimized for rendering perf
// ============================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const cytoscapeStylesheet: any[] = [
  {
    selector: '.detail-node',
    style: {
      label: 'data(shortLabel)',
      'font-size': '9px',
      'text-wrap': 'wrap',
      'text-max-width': '100px',
      color: '#c8cad4',
      'text-valign': 'bottom',
      'text-margin-y': 4,
      'text-background-color': '#181b24',
      'text-background-opacity': 0.9,
      'text-background-padding': '2px',
      'text-background-shape': 'roundrectangle',
      'background-color': 'mapData(confidence, 0, 1, #f87171, #4ade80)',
      width: 'data(nodeSize)',
      height: 'data(nodeSize)',
      'border-width': 1.5,
      'border-color': '#333750',
      'min-zoomed-font-size': 10,
    },
  },
  {
    selector: '.hub-node',
    style: {
      label: 'data(shortLabel)',
      'font-size': '10px',
      'font-weight': 'bold',
      'text-wrap': 'wrap',
      'text-max-width': '130px',
      color: '#e8eaf0',
      'text-outline-color': '#0f1117',
      'text-outline-width': 1,
      'text-valign': 'bottom',
      'text-margin-y': 5,
      'text-background-color': '#181b24',
      'text-background-opacity': 0.9,
      'text-background-padding': '3px',
      'text-background-shape': 'roundrectangle',
      'background-color': 'mapData(confidence, 0, 1, #f87171, #4ade80)',
      width: 'data(nodeSize)',
      height: 'data(nodeSize)',
      'border-width': 2.5,
      'border-color': '#818cf8',
      'min-zoomed-font-size': 8,
    },
  },
  {
    selector: '.orphan-node',
    style: {
      label: 'data(shortLabel)',
      shape: 'roundrectangle',
      width: 'data(nodeSize)',
      height: 'data(nodeSize)',
      'background-color': '#1e2235',
      'background-opacity': 0.9,
      'border-width': 2,
      'border-color': '#3d4260',
      'border-style': 'dashed',
      'font-size': '9px',
      'font-weight': 'bold',
      color: '#d0d4e0',
      'text-outline-color': '#0f1117',
      'text-outline-width': 1,
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '80px',
      padding: '4px',
      'min-zoomed-font-size': 8,
    },
  },
  {
    selector: 'node:selected',
    style: {
      'border-color': '#c084fc',
      'border-width': 3,
    },
  },
  // Default edge (fallback for unknown types)
  {
    selector: 'edge',
    style: {
      'line-color': '#60a5fa', 'target-arrow-color': '#60a5fa',
      'target-arrow-shape': 'triangle', 'curve-style': 'straight',
      width: 1.5, opacity: 1,
    },
  },
  {
    selector: 'edge[type="depends_on"]',
    style: {
      'line-color': '#60a5fa', 'target-arrow-color': '#60a5fa',
    },
  },
  {
    selector: 'edge[type="supports"]',
    style: {
      'line-color': '#4ade80', 'target-arrow-color': '#4ade80',
    },
  },
  {
    selector: 'edge[type="contradicts"]',
    style: {
      'line-color': '#f87171', 'target-arrow-color': '#f87171',
      'line-style': 'dashed', width: 2,
    },
  },
  {
    selector: 'edge[type="elaborates"]',
    style: {
      'line-color': '#6b7394', 'target-arrow-color': '#6b7394',
      width: 1,
    },
  },
  {
    selector: '.multi-edge',
    style: {
      width: 2.5,
      label: 'data(label)',
      'font-size': '9px',
      color: '#6b7394',
      'text-outline-color': '#0f1117',
      'text-outline-width': 1,
      'min-zoomed-font-size': 12,
    },
  },
]

// ============================================================
//  Tooltip
// ============================================================

function createTooltipEl(): HTMLDivElement {
  const el = document.createElement('div')
  el.style.cssText = `
    position:absolute; pointer-events:none; z-index:100;
    background:#222533; border:1px solid #333750; border-radius:6px;
    padding:8px 12px; max-width:340px; box-shadow:0 4px 12px rgba(0,0,0,0.5);
    transform:translate(-50%,-100%); opacity:0; transition:opacity 0.15s;
    font-family:Inter,system-ui,sans-serif;
  `
  const textDiv = document.createElement('div')
  textDiv.style.cssText = 'font-size:13px; line-height:1.4; color:#e8eaf0; word-break:break-word;'
  const metaDiv = document.createElement('div')
  metaDiv.style.cssText = 'margin-top:4px; font-size:11px; display:flex; gap:8px; flex-wrap:wrap;'
  el.appendChild(textDiv)
  el.appendChild(metaDiv)
  return el
}

function confColor(c: number): string {
  if (c >= 0.8) return '#4ade80'
  if (c >= 0.5) return '#facc15'
  return '#f87171'
}

// ============================================================
//  Component
// ============================================================

/** Compute a sensible initial budget for a given node count. */
function autoBudget(nodeCount: number): number {
  if (nodeCount <= 80) return nodeCount
  return Math.min(Math.max(20, Math.round(Math.sqrt(nodeCount) * 3)), nodeCount)
}

export default function GraphView() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Budget is the slider target. debouncedBudget drives the actual layout.
  const [budget, setBudget] = useState(0)
  const [debouncedBudget, setDebouncedBudget] = useState(0)
  const [visibleCount, setVisibleCount] = useState(0)
  const [autoZoomLOD, setAutoZoomLOD] = useState(true)

  const [selected, setSelected] = useState<NodeDetail | null>(null)
  const [explain, setExplain] = useState<ExplainResponse | null>(null)
  const [trace, setTrace] = useState<TraceResponse | null>(null)
  const [layoutRunning, setLayoutRunning] = useState(false)

  const cyRef = useRef<Core | null>(null)
  const topologyRef = useRef<GraphTopology | null>(null)
  const canvasContainerRef = useRef<HTMLDivElement | null>(null)
  const tooltipRef = useRef<HTMLDivElement | null>(null)
  const cyInitializedRef = useRef(false)
  const appliedBudgetRef = useRef(0)
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const zoomDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isSliderDragging = useRef(false)
  // Guards against zoom→LOD→layout→fit→zoom feedback loops
  const layoutInProgressRef = useRef(false)
  // Tracks whether the current apply was triggered by zoom LOD (to skip fit)
  const zoomTriggeredRef = useRef(false)

  // Debounce: slider updates budget instantly (for display), but layout waits 300ms
  useEffect(() => {
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    debounceTimerRef.current = setTimeout(() => {
      setDebouncedBudget(budget)
    }, 300)
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    }
  }, [budget])

  // Fetch graph data
  useEffect(() => {
    getGraphData()
      .then((data) => {
        setGraphData(data)
        const topo = buildTopology(data)
        topologyRef.current = topo
        const init = autoBudget(topo.nodeCount)
        setBudget(init)
        setDebouncedBudget(init)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  // Compute elements for current debounced budget
  const buildResult = useMemo(() => {
    if (!topologyRef.current || debouncedBudget <= 0) return null
    return buildElementsForBudget(topologyRef.current, debouncedBudget)
  }, [debouncedBudget, graphData]) // eslint-disable-line react-hooks/exhaustive-deps

  // Apply elements to Cytoscape
  const applyElements = useCallback((cy: Core, doLayout: boolean) => {
    if (!buildResult) return

    // If this was triggered by zoom LOD, save viewport to restore after layout
    const isZoomTriggered = zoomTriggeredRef.current
    zoomTriggeredRef.current = false

    // Save current viewport before modifying elements
    const savedZoom = cy.zoom()
    const savedPan = { ...cy.pan() }

    cy.batch(() => {
      cy.elements().remove()
      cy.add(buildResult.elements)
    })
    setVisibleCount(buildResult.visibleCount)

    if (doLayout && cy.nodes().length > 0) {
      const nodeCount = cy.nodes().length
      const shouldAnimate = !isZoomTriggered && nodeCount < 100

      layoutInProgressRef.current = true
      setLayoutRunning(true)

      // When zoom-triggered, don't let fcose fit (it resets viewport)
      const opts = getLayoutOptions(nodeCount, shouldAnimate)
      if (isZoomTriggered) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (opts as any).fit = false
      }

      const layout = cy.layout(opts)
      layout.on('layoutstop', () => {
        if (isZoomTriggered) {
          // Restore the viewport the user was looking at
          cy.zoom(savedZoom)
          cy.pan(savedPan)
        } else {
          // First load or manual action: fit to view
          cy.fit(undefined, 30)
        }
        setLayoutRunning(false)
        // Allow zoom LOD to fire again after a brief cooldown
        setTimeout(() => { layoutInProgressRef.current = false }, 500)
      })
      layout.run()
    } else {
      layoutInProgressRef.current = false
    }
  }, [buildResult])

  // When debounced budget changes, update the graph
  useEffect(() => {
    const cy = cyRef.current
    if (!cy || !buildResult) return
    if (debouncedBudget === appliedBudgetRef.current) return
    appliedBudgetRef.current = debouncedBudget
    applyElements(cy, true)
  }, [debouncedBudget, buildResult, applyElements])

  // Cleanup
  useEffect(() => {
    return () => {
      cyInitializedRef.current = false
      cyRef.current = null
      if (tooltipRef.current?.parentNode) {
        tooltipRef.current.parentNode.removeChild(tooltipRef.current)
        tooltipRef.current = null
      }
    }
  }, [])

  // Zoom-based dynamic LOD: when user zooms in, increase budget; zoom out, decrease
  const handleZoomLOD = useCallback((cy: Core) => {
    // Don't fire during layout or slider drag — prevents feedback loops
    if (!autoZoomLOD || isSliderDragging.current || layoutInProgressRef.current) return
    const topo = topologyRef.current
    if (!topo || topo.nodeCount <= 80) return

    if (zoomDebounceRef.current) clearTimeout(zoomDebounceRef.current)
    zoomDebounceRef.current = setTimeout(() => {
      // Double-check guard after debounce — layout may have started during wait
      if (layoutInProgressRef.current) return

      const extent = cy.extent()
      const viewW = extent.w
      const viewH = extent.h

      // Viewport area in graph coordinates (smaller when zoomed in)
      const viewArea = viewW * viewH
      // Total graph bounding box
      const bb = cy.elements().boundingBox()
      const totalArea = Math.max(1, bb.w * bb.h)

      // Fraction of graph visible
      const fractionVisible = Math.min(1, viewArea / totalArea)

      const minBudget = autoBudget(topo.nodeCount)
      const maxBudget = topo.nodeCount

      let newBudget: number
      if (fractionVisible < 0.1) {
        newBudget = Math.round(maxBudget * 0.8)
      } else if (fractionVisible < 0.3) {
        const t = (0.3 - fractionVisible) / 0.2
        newBudget = Math.round(minBudget + (maxBudget * 0.6 - minBudget) * t)
      } else if (fractionVisible > 0.8) {
        newBudget = minBudget
      } else {
        const t = (0.8 - fractionVisible) / 0.5
        newBudget = Math.round(minBudget + (maxBudget * 0.3 - minBudget) * t)
      }

      // Hysteresis: only change if delta > 15% of current budget
      const currentBudget = appliedBudgetRef.current
      const delta = Math.abs(newBudget - currentBudget)
      if (delta > currentBudget * 0.15 && delta > 5) {
        // Mark this as zoom-triggered so applyElements preserves viewport
        zoomTriggeredRef.current = true
        setBudget(newBudget)
      }
    }, 500) // 500ms debounce for zoom LOD
  }, [autoZoomLOD])

  // Cytoscape initialization
  const handleCyRef = useCallback((cy: Core) => {
    cyRef.current = cy
    if (cyInitializedRef.current) return
    cyInitializedRef.current = true

    const container = canvasContainerRef.current
    if (container && !tooltipRef.current) {
      const tip = createTooltipEl()
      container.appendChild(tip)
      tooltipRef.current = tip
    }

    // Node tap
    cy.on('tap', 'node', async (evt) => {
      const nodeId = (evt.target as NodeSingular).id()
      if (nodeId.startsWith('__orphan__')) {
        const topo = topologyRef.current
        if (topo) {
          setBudget((prev) => Math.min(prev + Math.round(topo.nodeCount * 0.15), topo.nodeCount))
        }
        return
      }
      setExplain(null)
      setTrace(null)
      try {
        const detail = await getNode(nodeId)
        setSelected(detail)
      } catch { /* ignore */ }
    })

    // Background tap
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelected(null)
        setExplain(null)
        setTrace(null)
      }
    })

    // Zoom-based LOD
    cy.on('zoom', () => handleZoomLOD(cy))
    // Also trigger on window resize
    const resizeObserver = new ResizeObserver(() => {
      cy.resize()
      handleZoomLOD(cy)
    })
    if (container) resizeObserver.observe(container)

    // Tooltip
    cy.on('mouseover', 'node', (evt) => {
      const tip = tooltipRef.current
      if (!tip || !container) return
      const node = evt.target as NodeSingular
      const pos = node.renderedPosition()
      const fullLabel: string = node.data('fullLabel') || node.data('label') || ''
      const conf: number = node.data('confidence') ?? 0
      const domain: string = node.data('domain') || ''
      const absorbed: number = node.data('absorbedCount') ?? 0

      const textDiv = tip.children[0] as HTMLDivElement
      const metaDiv = tip.children[1] as HTMLDivElement
      textDiv.textContent = fullLabel
      let metaHtml = `<span style="color:${confColor(conf)}">${(conf * 100).toFixed(0)}%</span>`
      if (domain) metaHtml += `<span style="color:#6b7394">${domain}</span>`
      if (absorbed > 0) metaHtml += `<span style="color:#818cf8">+${absorbed} absorbed</span>`
      metaDiv.innerHTML = metaHtml

      tip.style.left = `${pos.x}px`
      tip.style.top = `${pos.y - 18}px`
      tip.style.opacity = '1'
    })
    cy.on('mouseout', 'node', () => {
      if (tooltipRef.current) tooltipRef.current.style.opacity = '0'
    })
    cy.on('pan zoom', () => {
      if (tooltipRef.current) tooltipRef.current.style.opacity = '0'
    })

    // Apply initial elements
    requestAnimationFrame(() => {
      if (buildResult && cy.nodes().length === 0) {
        applyElements(cy, true)
      }
    })
  }, [buildResult, applyElements, handleZoomLOD])

  // Sidebar resize
  const prevSelectedRef = useRef<NodeDetail | null>(null)
  useEffect(() => {
    const hadSidebar = prevSelectedRef.current !== null
    const hasSidebar = selected !== null
    prevSelectedRef.current = selected
    if (hadSidebar !== hasSidebar && cyRef.current) {
      setTimeout(() => { cyRef.current?.resize() }, 50)
    }
  }, [selected])

  // Actions
  async function loadExplain(id: string) {
    try { setExplain(await explainNode(id)) }
    catch (e) { setExplain({ node_id: id, explanation: `Error: ${(e as Error).message}` }) }
  }
  async function loadTrace(id: string) {
    try { setTrace(await traceNode(id)) }
    catch (e) { setTrace({ node_id: id, trace: `Error: ${(e as Error).message}` }) }
  }

  function fitGraph() { cyRef.current?.fit(undefined, 30) }

  function relayout() {
    const cy = cyRef.current
    if (!cy || cy.nodes().length === 0) return
    layoutInProgressRef.current = true
    setLayoutRunning(true)
    const nodeCount = cy.nodes().length
    const layout = cy.layout(getLayoutOptions(nodeCount, nodeCount < 100))
    layout.on('layoutstop', () => {
      setLayoutRunning(false)
      cy.fit(undefined, 30)
      setTimeout(() => { layoutInProgressRef.current = false }, 500)
    })
    layout.run()
  }

  function showAll() {
    if (!topologyRef.current) return
    setAutoZoomLOD(false)
    setBudget(topologyRef.current.nodeCount)
  }

  function resetToAuto() {
    const topo = topologyRef.current
    if (!topo) return
    setAutoZoomLOD(true)
    setBudget(autoBudget(topo.nodeCount))
  }

  const emptyElements = useMemo<ElementDefinition[]>(() => [], [])

  if (loading) return <div className="page-content"><div className="loading-overlay"><div className="spinner" /> Loading graph...</div></div>
  if (error) return <div className="page-content"><div className="error-message">Error: {error}</div></div>
  if (!graphData) return <div className="page-content"><div className="empty-state">No graph data</div></div>

  const totalNodes = topologyRef.current?.nodeCount ?? 0

  return (
    <div className="page-full">
      <div style={styles.graphColumn}>
        {/* Toolbar */}
        <div style={styles.toolbar}>
          <h1 style={{ fontSize: 18, fontWeight: 600, whiteSpace: 'nowrap' }}>Knowledge Graph</h1>

          <div style={{ display: 'flex', gap: 6 }}>
            <button className="btn" onClick={fitGraph} title="Fit to viewport">Fit</button>
            <button className="btn" onClick={relayout} title="Re-run layout" disabled={layoutRunning}>
              {layoutRunning ? 'Laying out...' : 'Re-layout'}
            </button>
            <button className="btn" onClick={showAll} title="Show all nodes (disables auto LOD)">All</button>
            <button className="btn" onClick={resetToAuto} title="Reset to auto budget with zoom LOD">Auto</button>
          </div>

          {/* Budget slider */}
          <div style={styles.sliderGroup}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Detail:</span>
            <input
              type="range"
              min={Math.min(3, totalNodes)}
              max={totalNodes}
              value={budget}
              onMouseDown={() => { isSliderDragging.current = true; setAutoZoomLOD(false) }}
              onMouseUp={() => { isSliderDragging.current = false }}
              onChange={(e) => setBudget(Number(e.target.value))}
              style={styles.slider}
              title={`Showing ${budget} of ${totalNodes} nodes`}
            />
            <span style={{
              fontSize: 11, fontWeight: 600, minWidth: 55, textAlign: 'right',
              color: budget >= totalNodes ? 'var(--conf-high)' : '#60a5fa',
            }}>
              {visibleCount}/{totalNodes}
            </span>
          </div>

          {/* Auto LOD indicator */}
          <button
            className="btn"
            onClick={() => setAutoZoomLOD(!autoZoomLOD)}
            style={{
              fontSize: 11,
              padding: '2px 8px',
              background: autoZoomLOD ? 'var(--accent)' : 'var(--bg-card)',
              color: autoZoomLOD ? '#fff' : 'var(--text-secondary)',
              border: `1px solid ${autoZoomLOD ? 'var(--accent)' : 'var(--border)'}`,
            }}
            title={autoZoomLOD ? 'Zoom-based LOD is ON' : 'Zoom-based LOD is OFF'}
          >
            {autoZoomLOD ? 'Zoom LOD' : 'Zoom LOD'}
          </button>

          {/* Legend */}
          <div style={styles.legend}>
            <span style={styles.legendItem}><span style={{ ...styles.legendDot, background: 'var(--edge-depends)' }} /> Depends</span>
            <span style={styles.legendItem}><span style={{ ...styles.legendDot, background: 'var(--edge-supports)' }} /> Supports</span>
            <span style={styles.legendItem}><span style={{ ...styles.legendDot, background: 'var(--edge-contradicts)' }} /> Contradicts</span>
            <span style={styles.legendItem}><span style={{ ...styles.legendDot, background: 'var(--edge-elaborates)' }} /> Elaborates</span>
          </div>
        </div>

        {/* Canvas */}
        <div ref={canvasContainerRef} style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'hidden' }}>
          {totalNodes === 0 ? (
            <div className="empty-state" style={{ height: '100%' }}>No nodes in graph. Ingest some text first.</div>
          ) : (
            <CytoscapeComponent
              elements={emptyElements}
              stylesheet={cytoscapeStylesheet}
              style={{ width: '100%', height: '100%', background: '#0f1117' }}
              cy={handleCyRef}
              minZoom={0.02}
              maxZoom={5}
              pixelRatio={1}
            />
          )}
          {layoutRunning && (
            <div style={styles.layoutOverlay}>
              <div className="spinner" style={{ width: 20, height: 20 }} />
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Computing layout...</span>
            </div>
          )}
        </div>

        <div style={styles.hint}>
          Drag the slider or zoom in/out to control detail. Click nodes to inspect. Hover for full text.
          {autoZoomLOD && ' (Zoom LOD active)'}
        </div>
      </div>

      {/* Sidebar */}
      {selected && (
        <div style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <h3 style={{ fontSize: 14, fontWeight: 600 }}>Node Detail</h3>
            <button style={styles.closeBtn} onClick={() => { setSelected(null); setExplain(null); setTrace(null) }}>&times;</button>
          </div>
          <div style={styles.sidebarBody}>
            <div style={styles.nodeText}>{selected.text}</div>
            <div style={styles.metaGrid}>
              <div style={styles.metaRow}><span>ID</span><code style={{ fontSize: 11 }}>{selected.id.slice(0, 12)}...</code></div>
              <div style={styles.metaRow}>
                <span>Confidence</span>
                <span style={{ color: selected.confidence >= 0.8 ? 'var(--conf-high)' : selected.confidence >= 0.5 ? 'var(--conf-medium)' : 'var(--conf-low)' }}>
                  {(selected.confidence * 100).toFixed(1)}%
                </span>
              </div>
              {selected.confidence_bounds && (
                <div style={styles.metaRow}>
                  <span>Bounds</span>
                  <span style={{ fontSize: 11 }}>[{selected.confidence_bounds.map(b => b.toFixed(3)).join(', ')}]</span>
                </div>
              )}
              <div style={styles.metaRow}><span>Domain</span><span className="badge badge-blue">{selected.domain}</span></div>
              <div style={styles.metaRow}><span>Type</span><span>{selected.node_type}</span></div>
              <div style={styles.metaRow}><span>Status</span><span>{selected.status}</span></div>
              <div style={styles.metaRow}><span>Predecessors</span><span>{selected.predecessor_nodes.length}</span></div>
              <div style={styles.metaRow}><span>Successors</span><span>{selected.successor_nodes.length}</span></div>
              <div style={styles.metaRow}>
                <span>Conflicts</span>
                <span style={{ color: selected.conflicts_with.length > 0 ? 'var(--conf-low)' : 'var(--text-muted)' }}>
                  {selected.conflicts_with.length}
                </span>
              </div>
              {selected.access_count > 0 && (
                <div style={styles.metaRow}><span>Access Count</span><span>{selected.access_count}</span></div>
              )}
            </div>
            {selected.sources.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={styles.sectionTitle}>Sources</div>
                {selected.sources.map((s, i) => (
                  <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
                    {s.title || s.source_type} (quality: {(s.quality_score * 100).toFixed(0)}%)
                  </div>
                ))}
              </div>
            )}
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <button className="btn" onClick={() => loadExplain(selected.id)}>Explain</button>
              <button className="btn" onClick={() => loadTrace(selected.id)}>Trace</button>
            </div>
            {explain && (
              <div style={{ marginTop: 12 }}>
                <div style={styles.sectionTitle}>Explanation</div>
                <pre style={styles.debugOutput}>{typeof explain.explanation === 'string' ? explain.explanation : JSON.stringify(explain.explanation, null, 2)}</pre>
              </div>
            )}
            {trace && (
              <div style={{ marginTop: 12 }}>
                <div style={styles.sectionTitle}>Trace</div>
                <pre style={styles.debugOutput}>{JSON.stringify(trace.trace, null, 2)}</pre>
              </div>
            )}
            {selected.metadata && Object.keys(selected.metadata).length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={styles.sectionTitle}>Metadata</div>
                <pre style={styles.debugOutput}>{JSON.stringify(selected.metadata, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================
//  Styles
// ============================================================

const styles: Record<string, React.CSSProperties> = {
  graphColumn: {
    flex: 1, display: 'flex', flexDirection: 'column',
    minWidth: 0, minHeight: 0, padding: '8px 16px',
  },
  toolbar: {
    display: 'flex', alignItems: 'center', gap: 12, padding: '4px 0', marginBottom: 4,
    flexWrap: 'wrap', flexShrink: 0,
  },
  sliderGroup: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '2px 10px',
    background: 'var(--bg-card)',
    borderRadius: 6,
    border: '1px solid var(--border)',
  },
  slider: {
    width: 120,
    accentColor: '#818cf8',
    cursor: 'pointer',
  },
  legend: { display: 'flex', gap: 10, fontSize: 11, color: 'var(--text-secondary)' },
  legendItem: { display: 'flex', alignItems: 'center', gap: 3 },
  legendDot: { width: 8, height: 8, borderRadius: '50%', display: 'inline-block' },
  hint: {
    fontSize: 11, color: 'var(--text-muted)', padding: '4px 0', textAlign: 'center',
    flexShrink: 0,
  },
  layoutOverlay: {
    position: 'absolute', top: 12, right: 12,
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 12px',
    background: 'rgba(26,29,39,0.9)',
    borderRadius: 6,
    border: '1px solid var(--border)',
    zIndex: 10,
  },
  sidebar: {
    width: 360, minWidth: 280, maxWidth: '40vw',
    borderLeft: '1px solid var(--border)',
    background: 'var(--bg-secondary)', overflowY: 'auto',
    flexShrink: 0,
  },
  sidebarHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '12px 16px', borderBottom: '1px solid var(--border)',
  },
  closeBtn: {
    background: 'none', border: 'none', color: 'var(--text-muted)',
    fontSize: 18, cursor: 'pointer', lineHeight: 1,
  },
  sidebarBody: { padding: 16 },
  nodeText: {
    fontSize: 14, lineHeight: 1.6, padding: '10px 12px',
    background: 'var(--bg-card)', borderRadius: 6, border: '1px solid var(--border)',
    marginBottom: 12,
  },
  metaGrid: { display: 'flex', flexDirection: 'column', gap: 6 },
  metaRow: {
    display: 'flex', justifyContent: 'space-between', fontSize: 13,
    color: 'var(--text-secondary)',
  },
  sectionTitle: {
    fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
    textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6,
  },
  debugOutput: {
    padding: '10px', background: 'var(--bg-primary)', borderRadius: 6,
    fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 200,
    overflowY: 'auto', border: '1px solid var(--border)',
  },
}
