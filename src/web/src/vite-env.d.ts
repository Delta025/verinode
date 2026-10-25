/// <reference types="vite/client" />

declare module 'react-cytoscapejs' {
  import type { Core, ElementDefinition, Stylesheet } from 'cytoscape'
  import type { Component } from 'react'

  interface CytoscapeComponentProps {
    elements: ElementDefinition[]
    stylesheet?: Stylesheet[]
    style?: React.CSSProperties
    cy?: (cy: Core) => void
    layout?: object
    minZoom?: number
    maxZoom?: number
    zoom?: number
    pan?: { x: number; y: number }
    [key: string]: unknown
  }

  export default class CytoscapeComponent extends Component<CytoscapeComponentProps> {}
}
