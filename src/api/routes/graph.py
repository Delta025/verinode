"""Graph-level routes: stats, visualization data, anomalies, history, graph loading, file browser."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

import src.api.state as state_mod
from src.api.schemas import (
    AnomalyItem,
    ChangelogEntry,
    CytoscapeEdge,
    CytoscapeEdgeData,
    CytoscapeNode,
    CytoscapeNodeData,
    FileBrowseResponse,
    FileEntry,
    GraphData,
    GraphStats,
    LoadGraphRequest,
    LoadGraphResponse,
    MergeGraphRequest,
    MergeGraphResponse,
    SaveGraphRequest,
    SaveGraphResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])


@router.get("/stats", response_model=GraphStats)
def get_stats():
    """Return graph statistics for the dashboard."""
    stats = state_mod.app_state.graph.get_statistics()
    return GraphStats(**stats)


@router.get("/graph/data", response_model=GraphData)
def get_graph_data():
    """Return full graph as Cytoscape.js-compatible nodes and edges."""
    graph = state_mod.app_state.graph
    nodes = []
    edges = []
    edge_id = 0

    for nid, node in graph.nodes.items():
        # Truncate label for display
        label = node.text[:60] + "..." if len(node.text) > 60 else node.text

        nodes.append(
            CytoscapeNode(
                data=CytoscapeNodeData(
                    id=nid,
                    label=label,
                    confidence=node.confidence,
                    domain=node.domain or "general",
                    node_type=node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
                    status=node.status.value if hasattr(node.status, "value") else str(node.status),
                )
            )
        )

        # Predecessor edges (dependency)
        for pred_id in node.predecessor_nodes:
            if pred_id in graph.nodes:
                edge_type = node.get_edge_label(pred_id)
                edges.append(
                    CytoscapeEdge(
                        data=CytoscapeEdgeData(
                            id=f"e{edge_id}",
                            source=pred_id,
                            target=nid,
                            type=edge_type,
                        )
                    )
                )
                edge_id += 1

        # Conflict edges (undirected, emit once per pair)
        for conflict_id in node.conflicts_with:
            if conflict_id in graph.nodes and nid < conflict_id:
                edges.append(
                    CytoscapeEdge(
                        data=CytoscapeEdgeData(
                            id=f"e{edge_id}",
                            source=nid,
                            target=conflict_id,
                            type="contradicts",
                        )
                    )
                )
                edge_id += 1

    return GraphData(nodes=nodes, edges=edges)


@router.get("/anomalies", response_model=List[AnomalyItem])
def get_anomalies(threshold: float = Query(0.15, ge=0.0, le=1.0)):
    """Detect confidence anomalies in the graph."""
    st = state_mod.app_state
    anomalies = st.confidence_engine.detect_anomalies(st.graph)
    items = []
    for a in anomalies:
        if a.get("severity", a.get("deviation", 0)) >= threshold:
            items.append(
                AnomalyItem(
                    node_id=a.get("node_id", ""),
                    node_text=a.get("node_text", a.get("text", "")),
                    anomaly_type=a.get("anomaly_type", a.get("type", "deviation")),
                    expected_confidence=a.get("expected_confidence", a.get("expected", 0)),
                    actual_confidence=a.get("actual_confidence", a.get("actual", 0)),
                    deviation=a.get("deviation", 0),
                    severity=a.get("severity", a.get("deviation", 0)),
                )
            )
    return items


@router.get("/history", response_model=List[ChangelogEntry])
def get_history(limit: int = Query(50, ge=1, le=500)):
    """Return recent changelog entries (SQLite graphs only)."""
    graph = state_mod.app_state.graph

    # Only SqliteGraph has get_changelog
    if hasattr(graph, "get_changelog"):
        entries = graph.get_changelog(limit=limit)
        return [
            ChangelogEntry(
                id=e.get("id", 0),
                timestamp=str(e.get("timestamp", "")),
                change_type=e.get("change_type", ""),
                node_id=e.get("node_id"),
                details=e.get("details", {}),
            )
            for e in entries
        ]
    return []


@router.post("/load-graph", response_model=LoadGraphResponse)
def load_graph(req: LoadGraphRequest):
    """Load a graph file at runtime (JSON or SQLite)."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

    try:
        from src.core.graph import KnowledgeGraph

        if path.suffix in (".db", ".sqlite", ".sqlite3"):
            from src.core.storage import SqliteGraph

            st.graph = SqliteGraph(str(path))
        else:
            st.graph = KnowledgeGraph.load(str(path))

        st.graph_path = str(path)
        st.reset_lazy()

        stats = st.graph.get_statistics()
        logger.info(f"Loaded graph from {req.path}: {stats['total_nodes']} nodes")

        return LoadGraphResponse(
            path=str(path),
            total_nodes=stats["total_nodes"],
            avg_confidence=stats["avg_confidence"],
            domains=len(stats["nodes_by_domain"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load graph: {e}")


@router.post("/save-graph", response_model=SaveGraphResponse)
def save_graph(req: SaveGraphRequest):
    """Save the current graph to a file."""
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    try:
        st.graph.save(req.path)
        logger.info(f"Saved graph to {req.path}: {len(st.graph)} nodes")
        return SaveGraphResponse(path=req.path, total_nodes=len(st.graph))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save graph: {e}")


@router.post("/merge-graph", response_model=MergeGraphResponse)
def merge_graph(req: MergeGraphRequest):
    """Merge another graph file into the current graph.

    Loads the graph at `path` and merges it into the currently loaded graph.
    The current graph becomes graph A (base), the file becomes graph B (overlay).
    """
    st = state_mod.app_state
    if st is None:
        raise HTTPException(status_code=503, detail="App state not initialized")

    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

    if req.strategy not in ("keep_higher_confidence", "keep_existing", "keep_imported"):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {req.strategy}")

    try:
        from src.core.export import merge_graphs
        from src.core.graph import KnowledgeGraph

        if path.suffix in (".db", ".sqlite", ".sqlite3"):
            from src.core.storage import SqliteGraph

            graph_b = SqliteGraph(str(path))
        else:
            graph_b = KnowledgeGraph.load(str(path))

        merged, stats = merge_graphs(
            st.graph,
            graph_b,
            conflict_strategy=req.strategy,
            merge_sources=req.merge_sources,
        )

        st.graph = merged
        st.reset_lazy()

        logger.info(
            f"Merged graph from {req.path}: +{stats.nodes_from_b_added} added, "
            f"{stats.nodes_from_b_merged} merged, {len(merged)} total"
        )

        return MergeGraphResponse(
            total_nodes=len(merged),
            nodes_from_a=stats.nodes_from_a,
            nodes_from_b_added=stats.nodes_from_b_added,
            nodes_from_b_merged=stats.nodes_from_b_merged,
            nodes_from_b_skipped=stats.nodes_from_b_skipped,
            edges_added=stats.edges_added,
            conflicts_added=stats.conflicts_added,
            sources_merged=stats.sources_merged,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge graph: {e}")


@router.get("/files/browse", response_model=FileBrowseResponse)
def browse_files(
    dir: Optional[str] = Query(None, description="Directory to browse"),
    filter: str = Query("graph", pattern="^(graph|all)$"),
):
    """Browse filesystem for graph files (.json, .db, .sqlite)."""
    if dir:
        target = Path(dir)
    else:
        target = Path.cwd()

    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir}")

    # Resolve to absolute and get parent
    target = target.resolve()
    parent = str(target.parent) if target.parent != target else None

    graph_exts = {".json", ".db", ".sqlite", ".sqlite3"}
    entries = []

    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            # Skip hidden files
            if item.name.startswith("."):
                continue
            is_dir = item.is_dir()
            if not is_dir and filter == "graph" and item.suffix.lower() not in graph_exts:
                continue
            try:
                stat = item.stat()
                modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
                size = stat.st_size if not is_dir else 0
            except OSError:
                modified = None
                size = 0
            entries.append(
                FileEntry(
                    name=item.name,
                    path=str(item),
                    is_dir=is_dir,
                    size=size,
                    modified=modified,
                )
            )
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    return FileBrowseResponse(
        current_dir=str(target),
        parent_dir=parent,
        entries=entries,
    )
