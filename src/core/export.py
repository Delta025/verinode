"""
Sub-graph export and import for portable domain slices.

Enables "Federation Lite" — packaging a domain slice as a portable
JSON file that can be imported into another Verinode instance.

Usage:
    export = export_subgraph(graph, "chemistry", confidence_threshold=0.6)
    json_str = export_to_json(export)

    # On another instance:
    imported = import_from_json(json_str)
    stats = import_subgraph(target_graph, imported)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .graph import KnowledgeGraph
from .node import KnowledgeNode

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = 1


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class SubGraphExport:
    """Portable domain slice."""

    schema_version: int = EXPORT_SCHEMA_VERSION
    domain: str = ""
    exported_at: str = ""
    source_graph: str = ""
    confidence_threshold: float = 0.5
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def export_subgraph(
    graph: KnowledgeGraph,
    domain: str,
    confidence_threshold: float = 0.5,
    source_graph: str = "",
) -> SubGraphExport:
    """
    Export nodes in a domain above a confidence threshold.

    Preserves all edges where both endpoints are in the exported set.

    Args:
        graph: Source knowledge graph.
        domain: Domain to export.
        confidence_threshold: Minimum confidence for inclusion.
        source_graph: Identifier for the source instance.

    Returns:
        SubGraphExport with serialized nodes and edges.
    """
    # Filter nodes by domain + confidence
    filtered_ids = set()
    serialized_nodes = []

    for node_id, node in graph.nodes.items():
        if node.domain == domain and node.confidence >= confidence_threshold:
            filtered_ids.add(node_id)
            # Serialize without embedding (too large for export)
            d = node.to_dict()
            d.pop("embedding", None)
            serialized_nodes.append(d)

    # Collect edges where both endpoints are in the filtered set
    edges = []
    for node_id in filtered_ids:
        node = graph.get_node(node_id)
        if not node:
            continue
        for succ_id in node.successor_nodes:
            if succ_id in filtered_ids:
                succ_node = graph.get_node(succ_id)
                rel_type = succ_node.get_edge_label(node_id) if succ_node else "depends_on"
                edges.append(
                    {
                        "source": node_id,
                        "target": succ_id,
                        "type": rel_type,
                    }
                )
        for conflict_id in node.conflicts_with:
            if conflict_id in filtered_ids:
                # Avoid duplicate conflict edges (A→B and B→A)
                if node_id < conflict_id:
                    edges.append(
                        {
                            "source": node_id,
                            "target": conflict_id,
                            "type": "conflicts",
                        }
                    )

    return SubGraphExport(
        schema_version=EXPORT_SCHEMA_VERSION,
        domain=domain,
        exported_at=datetime.now(timezone.utc).isoformat(),
        source_graph=source_graph,
        confidence_threshold=confidence_threshold,
        nodes=serialized_nodes,
        edges=edges,
        node_count=len(serialized_nodes),
        edge_count=len(edges),
    )


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------


def export_to_json(export: SubGraphExport, indent: int = 2) -> str:
    """Serialize a SubGraphExport to a JSON string."""
    return json.dumps(
        {
            "schema_version": export.schema_version,
            "domain": export.domain,
            "exported_at": export.exported_at,
            "source_graph": export.source_graph,
            "confidence_threshold": export.confidence_threshold,
            "node_count": export.node_count,
            "edge_count": export.edge_count,
            "nodes": export.nodes,
            "edges": export.edges,
        },
        indent=indent,
    )


def import_from_json(json_str: str) -> SubGraphExport:
    """Deserialize a SubGraphExport from a JSON string."""
    data = json.loads(json_str)

    if data.get("schema_version", 0) > EXPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Export schema version {data['schema_version']} is newer than "
            f"supported version {EXPORT_SCHEMA_VERSION}. "
            f"Please upgrade Verinode."
        )

    return SubGraphExport(
        schema_version=data.get("schema_version", 1),
        domain=data.get("domain", ""),
        exported_at=data.get("exported_at", ""),
        source_graph=data.get("source_graph", ""),
        confidence_threshold=data.get("confidence_threshold", 0.5),
        nodes=data.get("nodes", []),
        edges=data.get("edges", []),
        node_count=data.get("node_count", 0),
        edge_count=data.get("edge_count", 0),
    )


# ------------------------------------------------------------------
# Import
# ------------------------------------------------------------------


def import_subgraph(
    graph: KnowledgeGraph,
    export: SubGraphExport,
    conflict_strategy: str = "keep_higher_confidence",
) -> Dict[str, Any]:
    """
    Import a sub-graph export into a target graph.

    Args:
        graph: Target knowledge graph.
        export: The exported sub-graph to import.
        conflict_strategy: How to resolve duplicates.
            - "keep_higher_confidence": keep whichever has higher confidence.
            - "keep_existing": always keep the existing node.
            - "keep_imported": always keep the imported node.

    Returns:
        Dict with {added, merged, skipped, conflicts} counts.
    """
    if conflict_strategy not in ("keep_higher_confidence", "keep_existing", "keep_imported"):
        raise ValueError(f"Unknown conflict_strategy: {conflict_strategy}")

    stats = {"added": 0, "merged": 0, "skipped": 0, "conflicts": 0}

    # Map old IDs to new IDs (for edge reconstruction)
    id_map: Dict[str, str] = {}

    skipped_edges = 0
    for node_data in export.nodes:
        if "id" not in node_data or "text" not in node_data:
            logger.warning("Skipping node with missing required fields: %s", list(node_data.keys()))
            continue
        old_id = node_data["id"]

        # Check if an equivalent node already exists (by text + domain)
        existing = _find_duplicate(graph, node_data)

        if existing:
            imported_conf = node_data.get("confidence", 0.5)

            if conflict_strategy == "keep_existing":
                id_map[old_id] = existing.id
                stats["skipped"] += 1
            elif conflict_strategy == "keep_imported":
                # Update existing node's confidence
                graph.update_node_confidence(existing.id, imported_conf)
                id_map[old_id] = existing.id
                stats["merged"] += 1
            else:
                # keep_higher_confidence
                if imported_conf > existing.confidence:
                    graph.update_node_confidence(existing.id, imported_conf)
                    stats["merged"] += 1
                else:
                    stats["skipped"] += 1
                id_map[old_id] = existing.id
        else:
            # New node — create it
            # Clear predecessor/successor to avoid dangling refs;
            # edges will be rebuilt below
            node_data_copy = dict(node_data)
            node_data_copy["predecessor_nodes"] = []
            node_data_copy["successor_nodes"] = []
            node_data_copy["conflicts_with"] = []

            node = KnowledgeNode.from_dict(node_data_copy)
            graph.add_node(node)
            id_map[old_id] = node.id
            stats["added"] += 1

    # Rebuild edges
    for edge in export.edges:
        source_old = edge["source"]
        target_old = edge["target"]
        edge_type = edge.get("type", "depends")

        source_new = id_map.get(source_old)
        target_new = id_map.get(target_old)

        if not source_new or not target_new:
            skipped_edges += 1
            continue
        if source_new == target_new:
            skipped_edges += 1
            continue

        source_node = graph.get_node(source_new)
        target_node = graph.get_node(target_new)

        if not source_node or not target_node:
            skipped_edges += 1
            continue

        if edge_type == "conflicts":
            graph.add_contradiction(source_new, target_new)
            stats["conflicts"] += 1
        else:
            # predecessor → successor edge
            if target_new not in source_node.successor_nodes:
                source_node.add_successor(target_new)
            if source_new not in target_node.predecessor_nodes:
                target_node.add_predecessor(source_new, edge_type)

    if skipped_edges:
        logger.warning("Import skipped %d edges with unmapped/missing node IDs", skipped_edges)

    return stats


def _find_duplicate(
    graph: KnowledgeGraph,
    node_data: Dict[str, Any],
) -> Optional[KnowledgeNode]:
    """
    Find an existing node that matches the imported node by exact text + domain.

    Simple text-based matching. In practice, embedding-based dedup
    would be better but requires the EmbeddingEngine to be initialized.
    """
    text = node_data.get("text", "").strip().lower()
    domain = node_data.get("domain", "general")

    for node in graph.get_by_domain(domain):
        if node.text.strip().lower() == text:
            return node

    return None


# ------------------------------------------------------------------
# Graph Merging
# ------------------------------------------------------------------


@dataclass
class MergeStats:
    """Statistics from a graph merge operation."""

    nodes_from_a: int = 0
    nodes_from_b_added: int = 0
    nodes_from_b_merged: int = 0
    nodes_from_b_skipped: int = 0
    edges_added: int = 0
    edges_skipped: int = 0
    conflicts_added: int = 0
    sources_merged: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "nodes_from_a": self.nodes_from_a,
            "nodes_from_b_added": self.nodes_from_b_added,
            "nodes_from_b_merged": self.nodes_from_b_merged,
            "nodes_from_b_skipped": self.nodes_from_b_skipped,
            "edges_added": self.edges_added,
            "edges_skipped": self.edges_skipped,
            "conflicts_added": self.conflicts_added,
            "sources_merged": self.sources_merged,
        }


def merge_graphs(
    graph_a: KnowledgeGraph,
    graph_b: KnowledgeGraph,
    conflict_strategy: str = "keep_higher_confidence",
    merge_sources: bool = True,
) -> tuple:
    """
    Merge two knowledge graphs into a new graph.

    Graph A is the "base" — all its nodes are copied first.
    Graph B is the "overlay" — its nodes are merged in, with dedup
    against A's nodes by exact text + domain match.

    Args:
        graph_a: Base graph (copied in full).
        graph_b: Overlay graph (merged with dedup).
        conflict_strategy: How to handle duplicate nodes.
            - "keep_higher_confidence": keep whichever has higher confidence.
            - "keep_existing": always keep graph_a's version.
            - "keep_imported": always keep graph_b's version.
        merge_sources: If True, merge source lists on duplicate nodes
            (union by title+source_type). If False, only update confidence.

    Returns:
        (merged_graph, MergeStats)
    """
    if conflict_strategy not in ("keep_higher_confidence", "keep_existing", "keep_imported"):
        raise ValueError(f"Unknown conflict_strategy: {conflict_strategy}")

    merged = KnowledgeGraph()
    stats = MergeStats()

    # --- Phase 1: Copy all of graph_a into merged ---
    # We serialize + deserialize to get clean copies with new object identity.
    id_map_a: Dict[str, str] = {}  # a_id -> merged_id (should be 1:1)

    for node_id, node in graph_a.nodes.items():
        node_data = node.to_dict()
        node_data.pop("embedding", None)
        # Clear edges — we'll rebuild them
        node_data["predecessor_nodes"] = []
        node_data["successor_nodes"] = []
        node_data["conflicts_with"] = []
        new_node = KnowledgeNode.from_dict(node_data)
        merged.add_node(new_node)
        id_map_a[node_id] = new_node.id
        stats.nodes_from_a += 1

    # Rebuild graph_a's edges
    for node_id, node in graph_a.nodes.items():
        new_id = id_map_a[node_id]
        for pred_id in node.predecessor_nodes:
            new_pred = id_map_a.get(pred_id)
            if new_pred:
                merged_node = merged.get_node(new_id)
                pred_node = merged.get_node(new_pred)
                if merged_node and pred_node:
                    rel_type = node.get_edge_label(pred_id)
                    if new_pred not in merged_node.predecessor_nodes:
                        merged_node.add_predecessor(new_pred, rel_type)
                    if new_id not in pred_node.successor_nodes:
                        pred_node.add_successor(new_id)
        for conflict_id in node.conflicts_with:
            new_conflict = id_map_a.get(conflict_id)
            if new_conflict and node_id < conflict_id:  # deduplicate
                merged.add_contradiction(new_id, new_conflict)

    # --- Phase 2: Merge graph_b into merged ---
    id_map_b: Dict[str, str] = {}  # b_id -> merged_id

    for node_id, node in graph_b.nodes.items():
        node_data = node.to_dict()
        node_data.pop("embedding", None)

        existing = _find_duplicate(merged, node_data)

        if existing:
            id_map_b[node_id] = existing.id
            imported_conf = node.confidence

            if conflict_strategy == "keep_existing":
                stats.nodes_from_b_skipped += 1
            elif conflict_strategy == "keep_imported":
                merged.update_node_confidence(existing.id, imported_conf)
                if merge_sources:
                    _merge_source_lists(existing, node)
                    stats.sources_merged += 1
                stats.nodes_from_b_merged += 1
            else:
                # keep_higher_confidence
                if imported_conf > existing.confidence:
                    merged.update_node_confidence(existing.id, imported_conf)
                    stats.nodes_from_b_merged += 1
                else:
                    stats.nodes_from_b_skipped += 1
                if merge_sources:
                    _merge_source_lists(existing, node)
                    stats.sources_merged += 1
        else:
            # New node
            node_data_copy = dict(node_data)
            node_data_copy["predecessor_nodes"] = []
            node_data_copy["successor_nodes"] = []
            node_data_copy["conflicts_with"] = []
            new_node = KnowledgeNode.from_dict(node_data_copy)
            merged.add_node(new_node)
            id_map_b[node_id] = new_node.id
            stats.nodes_from_b_added += 1

    # Rebuild graph_b's edges in the merged graph
    for node_id, node in graph_b.nodes.items():
        new_id = id_map_b.get(node_id)
        if not new_id:
            continue
        for pred_id in node.predecessor_nodes:
            new_pred = id_map_b.get(pred_id)
            if not new_pred:
                stats.edges_skipped += 1
                continue
            merged_node = merged.get_node(new_id)
            pred_node = merged.get_node(new_pred)
            if not merged_node or not pred_node:
                stats.edges_skipped += 1
                continue
            if new_pred not in merged_node.predecessor_nodes:
                rel_type = node.get_edge_label(pred_id)
                merged_node.add_predecessor(new_pred, rel_type)
                if new_id not in pred_node.successor_nodes:
                    pred_node.add_successor(new_id)
                stats.edges_added += 1
        for conflict_id in node.conflicts_with:
            new_conflict = id_map_b.get(conflict_id)
            if not new_conflict:
                stats.edges_skipped += 1
                continue
            if new_id == new_conflict:
                continue
            merged_node = merged.get_node(new_id)
            conflict_node = merged.get_node(new_conflict)
            if not merged_node or not conflict_node:
                stats.edges_skipped += 1
                continue
            if new_conflict not in merged_node.conflicts_with:
                merged.add_contradiction(new_id, new_conflict)
                stats.conflicts_added += 1

    total_merged = len(merged)
    logger.info(
        "Merged graphs: %d (from A) + %d added + %d merged + %d skipped = %d total",
        stats.nodes_from_a,
        stats.nodes_from_b_added,
        stats.nodes_from_b_merged,
        stats.nodes_from_b_skipped,
        total_merged,
    )

    return merged, stats


def _merge_source_lists(existing: KnowledgeNode, incoming: KnowledgeNode) -> None:
    """
    Merge source lists from incoming node into existing node (union by title+source_type).
    Also updates last_validated and sums access_count.
    """
    existing_keys = {(s.title, s.source_type) for s in existing.sources}
    for source in incoming.sources:
        key = (source.title, source.source_type)
        if key not in existing_keys:
            existing.sources.append(source)
            existing_keys.add(key)

    # Keep the more recent last_validated
    if incoming.last_validated and existing.last_validated:
        if incoming.last_validated > existing.last_validated:
            existing.last_validated = incoming.last_validated
    elif incoming.last_validated:
        existing.last_validated = incoming.last_validated

    # Sum access counts
    existing.access_count += incoming.access_count
