"""
Knowledge graph implementation.
Manages all nodes and their relationships.
"""

import json
import logging
from typing import Dict, List, Optional, Set

import numpy as np

from .node import KnowledgeNode, NodeStatus, NodeType

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Main knowledge graph structure.

    Stores all knowledge nodes and provides operations for:
    - Adding/removing nodes
    - Finding relationships
    - Detecting contradictions
    - Querying by various criteria
    """

    def __init__(self):
        # Core structure
        self.nodes: Dict[str, KnowledgeNode] = {}

        # Indexes for fast access
        self.domain_index: Dict[str, Set[str]] = {}  # domain -> node_ids
        self.type_index: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
        self.status_index: Dict[NodeStatus, Set[str]] = {s: set() for s in NodeStatus}

        # For tracking
        self.total_nodes = 0

    def add_node(self, node: KnowledgeNode) -> None:
        """
        Add a node to the graph.

        Args:
            node: KnowledgeNode to add
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists")

        self.nodes[node.id] = node
        self.total_nodes += 1

        # Update indexes
        if node.domain not in self.domain_index:
            self.domain_index[node.domain] = set()
        self.domain_index[node.domain].add(node.id)

        self.type_index[node.node_type].add(node.id)
        self.status_index[node.status].add(node.id)

        # Update predecessor/successor relationships
        for pred_id in list(node.predecessor_nodes):  # copy for safe mutation
            if pred_id in self.nodes:
                if self.would_create_cycle(pred_id, node.id):
                    logger.warning(
                        "Skipping edge %s -> %s: would create cycle",
                        pred_id,
                        node.id,
                    )
                    node.predecessor_nodes.remove(pred_id)
                    node.edge_labels.pop(pred_id, None)
                    continue
                self.nodes[pred_id].add_successor(node.id)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the graph.

        Args:
            node_id: ID of node to remove
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove from indexes
        self.domain_index[node.domain].discard(node_id)
        self.type_index[node.node_type].discard(node_id)
        self.status_index[node.status].discard(node_id)

        # Update relationships (use try/except for robustness against
        # inconsistent graph state where the back-link may already be gone)
        for pred_id in node.predecessor_nodes:
            if pred_id in self.nodes:
                try:
                    self.nodes[pred_id].successor_nodes.remove(node_id)
                except ValueError:
                    pass  # back-link already missing

        for succ_id in node.successor_nodes:
            if succ_id in self.nodes:
                try:
                    self.nodes[succ_id].predecessor_nodes.remove(node_id)
                except ValueError:
                    pass  # back-link already missing
                self.nodes[succ_id].edge_labels.pop(node_id, None)

        del self.nodes[node_id]
        self.total_nodes -= 1

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def update_node_confidence(self, node_id: str, new_confidence: float) -> None:
        """
        Update a node's confidence score.

        Args:
            node_id: Node to update
            new_confidence: New confidence value [0, 1]
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        old_status = node.status

        node.confidence = new_confidence
        node.update_status()

        # Update status index if status changed
        if node.status != old_status:
            self.status_index[old_status].discard(node_id)
            self.status_index[node.status].add(node_id)

    def get_by_domain(self, domain: str) -> List[KnowledgeNode]:
        """Get all nodes in a specific domain."""
        node_ids = self.domain_index.get(domain, set())
        return [self.nodes[nid] for nid in node_ids]

    def get_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type."""
        node_ids = self.type_index[node_type]
        return [self.nodes[nid] for nid in node_ids]

    def get_by_status(self, status: NodeStatus) -> List[KnowledgeNode]:
        """Get all nodes with a specific status."""
        node_ids = self.status_index[status]
        return [self.nodes[nid] for nid in node_ids]

    def get_predecessors(self, node_id: str) -> List[KnowledgeNode]:
        """Get all nodes that this node depends on."""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        return [self.nodes[pid] for pid in node.predecessor_nodes if pid in self.nodes]

    def get_successors(self, node_id: str) -> List[KnowledgeNode]:
        """Get all nodes that depend on this node."""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        return [self.nodes[sid] for sid in node.successor_nodes if sid in self.nodes]

    def get_descendants(self, node_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """
        Get all nodes that transitively depend on this node.

        Uses BFS to find all descendants.

        Args:
            node_id: Starting node.
            max_depth: If set, stop expanding after this many hops.
                None means unbounded (default, backward compatible).
        """
        if node_id not in self.nodes:
            return set()

        descendants = set()
        queue: list = [(node_id, 0)]
        visited = {node_id}

        while queue:
            current, depth = queue.pop(0)
            node = self.nodes[current]

            if max_depth is not None and depth >= max_depth:
                continue

            for succ_id in node.successor_nodes:
                if succ_id not in visited:
                    visited.add(succ_id)
                    descendants.add(succ_id)
                    queue.append((succ_id, depth + 1))

        return descendants

    def get_ancestors(self, node_id: str) -> Set[str]:
        """
        Get all nodes that this node transitively depends on.

        Uses BFS to find all ancestors.
        """
        if node_id not in self.nodes:
            return set()

        ancestors = set()
        queue = [node_id]
        visited = {node_id}

        while queue:
            current = queue.pop(0)
            node = self.nodes[current]

            for pred_id in node.predecessor_nodes:
                if pred_id not in visited:
                    visited.add(pred_id)
                    ancestors.add(pred_id)
                    queue.append(pred_id)

        return ancestors

    def would_create_cycle(self, predecessor_id: str, successor_id: str) -> bool:
        """
        Check if adding an edge predecessor_id -> successor_id would create a cycle.

        A cycle exists if successor_id is already an ancestor of predecessor_id
        (i.e., there is already a path from successor_id to predecessor_id).
        """
        if predecessor_id == successor_id:
            return True
        if predecessor_id not in self.nodes or successor_id not in self.nodes:
            return False
        return successor_id in self.get_ancestors(predecessor_id)

    def find_potential_contradictions(self, node: KnowledgeNode, threshold: float = 0.7) -> List[KnowledgeNode]:
        """
        Find nodes that might contradict the given node.

        This is a simple implementation that looks for nodes with similar text
        but different information. In practice, you'd want to use embeddings
        and more sophisticated similarity measures.

        Args:
            node: Node to check for contradictions
            threshold: Confidence threshold for contradiction

        Returns:
            List of potentially contradicting nodes
        """
        # For now, this is a placeholder
        # In practice, you'd use embeddings to find similar nodes
        # and then use NLI or other methods to detect contradiction

        # Simple keyword-based approach
        contradictions = []
        node_words = set(node.text.lower().split())

        for other_id, other in self.nodes.items():
            if other_id == node.id:
                continue

            if other.domain != node.domain:
                continue

            other_words = set(other.text.lower().split())
            overlap = len(node_words & other_words)

            # If they share many words but have different confidence,
            # they might be contradictory
            if overlap > 3:
                # Check for obvious negation patterns
                if ("not" in node.text.lower()) != ("not" in other.text.lower()):
                    contradictions.append(other)

        return contradictions

    def add_contradiction(self, node_id1: str, node_id2: str) -> None:
        """Mark two nodes as contradicting each other."""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.nodes[node_id1].add_conflict(node_id2)
            self.nodes[node_id2].add_conflict(node_id1)

    def get_statistics(self) -> Dict:
        """Get statistics about the graph."""
        return {
            "total_nodes": self.total_nodes,
            "nodes_by_domain": {d: len(ids) for d, ids in self.domain_index.items()},
            "nodes_by_type": {t.value: len(ids) for t, ids in self.type_index.items()},
            "nodes_by_status": {s.value: len(ids) for s, ids in self.status_index.items()},
            "avg_confidence": np.mean([n.confidence for n in self.nodes.values()]) if self.nodes else 0,
            "nodes_with_conflicts": sum(1 for n in self.nodes.values() if n.conflicts_with),
        }

    def save(self, filepath: str) -> None:
        """
        Save graph to JSON file.

        Args:
            filepath: Path to save file
        """
        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "metadata": {"total_nodes": self.total_nodes, "statistics": self.get_statistics()},
        }

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except (OSError, TypeError) as exc:
            logger.error("Failed to save graph to %s: %s", filepath, exc)
            raise

    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """
        Load graph from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded KnowledgeGraph
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load graph from %s: %s", filepath, exc)
            raise

        graph = cls()

        # Load all nodes (skip invalid entries)
        for node_data in data.get("nodes", []):
            try:
                node = KnowledgeNode.from_dict(node_data)
                graph.add_node(node)
            except Exception:
                logger.warning("Skipping invalid node data during load: %s", node_data.get("id", "<unknown>"))

        return graph

    def __len__(self) -> int:
        return self.total_nodes

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def __str__(self) -> str:
        return f"KnowledgeGraph(nodes={self.total_nodes}, domains={len(self.domain_index)})"
