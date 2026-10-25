"""
Confidence engine for calculating and propagating confidence scores.
"""

from typing import Dict, List, Optional, Set
import numpy as np
from collections import deque

from .node import KnowledgeNode, Derivation
from .graph import KnowledgeGraph


class ConfidenceEngine:
    """
    Handles confidence calculation and propagation through the knowledge graph.
    
    Implements various strategies for combining evidence and propagating
    confidence through dependency chains.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize confidence engine.
        
        Args:
            config: Configuration dict with hyperparameters
        """
        self.config = config or {}
        
        # Hyperparameters
        self.inference_strength = self.config.get('inference_strength', 0.9)
        self.depth_decay = self.config.get('depth_decay', 0.95)
        self.convergence_boost = self.config.get('convergence_boost', 1.2)
        self.min_confidence = self.config.get('min_confidence', 0.1)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.15)
    
    def calculate_node_confidence(
        self,
        node: KnowledgeNode,
        graph: KnowledgeGraph
    ) -> float:
        """
        Calculate confidence for a node based on its predecessors and sources.
        
        Args:
            node: Node to calculate confidence for
            graph: Knowledge graph containing the node
            
        Returns:
            Calculated confidence score [0, 1]
        """
        # If no predecessors, confidence comes from sources
        if not node.predecessor_nodes:
            return self._confidence_from_sources(node)
        
        # Get predecessor confidences
        pred_confidences = []
        for pred_id in node.predecessor_nodes:
            pred = graph.get_node(pred_id)
            if pred:
                pred_confidences.append(pred.confidence)
        
        if not pred_confidences:
            return self._confidence_from_sources(node)
        
        # Combine based on derivation type
        if node.derivation.derivation_type == "chain":
            # Chain reasoning: limited by weakest link
            combined = min(pred_confidences) * self.inference_strength
        
        elif node.derivation.derivation_type == "convergence":
            # Multiple independent sources: noisy-OR
            combined = self._noisy_or(pred_confidences)
            combined *= self.convergence_boost
        
        elif node.derivation.derivation_type == "inference":
            # Logical inference with strength
            combined = min(pred_confidences) * self.inference_strength
        
        else:
            # Default: average with decay
            combined = np.mean(pred_confidences) * self.inference_strength
        
        # Apply depth decay
        depth = node.derivation.inference_depth
        decay_factor = self.depth_decay ** depth
        combined *= decay_factor
        
        # Boost from direct sources if available
        if node.sources:
            source_conf = self._confidence_from_sources(node)
            # Combine derived and source-based confidence
            combined = max(combined, source_conf * 0.8)
        
        # Clamp to valid range
        return max(self.min_confidence, min(0.99, combined))
    
    def _confidence_from_sources(self, node: KnowledgeNode) -> float:
        """Calculate confidence based on source quality."""
        if not node.sources:
            return 0.5  # Default for unsourced nodes
        
        # Average quality of sources
        qualities = [s.quality_score for s in node.sources]
        return np.mean(qualities)
    
    def _noisy_or(self, confidences: List[float]) -> float:
        """
        Combine independent evidence using noisy-OR.
        
        If multiple independent sources support a claim,
        the combined confidence is higher than any individual source.
        
        P(A|E1,E2,...) = 1 - Product(1 - P(A|Ei))
        """
        product = 1.0
        for conf in confidences:
            product *= (1.0 - conf)
        return 1.0 - product
    
    def propagate_confidence_update(
        self,
        graph: KnowledgeGraph,
        updated_node_id: str,
        max_hops: Optional[int] = None,
    ) -> Set[str]:
        """
        Propagate confidence changes through the graph.

        When a node's confidence changes, update all dependent nodes
        (successors) using topological sort to ensure correct order.

        Args:
            graph: Knowledge graph
            updated_node_id: ID of node that was updated
            max_hops: If set, limit propagation to this many hops
                from the updated node. None means unbounded (default).

        Returns:
            Set of node IDs that were affected
        """
        # Get all descendants (nodes that depend on this one)
        descendants = graph.get_descendants(updated_node_id, max_depth=max_hops)
        
        if not descendants:
            return set()
        
        # Topologically sort descendants to process in dependency order
        sorted_nodes = self._topological_sort(graph, descendants)
        
        affected = set()
        
        # Update confidence for each node in order
        for node_id in sorted_nodes:
            node = graph.get_node(node_id)
            if not node:
                continue
            
            old_confidence = node.confidence
            new_confidence = self.calculate_node_confidence(node, graph)
            
            # Only update if change is significant
            if abs(new_confidence - old_confidence) > 0.01:
                graph.update_node_confidence(node_id, new_confidence)
                affected.add(node_id)
        
        return affected
    
    def _topological_sort(
        self,
        graph: KnowledgeGraph,
        node_ids: Set[str]
    ) -> List[str]:
        """
        Topologically sort nodes to process in dependency order.
        
        Args:
            graph: Knowledge graph
            node_ids: Set of node IDs to sort
            
        Returns:
            List of node IDs in topological order
        """
        # Count in-degrees (number of predecessors in the subset)
        in_degree = {}
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if node:
                # Only count predecessors that are in our subset
                in_degree[node_id] = sum(
                    1 for pid in node.predecessor_nodes if pid in node_ids
                )
        
        # Start with nodes that have no dependencies in the subset
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Decrease in-degree for successors
            node = graph.get_node(current)
            if node:
                for succ_id in node.successor_nodes:
                    if succ_id in in_degree:
                        in_degree[succ_id] -= 1
                        if in_degree[succ_id] == 0:
                            queue.append(succ_id)
        
        return result
    
    def detect_anomalies(self, graph: KnowledgeGraph) -> List[Dict]:
        """
        Detect nodes with anomalous confidence patterns.
        
        Identifies nodes where the actual confidence significantly
        differs from what would be expected based on predecessors.
        
        Args:
            graph: Knowledge graph to check
            
        Returns:
            List of anomaly reports
        """
        anomalies = []

        for node_id, node in graph.nodes.items():
            if node.predecessor_nodes:
                # Node with predecessors: compare against propagated confidence
                expected_conf = self.calculate_node_confidence(node, graph)
            elif node.sources:
                # Source-only node: compare against source quality
                expected_conf = self._confidence_from_sources(node)
            else:
                continue

            actual_conf = node.confidence
            deviation = actual_conf - expected_conf

            # Flag if deviation exceeds threshold
            if abs(deviation) > self.anomaly_threshold:
                anomaly_type = "overconfident" if deviation > 0 else "underconfident"
                
                anomalies.append({
                    'node_id': node_id,
                    'node_text': node.text,
                    'anomaly_type': anomaly_type,
                    'expected_confidence': expected_conf,
                    'actual_confidence': actual_conf,
                    'deviation': deviation,
                    'severity': abs(deviation)
                })
        
        # Sort by severity
        anomalies.sort(key=lambda x: x['severity'], reverse=True)
        
        return anomalies
    
    def trace_confidence_source(
        self,
        graph: KnowledgeGraph,
        node_id: str
    ) -> Dict:
        """
        Trace where a node's confidence ultimately comes from.
        
        Follows the dependency chain back to source nodes (nodes with no predecessors).
        
        Args:
            graph: Knowledge graph
            node_id: Node to trace
            
        Returns:
            Dict with trace information
        """
        node = graph.get_node(node_id)
        if not node:
            return {}
        
        # Find all ancestor nodes (transitive dependencies)
        ancestors = graph.get_ancestors(node_id)
        
        # Identify source nodes (no predecessors)
        source_nodes = []
        for anc_id in ancestors:
            anc = graph.get_node(anc_id)
            if anc and not anc.predecessor_nodes:
                source_nodes.append({
                    'id': anc_id,
                    'text': anc.text,
                    'confidence': anc.confidence,
                    'sources': [str(s) for s in anc.sources]
                })
        
        return {
            'node_id': node_id,
            'node_confidence': node.confidence,
            'inference_depth': node.derivation.inference_depth,
            'num_ancestors': len(ancestors),
            'source_nodes': source_nodes,
            'min_source_confidence': min(s['confidence'] for s in source_nodes) if source_nodes else None,
            'max_source_confidence': max(s['confidence'] for s in source_nodes) if source_nodes else None
        }
    
    def estimate_confidence_impact(
        self,
        graph: KnowledgeGraph,
        node_id: str,
        new_confidence: float,
        max_hops: Optional[int] = None,
    ) -> Dict:
        """
        Estimate the impact of changing a node's confidence.

        Args:
            graph: Knowledge graph
            node_id: Node to change
            new_confidence: Proposed new confidence
            max_hops: If set, limit impact analysis to this many hops.

        Returns:
            Dict with impact analysis
        """
        node = graph.get_node(node_id)
        if not node:
            return {}

        old_confidence = node.confidence
        descendants = graph.get_descendants(node_id, max_depth=max_hops)
        
        # Estimate how many nodes would be affected
        # and by how much
        estimated_changes = []
        
        for desc_id in descendants:
            desc = graph.get_node(desc_id)
            if desc:
                # Simple estimate: proportional change
                change_factor = new_confidence / old_confidence if old_confidence > 0 else 1.0
                estimated_new = desc.confidence * change_factor
                estimated_new = max(self.min_confidence, min(0.99, estimated_new))
                
                estimated_changes.append({
                    'node_id': desc_id,
                    'current_confidence': desc.confidence,
                    'estimated_confidence': estimated_new,
                    'change': estimated_new - desc.confidence
                })
        
        return {
            'node_id': node_id,
            'current_confidence': old_confidence,
            'proposed_confidence': new_confidence,
            'affected_nodes': len(descendants),
            'estimated_changes': estimated_changes,
            'max_positive_change': max((c['change'] for c in estimated_changes), default=0),
            'max_negative_change': min((c['change'] for c in estimated_changes), default=0)
        }
