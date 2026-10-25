"""
Core knowledge node implementation.
Each node represents a single piece of knowledge with confidence and provenance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import uuid


class NodeType(Enum):
    """Types of knowledge nodes."""
    FACT = "fact"              # Direct assertion
    RULE = "rule"              # General principle
    OBSERVATION = "observation"  # Specific measurement
    INFERENCE = "inference"    # Derived conclusion
    DEFINITION = "definition"  # Definitional statement


class NodeStatus(Enum):
    """Status of knowledge node."""
    ESTABLISHED = "established"      # High confidence, unlikely to change
    WELL_SUPPORTED = "well_supported"  # Solid evidence, could be refined
    PROVISIONAL = "provisional"      # Preliminary, needs more evidence
    CONTESTED = "contested"          # Genuine disagreement exists
    SUPERSEDED = "superseded"        # Replaced by better understanding
    FALSIFIED = "falsified"          # Contradicted by evidence


class SourceTier(Enum):
    """Source quality tiers for ingestion ordering and confidence gating."""
    AUTHORITATIVE = "authoritative"  # quality_score >= 0.85
    STANDARD = "standard"            # 0.5 <= quality_score < 0.85
    UNVERIFIED = "unverified"        # quality_score < 0.5

    @classmethod
    def from_quality_score(cls, score: float) -> 'SourceTier':
        """Derive tier from a numeric quality score."""
        if score >= 0.85:
            return cls.AUTHORITATIVE
        elif score >= 0.5:
            return cls.STANDARD
        else:
            return cls.UNVERIFIED

    @property
    def confidence_cap(self) -> float:
        """Maximum confidence allowed for this tier. 1.0 means no cap."""
        caps = {
            SourceTier.AUTHORITATIVE: 1.0,
            SourceTier.STANDARD: 0.8,
            SourceTier.UNVERIFIED: 0.5,
        }
        return caps[self]

    @property
    def sort_priority(self) -> int:
        """Lower number = higher priority (processed first)."""
        priorities = {
            SourceTier.AUTHORITATIVE: 0,
            SourceTier.STANDARD: 1,
            SourceTier.UNVERIFIED: 2,
        }
        return priorities[self]


@dataclass
class Source:
    """Information about the source of knowledge."""
    source_type: str  # "textbook", "paper", "experiment", "web", etc.
    title: str
    author: Optional[str] = None
    date: Optional[datetime] = None
    url: Optional[str] = None
    quality_score: float = 0.5  # [0, 1]

    @property
    def tier(self) -> SourceTier:
        """Derive the source tier from quality_score."""
        return SourceTier.from_quality_score(self.quality_score)

    def __str__(self) -> str:
        parts = [self.title]
        if self.author:
            parts.append(f"by {self.author}")
        if self.date:
            parts.append(f"({self.date.year})")
        return " ".join(parts)


@dataclass
class Derivation:
    """Information about how knowledge was derived."""
    method: str  # "extracted", "inferred", "experimental", "asserted"
    inference_depth: int = 0  # Steps from axioms/base facts
    derivation_type: str = "extracted"  # "chain", "convergence", "inference"
    notes: str = ""


class KnowledgeNode:
    """
    Represents a single piece of knowledge.
    
    This is the fundamental unit in the knowledge graph. Each node contains:
    - The actual knowledge (text statement)
    - Confidence score and bounds
    - Provenance (sources and dependencies)
    - Graph relationships (predecessors and successors)
    - Metadata (domain, timestamps, usage)
    """
    
    def __init__(
        self,
        text: str,
        node_type: NodeType = NodeType.FACT,
        confidence: float = 0.5,
        sources: Optional[List[Source]] = None,
        predecessor_nodes: Optional[List[str]] = None,
        domain: str = "general",
        embedding: Optional[np.ndarray] = None,
        derivation: Optional[Derivation] = None,
        node_id: Optional[str] = None
    ):
        # Identity
        self.id: str = node_id or str(uuid.uuid4())
        
        # Content
        self.text: str = text
        self.embedding: Optional[np.ndarray] = embedding
        self.node_type: NodeType = node_type
        
        # Confidence
        self._confidence: float = max(0.0, min(1.0, confidence))
        self.confidence_bounds: Tuple[float, float] = (
            max(0.0, confidence - 0.05),
            min(1.0, confidence + 0.05)
        )
        
        # Provenance
        self.sources: List[Source] = sources or []
        self.predecessor_nodes: List[str] = predecessor_nodes or []
        self.edge_labels: Dict[str, str] = {}  # predecessor_id -> relation_type
        self.successor_nodes: List[str] = []  # Filled by graph
        self.derivation: Derivation = derivation or Derivation(method="asserted")
        
        # Metadata
        self.domain: str = domain
        self.created_at: datetime = datetime.now()
        self.last_validated: datetime = datetime.now()
        self.access_count: int = 0
        self.metadata: Dict[str, Any] = {}
        
        # Status
        self.status: NodeStatus = self._determine_initial_status()
        self.conflicts_with: List[str] = []
    
    @property
    def confidence(self) -> float:
        """Get confidence score."""
        return self._confidence
    
    @confidence.setter
    def confidence(self, value: float):
        """Set confidence score with bounds checking."""
        self._confidence = max(0.0, min(1.0, value))
        self.confidence_bounds = (
            max(0.0, value - 0.05),
            min(1.0, value + 0.05)
        )
    
    def _determine_initial_status(self) -> NodeStatus:
        """Determine initial status based on confidence."""
        if self.confidence >= 0.95:
            return NodeStatus.ESTABLISHED
        elif self.confidence >= 0.80:
            return NodeStatus.WELL_SUPPORTED
        elif self.confidence >= 0.60:
            return NodeStatus.PROVISIONAL
        else:
            return NodeStatus.CONTESTED
    
    def update_status(self):
        """Update status based on current confidence and conflicts."""
        if self.conflicts_with:
            self.status = NodeStatus.CONTESTED
        elif self.confidence >= 0.95:
            self.status = NodeStatus.ESTABLISHED
        elif self.confidence >= 0.80:
            self.status = NodeStatus.WELL_SUPPORTED
        elif self.confidence >= 0.60:
            self.status = NodeStatus.PROVISIONAL
        else:
            self.status = NodeStatus.CONTESTED
    
    def add_source(self, source: Source):
        """Add a new source for this knowledge."""
        self.sources.append(source)
    
    def add_predecessor(self, node_id: str, relation_type: str = "depends_on"):
        """Add a node that this knowledge depends on, with an optional relation label."""
        if node_id not in self.predecessor_nodes:
            self.predecessor_nodes.append(node_id)
        self.edge_labels[node_id] = relation_type
    
    def add_successor(self, node_id: str):
        """Add a node that depends on this knowledge."""
        if node_id not in self.successor_nodes:
            self.successor_nodes.append(node_id)
    
    def get_edge_label(self, predecessor_id: str) -> str:
        """Get the relationship type for a predecessor edge. Defaults to 'depends_on'."""
        return self.edge_labels.get(predecessor_id, "depends_on")

    def add_conflict(self, node_id: str):
        """Mark another node as conflicting with this one."""
        if node_id not in self.conflicts_with:
            self.conflicts_with.append(node_id)
            self.update_status()
    
    def record_access(self):
        """Record that this node was accessed."""
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'node_type': self.node_type.value,
            'confidence': self.confidence,
            'confidence_bounds': self.confidence_bounds,
            'sources': [
                {
                    'source_type': s.source_type,
                    'title': s.title,
                    'author': s.author,
                    'date': s.date.isoformat() if s.date else None,
                    'url': s.url,
                    'quality_score': s.quality_score
                }
                for s in self.sources
            ],
            'predecessor_nodes': self.predecessor_nodes,
            'edge_labels': self.edge_labels,
            'successor_nodes': self.successor_nodes,
            'derivation': {
                'method': self.derivation.method,
                'inference_depth': self.derivation.inference_depth,
                'derivation_type': self.derivation.derivation_type,
                'notes': self.derivation.notes
            },
            'domain': self.domain,
            'created_at': self.created_at.isoformat(),
            'last_validated': self.last_validated.isoformat(),
            'access_count': self.access_count,
            'status': self.status.value,
            'conflicts_with': self.conflicts_with,
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create node from dictionary."""
        # Reconstruct sources
        sources = [
            Source(
                source_type=s['source_type'],
                title=s['title'],
                author=s.get('author'),
                date=datetime.fromisoformat(s['date']) if s.get('date') else None,
                url=s.get('url'),
                quality_score=s.get('quality_score', 0.5)
            )
            for s in data.get('sources', [])
        ]
        
        # Reconstruct derivation
        deriv_data = data.get('derivation', {})
        derivation = Derivation(
            method=deriv_data.get('method', 'asserted'),
            inference_depth=deriv_data.get('inference_depth', 0),
            derivation_type=deriv_data.get('derivation_type', 'extracted'),
            notes=deriv_data.get('notes', '')
        )
        
        # Create node
        node = cls(
            text=data['text'],
            node_type=NodeType(data['node_type']),
            confidence=data['confidence'],
            sources=sources,
            predecessor_nodes=data.get('predecessor_nodes', []),
            domain=data.get('domain', 'general'),
            embedding=np.array(data['embedding']) if data.get('embedding') else None,
            derivation=derivation,
            node_id=data['id']
        )
        
        # Restore additional fields
        node.successor_nodes = data.get('successor_nodes', [])
        node.edge_labels = data.get('edge_labels', {})
        node.confidence_bounds = tuple(data.get('confidence_bounds', (node.confidence - 0.05, node.confidence + 0.05)))
        if 'created_at' in data:
            try:
                node.created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                pass  # keep default
        if 'last_validated' in data:
            try:
                node.last_validated = datetime.fromisoformat(data['last_validated'])
            except (ValueError, TypeError):
                pass  # keep default
        node.access_count = data.get('access_count', 0)
        if 'status' in data:
            try:
                node.status = NodeStatus(data['status'])
            except ValueError:
                pass  # keep default
        node.conflicts_with = data.get('conflicts_with', [])
        node.metadata = data.get('metadata', {})
        
        return node
    
    def __str__(self) -> str:
        return f"Node({self.text[:50]}..., conf={self.confidence:.2f}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return f"KnowledgeNode(id={self.id}, text='{self.text[:30]}...', confidence={self.confidence:.2f})"
