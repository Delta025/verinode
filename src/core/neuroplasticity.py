"""
Neuroplasticity engine for adaptive graph maintenance.

Implements three biologically-inspired mechanisms for knowledge graph
quality maintenance:

1. **Staleness decay**: Nodes that haven't been accessed or validated
   recently experience gradual confidence decay. This models the
   intuition that unverified knowledge becomes less reliable over time.

2. **Pathway strengthening**: Nodes that are frequently accessed or
   successfully validated receive confidence boosts, modeling how
   reinforced memories become stronger.

3. **Doubt flagging**: Nodes that have been contradicted, repeatedly
   questioned but not supported, or have anomalous access patterns
   are flagged for review.

Usage:
    engine = NeuroplasticityEngine()

    # Apply staleness decay to the entire graph
    decayed = engine.apply_staleness_decay(graph)

    # Strengthen a frequently-accessed node
    engine.strengthen_pathway(graph, node_id)

    # Flag doubtful nodes for review
    flagged = engine.flag_doubtful_nodes(graph)

    # Run all maintenance in one call
    report = engine.run_maintenance(graph)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set

from .graph import KnowledgeGraph
from .node import KnowledgeNode, NodeStatus

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class DecayResult:
    """Result of applying staleness decay to a single node."""

    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    days_since_validated: float
    decay_applied: float  # The multiplicative decay factor
    effective_half_life: float = 90.0  # Actual half-life used (may differ from base)
    num_sources: int = 0  # Number of corroborating sources


@dataclass
class StrengthResult:
    """Result of strengthening a pathway."""

    node_id: str
    node_text: str
    original_confidence: float
    new_confidence: float
    access_count: int
    boost_applied: float


@dataclass
class DoubtFlag:
    """A node flagged for review."""

    node_id: str
    node_text: str
    confidence: float
    reason: str
    severity: str  # "low", "medium", "high"


@dataclass
class MaintenanceReport:
    """Summary of a full maintenance cycle."""

    timestamp: datetime
    nodes_examined: int
    nodes_decayed: int
    total_decay_amount: float
    nodes_strengthened: int
    total_strength_boost: float
    nodes_flagged: int
    flags: List[DoubtFlag] = field(default_factory=list)
    decay_details: List[DecayResult] = field(default_factory=list)
    strength_details: List[StrengthResult] = field(default_factory=list)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class NeuroplasticityEngine:
    """
    Adaptive graph maintenance engine.

    Applies biologically-inspired learning mechanisms to keep the
    knowledge graph's confidence landscape accurate and current.
    """

    def __init__(
        self,
        *,
        # Staleness decay parameters
        decay_half_life_days: float = 90.0,
        decay_min_confidence: float = 0.1,
        decay_grace_period_days: float = 7.0,
        decay_protected_statuses: Optional[Set[NodeStatus]] = None,
        # Corroboration-adjusted decay parameters
        corroboration_decay_enabled: bool = True,
        corroboration_scale_factor: float = 100.0,
        corroboration_min_multiplier: float = 0.5,
        corroboration_max_multiplier: float = 2.0,
        # Pathway strengthening parameters
        strengthen_boost_per_access: float = 0.005,
        strengthen_max_boost: float = 0.15,
        strengthen_access_threshold: int = 3,
        strengthen_validation_boost: float = 0.02,
        # Doubt flagging parameters
        doubt_conflict_threshold: int = 1,
        doubt_low_confidence_threshold: float = 0.3,
        doubt_stale_days_threshold: float = 180.0,
        # Optional confidence propagation after maintenance
        confidence_engine=None,
        propagation_max_hops: int = 3,
    ):
        """
        Args:
            decay_half_life_days: Days for confidence to halve from staleness.
                Higher = slower decay. Default 90 days.
            decay_min_confidence: Floor below which staleness decay won't
                push confidence. Default 0.1.
            decay_grace_period_days: Days after validation before decay starts.
                Default 7 days (no decay in the first week).
            decay_protected_statuses: Node statuses exempt from decay.
                Default: {ESTABLISHED} (very high confidence nodes are stable).

            corroboration_decay_enabled: Whether to adjust decay rate based
                on number of independent sources. Default True.
            corroboration_scale_factor: Graph size at which a node is
                expected to have 2 independent sources. Default 100.
            corroboration_min_multiplier: Minimum half-life multiplier
                (fastest possible decay). Default 0.5 (half the base).
            corroboration_max_multiplier: Maximum half-life multiplier
                (slowest possible decay). Default 2.0 (double the base).

            strengthen_boost_per_access: Confidence boost per access beyond
                the threshold. Default 0.005 (0.5% per access).
            strengthen_max_boost: Maximum total confidence boost from
                pathway strengthening. Default 0.15 (15%).
            strengthen_access_threshold: Minimum accesses before strengthening
                kicks in. Default 3.
            strengthen_validation_boost: Extra confidence boost when a node
                is explicitly validated (last_validated updated). Default 0.02.

            doubt_conflict_threshold: Number of conflicts to trigger doubt.
                Default 1.
            doubt_low_confidence_threshold: Confidence below which a node
                is flagged as doubtful. Default 0.3.
            doubt_stale_days_threshold: Days without validation to flag as
                stale doubt. Default 180 (6 months).
        """
        # Decay
        self.decay_half_life_days = decay_half_life_days
        self.decay_min_confidence = decay_min_confidence
        self.decay_grace_period_days = decay_grace_period_days
        self.decay_protected_statuses = decay_protected_statuses or {NodeStatus.ESTABLISHED}

        # Corroboration-adjusted decay
        self.corroboration_decay_enabled = corroboration_decay_enabled
        self.corroboration_scale_factor = corroboration_scale_factor
        self.corroboration_min_multiplier = corroboration_min_multiplier
        self.corroboration_max_multiplier = corroboration_max_multiplier

        # Strengthening
        self.strengthen_boost_per_access = strengthen_boost_per_access
        self.strengthen_max_boost = strengthen_max_boost
        self.strengthen_access_threshold = strengthen_access_threshold
        self.strengthen_validation_boost = strengthen_validation_boost

        # Doubt
        self.doubt_conflict_threshold = doubt_conflict_threshold
        self.doubt_low_confidence_threshold = doubt_low_confidence_threshold
        self.doubt_stale_days_threshold = doubt_stale_days_threshold

        # Optional propagation
        self.confidence_engine = confidence_engine
        self.propagation_max_hops = propagation_max_hops

    # ------------------------------------------------------------------
    # Corroboration-adjusted decay
    # ------------------------------------------------------------------

    def _corroboration_multiplier(
        self,
        node: KnowledgeNode,
        graph_size: int,
    ) -> float:
        """
        Compute a half-life multiplier based on corroboration vs graph size.

        The expected number of sources scales logarithmically with graph
        size: in a small graph, single-source facts are normal; in a large
        graph, uncorroborated facts are suspicious.

        Formula:
            expected = max(1.0, 1.0 + log2(graph_size / scale_factor))
            ratio = num_sources / expected
            multiplier = clamp(ratio, min_multiplier, max_multiplier)

        Returns:
            Multiplier for decay_half_life_days. Values < 1.0 mean faster
            decay; values > 1.0 mean slower decay.
        """
        num_sources = max(1, len(node.sources))

        if graph_size <= 0:
            return 1.0

        ratio = graph_size / self.corroboration_scale_factor
        if ratio <= 0:
            return 1.0

        expected_sources = max(1.0, 1.0 + math.log2(ratio))
        corroboration_ratio = num_sources / expected_sources

        return max(
            self.corroboration_min_multiplier,
            min(self.corroboration_max_multiplier, corroboration_ratio),
        )

    # ------------------------------------------------------------------
    # Staleness decay
    # ------------------------------------------------------------------

    def apply_staleness_decay(
        self,
        graph: KnowledgeGraph,
        now: Optional[datetime] = None,
    ) -> List[DecayResult]:
        """
        Apply staleness decay to all eligible nodes in the graph.

        Uses exponential decay based on time since last validation:
            decay_factor = 2^(-days_since_validated / half_life)

        Nodes are modified in-place. Confidence never drops below
        ``decay_min_confidence``.

        Args:
            graph: Knowledge graph to process.
            now: Current timestamp (for testing). Defaults to datetime.now().

        Returns:
            List of DecayResult for nodes that had their confidence reduced.
        """
        if now is None:
            now = datetime.now()

        results: List[DecayResult] = []
        graph_size = len(graph)

        for node_id, node in graph.nodes.items():
            # Skip protected statuses
            if node.status in self.decay_protected_statuses:
                continue

            # Calculate days since last validation
            try:
                delta = now - node.last_validated
                days = delta.total_seconds() / 86400.0
            except (TypeError, AttributeError):
                continue

            # Skip if within grace period
            if days <= self.decay_grace_period_days:
                continue

            # Calculate effective days (subtract grace period)
            effective_days = days - self.decay_grace_period_days

            # Corroboration-adjusted half-life
            if self.corroboration_decay_enabled:
                multiplier = self._corroboration_multiplier(node, graph_size)
                effective_half_life = self.decay_half_life_days * multiplier
            else:
                effective_half_life = self.decay_half_life_days

            # Exponential decay: 2^(-t/half_life)
            decay_factor = math.pow(2, -effective_days / effective_half_life)

            # Apply decay
            original_conf = node.confidence
            new_conf = max(
                self.decay_min_confidence,
                original_conf * decay_factor,
            )

            if abs(new_conf - original_conf) > 0.001:
                node.confidence = new_conf
                node.update_status()

                results.append(
                    DecayResult(
                        node_id=node_id,
                        node_text=node.text,
                        original_confidence=original_conf,
                        new_confidence=new_conf,
                        days_since_validated=days,
                        decay_applied=decay_factor,
                        effective_half_life=effective_half_life,
                        num_sources=len(node.sources),
                    )
                )

        logger.info(
            "Staleness decay: %d/%d nodes decayed",
            len(results),
            len(graph),
        )
        return results

    # ------------------------------------------------------------------
    # Pathway strengthening
    # ------------------------------------------------------------------

    def strengthen_pathway(
        self,
        graph: KnowledgeGraph,
        node_id: str,
        *,
        is_validation: bool = False,
    ) -> Optional[StrengthResult]:
        """
        Strengthen a single node's confidence based on access patterns.

        Called when a node is accessed (e.g., during verification or query).
        Records the access and optionally boosts confidence.

        The boost is based on access count beyond the threshold, capped
        at ``strengthen_max_boost`` total increase from the node's
        original confidence at creation.

        Args:
            graph: Knowledge graph containing the node.
            node_id: ID of the node to strengthen.
            is_validation: If True, apply extra validation boost and
                update ``last_validated``.

        Returns:
            StrengthResult if confidence was changed, None otherwise.
        """
        node = graph.get_node(node_id)
        if not node:
            return None

        # Record the access
        node.record_access()

        original_conf = node.confidence

        # Check if strengthening applies
        if node.access_count < self.strengthen_access_threshold:
            return None

        # Calculate boost based on accesses beyond threshold
        excess_accesses = node.access_count - self.strengthen_access_threshold
        access_boost = min(
            excess_accesses * self.strengthen_boost_per_access,
            self.strengthen_max_boost,
        )

        # Extra boost for explicit validation
        validation_boost = self.strengthen_validation_boost if is_validation else 0.0

        total_boost = access_boost + validation_boost

        new_conf = min(1.0, original_conf + total_boost)

        if is_validation:
            node.last_validated = datetime.now()

        if abs(new_conf - original_conf) > 0.001:
            node.confidence = new_conf
            node.update_status()

            return StrengthResult(
                node_id=node_id,
                node_text=node.text,
                original_confidence=original_conf,
                new_confidence=new_conf,
                access_count=node.access_count,
                boost_applied=total_boost,
            )

        return None

    def strengthen_batch(
        self,
        graph: KnowledgeGraph,
        node_ids: List[str],
        *,
        is_validation: bool = False,
    ) -> List[StrengthResult]:
        """
        Strengthen multiple nodes at once.

        Args:
            graph: Knowledge graph.
            node_ids: IDs of nodes to strengthen.
            is_validation: Whether this is a validation event.

        Returns:
            List of StrengthResult for nodes that were actually boosted.
        """
        results = []
        for node_id in node_ids:
            result = self.strengthen_pathway(graph, node_id, is_validation=is_validation)
            if result is not None:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Doubt flagging
    # ------------------------------------------------------------------

    def flag_doubtful_nodes(
        self,
        graph: KnowledgeGraph,
        now: Optional[datetime] = None,
    ) -> List[DoubtFlag]:
        """
        Identify nodes that should be reviewed for quality issues.

        Flags nodes based on:
        - Having conflicts with other nodes
        - Very low confidence
        - Being stale (not validated in a long time)
        - Being contested without resolution

        Args:
            graph: Knowledge graph to examine.
            now: Current timestamp (for testing).

        Returns:
            List of DoubtFlag objects describing flagged nodes.
        """
        if now is None:
            now = datetime.now()

        flags: List[DoubtFlag] = []

        for node_id, node in graph.nodes.items():
            # Check for conflicts
            if len(node.conflicts_with) >= self.doubt_conflict_threshold:
                # Verify conflicts still exist in graph
                active_conflicts = [cid for cid in node.conflicts_with if cid in graph.nodes]
                if len(active_conflicts) >= self.doubt_conflict_threshold:
                    severity = "high" if len(active_conflicts) >= 3 else "medium"
                    flags.append(
                        DoubtFlag(
                            node_id=node_id,
                            node_text=node.text,
                            confidence=node.confidence,
                            reason=f"Has {len(active_conflicts)} active conflict(s)",
                            severity=severity,
                        )
                    )

            # Check for very low confidence
            if node.confidence < self.doubt_low_confidence_threshold:
                flags.append(
                    DoubtFlag(
                        node_id=node_id,
                        node_text=node.text,
                        confidence=node.confidence,
                        reason=f"Very low confidence ({node.confidence:.2f})",
                        severity="medium",
                    )
                )

            # Check for staleness
            try:
                delta = now - node.last_validated
                days = delta.total_seconds() / 86400.0
            except (TypeError, AttributeError):
                days = 0

            if days >= self.doubt_stale_days_threshold:
                flags.append(
                    DoubtFlag(
                        node_id=node_id,
                        node_text=node.text,
                        confidence=node.confidence,
                        reason=f"Not validated in {days:.0f} days",
                        severity="low",
                    )
                )

            # Check for contested status without resolution
            if node.status == NodeStatus.CONTESTED and node.access_count == 0:
                flags.append(
                    DoubtFlag(
                        node_id=node_id,
                        node_text=node.text,
                        confidence=node.confidence,
                        reason="Contested and never accessed",
                        severity="medium",
                    )
                )

        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        flags.sort(key=lambda f: (severity_order.get(f.severity, 3), -f.confidence))

        logger.info(
            "Doubt flagging: %d nodes flagged (high=%d, medium=%d, low=%d)",
            len(flags),
            sum(1 for f in flags if f.severity == "high"),
            sum(1 for f in flags if f.severity == "medium"),
            sum(1 for f in flags if f.severity == "low"),
        )
        return flags

    # ------------------------------------------------------------------
    # Combined maintenance
    # ------------------------------------------------------------------

    def run_maintenance(
        self,
        graph: KnowledgeGraph,
        now: Optional[datetime] = None,
    ) -> MaintenanceReport:
        """
        Run a full maintenance cycle: decay, strengthen, flag.

        This is the primary entry point for periodic graph maintenance.
        Call it on a schedule (e.g., daily or weekly) to keep the graph's
        confidence landscape accurate and current.

        The order is deliberate:
        1. Apply staleness decay first (reduces stale node confidence).
        2. Apply pathway strengthening (boosts well-accessed nodes).
        3. Flag doubtful nodes (based on post-maintenance state).

        Note: Pathway strengthening in maintenance mode only applies
        to nodes that have already accumulated accesses. For real-time
        strengthening, call ``strengthen_pathway()`` directly when
        nodes are accessed.

        Args:
            graph: Knowledge graph to maintain.
            now: Current timestamp (for testing).

        Returns:
            MaintenanceReport with detailed results.
        """
        if now is None:
            now = datetime.now()

        # Step 1: Staleness decay
        decay_results = self.apply_staleness_decay(graph, now=now)
        total_decay = sum(r.original_confidence - r.new_confidence for r in decay_results)

        # Step 2: Pathway strengthening
        # In maintenance mode, we strengthen nodes that have high access
        # counts but haven't been boosted yet
        strength_results: List[StrengthResult] = []
        for node_id, node in graph.nodes.items():
            if node.access_count >= self.strengthen_access_threshold:
                result = self.strengthen_pathway(graph, node_id)
                if result is not None:
                    strength_results.append(result)
        total_boost = sum(r.new_confidence - r.original_confidence for r in strength_results)

        # Step 2.5: Propagate confidence changes from decay/strengthening
        if self.confidence_engine is not None:
            changed_ids = set()
            for r in decay_results:
                changed_ids.add(r.node_id)
            for r in strength_results:
                changed_ids.add(r.node_id)
            for nid in changed_ids:
                self.confidence_engine.propagate_confidence_update(
                    graph,
                    nid,
                    max_hops=self.propagation_max_hops,
                )

        # Step 3: Doubt flagging (on post-maintenance state)
        flags = self.flag_doubtful_nodes(graph, now=now)

        report = MaintenanceReport(
            timestamp=now,
            nodes_examined=len(graph),
            nodes_decayed=len(decay_results),
            total_decay_amount=total_decay,
            nodes_strengthened=len(strength_results),
            total_strength_boost=total_boost,
            nodes_flagged=len(flags),
            flags=flags,
            decay_details=decay_results,
            strength_details=strength_results,
        )

        logger.info(
            "Maintenance complete: %d nodes examined, %d decayed, %d strengthened, %d flagged",
            report.nodes_examined,
            report.nodes_decayed,
            report.nodes_strengthened,
            report.nodes_flagged,
        )

        return report
