"""
Simple demo of the knowledge graph system.

This example creates a small chemistry knowledge base and demonstrates:
- Creating nodes with confidence scores
- Establishing dependencies between facts
- Propagating confidence through the graph
- Querying and reasoning
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import (
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    Source,
    Derivation,
    ConfidenceEngine
)


def create_chemistry_knowledge_base():
    """Create a simple chemistry knowledge base."""
    
    # Initialize graph and confidence engine
    graph = KnowledgeGraph()
    conf_engine = ConfidenceEngine()
    
    print("Building chemistry knowledge base...\n")
    
    # === Base facts (axioms) ===
    
    # Fact 1: Water is H2O
    water_h2o = KnowledgeNode(
        text="Water is composed of H2O molecules",
        node_type=NodeType.FACT,
        confidence=0.99,
        domain="chemistry",
        sources=[
            Source(
                source_type="textbook",
                title="General Chemistry",
                author="Linus Pauling",
                quality_score=0.95
            )
        ]
    )
    graph.add_node(water_h2o)
    print(f"Added: {water_h2o.text}")
    print(f"  Confidence: {water_h2o.confidence}")
    print()
    
    # Fact 2: Boiling is phase transition
    boiling_def = KnowledgeNode(
        text="Boiling occurs when vapor pressure equals atmospheric pressure",
        node_type=NodeType.DEFINITION,
        confidence=0.95,
        domain="chemistry",
        sources=[
            Source(
                source_type="textbook",
                title="Thermodynamics",
                quality_score=0.93
            )
        ]
    )
    graph.add_node(boiling_def)
    print(f"Added: {boiling_def.text}")
    print(f"  Confidence: {boiling_def.confidence}")
    print()
    
    # Fact 3: Atmospheric pressure at sea level
    atm_pressure = KnowledgeNode(
        text="Atmospheric pressure at sea level is 1 atm (101.325 kPa)",
        node_type=NodeType.OBSERVATION,
        confidence=0.98,
        domain="physics",
        sources=[
            Source(
                source_type="standard",
                title="NIST Standard Reference",
                quality_score=0.99
            )
        ]
    )
    graph.add_node(atm_pressure)
    print(f"Added: {atm_pressure.text}")
    print(f"  Confidence: {atm_pressure.confidence}")
    print()
    
    # === Derived facts ===
    
    # Derived Fact: Water boiling point
    water_boiling = KnowledgeNode(
        text="Water boils at 100°C at sea level (1 atm pressure)",
        node_type=NodeType.FACT,
        confidence=0.8,  # Initial, will be recalculated
        domain="chemistry",
        predecessor_nodes=[water_h2o.id, boiling_def.id, atm_pressure.id],
        derivation=Derivation(
            method="inferred",
            inference_depth=1,
            derivation_type="convergence",
            notes="Derived from thermodynamic principles and experimental validation"
        ),
        sources=[
            Source(
                source_type="experiment",
                title="Experimental measurements (NIST)",
                quality_score=0.97
            )
        ]
    )
    graph.add_node(water_boiling)
    print(f"Added (derived): {water_boiling.text}")
    print(f"  Initial confidence: {water_boiling.confidence}")
    
    # Recalculate confidence based on predecessors
    new_conf = conf_engine.calculate_node_confidence(water_boiling, graph)
    graph.update_node_confidence(water_boiling.id, new_conf)
    print(f"  Recalculated confidence: {water_boiling.confidence}")
    print()
    
    # Another derived fact: Cooking at altitude
    altitude_cooking = KnowledgeNode(
        text="Cooking at high altitude requires adjustments because water boils at lower temperatures",
        node_type=NodeType.INFERENCE,
        confidence=0.7,
        domain="chemistry",
        predecessor_nodes=[water_boiling.id, atm_pressure.id],
        derivation=Derivation(
            method="inferred",
            inference_depth=2,
            derivation_type="chain",
            notes="Follows from boiling point dependency on pressure"
        )
    )
    graph.add_node(altitude_cooking)
    print(f"Added (derived): {altitude_cooking.text}")
    print(f"  Initial confidence: {altitude_cooking.confidence}")
    
    new_conf = conf_engine.calculate_node_confidence(altitude_cooking, graph)
    graph.update_node_confidence(altitude_cooking.id, new_conf)
    print(f"  Recalculated confidence: {altitude_cooking.confidence}")
    print()
    
    return graph, conf_engine


def demonstrate_confidence_propagation(graph, conf_engine):
    """Demonstrate confidence propagation."""
    
    print("\n" + "="*70)
    print("DEMONSTRATING CONFIDENCE PROPAGATION")
    print("="*70 + "\n")
    
    # Find a base fact to update
    base_facts = [n for n in graph.nodes.values() if not n.predecessor_nodes]
    if not base_facts:
        print("No base facts found")
        return
    
    base_fact = base_facts[0]
    print(f"Updating confidence of base fact:")
    print(f"  '{base_fact.text}'")
    print(f"  Current confidence: {base_fact.confidence}")
    
    # Simulate new evidence that increases confidence
    old_confidence = base_fact.confidence
    new_confidence = min(0.99, old_confidence + 0.05)
    
    print(f"  New confidence: {new_confidence} (simulating new evidence)")
    print()
    
    # Update and propagate
    graph.update_node_confidence(base_fact.id, new_confidence)
    affected = conf_engine.propagate_confidence_update(graph, base_fact.id)
    
    print(f"Confidence propagated to {len(affected)} dependent nodes:")
    for node_id in affected:
        node = graph.get_node(node_id)
        if node:
            print(f"  - '{node.text[:60]}...'")
            print(f"    New confidence: {node.confidence:.3f}")
    print()


def demonstrate_anomaly_detection(graph, conf_engine):
    """Demonstrate anomaly detection."""
    
    print("\n" + "="*70)
    print("DEMONSTRATING ANOMALY DETECTION")
    print("="*70 + "\n")
    
    # Create a node with suspiciously high confidence
    suspicious_node = KnowledgeNode(
        text="This is a suspicious claim with artificially high confidence",
        node_type=NodeType.FACT,
        confidence=0.95,  # Very high!
        domain="chemistry",
        predecessor_nodes=list(graph.nodes.keys())[:2],  # Based on existing nodes
        derivation=Derivation(
            method="extracted",
            inference_depth=2,
            derivation_type="chain"
        ),
        sources=[
            Source(
                source_type="blog",
                title="Random Blog Post",
                quality_score=0.3  # Low quality source!
            )
        ]
    )
    
    graph.add_node(suspicious_node)
    print(f"Added suspicious node: '{suspicious_node.text}'")
    print(f"  Confidence: {suspicious_node.confidence}")
    print(f"  Source quality: {suspicious_node.sources[0].quality_score}")
    print()
    
    # Detect anomalies
    anomalies = conf_engine.detect_anomalies(graph)
    
    print(f"Found {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"\n  Node: '{anomaly['node_text'][:60]}...'")
        print(f"  Type: {anomaly['anomaly_type']}")
        print(f"  Expected confidence: {anomaly['expected_confidence']:.3f}")
        print(f"  Actual confidence: {anomaly['actual_confidence']:.3f}")
        print(f"  Deviation: {anomaly['deviation']:.3f}")
        print(f"  Severity: {anomaly['severity']:.3f}")


def demonstrate_tracing(graph, conf_engine):
    """Demonstrate confidence tracing."""
    
    print("\n" + "="*70)
    print("DEMONSTRATING CONFIDENCE TRACING")
    print("="*70 + "\n")
    
    # Find a derived fact
    derived_facts = [n for n in graph.nodes.values() if n.predecessor_nodes]
    if not derived_facts:
        print("No derived facts found")
        return
    
    derived_fact = derived_facts[0]
    print(f"Tracing confidence for:")
    print(f"  '{derived_fact.text}'")
    print()
    
    trace = conf_engine.trace_confidence_source(graph, derived_fact.id)
    
    print(f"Trace results:")
    print(f"  Current confidence: {trace['node_confidence']:.3f}")
    print(f"  Inference depth: {trace['inference_depth']}")
    print(f"  Number of ancestors: {trace['num_ancestors']}")
    print(f"  Source confidence range: [{trace['min_source_confidence']:.3f}, {trace['max_source_confidence']:.3f}]")
    print()
    print(f"  Ultimate source nodes:")
    for source in trace['source_nodes']:
        print(f"    - '{source['text'][:50]}...' (conf: {source['confidence']:.3f})")


def demonstrate_statistics(graph):
    """Show graph statistics."""
    
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*70 + "\n")
    
    stats = graph.get_statistics()
    
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Nodes with conflicts: {stats['nodes_with_conflicts']}")
    print()
    
    print("Nodes by domain:")
    for domain, count in stats['nodes_by_domain'].items():
        print(f"  {domain}: {count}")
    print()
    
    print("Nodes by type:")
    for node_type, count in stats['nodes_by_type'].items():
        print(f"  {node_type}: {count}")
    print()
    
    print("Nodes by status:")
    for status, count in stats['nodes_by_status'].items():
        print(f"  {status}: {count}")


def main():
    """Run the demo."""
    
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH AI - CHEMISTRY DEMO")
    print("="*70 + "\n")
    
    # Create knowledge base
    graph, conf_engine = create_chemistry_knowledge_base()
    
    # Demonstrate features
    demonstrate_confidence_propagation(graph, conf_engine)
    demonstrate_anomaly_detection(graph, conf_engine)
    demonstrate_tracing(graph, conf_engine)
    demonstrate_statistics(graph)
    
    # Save graph
    output_path = Path(__file__).parent.parent / "data" / "chemistry_demo.json"
    output_path.parent.mkdir(exist_ok=True)
    graph.save(str(output_path))
    print(f"\n\nGraph saved to: {output_path}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
