"""
Basic tests for the knowledge graph system.

Run with: python -m pytest tests/test_basic.py
Or: python tests/test_basic.py
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


def test_node_creation():
    """Test creating a knowledge node."""
    node = KnowledgeNode(
        text="Water is H2O",
        node_type=NodeType.FACT,
        confidence=0.99,
        domain="chemistry"
    )
    
    assert node.text == "Water is H2O"
    assert node.confidence == 0.99
    assert node.node_type == NodeType.FACT
    assert node.domain == "chemistry"
    print("✓ Node creation test passed")


def test_graph_operations():
    """Test basic graph operations."""
    graph = KnowledgeGraph()
    
    node1 = KnowledgeNode(text="Fact 1", confidence=0.9)
    node2 = KnowledgeNode(text="Fact 2", confidence=0.8)
    
    graph.add_node(node1)
    graph.add_node(node2)
    
    assert len(graph) == 2
    assert node1.id in graph
    assert graph.get_node(node1.id) == node1
    
    print("✓ Graph operations test passed")


def test_dependencies():
    """Test node dependencies."""
    graph = KnowledgeGraph()
    
    base = KnowledgeNode(
        text="Base fact",
        confidence=0.95
    )
    
    derived = KnowledgeNode(
        text="Derived fact",
        confidence=0.8,
        predecessor_nodes=[base.id]
    )
    
    graph.add_node(base)
    graph.add_node(derived)
    
    # Check relationships
    preds = graph.get_predecessors(derived.id)
    succs = graph.get_successors(base.id)
    
    assert len(preds) == 1
    assert preds[0].id == base.id
    assert len(succs) == 1
    assert succs[0].id == derived.id
    
    print("✓ Dependencies test passed")


def test_confidence_propagation():
    """Test confidence propagation."""
    graph = KnowledgeGraph()
    conf_engine = ConfidenceEngine()
    
    # Create chain: A → B → C
    node_a = KnowledgeNode(text="A", confidence=0.9)
    node_b = KnowledgeNode(
        text="B",
        confidence=0.5,
        predecessor_nodes=[node_a.id],
        derivation=Derivation(
            method="inferred",
            inference_depth=1,
            derivation_type="chain"
        )
    )
    node_c = KnowledgeNode(
        text="C",
        confidence=0.5,
        predecessor_nodes=[node_b.id],
        derivation=Derivation(
            method="inferred",
            inference_depth=2,
            derivation_type="chain"
        )
    )
    
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    
    # Recalculate confidence for B
    new_conf_b = conf_engine.calculate_node_confidence(node_b, graph)
    assert new_conf_b < 0.9  # Should be less than A due to inference strength
    assert new_conf_b > 0.0
    
    # Recalculate confidence for C
    graph.update_node_confidence(node_b.id, new_conf_b)
    new_conf_c = conf_engine.calculate_node_confidence(node_c, graph)
    assert new_conf_c < new_conf_b  # Should be less than B due to depth decay
    
    print("✓ Confidence propagation test passed")


def test_convergent_evidence():
    """Test confidence with multiple supporting nodes."""
    graph = KnowledgeGraph()
    conf_engine = ConfidenceEngine()
    
    # Create convergent structure: A → C, B → C
    node_a = KnowledgeNode(text="A", confidence=0.8)
    node_b = KnowledgeNode(text="B", confidence=0.7)
    node_c = KnowledgeNode(
        text="C",
        confidence=0.5,
        predecessor_nodes=[node_a.id, node_b.id],
        derivation=Derivation(
            method="inferred",
            derivation_type="convergence"
        )
    )
    
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    
    # Confidence should be higher than either A or B alone
    new_conf = conf_engine.calculate_node_confidence(node_c, graph)
    assert new_conf > 0.8  # Boosted by convergence
    
    print("✓ Convergent evidence test passed")


def test_anomaly_detection():
    """Test anomaly detection."""
    graph = KnowledgeGraph()
    conf_engine = ConfidenceEngine()
    
    # Create normal node
    normal = KnowledgeNode(
        text="Normal fact",
        confidence=0.8,
        sources=[Source(source_type="textbook", title="Good Source", quality_score=0.85)]
    )
    graph.add_node(normal)
    
    # Create suspicious node with artificially high confidence
    suspicious = KnowledgeNode(
        text="Suspicious fact",
        confidence=0.95,  # Very high!
        sources=[Source(source_type="blog", title="Random Blog", quality_score=0.3)]  # Low quality!
    )
    graph.add_node(suspicious)
    
    # Detect anomalies
    anomalies = conf_engine.detect_anomalies(graph)
    
    # Should find the suspicious node
    assert len(anomalies) > 0
    found_suspicious = any(a['node_id'] == suspicious.id for a in anomalies)
    assert found_suspicious
    
    print("✓ Anomaly detection test passed")


def test_serialization():
    """Test saving and loading graph."""
    import tempfile
    import os
    
    graph = KnowledgeGraph()
    
    node = KnowledgeNode(
        text="Test fact",
        confidence=0.9,
        domain="test"
    )
    graph.add_node(node)
    
    # Save
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    try:
        graph.save(filepath)
        
        # Load
        loaded_graph = KnowledgeGraph.load(filepath)
        
        assert len(loaded_graph) == len(graph)
        assert node.id in loaded_graph
        loaded_node = loaded_graph.get_node(node.id)
        assert loaded_node.text == node.text
        assert loaded_node.confidence == node.confidence
        
        print("✓ Serialization test passed")
    finally:
        os.unlink(filepath)


def run_all_tests():
    """Run all tests."""
    print("\nRunning tests...\n")
    
    test_node_creation()
    test_graph_operations()
    test_dependencies()
    test_confidence_propagation()
    test_convergent_evidence()
    test_anomaly_detection()
    test_serialization()
    
    print("\n✅ All tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
