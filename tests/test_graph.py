"""
Tests for KnowledgeGraph: CRUD operations, index consistency, traversal,
statistics, and save/load round-trips.
"""

import json

import pytest

from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeStatus, NodeType

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_node(text, confidence=0.8, domain="general", node_id=None, node_type=NodeType.FACT, predecessors=None):
    return KnowledgeNode(
        text=text,
        node_type=node_type,
        confidence=confidence,
        domain=domain,
        predecessor_nodes=predecessors or [],
        node_id=node_id,
    )


# ------------------------------------------------------------------
# Add / Remove
# ------------------------------------------------------------------


class TestAddRemove:
    def test_add_node(self):
        graph = KnowledgeGraph()
        node = make_node("Test fact", node_id="n1")
        graph.add_node(node)
        assert len(graph) == 1
        assert "n1" in graph

    def test_add_duplicate_raises(self):
        graph = KnowledgeGraph()
        node = make_node("Test", node_id="dup")
        graph.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(make_node("Other", node_id="dup"))

    def test_remove_node(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("Test", node_id="rm"))
        graph.remove_node("rm")
        assert len(graph) == 0
        assert "rm" not in graph

    def test_remove_nonexistent(self):
        """Removing a missing node is a no-op."""
        graph = KnowledgeGraph()
        graph.remove_node("ghost")  # should not raise
        assert len(graph) == 0

    def test_remove_updates_relationships(self):
        """Removing a node cleans up predecessor/successor links."""
        graph = KnowledgeGraph()
        root = make_node("Root", node_id="root")
        graph.add_node(root)

        child = make_node("Child", node_id="child", predecessors=["root"])
        graph.add_node(child)

        assert "child" in root.successor_nodes
        assert "root" in child.predecessor_nodes

        graph.remove_node("root")
        assert "root" not in child.predecessor_nodes

    def test_add_wires_predecessor_successors(self):
        """Adding a node with predecessors auto-wires successor links."""
        graph = KnowledgeGraph()
        root = make_node("Root", node_id="root")
        graph.add_node(root)

        child = make_node("Child", node_id="child", predecessors=["root"])
        graph.add_node(child)

        assert "child" in root.successor_nodes


# ------------------------------------------------------------------
# Index consistency
# ------------------------------------------------------------------


class TestIndexConsistency:
    def test_domain_index_on_add(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", domain="chemistry", node_id="a"))
        graph.add_node(make_node("B", domain="chemistry", node_id="b"))
        graph.add_node(make_node("C", domain="physics", node_id="c"))

        assert len(graph.domain_index["chemistry"]) == 2
        assert len(graph.domain_index["physics"]) == 1

    def test_domain_index_on_remove(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", domain="chemistry", node_id="a"))
        graph.remove_node("a")

        assert len(graph.domain_index["chemistry"]) == 0

    def test_type_index_on_add(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_type=NodeType.FACT, node_id="a"))
        graph.add_node(make_node("B", node_type=NodeType.RULE, node_id="b"))

        assert "a" in graph.type_index[NodeType.FACT]
        assert "b" in graph.type_index[NodeType.RULE]

    def test_type_index_on_remove(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_type=NodeType.FACT, node_id="a"))
        graph.remove_node("a")

        assert "a" not in graph.type_index[NodeType.FACT]

    def test_status_index_on_add(self):
        """Nodes get indexed by their initial status."""
        graph = KnowledgeGraph()
        high = make_node("High", confidence=0.96, node_id="high")
        low = make_node("Low", confidence=0.3, node_id="low")
        graph.add_node(high)
        graph.add_node(low)

        assert "high" in graph.status_index[NodeStatus.ESTABLISHED]
        assert "low" in graph.status_index[NodeStatus.CONTESTED]

    def test_status_index_on_confidence_update(self):
        """Changing confidence moves the node between status indexes."""
        graph = KnowledgeGraph()
        node = make_node("Test", confidence=0.96, node_id="n1")
        graph.add_node(node)
        assert "n1" in graph.status_index[NodeStatus.ESTABLISHED]

        graph.update_node_confidence("n1", 0.3)
        assert "n1" not in graph.status_index[NodeStatus.ESTABLISHED]
        assert "n1" in graph.status_index[NodeStatus.CONTESTED]

    def test_status_index_on_remove(self):
        graph = KnowledgeGraph()
        node = make_node("Test", confidence=0.96, node_id="n1")
        graph.add_node(node)
        graph.remove_node("n1")

        assert "n1" not in graph.status_index[NodeStatus.ESTABLISHED]

    def test_total_nodes_consistent(self):
        """total_nodes stays in sync with nodes dict."""
        graph = KnowledgeGraph()
        for i in range(5):
            graph.add_node(make_node(f"Node {i}", node_id=f"n{i}"))
        assert graph.total_nodes == 5

        graph.remove_node("n2")
        graph.remove_node("n4")
        assert graph.total_nodes == 3
        assert len(graph.nodes) == 3

    def test_rapid_add_remove_cycle(self):
        """Index stays consistent through add/remove cycles."""
        graph = KnowledgeGraph()
        for i in range(10):
            graph.add_node(make_node(f"Node {i}", domain="test", node_id=f"n{i}"))
        for i in range(0, 10, 2):  # remove evens
            graph.remove_node(f"n{i}")

        assert len(graph) == 5
        assert len(graph.domain_index["test"]) == 5
        # Check all remaining nodes are odd
        for i in range(1, 10, 2):
            assert f"n{i}" in graph


# ------------------------------------------------------------------
# Query methods
# ------------------------------------------------------------------


class TestQueryMethods:
    def _build_graph(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("Water boils", domain="chemistry", confidence=0.9, node_id="chem1"))
        graph.add_node(
            make_node(
                "H2O formula", domain="chemistry", confidence=0.95, node_id="chem2", node_type=NodeType.DEFINITION
            )
        )
        graph.add_node(make_node("F=ma", domain="physics", confidence=0.99, node_id="phys1"))
        return graph

    def test_get_by_domain(self):
        graph = self._build_graph()
        chem = graph.get_by_domain("chemistry")
        assert len(chem) == 2

    def test_get_by_domain_empty(self):
        graph = self._build_graph()
        bio = graph.get_by_domain("biology")
        assert bio == []

    def test_get_by_type(self):
        graph = self._build_graph()
        defs = graph.get_by_type(NodeType.DEFINITION)
        assert len(defs) == 1
        assert defs[0].id == "chem2"

    def test_get_by_status(self):
        graph = self._build_graph()
        established = graph.get_by_status(NodeStatus.ESTABLISHED)
        # phys1 (0.99) and chem2 (0.95) are established
        ids = {n.id for n in established}
        assert "phys1" in ids
        assert "chem2" in ids

    def test_get_node_exists(self):
        graph = self._build_graph()
        assert graph.get_node("chem1") is not None

    def test_get_node_missing(self):
        graph = self._build_graph()
        assert graph.get_node("ghost") is None


# ------------------------------------------------------------------
# Traversal
# ------------------------------------------------------------------


class TestTraversal:
    def _build_chain(self):
        """root -> mid -> leaf"""
        graph = KnowledgeGraph()
        root = make_node("Root", node_id="root")
        graph.add_node(root)
        mid = make_node("Mid", node_id="mid", predecessors=["root"])
        graph.add_node(mid)
        leaf = make_node("Leaf", node_id="leaf", predecessors=["mid"])
        graph.add_node(leaf)
        return graph

    def test_get_predecessors(self):
        graph = self._build_chain()
        preds = graph.get_predecessors("mid")
        assert len(preds) == 1
        assert preds[0].id == "root"

    def test_get_successors(self):
        graph = self._build_chain()
        succs = graph.get_successors("root")
        assert len(succs) == 1
        assert succs[0].id == "mid"

    def test_get_descendants(self):
        graph = self._build_chain()
        desc = graph.get_descendants("root")
        assert desc == {"mid", "leaf"}

    def test_get_ancestors(self):
        graph = self._build_chain()
        anc = graph.get_ancestors("leaf")
        assert anc == {"root", "mid"}

    def test_descendants_of_leaf(self):
        graph = self._build_chain()
        assert graph.get_descendants("leaf") == set()

    def test_ancestors_of_root(self):
        graph = self._build_chain()
        assert graph.get_ancestors("root") == set()

    def test_traversal_nonexistent(self):
        graph = KnowledgeGraph()
        assert graph.get_descendants("ghost") == set()
        assert graph.get_ancestors("ghost") == set()
        assert graph.get_predecessors("ghost") == []
        assert graph.get_successors("ghost") == []


# ------------------------------------------------------------------
# Contradictions
# ------------------------------------------------------------------


class TestContradictions:
    def test_add_contradiction(self):
        graph = KnowledgeGraph()
        n1 = make_node("Water boils at 100C", node_id="a")
        n2 = make_node("Water boils at 50C", node_id="b")
        graph.add_node(n1)
        graph.add_node(n2)

        graph.add_contradiction("a", "b")
        assert "b" in n1.conflicts_with
        assert "a" in n2.conflicts_with

    def test_add_contradiction_nonexistent(self):
        """Adding contradiction with missing node is a no-op."""
        graph = KnowledgeGraph()
        n1 = make_node("Test", node_id="a")
        graph.add_node(n1)
        graph.add_contradiction("a", "ghost")
        assert len(n1.conflicts_with) == 0


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------


class TestStatistics:
    def test_statistics_on_empty(self):
        graph = KnowledgeGraph()
        stats = graph.get_statistics()
        assert stats["total_nodes"] == 0

    def test_statistics_counts(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", domain="chem", confidence=0.8, node_id="a"))
        graph.add_node(make_node("B", domain="chem", confidence=0.9, node_id="b"))
        graph.add_node(make_node("C", domain="phys", confidence=0.7, node_id="c"))

        stats = graph.get_statistics()
        assert stats["total_nodes"] == 3
        assert stats["nodes_by_domain"]["chem"] == 2
        assert stats["nodes_by_domain"]["phys"] == 1

    def test_statistics_avg_confidence(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", confidence=0.6, node_id="a"))
        graph.add_node(make_node("B", confidence=0.8, node_id="b"))

        stats = graph.get_statistics()
        assert abs(stats["avg_confidence"] - 0.7) < 0.01


# ------------------------------------------------------------------
# Save / Load round trip
# ------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_round_trip(self, tmp_path):
        """Graph survives a save/load cycle."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("Water boils at 100C", domain="chemistry", confidence=0.9, node_id="n1"))
        graph.add_node(
            make_node("H2O is water", domain="chemistry", confidence=0.95, node_id="n2", predecessors=["n1"])
        )

        filepath = str(tmp_path / "test_graph.json")
        graph.save(filepath)

        loaded = KnowledgeGraph.load(filepath)
        assert len(loaded) == 2
        assert loaded.get_node("n1") is not None
        assert loaded.get_node("n2") is not None
        assert loaded.get_node("n2").predecessor_nodes == ["n1"]

    def test_save_load_preserves_domains(self, tmp_path):
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", domain="chem", node_id="a"))
        graph.add_node(make_node("B", domain="phys", node_id="b"))

        filepath = str(tmp_path / "test_graph.json")
        graph.save(filepath)

        loaded = KnowledgeGraph.load(filepath)
        assert len(loaded.get_by_domain("chem")) == 1
        assert len(loaded.get_by_domain("phys")) == 1


# ------------------------------------------------------------------
# Dunder methods
# ------------------------------------------------------------------


class TestDunderMethods:
    def test_len(self):
        graph = KnowledgeGraph()
        assert len(graph) == 0
        graph.add_node(make_node("Test", node_id="n1"))
        assert len(graph) == 1

    def test_contains(self):
        graph = KnowledgeGraph()
        graph.add_node(make_node("Test", node_id="n1"))
        assert "n1" in graph
        assert "ghost" not in graph

    def test_str(self):
        graph = KnowledgeGraph()
        s = str(graph)
        assert "nodes=" in s
        assert "domains=" in s


# ------------------------------------------------------------------
# Defensive save/load
# ------------------------------------------------------------------


class TestSaveLoadDefensive:
    def test_load_skips_invalid_node_data(self, tmp_path):
        """Load skips nodes missing required fields without crashing."""
        data = {
            "nodes": [
                {
                    "id": "good",
                    "text": "Valid fact",
                    "node_type": "fact",
                    "confidence": 0.8,
                    "domain": "general",
                    "predecessor_nodes": [],
                    "successor_nodes": [],
                    "conflicts_with": [],
                    "sources": [],
                    "derivation": {
                        "method": "asserted",
                        "derivation_type": "extracted",
                        "inference_depth": 0,
                        "notes": "",
                    },
                    "embedding": None,
                    "metadata": {},
                    "access_count": 0,
                },
                # Missing 'text' — should be skipped
                {"id": "bad", "confidence": 0.5},
            ]
        }
        filepath = str(tmp_path / "partial.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = KnowledgeGraph.load(filepath)
        # At least the good node should be loaded; bad one may or may not crash
        # depending on from_dict tolerance. The key is no crash.
        assert loaded.get_node("good") is not None

    def test_load_corrupt_json_raises(self, tmp_path):
        """Loading a file with invalid JSON raises an error."""
        filepath = str(tmp_path / "bad.json")
        with open(filepath, "w") as f:
            f.write("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            KnowledgeGraph.load(filepath)

    def test_save_to_invalid_path_raises(self):
        """Saving to a non-existent directory raises OSError."""
        graph = KnowledgeGraph()
        with pytest.raises(OSError):
            graph.save("/nonexistent/path/that/does/not/exist/graph.json")

    def test_load_empty_nodes_list(self, tmp_path):
        """Loading a graph with empty nodes list returns empty graph."""
        filepath = str(tmp_path / "empty.json")
        with open(filepath, "w") as f:
            json.dump({"nodes": []}, f)

        loaded = KnowledgeGraph.load(filepath)
        assert len(loaded) == 0


# ------------------------------------------------------------------
# Remove with inconsistent back-links
# ------------------------------------------------------------------


class TestRemoveDefensive:
    def test_remove_node_with_missing_backlink(self):
        """Removing a node when back-link is already gone doesn't crash."""
        graph = KnowledgeGraph()
        root = make_node("Root", node_id="root")
        child = make_node("Child", node_id="child", predecessors=["root"])
        graph.add_node(root)
        graph.add_node(child)

        # Manually break the back-link to simulate inconsistency
        root.successor_nodes.remove("child")

        # Should not raise
        graph.remove_node("child")
        assert "child" not in graph

    def test_remove_node_predecessor_backlink_missing(self):
        """Removing a node when predecessor's successor list is inconsistent."""
        graph = KnowledgeGraph()
        root = make_node("Root", node_id="root")
        child = make_node("Child", node_id="child", predecessors=["root"])
        graph.add_node(root)
        graph.add_node(child)

        # Break the other direction
        child.predecessor_nodes.append("ghost")  # node that doesn't exist

        graph.remove_node("child")
        assert "child" not in graph


# ------------------------------------------------------------------
# Potential contradictions
# ------------------------------------------------------------------


class TestFindPotentialContradictions:
    def test_finds_negation_contradiction(self):
        graph = KnowledgeGraph()
        n1 = make_node("Water is a good conductor of electricity", domain="physics", node_id="p1")
        n2 = make_node("Water is not a good conductor of electricity", domain="physics", node_id="p2")
        graph.add_node(n1)
        graph.add_node(n2)

        contradictions = graph.find_potential_contradictions(n1)
        ids = {c.id for c in contradictions}
        assert "p2" in ids

    def test_no_contradiction_across_domains(self):
        """Nodes in different domains are not compared."""
        graph = KnowledgeGraph()
        n1 = make_node("Water freezes at zero degrees", domain="chemistry", node_id="c1")
        n2 = make_node("Water does not freeze at zero degrees", domain="physics", node_id="p1")
        graph.add_node(n1)
        graph.add_node(n2)

        contradictions = graph.find_potential_contradictions(n1)
        assert len(contradictions) == 0

    def test_no_contradiction_with_self(self):
        """A node should not contradict itself."""
        graph = KnowledgeGraph()
        n1 = make_node("Water freezes at zero", domain="chem", node_id="n1")
        graph.add_node(n1)

        contradictions = graph.find_potential_contradictions(n1)
        assert all(c.id != "n1" for c in contradictions)


# ------------------------------------------------------------------
# Cycle detection
# ------------------------------------------------------------------


class TestCycleDetection:
    """Test the would_create_cycle method and cycle guards in add_node."""

    def test_would_create_cycle_simple(self):
        """A -> B exists, adding B -> A would create a cycle."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_id="a"))
        graph.add_node(make_node("B", node_id="b", predecessors=["a"]))
        assert graph.would_create_cycle("b", "a") is True

    def test_would_create_cycle_transitive(self):
        """A -> B -> C exists, adding C -> A would create a cycle."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_id="a"))
        graph.add_node(make_node("B", node_id="b", predecessors=["a"]))
        graph.add_node(make_node("C", node_id="c", predecessors=["b"]))
        assert graph.would_create_cycle("c", "a") is True

    def test_no_cycle_valid_edge(self):
        """A -> B exists, adding A -> C is not a cycle."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_id="a"))
        graph.add_node(make_node("B", node_id="b", predecessors=["a"]))
        graph.add_node(make_node("C", node_id="c"))
        assert graph.would_create_cycle("a", "c") is False

    def test_self_loop_is_cycle(self):
        """A node pointing to itself is a cycle."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_id="a"))
        assert graph.would_create_cycle("a", "a") is True

    def test_add_node_skips_cyclic_predecessor(self):
        """Adding a node whose predecessor would create a cycle silently drops that edge."""
        graph = KnowledgeGraph()
        a = make_node("A", node_id="a")
        graph.add_node(a)
        b = make_node("B", node_id="b", predecessors=["a"])
        graph.add_node(b)

        # Now add C with predecessor B, and also make B have predecessor C
        # by directly setting up the scenario: add C with predecessor "a"
        # so graph is a -> b, a -> c, then try to add edge c -> a via new node
        # Actually, test the direct case: A -> B already, now add node with B -> A
        # by creating a new node "a2" pretending to be an ancestor
        # Simplest: mutate B to have predecessor C, then add C with predecessor B
        # This is simplest via apply_dependencies tested separately.

        # Test via add_node: create node "c" with predecessors ["b"],
        # then create a new node "d" with predecessors ["c", "a"]
        c = make_node("C", node_id="c", predecessors=["b"])
        graph.add_node(c)
        # a -> b -> c, now try d with predecessor "c" — that's fine
        d = make_node("D", node_id="d", predecessors=["c"])
        graph.add_node(d)
        assert "c" in d.predecessor_nodes

        # Now the real test: try to add a node whose predecessor would close a cycle
        # Graph: a -> b -> c -> d
        # Add node with predecessors=["d"] and node_id that we then try to link back
        # Easiest: add a new node with predecessors containing an ancestor
        # Since add_node can't create back-edges (node is new), test with
        # node that has the root "a" as predecessor (which is fine, just a diamond)
        # The cycle case requires apply_dependencies (tested separately)
        # For add_node specifically: test that a node with a self-referencing predecessor is blocked
        self_ref = make_node("Self", node_id="self-ref", predecessors=["self-ref"])
        graph.add_node(self_ref)
        # Self-loop should be removed
        assert "self-ref" not in self_ref.predecessor_nodes

    def test_nonexistent_node_no_cycle(self):
        """Cycle check with nonexistent node returns False."""
        graph = KnowledgeGraph()
        graph.add_node(make_node("A", node_id="a"))
        assert graph.would_create_cycle("a", "nonexistent") is False
        assert graph.would_create_cycle("nonexistent", "a") is False


# ------------------------------------------------------------------
# Edge labels in graph
# ------------------------------------------------------------------


class TestEdgeLabelsInGraph:
    """Test edge label preservation through graph operations."""

    def test_save_load_preserves_edge_labels(self, tmp_path):
        """Edge labels survive JSON save/load round-trip."""
        graph = KnowledgeGraph()
        root = make_node("Root", node_id="root")
        graph.add_node(root)
        child = make_node("Child", node_id="child", predecessors=["root"])
        child.edge_labels["root"] = "supports"
        graph.add_node(child)

        filepath = str(tmp_path / "test_graph.json")
        graph.save(filepath)
        loaded = KnowledgeGraph.load(filepath)
        assert loaded.get_node("child").get_edge_label("root") == "supports"

    def test_remove_node_cleans_edge_labels(self):
        """Removing a predecessor cleans up edge_labels on successors."""
        graph = KnowledgeGraph()
        a = make_node("A", node_id="a")
        graph.add_node(a)
        b = make_node("B", node_id="b", predecessors=["a"])
        b.edge_labels["a"] = "supports"
        graph.add_node(b)

        graph.remove_node("a")
        assert "a" not in graph.get_node("b").edge_labels
