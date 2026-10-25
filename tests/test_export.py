"""
Tests for the sub-graph export/import module (src/core/export.py).

Tests cover export filtering, edge preservation, JSON round-trip,
import conflict strategies, schema versioning, and edge cases.
"""

import json

import pytest

from src.core.export import (
    EXPORT_SCHEMA_VERSION,
    MergeStats,
    SubGraphExport,
    export_subgraph,
    export_to_json,
    import_from_json,
    import_subgraph,
    merge_graphs,
)
from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, Source

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_node(text, confidence=0.8, domain="chemistry", node_id=None):
    """Create a test node."""
    return KnowledgeNode(
        text=text,
        confidence=confidence,
        domain=domain,
        node_id=node_id or text[:8].replace(" ", "_"),
        sources=[Source(source_type="textbook", title="Test Source", quality_score=0.9)],
    )


def build_test_graph():
    """Build a graph with chemistry and biology nodes + edges."""
    graph = KnowledgeGraph()

    # Chemistry nodes
    water = make_node("Water boils at 100 degrees Celsius", 0.95, "chemistry", "water_bp")
    steam = make_node("Steam is water in gas phase", 0.85, "chemistry", "steam")
    ice = make_node("Ice is water in solid phase", 0.80, "chemistry", "ice")
    gold = make_node("Gold has symbol Au", 0.90, "chemistry", "gold")
    low_conf = make_node("Mercury is a liquid metal", 0.3, "chemistry", "mercury")

    # Biology node (different domain)
    dna = make_node("DNA has a double helix", 0.90, "biology", "dna")

    for n in [water, steam, ice, gold, low_conf, dna]:
        graph.add_node(n)

    # Edges: water → steam, water → ice
    steam.add_predecessor(water.id)
    water.add_successor(steam.id)
    ice.add_predecessor(water.id)
    water.add_successor(ice.id)

    # Conflict: gold → mercury (hypothetical)
    graph.add_contradiction("gold", "mercury")

    return graph


# ------------------------------------------------------------------
# Tests: export_subgraph
# ------------------------------------------------------------------


class TestExportSubgraph:
    def test_filters_by_domain(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry")

        # Should only have chemistry nodes
        for node in export.nodes:
            assert node["domain"] == "chemistry"

        # Biology node should not be included
        node_ids = {n["id"] for n in export.nodes}
        assert "dna" not in node_ids

    def test_filters_by_confidence(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry", confidence_threshold=0.5)

        # Mercury (0.3 confidence) should be excluded
        node_ids = {n["id"] for n in export.nodes}
        assert "mercury" not in node_ids
        assert "water_bp" in node_ids

    def test_includes_all_above_threshold(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry", confidence_threshold=0.0)

        # All chemistry nodes should be included
        node_ids = {n["id"] for n in export.nodes}
        assert "mercury" in node_ids
        assert "water_bp" in node_ids

    def test_preserves_internal_edges(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry", confidence_threshold=0.5)

        # water → steam and water → ice edges should exist
        edge_pairs = {(e["source"], e["target"]) for e in export.edges if e.get("type") != "conflicts"}
        assert ("water_bp", "steam") in edge_pairs
        assert ("water_bp", "ice") in edge_pairs

    def test_excludes_cross_domain_edges(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry")

        # No edges should reference the biology node
        for edge in export.edges:
            assert edge["source"] != "dna"
            assert edge["target"] != "dna"

    def test_counts_correct(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry", confidence_threshold=0.5)

        assert export.node_count == len(export.nodes)
        assert export.edge_count == len(export.edges)
        assert export.domain == "chemistry"

    def test_empty_domain(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "physics")

        assert export.node_count == 0
        assert export.edge_count == 0
        assert export.nodes == []

    def test_no_embeddings_in_export(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry")

        for node in export.nodes:
            assert "embedding" not in node

    def test_conflict_edges_exported(self):
        graph = build_test_graph()
        # Use threshold 0 to include mercury
        export = export_subgraph(graph, "chemistry", confidence_threshold=0.0)

        conflict_edges = [e for e in export.edges if e.get("type") == "conflicts"]
        assert len(conflict_edges) >= 1
        # Check the conflict is between gold and mercury
        conflict_pair = {conflict_edges[0]["source"], conflict_edges[0]["target"]}
        assert conflict_pair == {"gold", "mercury"}


# ------------------------------------------------------------------
# Tests: JSON round-trip
# ------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_export_import_round_trip(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry", confidence_threshold=0.5)

        json_str = export_to_json(export)
        reimported = import_from_json(json_str)

        assert reimported.domain == export.domain
        assert reimported.node_count == export.node_count
        assert reimported.edge_count == export.edge_count
        assert reimported.schema_version == EXPORT_SCHEMA_VERSION
        assert len(reimported.nodes) == len(export.nodes)

    def test_json_is_valid(self):
        graph = build_test_graph()
        export = export_subgraph(graph, "chemistry")
        json_str = export_to_json(export)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "schema_version" in data
        assert "nodes" in data
        assert "edges" in data

    def test_schema_version_check_passes(self):
        json_str = json.dumps({"schema_version": EXPORT_SCHEMA_VERSION, "nodes": [], "edges": []})
        export = import_from_json(json_str)
        assert export.schema_version == EXPORT_SCHEMA_VERSION

    def test_schema_version_check_fails(self):
        json_str = json.dumps({"schema_version": EXPORT_SCHEMA_VERSION + 1, "nodes": [], "edges": []})
        with pytest.raises(ValueError, match="newer than supported"):
            import_from_json(json_str)


# ------------------------------------------------------------------
# Tests: import_subgraph
# ------------------------------------------------------------------


class TestImportSubgraph:
    def _make_export(self, nodes=None, edges=None, domain="chemistry"):
        return SubGraphExport(
            schema_version=EXPORT_SCHEMA_VERSION,
            domain=domain,
            exported_at="2026-02-11T00:00:00Z",
            nodes=nodes or [],
            edges=edges or [],
            node_count=len(nodes or []),
            edge_count=len(edges or []),
        )

    def test_import_new_nodes(self):
        target = KnowledgeGraph()
        node_data = make_node("Argon is a noble gas", 0.85, "chemistry", "argon").to_dict()
        node_data.pop("embedding", None)

        export = self._make_export(nodes=[node_data])
        stats = import_subgraph(target, export)

        assert stats["added"] == 1
        assert len(target) == 1

    def test_import_duplicate_keep_existing(self):
        target = KnowledgeGraph()
        existing = make_node("Argon is a noble gas", 0.85, "chemistry", "argon_existing")
        target.add_node(existing)

        node_data = make_node("Argon is a noble gas", 0.90, "chemistry", "argon_import").to_dict()
        node_data.pop("embedding", None)

        export = self._make_export(nodes=[node_data])
        stats = import_subgraph(target, export, conflict_strategy="keep_existing")

        assert stats["skipped"] == 1
        assert stats["added"] == 0
        assert len(target) == 1
        # Confidence should remain at original
        assert existing.confidence == 0.85

    def test_import_duplicate_keep_imported(self):
        target = KnowledgeGraph()
        existing = make_node("Argon is a noble gas", 0.85, "chemistry", "argon_existing")
        target.add_node(existing)

        node_data = make_node("Argon is a noble gas", 0.90, "chemistry", "argon_import").to_dict()
        node_data.pop("embedding", None)

        export = self._make_export(nodes=[node_data])
        stats = import_subgraph(target, export, conflict_strategy="keep_imported")

        assert stats["merged"] == 1
        # Confidence should be updated to imported value
        assert existing.confidence == 0.90

    def test_import_duplicate_keep_higher_confidence(self):
        target = KnowledgeGraph()
        existing = make_node("Argon is a noble gas", 0.85, "chemistry", "argon_existing")
        target.add_node(existing)

        # Import has higher confidence
        node_data = make_node("Argon is a noble gas", 0.95, "chemistry", "argon_import").to_dict()
        node_data.pop("embedding", None)

        export = self._make_export(nodes=[node_data])
        stats = import_subgraph(target, export, conflict_strategy="keep_higher_confidence")

        assert stats["merged"] == 1
        assert existing.confidence == 0.95

    def test_import_duplicate_keep_higher_existing_wins(self):
        target = KnowledgeGraph()
        existing = make_node("Argon is a noble gas", 0.95, "chemistry", "argon_existing")
        target.add_node(existing)

        # Import has lower confidence
        node_data = make_node("Argon is a noble gas", 0.70, "chemistry", "argon_import").to_dict()
        node_data.pop("embedding", None)

        export = self._make_export(nodes=[node_data])
        stats = import_subgraph(target, export, conflict_strategy="keep_higher_confidence")

        assert stats["skipped"] == 1
        assert existing.confidence == 0.95

    def test_import_rebuilds_edges(self):
        target = KnowledgeGraph()

        water_data = make_node("Water boils at 100C", 0.95, "chemistry", "w").to_dict()
        steam_data = make_node("Steam is gas phase water", 0.85, "chemistry", "s").to_dict()
        water_data.pop("embedding", None)
        steam_data.pop("embedding", None)

        export = self._make_export(
            nodes=[water_data, steam_data],
            edges=[{"source": "w", "target": "s"}],
        )
        stats = import_subgraph(target, export)

        assert stats["added"] == 2
        # Check edge was rebuilt
        water_node = target.get_node("w")
        steam_node = target.get_node("s")
        assert "s" in water_node.successor_nodes
        assert "w" in steam_node.predecessor_nodes

    def test_import_conflict_edges(self):
        target = KnowledgeGraph()

        a_data = make_node("Fact A", 0.8, "chemistry", "a").to_dict()
        b_data = make_node("Fact B", 0.8, "chemistry", "b").to_dict()
        a_data.pop("embedding", None)
        b_data.pop("embedding", None)

        export = self._make_export(
            nodes=[a_data, b_data],
            edges=[{"source": "a", "target": "b", "type": "conflicts"}],
        )
        stats = import_subgraph(target, export)

        assert stats["conflicts"] == 1
        a_node = target.get_node("a")
        b_node = target.get_node("b")
        assert "b" in a_node.conflicts_with
        assert "a" in b_node.conflicts_with

    def test_import_returns_summary(self):
        target = KnowledgeGraph()
        existing = make_node("Argon is a noble gas", 0.85, "chemistry", "argon")
        target.add_node(existing)

        new_data = make_node("Neon is a noble gas", 0.80, "chemistry", "neon").to_dict()
        dup_data = make_node("Argon is a noble gas", 0.90, "chemistry", "argon_dup").to_dict()
        new_data.pop("embedding", None)
        dup_data.pop("embedding", None)

        export = self._make_export(nodes=[new_data, dup_data])
        stats = import_subgraph(target, export)

        assert "added" in stats
        assert "merged" in stats
        assert "skipped" in stats
        assert "conflicts" in stats
        assert stats["added"] + stats["merged"] + stats["skipped"] == 2

    def test_invalid_conflict_strategy(self):
        target = KnowledgeGraph()
        export = self._make_export()
        with pytest.raises(ValueError, match="Unknown conflict_strategy"):
            import_subgraph(target, export, conflict_strategy="invalid")

    def test_import_empty_export(self):
        target = KnowledgeGraph()
        export = self._make_export()
        stats = import_subgraph(target, export)
        assert stats["added"] == 0
        assert stats["merged"] == 0
        assert len(target) == 0

    def test_import_skips_node_missing_required_fields(self):
        """Nodes missing 'id' or 'text' should be skipped, not crash."""
        target = KnowledgeGraph()

        # Create a valid node dict
        valid_node_data = make_node("Valid fact", 0.8, "science", "valid-node").to_dict()
        valid_node_data.pop("embedding", None)

        export = self._make_export(
            nodes=[
                {"text": "orphan text", "confidence": 0.5},  # missing "id"
                {"id": "bad-node-001"},  # missing "text"
                valid_node_data,  # valid
            ]
        )

        stats = import_subgraph(target, export)
        # Only the valid node should be added
        assert stats["added"] == 1
        assert len(target) == 1


# ------------------------------------------------------------------
# Tests: merge_graphs
# ------------------------------------------------------------------


def build_graph_a():
    """Build a small chemistry graph for merge testing."""
    graph = KnowledgeGraph()
    water = make_node("Water boils at 100 degrees Celsius", 0.95, "chemistry", "water")
    steam = make_node("Steam is water in gas phase", 0.85, "chemistry", "steam")
    gold = make_node("Gold has symbol Au", 0.90, "chemistry", "gold")
    for n in [water, steam, gold]:
        graph.add_node(n)
    # Edge: water → steam
    steam.add_predecessor(water.id)
    water.add_successor(steam.id)
    return graph


def build_graph_b():
    """Build a second graph with some overlapping and some new nodes."""
    graph = KnowledgeGraph()
    # Overlapping with graph_a (same text+domain)
    water = make_node("Water boils at 100 degrees Celsius", 0.80, "chemistry", "water_b")
    # New nodes
    iron = make_node("Iron has symbol Fe", 0.88, "chemistry", "iron")
    rust = make_node("Rust is iron oxide", 0.75, "chemistry", "rust")
    for n in [water, iron, rust]:
        graph.add_node(n)
    # Edge: iron → rust
    rust.add_predecessor(iron.id)
    iron.add_successor(rust.id)
    return graph


class TestMergeGraphs:
    def test_basic_merge_counts(self):
        a = build_graph_a()
        b = build_graph_b()
        merged, stats = merge_graphs(a, b)

        assert stats.nodes_from_a == 3
        assert stats.nodes_from_b_added == 2  # iron, rust
        # Water is a duplicate — lower confidence, so skipped
        assert stats.nodes_from_b_skipped == 1
        assert len(merged) == 5  # 3 from A + 2 new from B

    def test_merged_graph_has_all_nodes(self):
        a = build_graph_a()
        b = build_graph_b()
        merged, _ = merge_graphs(a, b)

        texts = {n.text.strip().lower() for n in merged.nodes.values()}
        assert "water boils at 100 degrees celsius" in texts
        assert "steam is water in gas phase" in texts
        assert "gold has symbol au" in texts
        assert "iron has symbol fe" in texts
        assert "rust is iron oxide" in texts

    def test_edges_from_a_preserved(self):
        a = build_graph_a()
        b = build_graph_b()
        merged, _ = merge_graphs(a, b)

        # Find water and steam in merged graph
        water_node = None
        steam_node = None
        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                water_node = n
            if "steam" in n.text.lower():
                steam_node = n

        assert water_node is not None
        assert steam_node is not None
        assert steam_node.id in water_node.successor_nodes
        assert water_node.id in steam_node.predecessor_nodes

    def test_edges_from_b_added(self):
        a = build_graph_a()
        b = build_graph_b()
        merged, stats = merge_graphs(a, b)

        # Find iron and rust in merged graph
        iron_node = None
        rust_node = None
        for n in merged.nodes.values():
            if "iron has symbol" in n.text.lower():
                iron_node = n
            if "rust" in n.text.lower():
                rust_node = n

        assert iron_node is not None
        assert rust_node is not None
        assert rust_node.id in iron_node.successor_nodes
        assert iron_node.id in rust_node.predecessor_nodes
        assert stats.edges_added >= 1

    def test_strategy_keep_higher_confidence(self):
        a = build_graph_a()  # water at 0.95
        b = build_graph_b()  # water at 0.80
        merged, stats = merge_graphs(a, b, conflict_strategy="keep_higher_confidence")

        # A's water (0.95) should win over B's water (0.80)
        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                assert n.confidence == 0.95
                break

        assert stats.nodes_from_b_skipped == 1

    def test_strategy_keep_higher_confidence_b_wins(self):
        a = build_graph_a()  # water at 0.95
        b = build_graph_b()
        # Boost B's water above A
        for n in b.nodes.values():
            if "water boils" in n.text.lower():
                n.confidence = 0.99
                break

        merged, stats = merge_graphs(a, b, conflict_strategy="keep_higher_confidence")

        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                assert n.confidence == 0.99
                break

        assert stats.nodes_from_b_merged == 1

    def test_strategy_keep_existing(self):
        a = build_graph_a()  # water at 0.95
        b = build_graph_b()  # water at 0.80
        merged, stats = merge_graphs(a, b, conflict_strategy="keep_existing")

        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                assert n.confidence == 0.95
                break

        assert stats.nodes_from_b_skipped == 1

    def test_strategy_keep_imported(self):
        a = build_graph_a()  # water at 0.95
        b = build_graph_b()  # water at 0.80
        merged, stats = merge_graphs(a, b, conflict_strategy="keep_imported")

        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                assert n.confidence == 0.80
                break

        assert stats.nodes_from_b_merged == 1

    def test_invalid_strategy_raises(self):
        a = build_graph_a()
        b = build_graph_b()
        with pytest.raises(ValueError, match="Unknown conflict_strategy"):
            merge_graphs(a, b, conflict_strategy="invalid")

    def test_merge_empty_graphs(self):
        a = KnowledgeGraph()
        b = KnowledgeGraph()
        merged, stats = merge_graphs(a, b)

        assert len(merged) == 0
        assert stats.nodes_from_a == 0
        assert stats.nodes_from_b_added == 0

    def test_merge_with_empty_b(self):
        a = build_graph_a()
        b = KnowledgeGraph()
        merged, stats = merge_graphs(a, b)

        assert len(merged) == 3
        assert stats.nodes_from_a == 3
        assert stats.nodes_from_b_added == 0

    def test_merge_with_empty_a(self):
        a = KnowledgeGraph()
        b = build_graph_b()
        merged, stats = merge_graphs(a, b)

        assert len(merged) == 3  # water, iron, rust
        assert stats.nodes_from_a == 0
        assert stats.nodes_from_b_added == 3

    def test_conflict_edges_merged(self):
        a = KnowledgeGraph()
        n1 = make_node("Fact A", 0.8, "chemistry", "a1")
        n2 = make_node("Fact B", 0.8, "chemistry", "a2")
        a.add_node(n1)
        a.add_node(n2)
        a.add_contradiction("a1", "a2")

        b = KnowledgeGraph()
        n3 = make_node("Fact C", 0.8, "chemistry", "b1")
        n4 = make_node("Fact D", 0.8, "chemistry", "b2")
        b.add_node(n3)
        b.add_node(n4)
        b.add_contradiction("b1", "b2")

        merged, stats = merge_graphs(a, b)

        assert len(merged) == 4
        assert stats.conflicts_added >= 1

        # Check that conflict edges from both graphs exist
        has_a_conflict = False
        has_b_conflict = False
        for n in merged.nodes.values():
            if n.text == "Fact A" and len(n.conflicts_with) > 0:
                has_a_conflict = True
            if n.text == "Fact C" and len(n.conflicts_with) > 0:
                has_b_conflict = True

        assert has_a_conflict
        assert has_b_conflict

    def test_merge_stats_to_dict(self):
        stats = MergeStats(
            nodes_from_a=10,
            nodes_from_b_added=5,
            nodes_from_b_merged=2,
            nodes_from_b_skipped=1,
            edges_added=3,
            edges_skipped=0,
            conflicts_added=1,
            sources_merged=2,
        )
        d = stats.to_dict()
        assert d["nodes_from_a"] == 10
        assert d["nodes_from_b_added"] == 5
        assert d["sources_merged"] == 2

    def test_source_merging_enabled(self):
        a = KnowledgeGraph()
        water_a = make_node("Water boils at 100 degrees Celsius", 0.95, "chemistry", "water_a")
        water_a.sources = [Source(source_type="textbook", title="Chemistry 101", quality_score=0.9)]
        a.add_node(water_a)

        b = KnowledgeGraph()
        water_b = make_node("Water boils at 100 degrees Celsius", 0.80, "chemistry", "water_b")
        water_b.sources = [Source(source_type="journal", title="Nature Chemistry", quality_score=0.95)]
        b.add_node(water_b)

        merged, stats = merge_graphs(a, b, merge_sources=True)

        # The merged water node should have both sources
        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                assert len(n.sources) == 2
                source_titles = {s.title for s in n.sources}
                assert "Chemistry 101" in source_titles
                assert "Nature Chemistry" in source_titles
                break

        assert stats.sources_merged == 1

    def test_source_merging_disabled(self):
        a = KnowledgeGraph()
        water_a = make_node("Water boils at 100 degrees Celsius", 0.95, "chemistry", "water_a")
        water_a.sources = [Source(source_type="textbook", title="Chemistry 101", quality_score=0.9)]
        a.add_node(water_a)

        b = KnowledgeGraph()
        water_b = make_node("Water boils at 100 degrees Celsius", 0.80, "chemistry", "water_b")
        water_b.sources = [Source(source_type="journal", title="Nature Chemistry", quality_score=0.95)]
        b.add_node(water_b)

        merged, stats = merge_graphs(a, b, merge_sources=False)

        # The merged water node should only have A's source
        for n in merged.nodes.values():
            if "water boils" in n.text.lower():
                assert len(n.sources) == 1
                assert n.sources[0].title == "Chemistry 101"
                break

        assert stats.sources_merged == 0

    def test_merged_graph_is_independent(self):
        """Mutations to the merged graph should not affect originals."""
        a = build_graph_a()
        b = build_graph_b()
        a_len = len(a)
        b_len = len(b)

        merged, _ = merge_graphs(a, b)

        # Add a node to merged — should not affect originals
        new_node = make_node("Helium is inert", 0.9, "chemistry", "helium")
        merged.add_node(new_node)

        assert len(a) == a_len
        assert len(b) == b_len
        assert len(merged) == 6  # 5 from merge + 1 added


# ------------------------------------------------------------------
# Edge label preservation
# ------------------------------------------------------------------


class TestEdgeLabelPreservation:
    """Test that edge labels survive export/import/merge."""

    def test_export_preserves_edge_types(self):
        """Edge types appear in exported edge dicts."""
        graph = build_test_graph()
        # Add an explicit edge label
        steam = graph.get_node("steam")
        steam.edge_labels["water_bp"] = "supports"

        export = export_subgraph(graph, "chemistry", confidence_threshold=0.0)

        # Find the water_bp -> steam edge
        edge_types = {(e["source"], e["target"]): e.get("type", "depends_on") for e in export.edges}
        assert edge_types.get(("water_bp", "steam")) == "supports"

    def test_import_restores_edge_types(self):
        """Edge types survive import from JSON."""
        from src.core.export import export_to_json, import_from_json

        graph = build_test_graph()
        steam = graph.get_node("steam")
        steam.edge_labels["water_bp"] = "supports"

        export = export_subgraph(graph, "chemistry", confidence_threshold=0.0)
        json_str = export_to_json(export)
        json_data = import_from_json(json_str)

        target = KnowledgeGraph()
        import_subgraph(target, json_data)

        # Find the steam node and check its edge label
        steam_nodes = [n for n in target.nodes.values() if "Steam" in n.text]
        assert len(steam_nodes) == 1
        steam_node = steam_nodes[0]

        # Find the water predecessor
        water_nodes = [n for n in target.nodes.values() if "Water boils" in n.text]
        assert len(water_nodes) == 1
        water_id = water_nodes[0].id

        assert steam_node.get_edge_label(water_id) == "supports"
