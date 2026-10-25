"""
Unit tests for the Engram CLI.

All tests use pre-built in-memory graphs and mocked engines.
No Ollama server needed.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli.main import cli
from src.cli.state import EngineState
from src.cli.formatting import (
    confidence_color,
    confidence_text,
    status_style,
    short_id,
    resolve_node_id,
)
from src.core.node import KnowledgeNode, NodeType, NodeStatus, Source, Derivation
from src.core.graph import KnowledgeGraph
from datetime import datetime, timedelta


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_graph() -> KnowledgeGraph:
    """Create a small test graph with 4 nodes."""
    graph = KnowledgeGraph()

    n1 = KnowledgeNode(
        text="Water boils at 100 degrees Celsius at sea level",
        node_type=NodeType.FACT,
        confidence=0.95,
        domain="chemistry",
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.95)],
    )
    n2 = KnowledgeNode(
        text="Water freezes at 0 degrees Celsius",
        node_type=NodeType.FACT,
        confidence=0.9,
        domain="chemistry",
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.95)],
    )
    n3 = KnowledgeNode(
        text="Steam is produced when water boils",
        node_type=NodeType.INFERENCE,
        confidence=0.8,
        domain="chemistry",
        predecessor_nodes=[n1.id],
        derivation=Derivation(method="inferred", inference_depth=1, derivation_type="chain"),
    )
    n4 = KnowledgeNode(
        text="Gold is a metal",
        node_type=NodeType.FACT,
        confidence=0.4,
        domain="chemistry",
        sources=[Source(source_type="web", title="RandomFacts.blog", quality_score=0.3)],
    )

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    return graph


def _save_temp_graph(graph: KnowledgeGraph) -> str:
    """Save a graph to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    graph.save(path)
    return path


def _build_stale_graph() -> KnowledgeGraph:
    """Create a graph with nodes at various staleness levels for maintenance tests."""
    now = datetime.now()
    graph = KnowledgeGraph()

    # Stale node — will be decayed
    n1 = KnowledgeNode(
        text="Gold melts at 1064 degrees Celsius",
        node_type=NodeType.FACT,
        confidence=0.85,
        domain="chemistry",
        node_id="stale-node",
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.9)],
    )
    n1.last_validated = now - timedelta(days=120)

    # Low confidence + conflicted node — will be doubt-flagged
    n2 = KnowledgeNode(
        text="Iron melts at 100 degrees Celsius",
        node_type=NodeType.FACT,
        confidence=0.25,
        domain="chemistry",
        node_id="low-conf",
        sources=[Source(source_type="web", title="BadSite", quality_score=0.2)],
    )
    n2.conflicts_with = ["stale-node"]
    n2.update_status()

    # Well-accessed node — will be strengthened
    n3 = KnowledgeNode(
        text="Water boils at 100 degrees Celsius",
        node_type=NodeType.FACT,
        confidence=0.70,
        domain="chemistry",
        node_id="popular-node",
        sources=[Source(source_type="textbook", title="Chem 101", quality_score=0.95)],
    )
    n3.access_count = 20

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    return graph


# ------------------------------------------------------------------
# Formatting unit tests
# ------------------------------------------------------------------

class TestFormattingHelpers:
    """Test low-level formatting utilities."""

    def test_confidence_color_green(self):
        assert confidence_color(0.9) == "green"

    def test_confidence_color_yellow(self):
        assert confidence_color(0.6) == "yellow"

    def test_confidence_color_red(self):
        assert confidence_color(0.3) == "red"

    def test_status_style(self):
        assert "green" in status_style("established")
        assert "yellow" in status_style("provisional")
        assert "red" in status_style("contested")

    def test_short_id(self):
        assert short_id("abcdefgh-1234-5678-9abc-def012345678") == "abcdefgh"

    def test_resolve_node_id_exact(self):
        graph = _build_graph()
        full_id = list(graph.nodes.keys())[0]
        assert resolve_node_id(graph, full_id) == full_id

    def test_resolve_node_id_prefix(self):
        graph = _build_graph()
        full_id = list(graph.nodes.keys())[0]
        prefix = full_id[:8]
        assert resolve_node_id(graph, prefix) == full_id

    def test_resolve_node_id_not_found(self):
        import click
        graph = _build_graph()
        with pytest.raises(click.BadParameter, match="No node found"):
            resolve_node_id(graph, "zzzzzzz")


# ------------------------------------------------------------------
# CLI: stats
# ------------------------------------------------------------------

class TestStatsCommand:
    """Test the 'stats' command."""

    def test_stats_empty_graph(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_stats_with_graph(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "stats"])
            assert result.exit_code == 0
            assert "4" in result.output  # total nodes
            assert "chemistry" in result.output
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: list
# ------------------------------------------------------------------

class TestListCommand:
    """Test the 'list' command."""

    def test_list_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_list_all(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "list"])
            assert result.exit_code == 0
            assert "Water boils" in result.output
        finally:
            os.unlink(path)

    def test_list_filter_type(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "list", "--type", "inference"])
            assert result.exit_code == 0
            assert "Steam" in result.output
            # Should not show non-inference nodes
            assert "Gold" not in result.output
        finally:
            os.unlink(path)

    def test_list_filter_invalid_type(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "list", "--type", "bogus"])
            assert result.exit_code != 0
        finally:
            os.unlink(path)

    def test_list_sort_text(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "list", "--sort", "text"])
            assert result.exit_code == 0
            # Gold should come before Water alphabetically
            gold_pos = result.output.find("Gold")
            water_pos = result.output.find("Water")
            assert gold_pos < water_pos
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: inspect
# ------------------------------------------------------------------

class TestInspectCommand:
    """Test the 'inspect' command."""

    def test_inspect_full_id(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        full_id = list(graph.nodes.keys())[0]
        node_text = graph.nodes[full_id].text
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "inspect", full_id])
            assert result.exit_code == 0
            assert full_id in result.output
            assert node_text[:30] in result.output
        finally:
            os.unlink(path)

    def test_inspect_prefix(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        full_id = list(graph.nodes.keys())[0]
        prefix = full_id[:8]
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "inspect", prefix])
            assert result.exit_code == 0
            assert full_id in result.output
        finally:
            os.unlink(path)

    def test_inspect_not_found(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "inspect", "zzzzzzz"])
            assert result.exit_code != 0
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: explain
# ------------------------------------------------------------------

class TestExplainCommand:
    """Test the 'explain' command."""

    def test_explain(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        full_id = list(graph.nodes.keys())[0]
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "explain", full_id[:8]])
            assert result.exit_code == 0
            assert "Confidence:" in result.output
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: trace
# ------------------------------------------------------------------

class TestTraceCommand:
    """Test the 'trace' command."""

    def test_trace(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        # Use the inference node (n3) which has predecessors
        nodes_list = list(graph.nodes.values())
        inference_node = [n for n in nodes_list if n.node_type == NodeType.INFERENCE][0]
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["--graph", path, "trace", inference_node.id[:8]]
            )
            assert result.exit_code == 0
            assert "Confidence Trace" in result.output
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: anomalies
# ------------------------------------------------------------------

class TestAnomaliesCommand:
    """Test the 'anomalies' command."""

    def test_anomalies_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["anomalies"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_anomalies_with_graph(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "anomalies"])
            assert result.exit_code == 0
            # May or may not find anomalies depending on graph state
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: save / load
# ------------------------------------------------------------------

class TestSaveLoadCommands:
    """Test save and load commands."""

    def test_save_and_load_round_trip(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)

        fd, path2 = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        try:
            runner = CliRunner()
            # Load from path, save to path2
            result = runner.invoke(cli, ["--graph", path, "save", path2])
            assert result.exit_code == 0
            assert "4 nodes" in result.output

            # Load from path2 and check stats
            result = runner.invoke(cli, ["--graph", path2, "stats"])
            assert result.exit_code == 0
            assert "4" in result.output
        finally:
            os.unlink(path)
            os.unlink(path2)

    def test_load_nonexistent(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["load", "/nonexistent/path.json"])
        assert result.exit_code != 0


# ------------------------------------------------------------------
# CLI: ingest (mocked)
# ------------------------------------------------------------------

class TestIngestCommand:
    """Test the 'ingest' command with mocked pipeline."""

    @staticmethod
    def _mock_ingestion_result(text="Water is H2O"):
        """Create a mock IngestionResult for testing."""
        from src.core.fact_extraction import ExtractedFact
        from src.core.pipeline import IngestionResult, IngestionVerdict

        mock_fact = ExtractedFact(
            text=text,
            fact_type="fact",
            confidence=0.9,
            domain="chemistry",
        )
        return IngestionResult(
            source_text=text,
            extraction_model="qwen2.5:7b",
            verdicts=[IngestionVerdict(fact=mock_fact, action="added", node_id="fake-id-1234")],
            nodes_added=1,
        )

    def test_ingest_text(self):
        """Ingest text with a fully mocked pipeline."""
        mock_result = self._mock_ingestion_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"nodes": [], "metadata": {}}, f)
            graph_path = f.name

        try:
            runner = CliRunner()
            mock_pipe = MagicMock()
            mock_pipe.ingest.return_value = mock_result

            # Patch the pipeline property on the EngineState class
            with patch.object(EngineState, "pipeline", new=property(lambda self: mock_pipe)):
                result = runner.invoke(cli, [
                    "--graph", graph_path,
                    "ingest",
                    "Water is H2O",
                    "--domain", "chemistry",
                ])
                assert result.exit_code == 0
                assert "ADDED" in result.output
                assert "Water is H2O" in result.output
        finally:
            os.unlink(graph_path)

    def test_ingest_from_file(self):
        """Ingest from a file with mocked pipeline."""
        mock_result = self._mock_ingestion_result("Gold is Au")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            tf.write("Gold has the symbol Au and atomic number 79.")
            text_path = tf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"nodes": [], "metadata": {}}, f)
            graph_path = f.name

        try:
            runner = CliRunner()
            mock_pipe = MagicMock()
            mock_pipe.ingest.return_value = mock_result

            with patch.object(EngineState, "pipeline", new=property(lambda self: mock_pipe)):
                result = runner.invoke(cli, [
                    "--graph", graph_path,
                    "ingest", text_path, "--file",
                    "--domain", "chemistry",
                ])
                assert result.exit_code == 0
                assert "ADDED" in result.output
        finally:
            os.unlink(text_path)
            os.unlink(graph_path)


# ------------------------------------------------------------------
# CLI: query (mocked)
# ------------------------------------------------------------------

class TestQueryCommand:
    """Test the 'query' command with a pre-built graph (no Ollama)."""

    def test_query_empty_graph(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "What is water?"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_query_with_graph(self):
        """Query a graph using keyword fallback (no embedding engine needed)."""
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "query", "boiling point of water"])
            assert result.exit_code == 0
            assert "Answer" in result.output or "Query" in result.output
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: hypothesis (mocked)
# ------------------------------------------------------------------

class TestHypothesisCommand:
    """Test the 'hypothesis' command."""

    def test_hypothesis_empty_graph(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hypothesis", "Water boils"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_hypothesis_with_graph(self):
        graph = _build_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["--graph", path, "hypothesis", "Water boils at 100 degrees"]
            )
            assert result.exit_code == 0
            assert "Hypothesis" in result.output
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: SQLite support
# ------------------------------------------------------------------

class TestSQLiteCLI:
    """Test CLI commands with SQLite graph backend."""

    @staticmethod
    def _create_sqlite_graph(db_path):
        """Create a pre-populated SQLite graph from _build_graph().

        Uses from_graph to avoid FK issues when nodes have successor_nodes.
        Returns list of node IDs.
        """
        from src.core.storage import SqliteGraph
        mem_graph = _build_graph()
        sg = SqliteGraph.from_graph(mem_graph, db_path)
        node_ids = list(sg.nodes.keys())
        sg.close()
        return node_ids

    @staticmethod
    def _invoke_and_close(runner, args):
        """Invoke CLI and close any SQLite graph left open by the CliRunner.

        CliRunner doesn't close the SqliteGraph opened by load_graph(),
        which leaves a file handle open on Windows. This helper closes it.
        """
        result = runner.invoke(cli, args, catch_exceptions=False)
        # CliRunner stores state in ctx.obj; we can't easily access it
        # but the SqliteGraph __del__ should close on GC.
        import gc
        gc.collect()
        return result

    def test_stats_with_sqlite_graph(self, tmp_path):
        """Stats command works with a .db graph file."""
        db_path = str(tmp_path / "test.db")
        self._create_sqlite_graph(db_path)

        runner = CliRunner()
        result = self._invoke_and_close(runner, ["--graph", db_path, "stats"])
        assert result.exit_code == 0
        assert "chemistry" in result.output
        assert "4" in result.output

    def test_list_with_sqlite_graph(self, tmp_path):
        """List command works with a .db graph file."""
        db_path = str(tmp_path / "test.db")
        self._create_sqlite_graph(db_path)

        runner = CliRunner()
        result = self._invoke_and_close(runner, ["--graph", db_path, "list"])
        assert result.exit_code == 0
        assert "Water boils" in result.output

    def test_save_format_sqlite(self, tmp_path):
        """Save converts JSON graph to SQLite when target has .db extension."""
        graph = _build_graph()
        json_path = _save_temp_graph(graph)
        db_path = str(tmp_path / "output.db")

        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", json_path, "save", db_path])
            assert result.exit_code == 0
            assert "SQLite" in result.output or "4 nodes" in result.output
            assert os.path.exists(db_path)

            # Verify the SQLite file is valid
            from src.core.storage import SqliteGraph
            sg = SqliteGraph(db_path)
            assert len(sg) == 4
            sg.close()
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_history_json_graph(self):
        """History command shows a message for non-SQLite graphs."""
        runner = CliRunner()
        result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "SQLite" in result.output or "only available" in result.output

    def test_history_sqlite_graph(self, tmp_path):
        """History command shows changelog entries for SQLite graphs."""
        from src.core.storage import SqliteGraph

        db_path = str(tmp_path / "test_history.db")
        sg = SqliteGraph(db_path)
        # Add nodes directly (not from_graph) so changelog entries are created
        sg.add_node(KnowledgeNode(text="Fact one", confidence=0.9, domain="test"))
        sg.add_node(KnowledgeNode(text="Fact two", confidence=0.8, domain="test"))
        sg.close()

        runner = CliRunner()
        result = self._invoke_and_close(runner, ["--graph", db_path, "history"])
        assert result.exit_code == 0
        assert "add_node" in result.output
        assert "Change History" in result.output

    def test_inspect_sqlite_graph(self, tmp_path):
        """Inspect command works with SQLite graph."""
        db_path = str(tmp_path / "test.db")
        node_ids = self._create_sqlite_graph(db_path)

        runner = CliRunner()
        result = self._invoke_and_close(runner, ["--graph", db_path, "inspect", node_ids[0][:8]])
        assert result.exit_code == 0
        assert node_ids[0] in result.output


# ------------------------------------------------------------------
# CLI: maintain
# ------------------------------------------------------------------

class TestMaintainCommand:
    """Test the 'maintain' command."""

    def test_maintain_empty_graph(self):
        """Maintain on an empty graph prints 'empty'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["maintain"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_maintain_with_stale_graph(self):
        """Maintain on a stale graph should produce a maintenance report."""
        graph = _build_stale_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "maintain"])
            assert result.exit_code == 0
            # Report should contain maintenance output
            assert "Maintenance" in result.output or "Decay" in result.output or "examined" in result.output
        finally:
            os.unlink(path)

    def test_maintain_dry_run(self):
        """Dry-run should display report but not modify the original graph file."""
        graph = _build_stale_graph()
        path = _save_temp_graph(graph)
        try:
            # Read original file content
            with open(path, "r") as f:
                original_content = f.read()

            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "maintain", "--dry-run"])
            assert result.exit_code == 0
            assert "DRY RUN" in result.output

            # File should not have been modified
            with open(path, "r") as f:
                after_content = f.read()
            assert original_content == after_content
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: doubt
# ------------------------------------------------------------------

class TestDoubtCommand:
    """Test the 'doubt' command."""

    def test_doubt_empty_graph(self):
        """Doubt on an empty graph prints 'empty'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doubt"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_doubt_with_conflicted_node(self):
        """Doubt on a graph with problematic nodes should show flags."""
        graph = _build_stale_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "doubt"])
            assert result.exit_code == 0
            # Should show doubt flags or "healthy" (both are valid)
            output_lower = result.output.lower()
            assert "doubt" in output_lower or "healthy" in output_lower or "flag" in output_lower
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# CLI: strengthen
# ------------------------------------------------------------------

class TestStrengthenCommand:
    """Test the 'strengthen' command."""

    def test_strengthen_existing_node(self):
        """Strengthen an existing node should succeed."""
        graph = _build_stale_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            # Use partial ID of the popular node
            result = runner.invoke(cli, ["--graph", path, "strengthen", "popular-"])
            assert result.exit_code == 0
            # Should show access info or boost info
            assert "access" in result.output.lower() or "boost" in result.output.lower() or "Strengthened" in result.output
        finally:
            os.unlink(path)

    def test_strengthen_with_validate(self):
        """Strengthen with --validate flag should succeed."""
        graph = _build_stale_graph()
        path = _save_temp_graph(graph)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["--graph", path, "strengthen", "--validate", "popular-"])
            assert result.exit_code == 0
            assert "access" in result.output.lower() or "boost" in result.output.lower() or "Strengthened" in result.output
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# Run all
# ------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
