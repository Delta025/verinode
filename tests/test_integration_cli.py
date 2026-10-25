"""
Integration tests for the Verinode CLI.

These tests require a running Ollama server with a pulled model.
Skipped by default; run with: pytest --run-integration

Each test exercises the CLI end-to-end using click's CliRunner,
with real Ollama extraction and reasoning.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli.main import cli

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _temp_graph_path():
    """Return a path for a temporary graph file (empty)."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    # Write an empty graph so it can be loaded
    with open(path, "w") as f:
        json.dump({"nodes": [], "metadata": {}}, f)
    return path


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestCLIIngestion:
    """Test ingestion via the CLI with live Ollama."""

    def test_ingest_and_stats(self):
        """Ingest text, then verify stats shows the added nodes."""
        path = _temp_graph_path()
        try:
            runner = CliRunner()
            # Ingest
            result = runner.invoke(
                cli,
                [
                    "--graph",
                    path,
                    "ingest",
                    "Water boils at 100 degrees Celsius at sea level. Water freezes at 0 degrees Celsius.",
                    "--domain",
                    "chemistry",
                    "--source-type",
                    "textbook",
                    "--source-title",
                    "Chemistry 101",
                    "--source-quality",
                    "0.9",
                ],
            )
            assert result.exit_code == 0, f"ingest failed: {result.output}"
            assert "ADDED" in result.output or "added" in result.output.lower()

            # Stats
            result = runner.invoke(cli, ["--graph", path, "stats"])
            assert result.exit_code == 0
            assert "chemistry" in result.output
        finally:
            os.unlink(path)

    def test_ingest_from_file(self):
        """Ingest from a file."""
        path = _temp_graph_path()
        fd, text_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(text_path, "w") as f:
            f.write("Gold has the symbol Au and atomic number 79.")

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--graph",
                    path,
                    "ingest",
                    text_path,
                    "--file",
                    "--domain",
                    "chemistry",
                ],
            )
            assert result.exit_code == 0, f"ingest --file failed: {result.output}"
        finally:
            os.unlink(path)
            os.unlink(text_path)


class TestCLIQuerying:
    """Test query and hypothesis via the CLI with a pre-ingested graph."""

    @pytest.fixture(autouse=True)
    def setup_graph(self, tmp_path):
        """Ingest some facts to create a populated graph."""
        self.graph_path = str(tmp_path / "test_graph.json")
        # Write empty graph
        with open(self.graph_path, "w") as f:
            json.dump({"nodes": [], "metadata": {}}, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--graph",
                self.graph_path,
                "ingest",
                "Water boils at 100 degrees Celsius. Ice melts at 0 degrees Celsius.",
                "--domain",
                "chemistry",
                "--source-quality",
                "0.9",
            ],
        )
        assert result.exit_code == 0, f"Setup ingest failed: {result.output}"

    def test_query(self):
        """Query the populated graph."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--graph",
                self.graph_path,
                "query",
                "What is the boiling point of water?",
            ],
        )
        assert result.exit_code == 0
        assert "Answer" in result.output or "Query" in result.output

    def test_hypothesis(self):
        """Test a hypothesis against the graph."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--graph",
                self.graph_path,
                "hypothesis",
                "Water boils at 100 degrees Celsius",
            ],
        )
        assert result.exit_code == 0
        assert "Hypothesis" in result.output

    def test_list_after_ingestion(self):
        """List nodes after ingestion."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--graph",
                self.graph_path,
                "list",
            ],
        )
        assert result.exit_code == 0
        # Should contain at least one chemistry node
        assert "chemistry" in result.output.lower() or "fact" in result.output.lower()


class TestCLISaveLoad:
    """Test save/load round-trip with live data."""

    def test_save_load_round_trip(self, tmp_path):
        """Ingest, save, load into new path, verify stats match."""
        graph_path = str(tmp_path / "original.json")
        with open(graph_path, "w") as f:
            json.dump({"nodes": [], "metadata": {}}, f)

        runner = CliRunner()

        # Ingest
        result = runner.invoke(
            cli,
            [
                "--graph",
                graph_path,
                "ingest",
                "Silver has the symbol Ag.",
                "--domain",
                "chemistry",
            ],
        )
        assert result.exit_code == 0

        # Save to new location
        copy_path = str(tmp_path / "copy.json")
        result = runner.invoke(
            cli,
            [
                "--graph",
                graph_path,
                "save",
                copy_path,
            ],
        )
        assert result.exit_code == 0

        # Load the copy and check stats
        result = runner.invoke(cli, ["--graph", copy_path, "stats"])
        assert result.exit_code == 0
        assert "chemistry" in result.output


class TestCLIAnomalies:
    """Test anomaly detection via the CLI."""

    def test_anomalies_on_ingested_graph(self, tmp_path):
        """Run anomaly detection on a live-ingested graph."""
        graph_path = str(tmp_path / "test.json")
        with open(graph_path, "w") as f:
            json.dump({"nodes": [], "metadata": {}}, f)

        runner = CliRunner()

        # Ingest
        result = runner.invoke(
            cli,
            [
                "--graph",
                graph_path,
                "ingest",
                "Iron is a metal. Iron has the symbol Fe.",
                "--domain",
                "chemistry",
                "--source-quality",
                "0.9",
            ],
        )
        assert result.exit_code == 0

        # Run anomalies
        result = runner.invoke(cli, ["--graph", graph_path, "anomalies"])
        assert result.exit_code == 0
        # Either "No anomalies" or a table
        assert "anomal" in result.output.lower() or "consistent" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-integration"])
