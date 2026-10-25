"""
Verinode CLI — click-based command-line interface.

Run via:  python -m src.cli [--graph PATH] [--model MODEL] COMMAND
"""

from pathlib import Path

import click

from .state import EngineState
from .formatting import (
    console,
    format_node_table,
    format_stats,
    format_anomalies,
    format_ingestion_result,
    format_query_result,
    format_hypothesis_result,
    format_verification_result,
    format_retraction_result,
    format_memory_result,
    format_export_result,
    format_import_result,
    format_merge_result,
    format_node_detail,
    format_confidence_trace,
    format_maintenance_report,
    format_doubt_flags,
    format_strengthen_result,
    resolve_node_id,
    short_id,
)

# Click context key for shared state
_STATE_KEY = "state"


# ------------------------------------------------------------------
# Root group
# ------------------------------------------------------------------

@click.group()
@click.option(
    "--graph", "-g",
    default="verinode_graph.json",
    envvar="VERINODE_GRAPH",
    show_default=True,
    help="Path to the graph file (.json or .db).",
)
@click.option(
    "--model", "-m",
    default="qwen3:14b",
    envvar="VERINODE_MODEL",
    show_default=True,
    help="Ollama model name for LLM operations.",
)
@click.pass_context
def cli(ctx, graph, model):
    """Verinode — knowledge graph with confidence tracking and reasoning."""
    ctx.ensure_object(dict)
    state = EngineState(graph_path=graph, model=model)
    # Auto-load graph if the file exists
    if Path(graph).exists():
        try:
            state.load_graph()
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load graph: {exc}[/yellow]")
    ctx.obj[_STATE_KEY] = state


def _get_state(ctx) -> EngineState:
    """Retrieve the shared ``EngineState`` from the click context."""
    return ctx.obj[_STATE_KEY]


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

@cli.command()
@click.pass_context
def stats(ctx):
    """Show graph statistics."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return
    s = state.graph.get_statistics()
    console.print(format_stats(s))


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------

@cli.command("list")
@click.option("--domain", "-d", default=None, help="Filter by domain.")
@click.option("--type", "-t", "node_type", default=None, help="Filter by type (fact/rule/...).")
@click.option("--status", "-s", default=None, help="Filter by status (established/provisional/...).")
@click.option("--limit", "-n", default=20, show_default=True, help="Max nodes to show.")
@click.option(
    "--sort",
    type=click.Choice(["confidence", "created", "text"], case_sensitive=False),
    default="confidence",
    show_default=True,
    help="Sort order.",
)
@click.pass_context
def list_nodes(ctx, domain, node_type, status, limit, sort):
    """List nodes in the graph (with optional filters)."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return

    from src.core.node import NodeType, NodeStatus

    nodes = list(state.graph.nodes.values())

    # Filters
    if domain:
        nodes = [n for n in nodes if n.domain == domain]
    if node_type:
        try:
            nt = NodeType(node_type)
        except ValueError:
            raise click.BadParameter(
                f"Unknown type '{node_type}'. "
                f"Choose from: {', '.join(t.value for t in NodeType)}"
            )
        nodes = [n for n in nodes if n.node_type == nt]
    if status:
        try:
            ns = NodeStatus(status)
        except ValueError:
            raise click.BadParameter(
                f"Unknown status '{status}'. "
                f"Choose from: {', '.join(s.value for s in NodeStatus)}"
            )
        nodes = [n for n in nodes if n.status == ns]

    # Sort
    if sort == "confidence":
        nodes.sort(key=lambda n: n.confidence, reverse=True)
    elif sort == "created":
        nodes.sort(key=lambda n: n.created_at, reverse=True)
    elif sort == "text":
        nodes.sort(key=lambda n: n.text.lower())

    nodes = nodes[:limit]

    if not nodes:
        console.print("[dim]No nodes match the filters.[/dim]")
        return

    title = "Nodes"
    filter_parts = []
    if domain:
        filter_parts.append(f"domain={domain}")
    if node_type:
        filter_parts.append(f"type={node_type}")
    if status:
        filter_parts.append(f"status={status}")
    if filter_parts:
        title += f" ({', '.join(filter_parts)})"

    console.print(format_node_table(nodes, title=title))


# ------------------------------------------------------------------
# inspect
# ------------------------------------------------------------------

@cli.command()
@click.argument("node_id")
@click.pass_context
def inspect(ctx, node_id):
    """Show detailed information about a node."""
    state = _get_state(ctx)
    full_id = resolve_node_id(state.graph, node_id)
    node = state.graph.get_node(full_id)
    console.print(format_node_detail(node, state.graph))


# ------------------------------------------------------------------
# explain
# ------------------------------------------------------------------

@cli.command()
@click.argument("node_id")
@click.pass_context
def explain(ctx, node_id):
    """Explain a node's confidence and evidence chains."""
    state = _get_state(ctx)
    full_id = resolve_node_id(state.graph, node_id)
    explanation = state.reasoning_engine.explain_node(state.graph, full_id)
    console.print(explanation)


# ------------------------------------------------------------------
# trace
# ------------------------------------------------------------------

@cli.command()
@click.argument("node_id")
@click.pass_context
def trace(ctx, node_id):
    """Trace a node's confidence back to root sources."""
    state = _get_state(ctx)
    full_id = resolve_node_id(state.graph, node_id)
    t = state.confidence_engine.trace_confidence_source(state.graph, full_id)
    console.print(format_confidence_trace(t))


# ------------------------------------------------------------------
# anomalies
# ------------------------------------------------------------------

@cli.command()
@click.option("--threshold", default=0.15, show_default=True, help="Anomaly threshold.")
@click.pass_context
def anomalies(ctx, threshold):
    """Detect confidence anomalies in the graph."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return

    engine = ConfidenceEngineWithThreshold(threshold)
    anoms = engine.detect_anomalies(state.graph)

    if not anoms:
        console.print("[green]No anomalies detected. The graph is internally consistent.[/green]")
        return

    console.print(format_anomalies(anoms))
    console.print(f"\n[bold]{len(anoms)}[/bold] anomaly(ies) found.")


class ConfidenceEngineWithThreshold:
    """Thin wrapper to allow custom anomaly threshold at runtime."""

    def __init__(self, threshold: float):
        from src.core.confidence import ConfidenceEngine as _CE

        self._engine = _CE(config={"anomaly_threshold": threshold})

    def detect_anomalies(self, graph):
        return self._engine.detect_anomalies(graph)


# ------------------------------------------------------------------
# save / load
# ------------------------------------------------------------------

@cli.command()
@click.argument("path", required=False)
@click.option(
    "--format", "-F", "fmt",
    type=click.Choice(["json", "sqlite"], case_sensitive=False),
    default=None,
    help="Force output format (json or sqlite). Inferred from extension if omitted.",
)
@click.pass_context
def save(ctx, path, fmt):
    """Save the graph to a file (JSON or SQLite)."""
    state = _get_state(ctx)

    # Determine target path
    target = path or state.graph_path

    # If --format given, override extension
    if fmt == "json" and not target.endswith(".json"):
        target = target.rsplit(".", 1)[0] + ".json" if "." in target else target + ".json"
    elif fmt == "sqlite" and not target.endswith((".db", ".sqlite", ".sqlite3")):
        target = target.rsplit(".", 1)[0] + ".db" if "." in target else target + ".db"

    state.save_graph(target)
    fmt_label = "SQLite" if target.endswith((".db", ".sqlite", ".sqlite3")) else "JSON"
    console.print(f"Graph saved to [bold]{target}[/bold] ({len(state.graph)} nodes, {fmt_label}).")


@cli.command()
@click.argument("path")
@click.pass_context
def load(ctx, path):
    """Load a graph from a file (JSON or SQLite)."""
    state = _get_state(ctx)
    if not Path(path).exists():
        raise click.BadParameter(f"File not found: {path}")
    state.load_graph(path)
    fmt_label = "SQLite" if state.is_sqlite else "JSON"
    console.print(f"Loaded graph from [bold]{path}[/bold] ({len(state.graph)} nodes, {fmt_label}).")


# ------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------

@cli.command()
@click.argument("text_or_path")
@click.option("--file", "-f", "is_file", is_flag=True, help="Treat argument as a file path.")
@click.option("--domain", "-d", default=None, help="Domain hint (e.g. chemistry).")
@click.option("--source-type", default="unknown", show_default=True, help="Source type.")
@click.option("--source-title", default="CLI input", show_default=True, help="Source title.")
@click.option(
    "--source-quality",
    default=0.7,
    show_default=True,
    type=float,
    help="Source quality [0-1].",
)
@click.option("--batch", is_flag=True, help="With --file, treat each paragraph as a separate passage.")
@click.pass_context
def ingest(ctx, text_or_path, is_file, domain, source_type, source_title, source_quality, batch):
    """Ingest text (or a file) into the knowledge graph."""
    state = _get_state(ctx)

    from src.core.node import Source

    source = Source(
        source_type=source_type,
        title=source_title,
        quality_score=source_quality,
    )

    # Read text
    if is_file:
        p = Path(text_or_path)
        if not p.exists():
            raise click.BadParameter(f"File not found: {text_or_path}")
        text = p.read_text(encoding="utf-8")
    else:
        text = text_or_path

    # Batch or single
    if batch and is_file:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            console.print("[dim]No paragraphs found in file.[/dim]")
            return
        console.print(f"Ingesting {len(paragraphs)} paragraph(s)...")
        result = state.pipeline.ingest_batch(paragraphs, source=source, domain=domain)
        for i, r in enumerate(result.results):
            console.print(format_ingestion_result(r))
        console.print(
            f"\n[bold]Batch totals:[/bold] {result.total_added} added, "
            f"{result.total_duplicates} merged, "
            f"{result.total_flagged} flagged, "
            f"{result.total_rejected} rejected"
        )
    else:
        result = state.pipeline.ingest(text, source=source, domain=domain)
        console.print(format_ingestion_result(result))

    # Auto-save (no-op for SQLite — changes already persisted)
    state.save_graph()
    if state.is_sqlite:
        console.print(f"[dim]Changes persisted to {state.graph_path} ({len(state.graph)} nodes).[/dim]")
    else:
        console.print(f"[dim]Graph saved to {state.graph_path} ({len(state.graph)} nodes).[/dim]")


# ------------------------------------------------------------------
# query
# ------------------------------------------------------------------

@cli.command()
@click.argument("question")
@click.option("--k", default=5, show_default=True, help="Number of candidate nodes.")
@click.option("--show-chains", is_flag=True, help="Show supporting reasoning chains.")
@click.pass_context
def query(ctx, question, k, show_chains):
    """Ask a natural-language question."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty. Ingest some text first.[/dim]")
        return

    state.ensure_embeddings_indexed()
    result = state.reasoning_engine.query(
        state.graph,
        question,
        embedding_engine=state.embedding_engine,
        k=k,
    )
    console.print(format_query_result(result, show_chains=show_chains))


# ------------------------------------------------------------------
# hypothesis
# ------------------------------------------------------------------

@cli.command()
@click.argument("claim")
@click.option("--k", default=10, show_default=True, help="Number of candidate nodes.")
@click.option("--show-evidence", is_flag=True, help="Show evidence chains.")
@click.pass_context
def hypothesis(ctx, claim, k, show_evidence):
    """Test a hypothesis / claim against the knowledge graph."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty. Ingest some text first.[/dim]")
        return

    state.ensure_embeddings_indexed()
    result = state.reasoning_engine.test_hypothesis(
        state.graph,
        claim,
        embedding_engine=state.embedding_engine,
        k=k,
    )
    console.print(format_hypothesis_result(result, show_evidence=show_evidence))


# ------------------------------------------------------------------
# verify
# ------------------------------------------------------------------

@cli.command()
@click.argument("claim")
@click.option("--k", default=10, show_default=True, help="Number of candidate nodes.")
@click.option("--show-evidence", is_flag=True, help="Show supporting/contradicting evidence.")
@click.pass_context
def verify(ctx, claim, k, show_evidence):
    """Verify an external claim against the knowledge graph (read-only)."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty. Ingest some text first.[/dim]")
        return

    state.ensure_embeddings_indexed()
    result = state.verification_engine.verify(claim, k=k)
    console.print(format_verification_result(result, show_evidence=show_evidence))


# ------------------------------------------------------------------
# what-if
# ------------------------------------------------------------------

@cli.command("what-if")
@click.argument("node_id")
@click.pass_context
def what_if(ctx, node_id):
    """Simulate retracting a node and show confidence cascade impact."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return

    full_id = resolve_node_id(state.graph, node_id)
    sim = state.verification_engine.simulate_retraction(full_id)

    panel_or_tuple = format_retraction_result(sim)
    if isinstance(panel_or_tuple, tuple):
        panel, table = panel_or_tuple
        console.print(panel)
        if table is not None:
            console.print(table)
    else:
        console.print(panel_or_tuple)


# ------------------------------------------------------------------
# maintain
# ------------------------------------------------------------------

@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would change without modifying the graph.")
@click.pass_context
def maintain(ctx, dry_run):
    """Run a full neuroplasticity maintenance cycle (decay + strengthen + flag)."""
    import copy

    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return

    if dry_run:
        from src.core.graph import KnowledgeGraph as KG
        graph_copy = KG()
        for node in state.graph.nodes.values():
            graph_copy.add_node(copy.deepcopy(node))
        report = state.neuroplasticity_engine.run_maintenance(graph_copy)
        console.print("[yellow]DRY RUN -- no changes were applied to the graph.[/yellow]")
    else:
        report = state.neuroplasticity_engine.run_maintenance(state.graph)
        state.save_graph()
        if state.is_sqlite:
            console.print(f"[dim]Changes persisted to {state.graph_path}.[/dim]")
        else:
            console.print(f"[dim]Graph saved to {state.graph_path} ({len(state.graph)} nodes).[/dim]")

    console.print(format_maintenance_report(report))


# ------------------------------------------------------------------
# doubt
# ------------------------------------------------------------------

@cli.command()
@click.pass_context
def doubt(ctx):
    """Show nodes flagged as doubtful (read-only diagnostic)."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return

    flags = state.neuroplasticity_engine.flag_doubtful_nodes(state.graph)

    if not flags:
        console.print("[green]No doubtful nodes found. The graph looks healthy.[/green]")
        return

    console.print(format_doubt_flags(flags))
    console.print(f"\n[bold]{len(flags)}[/bold] doubt flag(s) found.")


# ------------------------------------------------------------------
# strengthen
# ------------------------------------------------------------------

@cli.command()
@click.argument("node_id")
@click.option("--validate", is_flag=True, help="Also mark the node as validated (update last_validated).")
@click.pass_context
def strengthen(ctx, node_id, validate):
    """Manually strengthen a node's confidence (record access + optional validation)."""
    state = _get_state(ctx)
    full_id = resolve_node_id(state.graph, node_id)

    result = state.neuroplasticity_engine.strengthen_pathway(
        state.graph, full_id, is_validation=validate,
    )

    if result is None:
        node = state.graph.get_node(full_id)
        console.print(
            f"Node [cyan]{short_id(full_id)}[/cyan] accessed "
            f"(count: {node.access_count}), but no confidence boost applied yet "
            f"(threshold: {state.neuroplasticity_engine.strengthen_access_threshold} accesses)."
        )
    else:
        console.print(format_strengthen_result(result))

    state.save_graph()
    if state.is_sqlite:
        console.print(f"[dim]Changes persisted to {state.graph_path}.[/dim]")
    else:
        console.print(f"[dim]Graph saved to {state.graph_path} ({len(state.graph)} nodes).[/dim]")


# ------------------------------------------------------------------
# history
# ------------------------------------------------------------------

@cli.command()
@click.option("--limit", "-n", default=20, show_default=True, help="Max entries to show.")
@click.pass_context
def history(ctx, limit):
    """Show recent graph change history (SQLite graphs only)."""
    state = _get_state(ctx)
    if not state.is_sqlite:
        console.print(
            "[yellow]History is only available for SQLite graphs (.db).[/yellow]\n"
            "[dim]Use --graph my_graph.db to enable persistent storage with history.[/dim]"
        )
        return

    from src.core.storage import SqliteGraph

    if not isinstance(state.graph, SqliteGraph):
        console.print("[dim]No changelog available.[/dim]")
        return

    entries = state.graph.get_changelog(limit=limit)
    if not entries:
        console.print("[dim]No history entries recorded yet.[/dim]")
        return

    from rich.table import Table, box

    table = Table(
        title="Change History",
        box=box.SIMPLE,
        pad_edge=False,
        show_lines=False,
    )
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Timestamp", no_wrap=True)
    table.add_column("Action", style="bold")
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("Details", style="dim")

    for entry in entries:
        node_id = entry.get("node_id") or ""
        node_short = node_id[:8] if node_id else ""
        details = entry.get("details", {})
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items()) if details else ""
        table.add_row(
            str(entry["id"]),
            entry["timestamp"][:19],  # trim microseconds
            entry["change_type"],
            node_short,
            detail_str[:60],
        )

    console.print(table)
    console.print(f"\n[dim]{len(entries)} entries shown (limit={limit}).[/dim]")


# ------------------------------------------------------------------
# memory-query
# ------------------------------------------------------------------

@cli.command("memory-query")
@click.argument("question")
@click.option("--domain", "-d", default=None, help="Filter by domain.")
@click.option("--k", default=10, show_default=True, help="Number of candidate nodes.")
@click.option("--chains", is_flag=True, help="Include reasoning chains.")
@click.pass_context
def memory_query(ctx, question, domain, k, chains):
    """Query the graph as LLM memory (structured JSON output)."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty. Ingest some text first.[/dim]")
        return

    state.ensure_embeddings_indexed()
    result = state.llm_memory.query(question, domain=domain, k=k, include_chains=chains)
    console.print(format_memory_result(result))


# ------------------------------------------------------------------
# export
# ------------------------------------------------------------------

@cli.command("export")
@click.argument("domain")
@click.option("--output", "-o", default=None, help="Output file path (default: <domain>_export.json).")
@click.option("--threshold", default=0.5, show_default=True, help="Minimum confidence threshold.")
@click.pass_context
def export_graph(ctx, domain, output, threshold):
    """Export a domain slice as a portable JSON file."""
    state = _get_state(ctx)
    if len(state.graph) == 0:
        console.print("[dim]Graph is empty.[/dim]")
        return

    from src.core.export import export_subgraph, export_to_json

    export = export_subgraph(state.graph, domain, confidence_threshold=threshold)
    console.print(format_export_result(export))

    if export.node_count == 0:
        console.print(f"[yellow]No nodes found in domain '{domain}' above threshold {threshold}.[/yellow]")
        return

    out_path = output or f"{domain}_export.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(export_to_json(export), encoding="utf-8")
    console.print(f"[green]Exported to {out_path}[/green]")


# ------------------------------------------------------------------
# import-graph
# ------------------------------------------------------------------

@cli.command("import-graph")
@click.argument("path")
@click.option(
    "--strategy",
    type=click.Choice(["keep_higher_confidence", "keep_existing", "keep_imported"], case_sensitive=False),
    default="keep_higher_confidence",
    show_default=True,
    help="Conflict resolution strategy.",
)
@click.pass_context
def import_graph(ctx, path, strategy):
    """Import a sub-graph export from a JSON file."""
    state = _get_state(ctx)

    if not Path(path).exists():
        raise click.BadParameter(f"File not found: {path}")

    from src.core.export import import_from_json, import_subgraph

    json_str = Path(path).read_text(encoding="utf-8")
    export = import_from_json(json_str)
    stats = import_subgraph(state.graph, export, conflict_strategy=strategy)

    console.print(format_import_result(stats))

    # Auto-save
    state.save_graph()
    console.print(f"[dim]Graph saved ({len(state.graph)} nodes).[/dim]")


# ------------------------------------------------------------------
# merge-graphs
# ------------------------------------------------------------------

@cli.command("merge-graphs")
@click.argument("graph_a_path")
@click.argument("graph_b_path")
@click.argument("output_path")
@click.option(
    "--strategy",
    type=click.Choice(["keep_higher_confidence", "keep_existing", "keep_imported"], case_sensitive=False),
    default="keep_higher_confidence",
    show_default=True,
    help="Conflict resolution strategy for duplicate nodes.",
)
@click.option(
    "--merge-sources/--no-merge-sources",
    default=True,
    show_default=True,
    help="Merge source lists on duplicate nodes.",
)
@click.pass_context
def merge_graphs_cmd(ctx, graph_a_path, graph_b_path, output_path, strategy, merge_sources):
    """Merge two knowledge graphs into a new graph file.

    GRAPH_A_PATH is the base graph. GRAPH_B_PATH is merged into it.
    The result is saved to OUTPUT_PATH.
    """
    for p, label in [(graph_a_path, "Graph A"), (graph_b_path, "Graph B")]:
        if not Path(p).exists():
            raise click.BadParameter(f"{label} not found: {p}")

    from src.core.graph import KnowledgeGraph
    from src.core.export import merge_graphs

    console.print(f"[dim]Loading graph A: {graph_a_path}[/dim]")
    graph_a = KnowledgeGraph.load(graph_a_path)
    console.print(f"[dim]  {len(graph_a)} nodes[/dim]")

    console.print(f"[dim]Loading graph B: {graph_b_path}[/dim]")
    graph_b = KnowledgeGraph.load(graph_b_path)
    console.print(f"[dim]  {len(graph_b)} nodes[/dim]")

    console.print(f"[dim]Merging (strategy={strategy}, merge_sources={merge_sources})...[/dim]")
    merged, stats = merge_graphs(graph_a, graph_b, conflict_strategy=strategy, merge_sources=merge_sources)

    console.print(format_merge_result(stats))

    merged.save(output_path)
    console.print(f"[green]Merged graph saved to {output_path} ({len(merged)} nodes).[/green]")
