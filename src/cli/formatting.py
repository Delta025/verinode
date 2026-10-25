"""
Rich-based output formatters for the Engram CLI.

Provides coloured tables, panels, and tree views for
knowledge graph data: nodes, statistics, anomalies,
ingestion results, query answers, and reasoning chains.
"""

from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich import box

# ---------------------------------------------------------------------------
# Console singleton
# ---------------------------------------------------------------------------

console = Console()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def confidence_color(value: float) -> str:
    """Return a rich colour tag string for a confidence value."""
    if value >= 0.8:
        return "green"
    elif value >= 0.5:
        return "yellow"
    else:
        return "red"


def confidence_text(value: float) -> Text:
    """Return a coloured Text object for a confidence value."""
    color = confidence_color(value)
    return Text(f"{value:.2f}", style=color)


def status_style(status_value: str) -> str:
    """Return a rich style string for a node status value."""
    mapping = {
        "established": "bold green",
        "well_supported": "green",
        "provisional": "yellow",
        "contested": "red",
        "superseded": "dim",
        "falsified": "strike red",
    }
    return mapping.get(status_value, "")


def action_style(action: str) -> str:
    """Return a rich style string for an ingestion verdict action."""
    mapping = {
        "added": "green",
        "duplicate": "cyan",
        "flagged": "yellow",
        "rejected": "red",
    }
    return mapping.get(action, "")


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------

def short_id(node_id: str) -> str:
    """Return the first 8 characters of a UUID."""
    return node_id[:8]


def resolve_node_id(graph, prefix: str) -> str:
    """
    Resolve a (possibly short) node ID prefix to a full ID.

    Raises ``click.BadParameter`` if no match or ambiguous.
    """
    import click

    # Exact match first
    if prefix in graph.nodes:
        return prefix

    # Prefix match
    matches = [nid for nid in graph.nodes if nid.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise click.BadParameter(f"No node found matching '{prefix}'")
    else:
        raise click.BadParameter(
            f"Ambiguous prefix '{prefix}' matches {len(matches)} nodes: "
            + ", ".join(short_id(m) for m in matches[:5])
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_node_table(
    nodes: list,
    title: str = "Nodes",
    show_domain: bool = True,
) -> Table:
    """Format a list of KnowledgeNode objects as a Rich Table."""
    table = Table(title=title, box=box.SIMPLE, show_lines=False, pad_edge=False)
    table.add_column("ID", style="dim", no_wrap=True, min_width=8)
    table.add_column("Text", no_wrap=True, ratio=1)
    table.add_column("Type", no_wrap=True, min_width=6)
    table.add_column("Conf", justify="right", no_wrap=True, min_width=4)
    table.add_column("Status", no_wrap=True, min_width=10)
    if show_domain:
        table.add_column("Domain", no_wrap=True, min_width=6)

    for node in nodes:
        conf = Text(f"{node.confidence:.2f}", style=confidence_color(node.confidence))
        status = Text(node.status.value, style=status_style(node.status.value))
        text = node.text[:50] + ("..." if len(node.text) > 50 else "")
        row = [
            short_id(node.id),
            text,
            node.node_type.value,
            conf,
            status,
        ]
        if show_domain:
            row.append(node.domain)
        table.add_row(*row)

    return table


def format_stats(stats: Dict[str, Any]) -> Panel:
    """Format graph.get_statistics() output as a Rich Panel."""
    # Summary
    lines = []
    lines.append(f"[bold]Total nodes:[/bold] {stats['total_nodes']}")
    avg = stats.get("avg_confidence", 0)
    lines.append(
        f"[bold]Avg confidence:[/bold] [{confidence_color(avg)}]{avg:.3f}[/]"
    )
    conflicts = stats.get("nodes_with_conflicts", 0)
    conflict_style = "red" if conflicts > 0 else "green"
    lines.append(
        f"[bold]Nodes with conflicts:[/bold] [{conflict_style}]{conflicts}[/]"
    )

    # Domain breakdown
    domains = stats.get("nodes_by_domain", {})
    if domains:
        lines.append("")
        lines.append("[bold]By domain:[/bold]")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            lines.append(f"  {domain}: {count}")

    # Type breakdown
    types = stats.get("nodes_by_type", {})
    if types:
        non_zero = {k: v for k, v in types.items() if v > 0}
        if non_zero:
            lines.append("")
            lines.append("[bold]By type:[/bold]")
            for ntype, count in sorted(non_zero.items(), key=lambda x: -x[1]):
                lines.append(f"  {ntype}: {count}")

    # Status breakdown
    statuses = stats.get("nodes_by_status", {})
    if statuses:
        non_zero = {k: v for k, v in statuses.items() if v > 0}
        if non_zero:
            lines.append("")
            lines.append("[bold]By status:[/bold]")
            for st, count in sorted(non_zero.items(), key=lambda x: -x[1]):
                style = status_style(st)
                lines.append(f"  [{style}]{st}[/]: {count}")

    return Panel("\n".join(lines), title="Graph Statistics", box=box.ROUNDED)


def format_anomalies(anomalies: List[Dict[str, Any]]) -> Table:
    """Format detect_anomalies() output as a Rich Table."""
    table = Table(title="Anomalies", box=box.SIMPLE, pad_edge=False)
    table.add_column("Node", no_wrap=True, ratio=1)
    table.add_column("Type", no_wrap=True)
    table.add_column("Expected", justify="right", no_wrap=True)
    table.add_column("Actual", justify="right", no_wrap=True)
    table.add_column("Dev", justify="right", no_wrap=True)
    table.add_column("Sev", justify="right", no_wrap=True)

    for a in anomalies:
        atype_style = "yellow" if a["anomaly_type"] == "overconfident" else "cyan"
        expected = a["expected_confidence"]
        actual = a["actual_confidence"]
        node_text = a["node_text"][:40] + ("..." if len(a["node_text"]) > 40 else "")
        table.add_row(
            node_text,
            Text(a["anomaly_type"], style=atype_style),
            Text(f"{expected:.3f}", style=confidence_color(expected)),
            Text(f"{actual:.3f}", style=confidence_color(actual)),
            f"{a['deviation']:+.3f}",
            Text(f"{a['severity']:.3f}", style="bold red" if a["severity"] > 0.3 else "yellow"),
        )

    return table


def format_ingestion_result(result) -> Panel:
    """Format an IngestionResult as a Rich Panel with verdict table."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("Action", width=10)
    table.add_column("Fact", max_width=55, overflow="ellipsis")
    table.add_column("Reason", max_width=40, overflow="ellipsis")

    for v in result.verdicts:
        act = Text(v.action.upper(), style=action_style(v.action))
        reason = v.reason or ""
        if v.action == "duplicate" and v.confidence_before is not None:
            reason = f"Boosted: {v.confidence_before:.3f} -> {v.confidence_after:.3f}"
        table.add_row(act, v.fact.text[:55], reason[:40])

    content = table
    summary = result.summary
    return Panel(content, title=f"Ingestion Result", subtitle=summary, box=box.ROUNDED)


def format_query_result(result, show_chains: bool = False) -> Panel:
    """Format a QueryResult as a Rich Panel."""
    lines = []
    conf_col = confidence_color(result.confidence)
    lines.append(
        f"[bold]Answer:[/bold] {result.answer_text}"
    )
    lines.append(
        f"[bold]Confidence:[/bold] [{conf_col}]{result.confidence:.3f}[/]"
    )

    if result.answer_node_id:
        lines.append(f"[dim]Node: {short_id(result.answer_node_id)}[/dim]")

    if show_chains and result.supporting_chains:
        lines.append("")
        lines.append(f"[bold]Supporting chains ({len(result.supporting_chains)}):[/bold]")
        for i, chain in enumerate(result.supporting_chains[:5]):
            chain_conf = confidence_color(chain.conclusion_confidence)
            lines.append(
                f"  Chain {i+1} "
                f"[{chain_conf}](conf={chain.conclusion_confidence:.3f})[/]:"
            )
            for step in chain.steps:
                marker = ">>" if step.step_type == "conclusion" else "  "
                s_col = confidence_color(step.confidence)
                lines.append(
                    f"    {marker} {step.node_text[:50]} "
                    f"[{s_col}][{step.confidence:.2f}][/]"
                )

    return Panel(
        "\n".join(lines),
        title=f"Query: {result.query[:60]}",
        box=box.ROUNDED,
    )


def format_hypothesis_result(result, show_evidence: bool = False) -> Panel:
    """Format a HypothesisResult as a Rich Panel."""
    verdict_style = "bold green" if result.supported else "bold red"
    verdict_text = "SUPPORTED" if result.supported else "NOT SUPPORTED"
    if result.confidence < 0.1 and not result.supported:
        verdict_text = "INSUFFICIENT EVIDENCE"
        verdict_style = "dim"

    lines = []
    lines.append(f"[bold]Verdict:[/bold] [{verdict_style}]{verdict_text}[/]")
    conf_col = confidence_color(result.confidence)
    lines.append(f"[bold]Confidence:[/bold] [{conf_col}]{result.confidence:.3f}[/]")

    if result.explanation:
        lines.append(f"[bold]Explanation:[/bold] {result.explanation}")

    sup_count = len(result.supporting_evidence)
    con_count = len(result.contradicting_evidence)
    lines.append(
        f"[bold]Evidence:[/bold] {sup_count} supporting, {con_count} contradicting"
    )

    if show_evidence:
        if result.supporting_evidence:
            lines.append("")
            lines.append("[green]Supporting evidence:[/green]")
            for chain in result.supporting_evidence[:3]:
                for step in chain.steps:
                    s_col = confidence_color(step.confidence)
                    lines.append(
                        f"  + {step.node_text[:55]} [{s_col}][{step.confidence:.2f}][/]"
                    )
        if result.contradicting_evidence:
            lines.append("")
            lines.append("[red]Contradicting evidence:[/red]")
            for chain in result.contradicting_evidence[:3]:
                for step in chain.steps:
                    s_col = confidence_color(step.confidence)
                    lines.append(
                        f"  - {step.node_text[:55]} [{s_col}][{step.confidence:.2f}][/]"
                    )

    return Panel(
        "\n".join(lines),
        title=f"Hypothesis: {result.hypothesis[:60]}",
        box=box.ROUNDED,
    )


def format_node_detail(node, graph) -> Panel:
    """Format a single KnowledgeNode in full detail."""
    lines = []
    lines.append(f"[bold]ID:[/bold] {node.id}")
    lines.append(f"[bold]Text:[/bold] {node.text}")
    lines.append(f"[bold]Type:[/bold] {node.node_type.value}")

    conf_col = confidence_color(node.confidence)
    lines.append(f"[bold]Confidence:[/bold] [{conf_col}]{node.confidence:.3f}[/]")
    lines.append(
        f"[bold]Bounds:[/bold] [{node.confidence_bounds[0]:.3f}, {node.confidence_bounds[1]:.3f}]"
    )

    st_style = status_style(node.status.value)
    lines.append(f"[bold]Status:[/bold] [{st_style}]{node.status.value}[/]")
    lines.append(f"[bold]Domain:[/bold] {node.domain}")

    # Sources
    if node.sources:
        lines.append(f"\n[bold]Sources ({len(node.sources)}):[/bold]")
        for s in node.sources:
            lines.append(
                f"  - {s} [dim](quality: {s.quality_score:.2f})[/dim]"
            )

    # Derivation
    lines.append(f"\n[bold]Derivation:[/bold] {node.derivation.method}")
    if node.derivation.notes:
        lines.append(f"  Notes: {node.derivation.notes}")
    if node.derivation.inference_depth > 0:
        lines.append(f"  Depth: {node.derivation.inference_depth}")

    # Predecessors
    if node.predecessor_nodes:
        lines.append(f"\n[bold]Depends on ({len(node.predecessor_nodes)}):[/bold]")
        for pid in node.predecessor_nodes:
            pred = graph.get_node(pid)
            if pred:
                p_col = confidence_color(pred.confidence)
                lines.append(
                    f"  - {short_id(pid)} {pred.text[:45]} [{p_col}][{pred.confidence:.2f}][/]"
                )
            else:
                lines.append(f"  - {short_id(pid)} [dim](not found)[/dim]")

    # Successors
    if node.successor_nodes:
        lines.append(f"\n[bold]Depended on by ({len(node.successor_nodes)}):[/bold]")
        for sid in node.successor_nodes:
            succ = graph.get_node(sid)
            if succ:
                s_col = confidence_color(succ.confidence)
                lines.append(
                    f"  - {short_id(sid)} {succ.text[:45]} [{s_col}][{succ.confidence:.2f}][/]"
                )
            else:
                lines.append(f"  - {short_id(sid)} [dim](not found)[/dim]")

    # Conflicts
    if node.conflicts_with:
        lines.append(f"\n[bold red]Conflicts ({len(node.conflicts_with)}):[/bold red]")
        for cid in node.conflicts_with:
            conflict = graph.get_node(cid)
            if conflict:
                lines.append(f"  - {short_id(cid)} {conflict.text[:50]}")
            else:
                lines.append(f"  - {short_id(cid)} [dim](not found)[/dim]")

    # Timestamps
    lines.append(f"\n[bold]Created:[/bold] {node.created_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"[bold]Validated:[/bold] {node.last_validated.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"[bold]Accesses:[/bold] {node.access_count}")

    return Panel("\n".join(lines), title="Node Detail", box=box.ROUNDED)


def format_verification_result(result, show_evidence: bool = False) -> Panel:
    """Format a VerificationResult as a Rich Panel."""
    # Verdict styling
    verdict_styles = {
        "supported": ("bold green", "SUPPORTED"),
        "contradicted": ("bold red", "CONTRADICTED"),
        "uncertain": ("bold yellow", "UNCERTAIN"),
        "insufficient_evidence": ("dim", "INSUFFICIENT EVIDENCE"),
    }
    style, label = verdict_styles.get(
        result.verdict.value, ("dim", result.verdict.value.upper())
    )

    lines = []
    lines.append(f"[bold]Verdict:[/bold] [{style}]{label}[/]")

    conf_col = confidence_color(result.confidence)
    lines.append(f"[bold]Confidence:[/bold] [{conf_col}]{result.confidence:.3f}[/]")

    if result.explanation:
        lines.append(f"[bold]Explanation:[/bold] {result.explanation}")

    sup_count = len(result.supporting_evidence)
    con_count = len(result.contradicting_evidence)
    lines.append(
        f"[bold]Evidence:[/bold] {sup_count} supporting, {con_count} contradicting"
    )
    lines.append(f"[bold]Coverage:[/bold] {result.coverage:.0%}")
    lines.append(f"[bold]Candidates checked:[/bold] {result.num_candidates_checked}")

    if show_evidence:
        if result.supporting_evidence:
            lines.append("")
            lines.append("[green]Supporting evidence:[/green]")
            for ev in result.supporting_evidence[:5]:
                ev_col = confidence_color(ev.confidence)
                lines.append(
                    f"  + {ev.node_text[:55]} [{ev_col}][{ev.confidence:.2f}][/]"
                )
                if ev.explanation:
                    lines.append(f"    [dim]{ev.explanation[:60]}[/dim]")
                if ev.chains:
                    lines.append(f"    [dim]{len(ev.chains)} evidence chain(s)[/dim]")

        if result.contradicting_evidence:
            lines.append("")
            lines.append("[red]Contradicting evidence:[/red]")
            for ev in result.contradicting_evidence[:5]:
                ev_col = confidence_color(ev.confidence)
                lines.append(
                    f"  - {ev.node_text[:55]} [{ev_col}][{ev.confidence:.2f}][/]"
                )
                if ev.explanation:
                    lines.append(f"    [dim]{ev.explanation[:60]}[/dim]")
                if ev.chains:
                    lines.append(f"    [dim]{len(ev.chains)} evidence chain(s)[/dim]")

    return Panel(
        "\n".join(lines),
        title=f"Verify: {result.claim[:60]}",
        box=box.ROUNDED,
    )


def format_retraction_result(sim) -> Panel:
    """Format a RetractionSimulation as a Rich Panel."""
    lines = []
    lines.append(f"[bold]Retracted node:[/bold] {short_id(sim.retracted_node_id)}")
    lines.append(f"[bold]Fact:[/bold] {sim.retracted_node_text}")

    conf_col = confidence_color(sim.original_confidence)
    lines.append(
        f"[bold]Original confidence:[/bold] [{conf_col}]{sim.original_confidence:.3f}[/]"
    )

    lines.append(f"\n[bold]Impact summary:[/bold]")
    lines.append(f"  Affected nodes: {sim.total_affected}")
    lines.append(f"  Max confidence drop: {sim.max_confidence_drop:+.3f}")
    lines.append(f"  Avg confidence drop: {sim.avg_confidence_drop:+.3f}")
    lines.append(f"  Orphaned nodes: {sim.orphaned_nodes}")

    if sim.affected_nodes:
        lines.append(f"\n[bold]Affected nodes (top 10):[/bold]")

        table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        table.add_column("Node", no_wrap=True, ratio=1)
        table.add_column("Before", justify="right", no_wrap=True)
        table.add_column("After", justify="right", no_wrap=True)
        table.add_column("Change", justify="right", no_wrap=True)
        table.add_column("Hops", justify="right", no_wrap=True)

        for impact in sorted(sim.affected_nodes, key=lambda x: x.change)[:10]:
            before_col = confidence_color(impact.original_confidence)
            after_col = confidence_color(impact.new_confidence)
            change_style = "red" if impact.change < 0 else "green"
            table.add_row(
                impact.node_text[:40],
                Text(f"{impact.original_confidence:.3f}", style=before_col),
                Text(f"{impact.new_confidence:.3f}", style=after_col),
                Text(f"{impact.change:+.3f}", style=change_style),
                str(impact.hops_from_retracted),
            )

        # For panels with tables we need to render differently
        # Return a panel with the summary text; the table will be printed separately
        summary_panel = Panel(
            "\n".join(lines),
            title="Retraction Simulation (What-If)",
            box=box.ROUNDED,
        )
        return summary_panel, table

    return Panel(
        "\n".join(lines),
        title="Retraction Simulation (What-If)",
        box=box.ROUNDED,
    ), None


def format_memory_result(result) -> Panel:
    """Format a MemoryQueryResult as a Rich Panel."""
    lines = []
    lines.append(f"[bold]Summary:[/bold] {result.summary}")

    conf_col = confidence_color(result.overall_confidence)
    lines.append(f"[bold]Confidence:[/bold] [{conf_col}]{result.overall_confidence:.3f}[/]")
    lines.append(f"[bold]Coverage:[/bold] {result.coverage}")
    lines.append(f"[bold]Graph size:[/bold] {result.total_graph_nodes} nodes")

    if result.facts:
        lines.append(f"\n[bold]Facts ({len(result.facts)}):[/bold]")
        for i, fact in enumerate(result.facts[:10]):
            f_col = confidence_color(fact.confidence)
            lines.append(
                f"  {i+1}. {fact.text[:55]} [{f_col}][{fact.confidence:.2f}][/] ({fact.domain})"
            )
            if fact.sources:
                src_titles = [s.get("title", "?") for s in fact.sources[:2]]
                lines.append(f"     [dim]Sources: {', '.join(src_titles)}[/dim]")

    return Panel(
        "\n".join(lines),
        title=f"Memory Query: {result.query[:60]}",
        box=box.ROUNDED,
    )


def format_export_result(export) -> Panel:
    """Format a SubGraphExport summary as a Rich Panel."""
    lines = []
    lines.append(f"[bold]Domain:[/bold] {export.domain}")
    lines.append(f"[bold]Nodes:[/bold] {export.node_count}")
    lines.append(f"[bold]Edges:[/bold] {export.edge_count}")
    lines.append(f"[bold]Confidence threshold:[/bold] {export.confidence_threshold:.2f}")
    lines.append(f"[bold]Exported at:[/bold] {export.exported_at[:19]}")

    return Panel(
        "\n".join(lines),
        title="Sub-graph Export",
        box=box.ROUNDED,
    )


def format_import_result(stats: Dict[str, Any]) -> Panel:
    """Format import_subgraph() statistics as a Rich Panel."""
    lines = []
    lines.append(f"[bold green]Added:[/bold green] {stats.get('added', 0)}")
    lines.append(f"[bold cyan]Merged:[/bold cyan] {stats.get('merged', 0)}")
    lines.append(f"[bold dim]Skipped:[/bold dim] {stats.get('skipped', 0)}")
    lines.append(f"[bold yellow]Conflicts:[/bold yellow] {stats.get('conflicts', 0)}")

    return Panel(
        "\n".join(lines),
        title="Sub-graph Import",
        box=box.ROUNDED,
    )


def format_merge_result(stats) -> Panel:
    """Format merge_graphs() statistics as a Rich Panel."""
    d = stats.to_dict() if hasattr(stats, 'to_dict') else stats
    lines = []
    lines.append(f"[bold]Nodes from graph A:[/bold] {d.get('nodes_from_a', 0)}")
    lines.append(f"[bold green]Added from B:[/bold green] {d.get('nodes_from_b_added', 0)}")
    lines.append(f"[bold cyan]Merged from B:[/bold cyan] {d.get('nodes_from_b_merged', 0)}")
    lines.append(f"[bold dim]Skipped from B:[/bold dim] {d.get('nodes_from_b_skipped', 0)}")
    lines.append(f"[bold green]Edges added:[/bold green] {d.get('edges_added', 0)}")
    lines.append(f"[bold dim]Edges skipped:[/bold dim] {d.get('edges_skipped', 0)}")
    lines.append(f"[bold yellow]Conflicts added:[/bold yellow] {d.get('conflicts_added', 0)}")
    lines.append(f"[bold cyan]Sources merged:[/bold cyan] {d.get('sources_merged', 0)}")

    return Panel(
        "\n".join(lines),
        title="Graph Merge",
        box=box.ROUNDED,
    )


def format_confidence_trace(trace: Dict[str, Any]) -> Panel:
    """Format trace_confidence_source() output."""
    if not trace:
        return Panel("[dim]No trace available.[/dim]", title="Confidence Trace")

    lines = []
    conf = trace.get("node_confidence", 0)
    conf_col = confidence_color(conf)
    lines.append(f"[bold]Node:[/bold] {short_id(trace['node_id'])}")
    lines.append(f"[bold]Confidence:[/bold] [{conf_col}]{conf:.3f}[/]")
    lines.append(f"[bold]Inference depth:[/bold] {trace.get('inference_depth', 0)}")
    lines.append(f"[bold]Total ancestors:[/bold] {trace.get('num_ancestors', 0)}")

    source_nodes = trace.get("source_nodes", [])
    if source_nodes:
        min_c = trace.get("min_source_confidence")
        max_c = trace.get("max_source_confidence")
        lines.append(
            f"\n[bold]Root sources ({len(source_nodes)}):[/bold]"
        )
        if min_c is not None:
            lines.append(
                f"  Confidence range: "
                f"[{confidence_color(min_c)}]{min_c:.3f}[/] — "
                f"[{confidence_color(max_c)}]{max_c:.3f}[/]"
            )
        for sn in source_nodes:
            s_col = confidence_color(sn["confidence"])
            lines.append(
                f"  - {short_id(sn['id'])} {sn['text'][:45]} "
                f"[{s_col}][{sn['confidence']:.2f}][/]"
            )
            for src_str in sn.get("sources", []):
                lines.append(f"    [dim]{src_str}[/dim]")
    else:
        lines.append("\n[dim]No root source nodes found (this may be a root node itself).[/dim]")

    return Panel("\n".join(lines), title="Confidence Trace", box=box.ROUNDED)


# ---------------------------------------------------------------------------
# Neuroplasticity maintenance
# ---------------------------------------------------------------------------

def severity_style(severity: str) -> str:
    """Return a rich style string for a doubt flag severity level."""
    return {"high": "bold red", "medium": "yellow", "low": "dim"}.get(severity, "")


def format_maintenance_report(report) -> Panel:
    """Format a MaintenanceReport as a Rich Panel with summary and detail lines."""
    lines = []
    lines.append(f"[bold]Timestamp:[/bold] {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"[bold]Nodes examined:[/bold] {report.nodes_examined}")
    lines.append("")
    lines.append(
        f"[bold]Decay:[/bold] {report.nodes_decayed} node(s), "
        f"total confidence lost: {report.total_decay_amount:.4f}"
    )
    lines.append(
        f"[bold]Strengthened:[/bold] {report.nodes_strengthened} node(s), "
        f"total boost: {report.total_strength_boost:.4f}"
    )
    lines.append(f"[bold]Flagged:[/bold] {report.nodes_flagged} node(s) need review")

    if report.decay_details:
        lines.append("")
        lines.append("[bold]Decayed nodes:[/bold]")
        for d in report.decay_details[:10]:
            old_col = confidence_color(d.original_confidence)
            new_col = confidence_color(d.new_confidence)
            lines.append(
                f"  {short_id(d.node_id)} {d.node_text[:40]:<40s} "
                f"[{old_col}]{d.original_confidence:.3f}[/] -> "
                f"[{new_col}]{d.new_confidence:.3f}[/] "
                f"({d.days_since_validated:.0f}d stale)"
            )
        if len(report.decay_details) > 10:
            lines.append(f"  [dim]... and {len(report.decay_details) - 10} more[/dim]")

    if report.strength_details:
        lines.append("")
        lines.append("[bold]Strengthened nodes:[/bold]")
        for s in report.strength_details[:10]:
            old_col = confidence_color(s.original_confidence)
            new_col = confidence_color(s.new_confidence)
            lines.append(
                f"  {short_id(s.node_id)} {s.node_text[:40]:<40s} "
                f"[{old_col}]{s.original_confidence:.3f}[/] -> "
                f"[{new_col}]{s.new_confidence:.3f}[/] "
                f"(accesses: {s.access_count})"
            )
        if len(report.strength_details) > 10:
            lines.append(f"  [dim]... and {len(report.strength_details) - 10} more[/dim]")

    if report.flags:
        lines.append("")
        lines.append("[bold]Doubt flags:[/bold]")
        for f in report.flags[:10]:
            sev = severity_style(f.severity)
            f_col = confidence_color(f.confidence)
            lines.append(
                f"  [{sev}]{f.severity.upper():<6s}[/] "
                f"{short_id(f.node_id)} {f.node_text[:35]:<35s} "
                f"[{f_col}]{f.confidence:.2f}[/] - {f.reason}"
            )
        if len(report.flags) > 10:
            lines.append(f"  [dim]... and {len(report.flags) - 10} more[/dim]")

    return Panel(
        "\n".join(lines),
        title="Neuroplasticity Maintenance",
        box=box.ROUNDED,
    )


def format_doubt_flags(flags) -> Table:
    """Format a list of DoubtFlag objects as a Rich Table."""
    table = Table(title="Doubtful Nodes", box=box.SIMPLE, pad_edge=False)
    table.add_column("Severity", no_wrap=True, min_width=8)
    table.add_column("ID", style="dim", no_wrap=True, min_width=8)
    table.add_column("Text", no_wrap=True, ratio=1)
    table.add_column("Conf", justify="right", no_wrap=True, min_width=5)
    table.add_column("Reason", no_wrap=True)

    for f in flags:
        sev = severity_style(f.severity)
        conf = confidence_text(f.confidence)
        text = f.node_text[:45] + ("..." if len(f.node_text) > 45 else "")
        table.add_row(
            Text(f.severity.upper(), style=sev),
            short_id(f.node_id),
            text,
            conf,
            f.reason[:50],
        )

    return table


def format_strengthen_result(result) -> Panel:
    """Format a StrengthResult as a Rich Panel."""
    old_col = confidence_color(result.original_confidence)
    new_col = confidence_color(result.new_confidence)
    lines = []
    lines.append(f"[bold]Node:[/bold] {short_id(result.node_id)}")
    lines.append(f"[bold]Text:[/bold] {result.node_text[:60]}")
    lines.append(
        f"[bold]Confidence:[/bold] [{old_col}]{result.original_confidence:.3f}[/] "
        f"-> [{new_col}]{result.new_confidence:.3f}[/]"
    )
    lines.append(f"[bold]Boost applied:[/bold] +{result.boost_applied:.4f}")
    lines.append(f"[bold]Access count:[/bold] {result.access_count}")

    return Panel("\n".join(lines), title="Node Strengthened", box=box.ROUNDED)
