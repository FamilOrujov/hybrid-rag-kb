"""Panel components for displaying structured data with modern styling."""

from __future__ import annotations

import re
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Modern box styles for panels
PANEL_BOX = box.ROUNDED
SUBTLE_BOX = box.SIMPLE
CHAT_BOX = box.ROUNDED


def create_stats_panel(data: dict[str, Any]) -> Panel:
    """Create a beautiful stats panel with modern styling."""

    # Create main table with modern styling
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Category", style="#C77DFF", width=14)
    table.add_column("Details", style="text")

    # SQLite stats
    sqlite = data.get("sqlite", {})
    sqlite_text = Text()
    sqlite_text.append("Documents: ", style="#888888")
    sqlite_text.append(str(sqlite.get("documents", "N/A")), style="#FFB347 bold")
    sqlite_text.append("  │  ", style="#555555")
    sqlite_text.append("Chunks: ", style="#888888")
    sqlite_text.append(str(sqlite.get("chunks", "N/A")), style="#FFB347 bold")
    sqlite_text.append("  │  ", style="#555555")
    sqlite_text.append("FTS5: ", style="#888888")
    sqlite_text.append(str(sqlite.get("chunks_fts", "N/A")), style="#FFB347 bold")
    table.add_row("SQLite", sqlite_text)

    # FAISS stats
    faiss = data.get("faiss", {})
    faiss_text = Text()
    faiss_text.append("Vectors: ", style="#888888")
    faiss_text.append(str(faiss.get("ntotal", "N/A")), style="#FFB347 bold")
    faiss_text.append("  │  ", style="#555555")
    faiss_text.append("Dimension: ", style="#888888")
    faiss_text.append(str(faiss.get("d", "N/A")), style="#FFB347 bold")
    faiss_text.append("  │  ", style="#555555")
    faiss_text.append("Type: ", style="#888888")
    faiss_text.append(str(faiss.get("index_type", "N/A")), style="#9D4EDD")
    table.add_row("FAISS", faiss_text)

    # GPU stats
    gpu = data.get("gpu", {})
    gpu_text = Text()
    gpu_enabled = gpu.get("configured_use_gpu", False)
    gpu_text.append("GPU: ", style="#888888")
    gpu_text.append(
        "✓ Enabled" if gpu_enabled else "✗ Disabled",
        style="#00E676 bold" if gpu_enabled else "#FFB347",
    )
    gpu_text.append("  │  ", style="#555555")
    gpu_text.append("Devices: ", style="#888888")
    gpu_text.append(str(gpu.get("gpu_count_visible_to_faiss", 0)), style="#FFB347 bold")
    table.add_row("GPU", gpu_text)

    # Models
    repro = data.get("repro", {})
    repro_text = Text()
    repro_text.append("Chat: ", style="#888888")
    repro_text.append(str(repro.get("chat_model", "N/A")), style="#FF8C42 bold")
    repro_text.append("  │  ", style="#555555")
    repro_text.append("Embed: ", style="#888888")
    repro_text.append(str(repro.get("embed_model", "N/A")), style="#FF8C42 bold")
    table.add_row("Models", repro_text)

    # Chunking
    chunk_text = Text()
    chunk_text.append("Size: ", style="#888888")
    chunk_text.append(str(repro.get("chunk_size", "N/A")), style="#FFB347 bold")
    chunk_text.append("  │  ", style="#555555")
    chunk_text.append("Overlap: ", style="#888888")
    chunk_text.append(str(repro.get("chunk_overlap", "N/A")), style="#FFB347 bold")
    table.add_row("Chunking", chunk_text)

    return Panel(
        table,
        title="[#C77DFF bold]System Statistics[/#C77DFF bold]",
        border_style="#5D4E6D",
        box=PANEL_BOX,
        padding=(1, 2),
    )


def format_answer_with_citations(answer: str) -> Text:
    """Format answer text with highlighted citations.

    Matches citation formats:
    - [cid:123]
    - [Source: filename | cid:123]
    - [Source: filename | cid:5, 4] (multiple cids)
    """
    text = Text()

    # Pattern to match all citation formats:
    # - [cid:123] - simple format
    # - [Source: ... | cid:...] - full format with filename
    # The pattern matches the entire bracketed citation
    pattern = r"\[Source:[^\]]+\]|\[cid:\d+(?:,\s*\d+)*\]"

    last_end = 0
    for match in re.finditer(pattern, answer):
        # Add text before the citation
        text.append(answer[last_end : match.start()], style="text")
        # Add the citation with highlighting (cyan/teal for visibility)
        text.append(match.group(0), style="citation")
        last_end = match.end()

    # Add remaining text
    text.append(answer[last_end:], style="text")

    return text


def create_user_message_bubble(query: str) -> Panel:
    """Create a user message bubble (right-aligned style)."""
    text = Text()
    text.append(query, style="#e8e8e8")

    return Panel(
        text,
        title="[#FF8C42 bold]You[/#FF8C42 bold]",
        title_align="left",
        border_style="#FF8C42",
        box=CHAT_BOX,
        padding=(0, 2),
        width=min(len(query) + 10, 100),
    )


def create_assistant_message_bubble(answer: str) -> Panel:
    """Create an assistant message bubble with citations highlighted."""
    # Format answer with highlighted citations
    formatted_answer = format_answer_with_citations(answer)

    return Panel(
        formatted_answer,
        title="[#C77DFF bold]Assistant[/#C77DFF bold]",
        title_align="left",
        border_style="#5D4E6D",
        box=CHAT_BOX,
        padding=(1, 2),
    )


def create_query_result_panel(answer: str, query: str) -> Group:
    """Create a chat-style panel for query results.

    Shows user query bubble followed by assistant response bubble.
    """
    elements = []

    # User message bubble
    user_bubble = create_user_message_bubble(query)
    elements.append(Align.right(user_bubble))
    elements.append(Text())  # Spacing

    # Assistant response bubble
    assistant_bubble = create_assistant_message_bubble(answer)
    elements.append(assistant_bubble)

    return Group(*elements)


def create_sources_panel(sources: list[dict[str, Any]], max_sources: int = 5) -> Panel:
    """Create a panel showing source citations with modern styling."""

    table = Table(show_header=True, header_style="#9D4EDD bold", box=SUBTLE_BOX, expand=True)
    table.add_column("CID", style="#FFB347", width=6, justify="right")
    table.add_column("Document", style="#FF8C42", width=25)
    table.add_column("Preview", style="#888888", overflow="ellipsis")

    for source in sources[:max_sources]:
        chunk_id = str(source.get("chunk_id", "?"))
        filename = source.get("filename", "Unknown")[:25]
        text = (
            source.get("text", "")[:80] + "..."
            if len(source.get("text", "")) > 80
            else source.get("text", "")
        )
        table.add_row(chunk_id, filename, text)

    if len(sources) > max_sources:
        table.add_row("...", f"+{len(sources) - max_sources} more", "")

    return Panel(
        table,
        title=f"[#9D4EDD bold]Sources ({len(sources)} chunks)[/#9D4EDD bold]",
        border_style="#5D4E6D",
        box=PANEL_BOX,
        padding=(0, 1),
    )


def create_debug_panel(debug_data: dict[str, Any], title: str = "Debug Info") -> Panel:
    """Create a panel for debug information."""

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="muted", width=20)
    table.add_column("Value", style="text")

    for key, value in debug_data.items():
        if isinstance(value, dict):
            value_str = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, list):
            value_str = f"[{len(value)} items]"
        else:
            value_str = str(value)
        table.add_row(key, value_str)

    return Panel(
        table,
        title=f"[muted]{title}[/muted]",
        border_style="muted",
        padding=(0, 1),
    )


def create_retrieval_debug_panel(data: dict[str, Any]) -> Panel:
    """Create a detailed panel for retrieval debugging."""

    from rich.console import Group

    elements = []

    # Summary
    debug = data.get("debug", {})
    summary = Text()
    summary.append("BM25 Hits: ", style="muted")
    summary.append(str(debug.get("bm25_hits", 0)), style="number")
    summary.append("  │  ", style="muted")
    summary.append("Vector Hits: ", style="muted")
    summary.append(str(debug.get("vec_hits", 0)), style="number")
    summary.append("  │  ", style="muted")
    summary.append("Fused: ", style="muted")
    summary.append(str(debug.get("fused_hits", 0)), style="number")
    elements.append(summary)
    elements.append(Text())

    # BM25 results table
    bm25_results = data.get("bm25", [])[:5]
    if bm25_results:
        bm25_table = Table(
            title="[secondary]BM25 Results[/secondary]",
            show_header=True,
            header_style="secondary",
            box=None,
        )
        bm25_table.add_column("CID", style="number", width=6)
        bm25_table.add_column("Score", style="warning", width=10)
        bm25_table.add_column("File", style="path", width=20)
        bm25_table.add_column("Preview", style="muted", overflow="ellipsis")

        for r in bm25_results:
            bm25_table.add_row(
                str(r.get("chunk_id", "?")),
                f"{r.get('bm25_score', 0):.4f}",
                str(r.get("filename", "?"))[:20],
                str(r.get("text", ""))[:40] + "...",
            )
        elements.append(bm25_table)
        elements.append(Text())

    # Vector results table
    vec_results = data.get("vector", [])[:5]
    if vec_results:
        vec_table = Table(
            title="[tertiary]Vector Results[/tertiary]",
            show_header=True,
            header_style="tertiary",
            box=None,
        )
        vec_table.add_column("CID", style="number", width=6)
        vec_table.add_column("Score", style="warning", width=10)
        vec_table.add_column("File", style="path", width=20)
        vec_table.add_column("Preview", style="muted", overflow="ellipsis")

        for r in vec_results:
            vec_table.add_row(
                str(r.get("chunk_id", "?")),
                f"{r.get('vec_score', 0):.4f}",
                str(r.get("filename", "?"))[:20],
                str(r.get("text", ""))[:40] + "...",
            )
        elements.append(vec_table)

    return Panel(
        Group(*elements),
        title="[primary]Retrieval Debug[/primary]",
        border_style="primary",
        padding=(1, 2),
    )


def create_ingest_result_panel(data: dict[str, Any]) -> Panel:
    """Create a panel for ingest results with modern styling."""

    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("Metric", style="#888888", width=22)
    table.add_column("Value", style="#e8e8e8")

    received = data.get("received", [])
    table.add_row("Files Received", f"[#FFB347 bold]{len(received)}[/#FFB347 bold]")

    for f in received[:5]:
        table.add_row("", f"[#555555]  └─[/#555555] [#FF8C42]{f}[/#FF8C42]")
    if len(received) > 5:
        table.add_row("", f"[#555555]  └─ +{len(received) - 5} more...[/#555555]")

    table.add_row(
        "Documents Added", f"[#00E676 bold]{data.get('documents_added', 0)}[/#00E676 bold]"
    )
    table.add_row("Chunks Created", f"[#00E676 bold]{data.get('chunks_added', 0)}[/#00E676 bold]")
    table.add_row("Vectors Indexed", f"[#00E676 bold]{data.get('vectors_added', 0)}[/#00E676 bold]")

    skipped = data.get("skipped", [])
    if skipped:
        table.add_row("Skipped (duplicates)", f"[#FFB347]{len(skipped)}[/#FFB347]")

    return Panel(
        table,
        title="[#00E676 bold]Ingestion Complete[/#00E676 bold]",
        border_style="#00E676",
        box=PANEL_BOX,
        padding=(1, 2),
    )
