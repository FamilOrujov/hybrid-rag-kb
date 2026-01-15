"""Panel components for displaying structured data."""

from __future__ import annotations

import re
from typing import Any
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.markdown import Markdown
from rich.syntax import Syntax
from cli.ui.console import console


def create_stats_panel(data: dict[str, Any]) -> Panel:
    """Create a beautiful stats panel."""
    
    # Create main table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Category", style="primary")
    table.add_column("Details", style="text")
    
    # SQLite stats
    sqlite = data.get("sqlite", {})
    sqlite_text = Text()
    sqlite_text.append("Documents: ", style="muted")
    sqlite_text.append(str(sqlite.get("documents", "N/A")), style="number")
    sqlite_text.append("  │  ", style="muted")
    sqlite_text.append("Chunks: ", style="muted")
    sqlite_text.append(str(sqlite.get("chunks", "N/A")), style="number")
    sqlite_text.append("  │  ", style="muted")
    sqlite_text.append("FTS5: ", style="muted")
    sqlite_text.append(str(sqlite.get("chunks_fts", "N/A")), style="number")
    table.add_row("SQLite", sqlite_text)
    
    # FAISS stats
    faiss = data.get("faiss", {})
    faiss_text = Text()
    faiss_text.append("Vectors: ", style="muted")
    faiss_text.append(str(faiss.get("ntotal", "N/A")), style="number")
    faiss_text.append("  │  ", style="muted")
    faiss_text.append("Dimension: ", style="muted")
    faiss_text.append(str(faiss.get("d", "N/A")), style="number")
    faiss_text.append("  │  ", style="muted")
    faiss_text.append("Type: ", style="muted")
    faiss_text.append(str(faiss.get("index_type", "N/A")), style="tertiary")
    table.add_row("FAISS", faiss_text)
    
    # GPU stats
    gpu = data.get("gpu", {})
    gpu_text = Text()
    gpu_enabled = gpu.get("configured_use_gpu", False)
    gpu_text.append("GPU: ", style="muted")
    gpu_text.append("✓ Enabled" if gpu_enabled else "✗ Disabled", 
                   style="success" if gpu_enabled else "warning")
    gpu_text.append("  │  ", style="muted")
    gpu_text.append("GPUs: ", style="muted")
    gpu_text.append(str(gpu.get("gpu_count_visible_to_faiss", 0)), style="number")
    table.add_row("GPU", gpu_text)
    
    # Repro/Config
    repro = data.get("repro", {})
    repro_text = Text()
    repro_text.append("Chat: ", style="muted")
    repro_text.append(str(repro.get("chat_model", "N/A")), style="secondary")
    repro_text.append("  │  ", style="muted")
    repro_text.append("Embed: ", style="muted")
    repro_text.append(str(repro.get("embed_model", "N/A")), style="secondary")
    table.add_row("Models", repro_text)
    
    # Chunking
    chunk_text = Text()
    chunk_text.append("Size: ", style="muted")
    chunk_text.append(str(repro.get("chunk_size", "N/A")), style="number")
    chunk_text.append("  │  ", style="muted")
    chunk_text.append("Overlap: ", style="muted")
    chunk_text.append(str(repro.get("chunk_overlap", "N/A")), style="number")
    table.add_row("Chunking", chunk_text)
    
    return Panel(
        table,
        title="[primary]System Statistics[/primary]",
        border_style="primary",
        padding=(1, 2),
    )


def format_answer_with_citations(answer: str) -> Text:
    """Format answer text with highlighted citations."""
    text = Text()
    
    # Pattern to match citations like [cid:123] or [Source: ... cid:123]
    pattern = r'\[(?:Source:[^\]]*)?cid:(\d+)\]|\[cid:(\d+)\]'
    
    last_end = 0
    for match in re.finditer(pattern, answer):
        # Add text before the citation
        text.append(answer[last_end:match.start()], style="text")
        # Add the citation with highlighting
        text.append(match.group(0), style="citation")
        last_end = match.end()
    
    # Add remaining text
    text.append(answer[last_end:], style="text")
    
    return text


def create_query_result_panel(answer: str, query: str) -> Panel:
    """Create a panel for query results."""
    
    # Format answer with highlighted citations
    formatted_answer = format_answer_with_citations(answer)
    
    return Panel(
        formatted_answer,
        title=f"[primary]💬 Response[/primary]",
        subtitle=f"[muted]Query: {query[:50]}{'...' if len(query) > 50 else ''}[/muted]",
        border_style="primary",
        padding=(1, 2),
    )


def create_sources_panel(sources: list[dict[str, Any]], max_sources: int = 5) -> Panel:
    """Create a panel showing source citations."""
    
    table = Table(show_header=True, header_style="primary", box=None)
    table.add_column("CID", style="number", width=6)
    table.add_column("Document", style="path", width=25)
    table.add_column("Preview", style="muted", overflow="ellipsis")
    
    for source in sources[:max_sources]:
        chunk_id = str(source.get("chunk_id", "?"))
        filename = source.get("filename", "Unknown")[:25]
        text = source.get("text", "")[:80] + "..." if len(source.get("text", "")) > 80 else source.get("text", "")
        table.add_row(chunk_id, filename, text)
    
    if len(sources) > max_sources:
        table.add_row("...", f"+{len(sources) - max_sources} more", "")
    
    return Panel(
        table,
        title=f"[tertiary]📚 Sources ({len(sources)} chunks)[/tertiary]",
        border_style="tertiary",
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
        title=f"[muted]🔧 {title}[/muted]",
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
        bm25_table = Table(title="[secondary]BM25 Results[/secondary]", show_header=True, header_style="secondary", box=None)
        bm25_table.add_column("CID", style="number", width=6)
        bm25_table.add_column("Score", style="warning", width=10)
        bm25_table.add_column("File", style="path", width=20)
        bm25_table.add_column("Preview", style="muted", overflow="ellipsis")
        
        for r in bm25_results:
            bm25_table.add_row(
                str(r.get("chunk_id", "?")),
                f"{r.get('bm25_score', 0):.4f}",
                str(r.get("filename", "?"))[:20],
                str(r.get("text", ""))[:40] + "..."
            )
        elements.append(bm25_table)
        elements.append(Text())
    
    # Vector results table
    vec_results = data.get("vector", [])[:5]
    if vec_results:
        vec_table = Table(title="[tertiary]Vector Results[/tertiary]", show_header=True, header_style="tertiary", box=None)
        vec_table.add_column("CID", style="number", width=6)
        vec_table.add_column("Score", style="warning", width=10)
        vec_table.add_column("File", style="path", width=20)
        vec_table.add_column("Preview", style="muted", overflow="ellipsis")
        
        for r in vec_results:
            vec_table.add_row(
                str(r.get("chunk_id", "?")),
                f"{r.get('vec_score', 0):.4f}",
                str(r.get("filename", "?"))[:20],
                str(r.get("text", ""))[:40] + "..."
            )
        elements.append(vec_table)
    
    return Panel(
        Group(*elements),
        title="[primary]🔬 Retrieval Debug[/primary]",
        border_style="primary",
        padding=(1, 2),
    )


def create_ingest_result_panel(data: dict[str, Any]) -> Panel:
    """Create a panel for ingest results."""
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="muted", width=20)
    table.add_column("Value", style="text")
    
    received = data.get("received", [])
    table.add_row("📁 Files Received", str(len(received)))
    
    for f in received[:5]:
        table.add_row("", f"  └─ {f}")
    if len(received) > 5:
        table.add_row("", f"  └─ +{len(received) - 5} more...")
    
    table.add_row("📄 Documents Added", str(data.get("documents_added", 0)))
    table.add_row("🔢 Chunks Created", str(data.get("chunks_added", 0)))
    table.add_row("🔍 Vectors Indexed", str(data.get("vectors_added", 0)))
    
    skipped = data.get("skipped", [])
    if skipped:
        table.add_row("⏭️ Skipped (duplicates)", str(len(skipped)))
    
    return Panel(
        table,
        title="[success]✅ Ingestion Complete[/success]",
        border_style="success",
        padding=(1, 2),
    )
