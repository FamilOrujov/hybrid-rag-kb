"""Debug commands - retrieval and citation debugging."""

from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error
from cli.ui.spinners import create_spinner


class DebugCommand(BaseCommand):
    """Debug retrieval and citations."""

    name = "debug"
    description = "Debug retrieval (BM25, vector, fusion) and citations"
    usage = "/debug <retrieval|citations> [query] [options]"
    aliases = ["dbg"]

    def execute(self, args: list[str]) -> bool:
        """Execute debug command."""
        flags, remaining = self.parse_flags(args)

        if not remaining:
            self._show_debug_help()
            return True

        subcommand = remaining[0].lower()
        query_args = remaining[1:]

        if subcommand in ["retrieval", "ret", "r"]:
            return self._debug_retrieval(query_args, flags)
        elif subcommand in ["citations", "cite", "c"]:
            return self._debug_citations(query_args, flags)
        else:
            # Treat first arg as query for retrieval debug
            return self._debug_retrieval(remaining, flags)

    def _show_debug_help(self) -> None:
        """Show debug command help."""
        console.print()
        console.print(
            Panel(
                Text.from_markup(
                    "[primary]Debug Commands[/primary]\n\n"
                    "[command]/debug retrieval[/command] [muted]<query>[/muted]\n"
                    "  Deep analysis of BM25, vector, and RRF fusion\n\n"
                    "[command]/debug citations[/command] [muted]<query>[/muted]\n"
                    "  Run query and validate citation accuracy\n\n"
                    "[muted]Options:[/muted]\n"
                    "  --bm25_k N      BM25 candidates (default: 20)\n"
                    "  --vec_k N       Vector candidates (default: 20)\n"
                    "  --top_k N       Final fused results (default: 8)\n"
                    "  --bm25_mode M   Query mode: heuristic|raw (default: heuristic)\n"
                    "  --rrf_k N       RRF constant (default: 60)\n"
                    "  --w_bm25 F      BM25 weight (default: 1.0)\n"
                    "  --w_vec F       Vector weight (default: 1.0)"
                ),
                title="[primary]Debug Commands[/primary]",
                border_style="primary",
                padding=(1, 2),
            )
        )

    def _debug_retrieval(self, args: list[str], flags: dict) -> bool:
        """Debug retrieval pipeline with detailed analysis."""
        # Get query
        if args:
            query = " ".join(args)
        else:
            query = Prompt.ask(
                "[primary]>[/primary] [text]Enter query for retrieval debug[/text]",
                console=console,
            ).strip()

        if not query:
            return True

        # Parse options
        bm25_k = int(flags.get("bm25_k", self.config.bm25_k))
        vec_k = int(flags.get("vec_k", self.config.vec_k))
        top_k = int(flags.get("top_k", self.config.top_k))
        bm25_mode = flags.get("bm25_mode", "heuristic")
        rrf_k = int(flags.get("rrf_k", 60))
        w_bm25 = float(flags.get("w_bm25", 1.0))
        w_vec = float(flags.get("w_vec", 1.0))

        # Make API request with extended parameters
        with create_spinner("Running deep retrieval analysis...", style="processing"):
            try:
                import httpx

                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        f"{self.api.base_url}/debug/retrieval",
                        json={
                            "query": query,
                            "bm25_k": bm25_k,
                            "vec_k": vec_k,
                            "top_k": top_k,
                            "bm25_mode": bm25_mode,
                            "bm25_max_terms": 10,
                            "rrf_k": rrf_k,
                            "w_bm25": w_bm25,
                            "w_vec": w_vec,
                        },
                    )
                    if response.status_code != 200:
                        print_error(f"API error: {response.status_code}")
                        return False
                    data = response.json()
            except httpx.ConnectError:
                print_error("Cannot connect to server. Is it running?")
                return False
            except Exception as e:
                print_error(f"Request failed: {e}")
                return False

        console.print()

        # 1. Query Analysis Panel
        self._show_query_analysis(data.get("query_analysis", {}))

        # 2. Database Context
        self._show_db_context(data.get("db_stats", {}), data.get("debug", {}))

        # 3. BM25 Results
        self._show_bm25_results(data.get("bm25", []))

        # 4. Vector Results (with error handling)
        vec_error = data.get("vector_error")
        if vec_error:
            self._show_vector_error(vec_error, data.get("debug", {}))
        else:
            self._show_vector_results(data.get("vector", []))

        # 5. Overlap Analysis
        self._show_overlap_analysis(data.get("overlap_analysis", {}))

        # 6. RRF Fusion Results
        self._show_fused_results(data.get("fused", []), data.get("rrf_params", {}))

        # 7. Timing Analysis
        self._show_timing_analysis(data.get("timings", {}))

        return True

    def _show_query_analysis(self, analysis: dict) -> None:
        """Display query tokenization and BM25 query construction."""
        console.print(
            Panel(
                self._build_query_analysis_content(analysis),
                title="[primary]Query Analysis[/primary]",
                border_style="primary",
                padding=(1, 2),
            )
        )
        console.print()

    def _build_query_analysis_content(self, analysis: dict) -> Text:
        """Build content for query analysis panel."""
        text = Text()

        # Original query
        text.append("Original Query: ", style="muted")
        text.append(f'"{analysis.get("original_query", "")}"', style="primary.bold")
        text.append("\n")

        # Original tokens
        text.append("Original Tokens: ", style="muted")
        orig_tokens = analysis.get("original_tokens", [])
        text.append(f"{len(orig_tokens)} tokens ", style="number")
        text.append(
            f"[{', '.join(orig_tokens[:10])}{'...' if len(orig_tokens) > 10 else ''}]",
            style="tertiary",
        )
        text.append("\n\n")

        # BM25 processing
        text.append("BM25 Mode: ", style="muted")
        text.append(analysis.get("bm25_mode", "heuristic"), style="secondary")
        text.append("\n")

        text.append("BM25 Query: ", style="muted")
        bm25_query = analysis.get("bm25_query", "")
        if bm25_query:
            text.append(f'"{bm25_query}"', style="success bold")
        else:
            text.append("(empty, no matching tokens)", style="warning")
        text.append("\n")

        # Token stats
        text.append("BM25 Tokens: ", style="muted")
        bm25_tokens = analysis.get("bm25_tokens", [])
        text.append(f"{len(bm25_tokens)} tokens ", style="number")
        if bm25_tokens:
            text.append(f"[{', '.join(bm25_tokens)}]", style="success")
        text.append("\n")

        removed = analysis.get("tokens_removed", 0)
        if removed > 0:
            text.append("Tokens Removed: ", style="muted")
            text.append(f"{removed} ", style="warning")
            text.append("(stopwords, short tokens)", style="muted")

        return text

    def _show_db_context(self, db_stats: dict, debug: dict) -> None:
        """Show database and index context."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="muted", width=20)
        table.add_column("Value", style="number", width=15)
        table.add_column("Metric", style="muted", width=20)
        table.add_column("Value", style="number", width=15)

        table.add_row(
            "Total Documents",
            str(db_stats.get("total_documents", "?")),
            "Total Chunks",
            str(db_stats.get("total_chunks", "?")),
        )
        table.add_row(
            "FTS5 Entries",
            str(db_stats.get("total_fts_entries", "?")),
            "FAISS Vectors",
            str(debug.get("faiss_ntotal", "?")),
        )
        table.add_row(
            "Query Embed Dim",
            str(debug.get("query_embedding_dim", "?")),
            "FAISS Index Dim",
            str(debug.get("faiss_index_dim", "?")),
        )

        console.print(
            Panel(
                table,
                title="[tertiary]Database Context[/tertiary]",
                border_style="tertiary",
                padding=(0, 1),
            )
        )
        console.print()

    def _show_bm25_results(self, results: list) -> None:
        """Display BM25 sparse retrieval results."""
        if not results:
            console.print(
                Panel(
                    Text(
                        "No BM25 results. The query tokens may not match any indexed text.\n"
                        "Try using --bm25_mode raw to include all tokens.",
                        style="warning",
                    ),
                    title="[secondary]BM25 Sparse Retrieval (0 hits)[/secondary]",
                    border_style="secondary",
                    padding=(1, 2),
                )
            )
            console.print()
            return

        table = Table(
            show_header=True,
            header_style="secondary.bold",
            border_style="muted",
            box=box.SIMPLE,
        )
        table.add_column("Rank", style="muted", width=5, justify="right")
        table.add_column("CID", style="number", width=6, justify="right")
        table.add_column("BM25 Score", style="warning", width=12, justify="right")
        table.add_column("Document", style="path", width=22)
        table.add_column("Chunk", style="muted", width=6, justify="right")
        table.add_column("Text Preview", style="text", overflow="ellipsis")

        for r in results[:10]:
            score = r.get("bm25_score", 0)
            # BM25 scores are negative (lower is better)
            score_style = "success" if score > -5 else "warning" if score > -10 else "error"

            table.add_row(
                str(r.get("bm25_rank", "?")),
                str(r.get("chunk_id", "?")),
                f"[{score_style}]{score:.4f}[/{score_style}]",
                str(r.get("filename", "?"))[:22],
                str(r.get("metadata", {}).get("chunk_index", "?")),
                self._clean_preview(r.get("text", ""), 50),
            )

        if len(results) > 10:
            table.add_row("...", f"+{len(results) - 10}", "", "", "", "")

        console.print(
            Panel(
                table,
                title=f"[secondary]BM25 Sparse Retrieval ({len(results)} hits)[/secondary]",
                border_style="secondary",
                padding=(0, 1),
            )
        )
        console.print()

    def _show_vector_error(self, error: str, debug: dict) -> None:
        """Display vector search error (e.g., dimension mismatch)."""
        text = Text()
        text.append("Vector Search Failed\n\n", style="error bold")
        text.append(error, style="warning")
        text.append("\n\n", style="text")

        # Show dimension info
        query_dim = debug.get("query_embedding_dim", "?")
        index_dim = debug.get("faiss_index_dim", "?")

        text.append("Details:\n", style="muted")
        text.append(f"  Query embedding dimension: {query_dim}\n", style="tertiary")
        text.append(f"  FAISS index dimension: {index_dim}\n", style="secondary")
        text.append("\n", style="text")

        text.append("Solution:\n", style="primary")
        text.append("  1. Run ", style="text")
        text.append("/reset", style="command")
        text.append(" to clear the database and index\n", style="text")
        text.append("  2. Run ", style="text")
        text.append("/restart", style="command")
        text.append(" to reinitialize\n", style="text")
        text.append("  3. Run ", style="text")
        text.append("/ingest", style="command")
        text.append(" to re-embed documents with the new model\n", style="text")

        console.print(
            Panel(
                text,
                title="[error]Vector Search Error[/error]",
                border_style="error",
                padding=(1, 2),
            )
        )
        console.print()

    def _show_vector_results(self, results: list) -> None:
        """Display vector dense retrieval results."""
        if not results:
            console.print(
                Panel(
                    Text("No vector results. FAISS index may be empty.", style="warning"),
                    title="[tertiary]Vector Dense Retrieval (0 hits)[/tertiary]",
                    border_style="tertiary",
                    padding=(1, 2),
                )
            )
            console.print()
            return

        table = Table(
            show_header=True,
            header_style="tertiary.bold",
            border_style="muted",
            box=box.SIMPLE,
        )
        table.add_column("Rank", style="muted", width=5, justify="right")
        table.add_column("CID", style="number", width=6, justify="right")
        table.add_column("Similarity", style="success", width=12, justify="right")
        table.add_column("Document", style="path", width=22)
        table.add_column("Chunk", style="muted", width=6, justify="right")
        table.add_column("Text Preview", style="text", overflow="ellipsis")

        for r in results[:10]:
            score = r.get("vec_score", 0)
            # Cosine similarity: higher is better, typically 0.3-0.8 range
            score_style = "success" if score > 0.5 else "warning" if score > 0.3 else "muted"

            table.add_row(
                str(r.get("vec_rank", "?")),
                str(r.get("chunk_id", "?")),
                f"[{score_style}]{score:.4f}[/{score_style}]",
                str(r.get("filename", "?"))[:22],
                str(r.get("metadata", {}).get("chunk_index", "?")),
                self._clean_preview(r.get("text", ""), 50),
            )

        if len(results) > 10:
            table.add_row("...", f"+{len(results) - 10}", "", "", "", "")

        console.print(
            Panel(
                table,
                title=f"[tertiary]Vector Dense Retrieval ({len(results)} hits)[/tertiary]",
                border_style="tertiary",
                padding=(0, 1),
            )
        )
        console.print()

    def _show_overlap_analysis(self, overlap: dict) -> None:
        """Show overlap between BM25 and vector results."""
        text = Text()

        bm25_count = overlap.get("bm25_result_count", 0)
        vec_count = overlap.get("vector_result_count", 0)
        overlap_count = overlap.get("overlap_count", 0)
        overlap_pct = overlap.get("overlap_percentage", 0)

        # Visual representation
        text.append("BM25 Results:   ", style="muted")
        text.append(f"{bm25_count:3d} ", style="secondary.bold")
        text.append("chunks\n", style="muted")

        text.append("Vector Results: ", style="muted")
        text.append(f"{vec_count:3d} ", style="tertiary.bold")
        text.append("chunks\n", style="muted")

        text.append("Overlap:        ", style="muted")
        text.append(f"{overlap_count:3d} ", style="primary.bold")
        text.append(f"chunks ({overlap_pct}%)\n\n", style="muted")

        # Overlap interpretation
        if overlap_count == 0 and bm25_count > 0 and vec_count > 0:
            text.append("Analysis: ", style="muted")
            text.append("No overlap between BM25 and vector results.\n", style="warning")
            text.append(
                "This suggests the query has different semantic vs lexical matches.", style="muted"
            )
        elif overlap_pct > 50:
            text.append("Analysis: ", style="muted")
            text.append("High overlap ", style="success")
            text.append("between retrieval methods. Both agree on relevance.", style="muted")
        elif overlap_pct > 20:
            text.append("Analysis: ", style="muted")
            text.append("Moderate overlap. ", style="primary")
            text.append(
                "Hybrid retrieval is adding value by combining different signals.", style="muted"
            )
        elif bm25_count == 0:
            text.append("Analysis: ", style="muted")
            text.append("BM25 returned no results. ", style="warning")
            text.append(
                "Query terms may not appear in documents. Vector search is providing all results.",
                style="muted",
            )
        else:
            text.append("Analysis: ", style="muted")
            text.append("Low overlap. ", style="tertiary")
            text.append("Semantic and lexical matches are quite different.", style="muted")

        # Show overlap IDs if any
        overlap_ids = overlap.get("overlap_ids", [])
        if overlap_ids:
            text.append("\n\nOverlapping CIDs: ", style="muted")
            text.append(", ".join(str(cid) for cid in overlap_ids[:15]), style="number")
            if len(overlap_ids) > 15:
                text.append(f" +{len(overlap_ids) - 15} more", style="muted")

        console.print(
            Panel(
                text,
                title="[primary]Retrieval Overlap Analysis[/primary]",
                border_style="primary",
                padding=(1, 2),
            )
        )
        console.print()

    def _show_fused_results(self, fused: list, rrf_params: dict) -> None:
        """Display RRF fused results with contribution breakdown."""
        if not fused:
            console.print(
                Panel(
                    Text("No fused results.", style="warning"),
                    title="[primary]RRF Fusion Results[/primary]",
                    border_style="primary",
                )
            )
            console.print()
            return

        # RRF parameters info
        params_text = Text()
        params_text.append("RRF Formula: ", style="muted")
        params_text.append("score = w_bm25/(k+rank_bm25) + w_vec/(k+rank_vec)", style="tertiary")
        params_text.append(f"\nParameters: k={rrf_params.get('rrf_k', 60)}, ", style="muted")
        params_text.append(f"w_bm25={rrf_params.get('w_bm25', 1.0)}, ", style="secondary")
        params_text.append(f"w_vec={rrf_params.get('w_vec', 1.0)}", style="tertiary")

        console.print(Panel(params_text, border_style="muted", padding=(0, 2)))
        console.print()

        # RRF results
        table = Table(
            title="[primary]Hybrid Fusion Results (RRF)[/primary]",
            border_style="primary",
            header_style="primary.bold",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Rank", style="primary.bold", width=5, justify="center")
        table.add_column("CID", style="number", width=6, justify="right")
        table.add_column("RRF Score", style="success", width=10, justify="right")
        table.add_column("BM25", style="secondary", width=8, justify="center")
        table.add_column("Vec", style="tertiary", width=8, justify="center")
        table.add_column("Source", style="muted", width=8, justify="center")
        table.add_column("Document", style="path", width=20)
        table.add_column("Preview", style="text", overflow="ellipsis")

        for i, r in enumerate(fused, 1):
            # Source indicator
            in_bm25 = r.get("in_bm25", False)
            in_vec = r.get("in_vector", False)
            if in_bm25 and in_vec:
                source = "[success]BOTH[/success]"
            elif in_bm25:
                source = "[secondary]BM25[/secondary]"
            else:
                source = "[tertiary]VEC[/tertiary]"

            # Rank displays
            bm25_rank = r.get("bm25_rank")
            vec_rank = r.get("vec_rank")
            bm25_str = f"#{bm25_rank}" if bm25_rank else "-"
            vec_str = f"#{vec_rank}" if vec_rank else "-"

            table.add_row(
                str(i),
                str(r.get("chunk_id", "?")),
                f"{r.get('fused_score', 0):.6f}",
                bm25_str,
                vec_str,
                source,
                str(r.get("filename", "?"))[:20],
                self._clean_preview(r.get("text", ""), 40),
            )

        console.print(
            Panel(
                table,
                title=f"[primary]RRF Fusion Results ({len(fused)} chunks)[/primary]",
                border_style="primary",
                padding=(0, 1),
            )
        )
        console.print()

        # Contribution breakdown for top 3
        self._show_rrf_breakdown(fused[:3], rrf_params)

    def _show_rrf_breakdown(self, top_results: list, rrf_params: dict) -> None:
        """Show detailed RRF score breakdown for top results."""
        if not top_results:
            return

        text = Text()
        text.append("Score Breakdown (Top 3):\n\n", style="primary.bold")

        for i, r in enumerate(top_results, 1):
            cid = r.get("chunk_id", "?")
            bm25_contrib = r.get("rrf_bm25_contribution", 0)
            vec_contrib = r.get("rrf_vec_contribution", 0)
            total = r.get("fused_score", 0)

            text.append(f"  #{i} CID {cid}: ", style="muted")
            text.append(f"{total:.6f}", style="success bold")
            text.append(" = ", style="muted")

            if bm25_contrib > 0:
                text.append(f"{bm25_contrib:.6f}", style="secondary")
                text.append(" (BM25)", style="muted")
            else:
                text.append("0", style="muted")
                text.append(" (BM25)", style="muted")

            text.append(" + ", style="muted")

            if vec_contrib > 0:
                text.append(f"{vec_contrib:.6f}", style="tertiary")
                text.append(" (Vec)", style="muted")
            else:
                text.append("0", style="muted")
                text.append(" (Vec)", style="muted")

            text.append("\n")

        console.print(Panel(text, border_style="muted", padding=(0, 2)))
        console.print()

    def _show_timing_analysis(self, timings: dict) -> None:
        """Display timing breakdown."""
        if not timings:
            return

        table = Table(
            show_header=True,
            header_style="muted",
            box=box.SIMPLE,
        )
        table.add_column("Stage", style="text", width=25)
        table.add_column("Time (ms)", style="number", width=12, justify="right")
        table.add_column("Bar", style="muted", width=30)

        total = timings.get("total_ms", 1)

        # Sort by time descending
        sorted_timings = sorted(
            [(k, v) for k, v in timings.items() if k != "total_ms"],
            key=lambda x: x[1],
            reverse=True,
        )

        for stage, ms in sorted_timings:
            # Create visual bar
            pct = (ms / total) * 100 if total > 0 else 0
            bar_len = int(pct / 5)  # 20 chars max
            bar = "[success]" + "█" * bar_len + "[/success]" + "░" * (20 - bar_len)

            # Clean up stage name
            stage_name = stage.replace("_ms", "").replace("_", " ").title()

            table.add_row(stage_name, f"{ms:.2f}", bar)

        table.add_row("", "", "")
        table.add_row("[bold]Total[/bold]", f"[bold]{total:.2f}[/bold]", "")

        console.print(
            Panel(
                table,
                title="[muted]Performance Timing[/muted]",
                border_style="muted",
                padding=(0, 1),
            )
        )

    def _clean_preview(self, text: str, max_len: int = 50) -> str:
        """Clean text for preview display."""
        # Remove newlines and extra whitespace
        cleaned = " ".join(text.split())
        if len(cleaned) > max_len:
            return cleaned[:max_len] + "..."
        return cleaned

    def _debug_citations(self, args: list[str], flags: dict) -> bool:
        """Debug citation validation."""
        # Get query
        if args:
            query = " ".join(args)
        else:
            query = Prompt.ask(
                "[primary]>[/primary] [text]Enter query for citation debug[/text]",
                console=console,
            ).strip()

        if not query:
            return True

        # Parse options
        bm25_k = int(flags.get("bm25_k", self.config.bm25_k))
        vec_k = int(flags.get("vec_k", self.config.vec_k))
        top_k = int(flags.get("top_k", self.config.top_k))
        bm25_mode = flags.get("bm25_mode", "heuristic")

        with create_spinner("Running citation debug...", style="processing"):
            response = self.api.debug_citations(
                query=query,
                bm25_k=bm25_k,
                vec_k=vec_k,
                top_k=top_k,
                bm25_mode=bm25_mode,
            )

        if not response.success:
            print_error(response.error or "Debug citations failed")
            return False

        data = response.data or {}

        console.print()
        self._show_citation_report(data)

        return True

    def _show_citation_report(self, data: dict) -> None:
        """Show citation validation report."""
        ok = data.get("ok", False)
        report = data.get("report", {})
        answer = data.get("answer", "")
        allowed = data.get("allowed_chunk_ids", [])

        # Status panel
        status_text = Text()
        if ok:
            status_text.append("PASS", style="success bold")
            status_text.append(" All citations are valid", style="success")
        else:
            status_text.append("FAIL", style="error bold")
            status_text.append(
                f" {report.get('reason', 'Citation validation failed')}", style="error"
            )

        console.print(
            Panel(
                status_text,
                border_style="success" if ok else "error",
                padding=(0, 2),
            )
        )

        # Report details
        console.print()

        report_table = Table(show_header=False, box=None, padding=(0, 2))
        report_table.add_column("Metric", style="muted", width=28)
        report_table.add_column("Value", style="text")

        report_table.add_row("Paragraphs", str(report.get("paragraph_count", 0)))
        report_table.add_row("Found Citations", str(report.get("found_citations", [])))
        report_table.add_row("Unique Citations", str(report.get("unique_citations_count", 0)))
        report_table.add_row("Invalid IDs", str(report.get("invalid_ids", [])))
        report_table.add_row("Missing Paragraphs", str(report.get("missing_paragraphs", [])))
        report_table.add_row(
            "Allowed Chunk IDs",
            f"[{', '.join(str(x) for x in allowed[:10])}{'...' if len(allowed) > 10 else ''}]",
        )

        console.print(
            Panel(
                report_table,
                title="[primary]Citation Report[/primary]",
                border_style="primary",
                padding=(1, 2),
            )
        )

        # Show answer preview
        if answer:
            preview = answer[:600] + "..." if len(answer) > 600 else answer
            console.print()
            console.print(
                Panel(
                    Text(preview, style="text"),
                    title="[muted]Answer Preview[/muted]",
                    border_style="muted",
                    padding=(1, 2),
                )
            )
