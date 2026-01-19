"""Query command - interactive chat with the RAG system."""

from __future__ import annotations

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error
from cli.ui.panels import (
    create_assistant_message_bubble,
    create_debug_panel,
    create_query_result_panel,
    create_sources_panel,
    create_user_message_bubble,
)
from cli.ui.spinners import create_spinner


class QueryCommand(BaseCommand):
    """Query the knowledge base with citations."""

    name = "query"
    description = "Ask questions and get answers with citations"
    usage = "/query [question] [--session SESSION] [--top_k N] [--show-sources]"
    aliases = ["ask", "q", "chat"]

    def execute(self, args: list[str]) -> bool:
        """Execute a query."""
        flags, remaining = self.parse_flags(args)

        # Get query from args or prompt
        if remaining:
            query = " ".join(remaining)
        else:
            query = self._prompt_query()
            if not query:
                return True

        # Parse options
        session_id = flags.get("session", self.config.session_id)
        top_k = int(flags.get("top_k", flags.get("k", self.config.top_k)))
        bm25_k = int(flags.get("bm25_k", self.config.bm25_k))
        vec_k = int(flags.get("vec_k", self.config.vec_k))
        memory_k = int(flags.get("memory_k", self.config.memory_k))
        show_sources = flags.get("show-sources", flags.get("sources", False))
        show_debug = flags.get("debug", self.config.show_debug)

        # Execute query
        with create_spinner("Thinking...", style="thinking"):
            response = self.api.query(
                query=query,
                session_id=session_id,
                bm25_k=bm25_k,
                vec_k=vec_k,
                top_k=top_k,
                memory_k=memory_k,
            )

        if not response.success:
            print_error(response.error or "Query failed")
            return False

        data = response.data or {}
        answer = data.get("answer", "No answer received")
        sources = data.get("sources", [])
        debug = data.get("debug", {})

        # Display results
        console.print()
        console.print(create_query_result_panel(answer, query))

        if show_sources and sources:
            console.print()
            console.print(create_sources_panel(sources))

        if show_debug and debug:
            console.print()
            console.print(create_debug_panel(debug))

        # Show citation summary
        self._show_citation_summary(answer, sources)

        return True

    def _prompt_query(self) -> str:
        """Prompt user for a query."""
        console.print()
        return Prompt.ask(
            "[primary]❯[/primary] [text]Enter your question[/text]",
            console=console,
        ).strip()

    def _show_citation_summary(self, answer: str, sources: list) -> None:
        """Show a brief citation summary."""
        import re

        # Extract citations
        pattern = r"\[(?:Source:[^\]]*)?cid:(\d+)\]|\[cid:(\d+)\]"
        matches = re.findall(pattern, answer)
        cited_ids = set()
        for m in matches:
            cited_ids.add(int(m[0] or m[1]))

        allowed_ids = {int(s.get("chunk_id", 0)) for s in sources if "chunk_id" in s}

        if cited_ids:
            text = Text()
            text.append("\n  ", style="muted")
            text.append(f"{len(cited_ids)} citations", style="citation")
            text.append(" from ", style="muted")
            text.append(f"{len(allowed_ids)} sources", style="tertiary")

            # Check for invalid citations
            invalid = cited_ids - allowed_ids
            if invalid:
                text.append(" (", style="muted")
                text.append(f"{len(invalid)} invalid", style="error")
                text.append(")", style="muted")

            console.print(text)


class InteractiveQueryCommand(QueryCommand):
    """Interactive chat mode with continuous querying."""

    name = "chat"
    description = "Start an interactive chat session"
    usage = "/chat [--session SESSION]"
    aliases = ["interactive"]

    def execute(self, args: list[str]) -> bool:
        """Start interactive chat mode."""
        flags, _ = self.parse_flags(args)
        session_id = flags.get("session", self.config.session_id)

        console.print()
        # Modern chat mode header
        header = Text()
        header.append("", style="#C77DFF")
        header.append("Interactive Chat Mode", style="#C77DFF bold")
        header.append("\n\n", style="")
        header.append("Type your questions and press Enter.\n", style="#888888")
        header.append("Commands: ", style="#555555")
        header.append("/sources", style="#FF8C42")
        header.append(" • ", style="#555555")
        header.append("/debug", style="#FF8C42")
        header.append(" • ", style="#555555")
        header.append("/clear", style="#FF8C42")
        header.append(" • ", style="#555555")
        header.append("/exit", style="#FF8C42")

        console.print(
            Panel(
                header,
                border_style="#5D4E6D",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

        show_sources = False
        show_debug = False

        while True:
            console.print()
            query = Prompt.ask(
                "[#FF8C42]❯[/#FF8C42]",
                console=console,
            ).strip()

            if not query:
                continue

            # Handle special commands
            if query.lower() in ["/exit", "/quit", "/q"]:
                console.print("[#888888]Exiting chat mode...[/#888888]")
                break
            elif query.lower() == "/sources":
                show_sources = not show_sources
                status = (
                    "[#00E676]enabled[/#00E676]" if show_sources else "[#888888]disabled[/#888888]"
                )
                console.print(f"[#888888]Sources display:[/#888888] {status}")
                continue
            elif query.lower() == "/debug":
                show_debug = not show_debug
                status = (
                    "[#00E676]enabled[/#00E676]" if show_debug else "[#888888]disabled[/#888888]"
                )
                console.print(f"[#888888]Debug display:[/#888888] {status}")
                continue
            elif query.lower() == "/clear":
                console.clear()
                continue

            # Show user message bubble
            console.print()
            console.print(Align.right(create_user_message_bubble(query)))
            console.print()

            # Execute query with spinner
            with create_spinner("Thinking...", style="thinking"):
                response = self.api.query(
                    query=query,
                    session_id=session_id,
                    bm25_k=self.config.bm25_k,
                    vec_k=self.config.vec_k,
                    top_k=self.config.top_k,
                    memory_k=self.config.memory_k,
                )

            if not response.success:
                print_error(response.error or "Query failed")
                continue

            data = response.data or {}
            answer = data.get("answer", "No answer received")
            sources = data.get("sources", [])
            debug = data.get("debug", {})

            # Show assistant response bubble
            console.print(create_assistant_message_bubble(answer))

            if show_sources and sources:
                console.print()
                console.print(create_sources_panel(sources))

            if show_debug and debug:
                console.print()
                console.print(create_debug_panel(debug))

            self._show_citation_summary(answer, sources)

        return True
