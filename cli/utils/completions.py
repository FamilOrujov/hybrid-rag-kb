"""Command completion for the CLI with enhanced styling."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

# Command definitions with descriptions for better UX
COMMANDS = {
    "/start": "Start the FastAPI server",
    "/stop": "Stop the running server",
    "/restart": "Restart the server",
    "/query": "Query the knowledge base",
    "/chat": "Interactive chat mode",
    "/ingest": "Ingest documents",
    "/stats": "Show system statistics",
    "/doctor": "Run health checks",
    "/reset": "Reset database and index",
    "/model": "Manage LLM models",
    "/debug": "Debug retrieval pipeline",
    "/chunk": "View chunk details",
    "/help": "Show help",
    "/clear": "Clear the screen",
    "/quit": "Exit the CLI",
    "/exit": "Exit the CLI",
}

COMMAND_LIST = list(COMMANDS.keys())

COMMAND_OPTIONS = {
    "/start": ["--host", "--port", "--reload", "--foreground", "--fg"],
    "/stop": [],
    "/restart": [],
    "/query": [
        "--session",
        "--top_k",
        "--bm25_k",
        "--vec_k",
        "--memory_k",
        "--show-sources",
        "--debug",
    ],
    "/chat": ["--session"],
    "/ingest": ["--dir"],
    "/stats": ["--json"],
    "/doctor": ["--verbose", "-v", "--fix"],
    "/reset": ["--force", "-f", "--db-only", "--index-only", "--keep-raw"],
    "/model": ["list", "set", "info", "--chat", "--embed"],
    "/debug": ["retrieval", "citations", "--bm25_k", "--vec_k", "--top_k", "--bm25_mode"],
    "/chunk": [],
    "/help": [c.lstrip("/") for c in COMMAND_LIST],
}


class CommandCompleter(Completer):
    """Completer for CLI commands with enhanced display."""

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Get completions for the current input with descriptions."""
        text = document.text_before_cursor
        words = text.split()

        if not text or text.isspace():
            # Show all commands with descriptions
            for cmd, desc in COMMANDS.items():
                yield Completion(
                    cmd,
                    start_position=0,
                    display=cmd,
                    display_meta=desc,
                )
            return

        if len(words) == 1 and not text.endswith(" "):
            # Completing command name with fuzzy matching
            word = words[0].lower()
            for cmd, desc in COMMANDS.items():
                if cmd.startswith(word) or cmd.lstrip("/").startswith(word.lstrip("/")):
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display=cmd,
                        display_meta=desc,
                    )
            return

        # Completing arguments
        if words:
            cmd = words[0].lower()
            if not cmd.startswith("/"):
                cmd = "/" + cmd

            options = COMMAND_OPTIONS.get(cmd, [])

            if text.endswith(" "):
                # Show all options with type hints
                for opt in options:
                    meta = self._get_option_meta(cmd, opt)
                    yield Completion(
                        opt,
                        start_position=0,
                        display=opt,
                        display_meta=meta,
                    )
            else:
                # Complete current word
                current = words[-1].lower()

                # File path completion
                if current.startswith("./") or current.startswith("/") or current.startswith("~"):
                    yield from self._complete_path(current)
                else:
                    for opt in options:
                        if opt.lower().startswith(current):
                            meta = self._get_option_meta(cmd, opt)
                            yield Completion(
                                opt,
                                start_position=-len(current),
                                display=opt,
                                display_meta=meta,
                            )

    def _get_option_meta(self, cmd: str, opt: str) -> str:
        """Get metadata description for command options."""
        option_meta = {
            "--host": "server host",
            "--port": "server port",
            "--reload": "auto-reload",
            "--foreground": "run in foreground",
            "--fg": "run in foreground",
            "--session": "session ID",
            "--top_k": "top K results",
            "--bm25_k": "BM25 candidates",
            "--vec_k": "vector candidates",
            "--memory_k": "memory context",
            "--show-sources": "show sources",
            "--debug": "debug mode",
            "--json": "JSON output",
            "--verbose": "verbose output",
            "-v": "verbose",
            "--fix": "auto-fix issues",
            "--force": "skip confirmation",
            "-f": "force",
            "--db-only": "database only",
            "--index-only": "index only",
            "--keep-raw": "keep raw files",
            "--chat": "chat model",
            "--embed": "embed model",
            "--bm25_mode": "BM25 mode",
            "list": "list models",
            "set": "set model",
            "info": "model info",
            "retrieval": "debug retrieval",
            "citations": "debug citations",
        }
        return option_meta.get(opt, "")

    def _complete_path(self, partial: str) -> Iterable[Completion]:
        """Complete file paths with file type indicators."""
        path = Path(partial).expanduser()

        if path.is_dir():
            parent = path
            prefix = ""
        else:
            parent = path.parent
            prefix = path.name

        if parent.exists():
            try:
                for item in sorted(
                    parent.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
                ):
                    if item.name.startswith(prefix) and not item.name.startswith("."):
                        completion = str(item)
                        if item.is_dir():
                            completion += "/"
                            meta = "folder"
                        else:
                            ext = item.suffix.lower()
                            meta = self._get_file_type_meta(ext)

                        yield Completion(
                            completion,
                            start_position=-len(partial),
                            display=item.name + ("/" if item.is_dir() else ""),
                            display_meta=meta,
                        )
            except PermissionError:
                pass

    def _get_file_type_meta(self, ext: str) -> str:
        """Get file type description for display."""
        file_types = {
            ".pdf": "PDF",
            ".txt": "text",
            ".md": "markdown",
            ".json": "JSON",
            ".csv": "CSV",
            ".py": "python",
        }
        return file_types.get(ext, "file")
