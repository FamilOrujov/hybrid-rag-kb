"""Command completion for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


COMMANDS = [
    "/start",
    "/stop",
    "/restart",
    "/query",
    "/chat",
    "/ingest",
    "/stats",
    "/doctor",
    "/reset",
    "/model",
    "/debug",
    "/chunk",
    "/help",
    "/clear",
    "/quit",
    "/exit",
]

COMMAND_OPTIONS = {
    "/start": ["--host", "--port", "--reload", "--foreground", "--fg"],
    "/stop": [],
    "/restart": [],
    "/query": ["--session", "--top_k", "--bm25_k", "--vec_k", "--memory_k", "--show-sources", "--debug"],
    "/chat": ["--session"],
    "/ingest": ["--dir"],
    "/stats": ["--json"],
    "/doctor": ["--verbose", "-v", "--fix"],
    "/reset": ["--force", "-f", "--db-only", "--index-only", "--keep-raw"],
    "/model": ["list", "set", "info", "--chat", "--embed"],
    "/debug": ["retrieval", "citations", "--bm25_k", "--vec_k", "--top_k", "--bm25_mode"],
    "/chunk": [],
    "/help": [c.lstrip("/") for c in COMMANDS],
}


class CommandCompleter(Completer):
    """Completer for CLI commands."""
    
    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions for the current input."""
        text = document.text_before_cursor
        words = text.split()
        
        if not text or text.isspace():
            # Show all commands
            for cmd in COMMANDS:
                yield Completion(cmd, start_position=0)
            return
        
        if len(words) == 1 and not text.endswith(" "):
            # Completing command name
            word = words[0].lower()
            for cmd in COMMANDS:
                if cmd.startswith(word) or cmd.lstrip("/").startswith(word.lstrip("/")):
                    yield Completion(cmd, start_position=-len(word))
            return
        
        # Completing arguments
        if words:
            cmd = words[0].lower()
            if not cmd.startswith("/"):
                cmd = "/" + cmd
            
            options = COMMAND_OPTIONS.get(cmd, [])
            
            if text.endswith(" "):
                # Show all options
                for opt in options:
                    yield Completion(opt, start_position=0)
            else:
                # Complete current word
                current = words[-1].lower()
                
                # File path completion
                if current.startswith("./") or current.startswith("/") or current.startswith("~"):
                    yield from self._complete_path(current)
                else:
                    for opt in options:
                        if opt.startswith(current):
                            yield Completion(opt, start_position=-len(current))
    
    def _complete_path(self, partial: str) -> Iterable[Completion]:
        """Complete file paths."""
        path = Path(partial).expanduser()
        
        if path.is_dir():
            parent = path
            prefix = ""
        else:
            parent = path.parent
            prefix = path.name
        
        if parent.exists():
            try:
                for item in parent.iterdir():
                    if item.name.startswith(prefix) and not item.name.startswith("."):
                        completion = str(item)
                        if item.is_dir():
                            completion += "/"
                        yield Completion(
                            completion,
                            start_position=-len(partial),
                            display=item.name,
                        )
            except PermissionError:
                pass
