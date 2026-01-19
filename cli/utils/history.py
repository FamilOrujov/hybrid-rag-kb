"""Command history management."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.history import FileHistory, InMemoryHistory


class CommandHistory:
    """Manages command history with optional file persistence."""

    def __init__(self, history_file: Path | None = None):
        self.history_file = history_file

        if history_file:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            self._history = FileHistory(str(history_file))
        else:
            self._history = InMemoryHistory()

    @property
    def history(self):
        """Get the underlying history object."""
        return self._history

    def add(self, command: str) -> None:
        """Add a command to history."""
        if command.strip():
            self._history.append_string(command)

    def get_recent(self, n: int = 10) -> list[str]:
        """Get the N most recent commands."""
        strings = list(self._history.get_strings())
        return strings[-n:] if strings else []

    def clear(self) -> None:
        """Clear command history."""
        if isinstance(self._history, InMemoryHistory):
            self._history._loaded_strings = []
        elif self.history_file and self.history_file.exists():
            self.history_file.unlink()
            self._history = FileHistory(str(self.history_file))
