"""Stats command - display system statistics."""

from __future__ import annotations

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error
from cli.ui.panels import create_stats_panel
from cli.ui.spinners import create_spinner


class StatsCommand(BaseCommand):
    """Display system statistics."""

    name = "stats"
    description = "Show system statistics (SQLite, FAISS, GPU, models)"
    usage = "/stats [--json]"
    aliases = ["status", "info"]

    def execute(self, args: list[str]) -> bool:
        """Display system stats."""
        flags, _ = self.parse_flags(args)
        as_json = flags.get("json", False)

        with create_spinner("Fetching statistics...", style="loading"):
            response = self.api.stats()

        if not response.success:
            print_error(response.error or "Failed to fetch stats")
            return False

        data = response.data or {}

        if as_json:
            import json

            console.print_json(json.dumps(data, indent=2))
        else:
            console.print()
            console.print(create_stats_panel(data))

            # Show sync status
            self._show_sync_status(data)

        return True

    def _show_sync_status(self, data: dict) -> None:
        """Show sync status between SQLite and FAISS."""
        sqlite = data.get("sqlite", {})
        faiss = data.get("faiss", {})

        chunks = sqlite.get("chunks", 0) or 0
        vectors = faiss.get("ntotal", 0) or 0

        console.print()
        if chunks == vectors:
            console.print(f"  [success]✔ Synced:[/success] {chunks} chunks = {vectors} vectors")
        elif chunks > vectors:
            diff = chunks - vectors
            console.print(
                f"  [warning]⚠ Out of sync:[/warning] {diff} chunks missing from FAISS index"
            )
        else:
            diff = vectors - chunks
            console.print(
                f"  [warning]⚠ Out of sync:[/warning] {diff} orphan vectors in FAISS index"
            )
