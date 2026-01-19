"""Reset command - remove database and index to start fresh."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.prompt import Confirm
from rich.table import Table

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_success, print_warning


class ResetCommand(BaseCommand):
    """Remove database and FAISS index to start fresh."""

    name = "reset"
    description = "Remove database and index files (start fresh)"
    usage = "/reset [--force] [--db-only] [--index-only] [--keep-raw]"
    aliases = ["clean", "wipe"]

    def execute(self, args: list[str]) -> bool:
        """Execute the reset command."""
        flags, _ = self.parse_flags(args)

        force = flags.get("force", flags.get("f", False))
        db_only = flags.get("db-only", flags.get("db", False))
        index_only = flags.get("index-only", flags.get("index", False))
        keep_raw = flags.get("keep-raw", False)

        # Determine what to delete
        delete_db = not index_only
        delete_index = not db_only
        delete_raw = not keep_raw and not db_only and not index_only

        # Paths
        db_path = Path(self.config.project_root) / "data" / "db" / "app.db"
        index_path = Path(self.config.project_root) / "data" / "index" / "faiss" / "index.faiss"
        raw_dir = Path(self.config.project_root) / "data" / "raw"

        # Check what exists
        items_to_delete = []

        if delete_db and db_path.exists():
            size = db_path.stat().st_size
            items_to_delete.append(("SQLite Database", db_path, self._format_size(size)))

        if delete_index and index_path.exists():
            size = index_path.stat().st_size
            items_to_delete.append(("FAISS Index", index_path, self._format_size(size)))

        if delete_raw and raw_dir.exists():
            files = list(raw_dir.glob("*"))
            if files:
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                items_to_delete.append(("Raw Files", raw_dir, f"{len(files)} files, {self._format_size(total_size)}"))

        if not items_to_delete:
            print_warning("Nothing to delete. Database and index are already clean.")
            return True

        # Show what will be deleted
        console.print()
        table = Table(
            title="[warning]⚠ Files to be deleted[/warning]",
            show_header=True,
            header_style="warning",
            border_style="warning",
        )
        table.add_column("Item", style="text")
        table.add_column("Path", style="muted")
        table.add_column("Size", style="number", justify="right")

        for name, path, size in items_to_delete:
            table.add_row(name, str(path), size)

        console.print(table)
        console.print()

        # Confirm unless --force
        if not force:
            confirmed = Confirm.ask(
                "[warning]Are you sure you want to delete these files?[/warning]",
                console=console,
                default=False,
            )
            if not confirmed:
                console.print("[muted]Cancelled[/muted]")
                return True

        # Delete files
        deleted = 0
        errors = []

        for name, path, _ in items_to_delete:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                deleted += 1
                console.print(f"  [success]✔[/success] Deleted {name}")
            except Exception as e:
                errors.append(f"{name}: {e}")
                console.print(f"  [error]✖[/error] Failed to delete {name}: {e}")

        console.print()

        if errors:
            print_warning(f"Deleted {deleted} item(s) with {len(errors)} error(s)")
        else:
            print_success(f"Deleted {deleted} item(s). Database is now clean.")
            console.print()
            console.print("  [warning]Important:[/warning] You must restart the server to reinitialize the database.")
            console.print("  [muted]Run:[/muted] [command]/restart[/command] [muted]before ingesting new files.[/muted]")

        return len(errors) == 0

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
