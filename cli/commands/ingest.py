"""Ingest command - upload documents to the knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error, print_success, print_warning
from cli.ui.panels import create_ingest_result_panel
from cli.ui.spinners import create_spinner


class IngestCommand(BaseCommand):
    """Ingest documents into the knowledge base."""
    
    name = "ingest"
    description = "Upload and process documents into the knowledge base"
    usage = "/ingest <file1> [file2] ... [--dir DIRECTORY]"
    aliases = ["upload", "add"]
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv"}
    
    def execute(self, args: list[str]) -> bool:
        """Execute document ingestion."""
        flags, remaining = self.parse_flags(args)
        
        # Collect files
        files: list[Path] = []
        
        # From directory flag
        if "dir" in flags:
            dir_path = Path(flags["dir"])
            if dir_path.is_dir():
                files.extend(self._scan_directory(dir_path))
            else:
                print_error(f"Directory not found: {dir_path}")
                return False
        
        # From arguments
        for arg in remaining:
            path = Path(arg)
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(self._scan_directory(path))
            elif "*" in arg or "?" in arg:
                # Glob pattern
                import glob
                for match in glob.glob(arg):
                    p = Path(match)
                    if p.is_file():
                        files.append(p)
            else:
                print_warning(f"File not found: {arg}")
        
        # If no files provided, prompt for input
        if not files:
            files = self._prompt_for_files()
            if not files:
                return True
        
        # Filter to supported extensions
        valid_files = [f for f in files if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        skipped = len(files) - len(valid_files)
        
        if skipped > 0:
            print_warning(f"Skipping {skipped} unsupported file(s)")
        
        if not valid_files:
            print_error("No valid files to ingest")
            return False
        
        # Show files to be ingested
        self._show_file_summary(valid_files)
        
        # Confirm
        if len(valid_files) > 1:
            if not Confirm.ask(
                f"\n[primary]Ingest {len(valid_files)} files?[/primary]",
                console=console,
                default=True,
            ):
                console.print("[muted]Cancelled[/muted]")
                return True
        
        # Perform ingestion
        console.print()
        with create_spinner(
            f"Ingesting {len(valid_files)} file(s)...",
            style="processing",
        ):
            response = self.api.ingest(valid_files)
        
        if not response.success:
            print_error(response.error or "Ingestion failed")
            return False
        
        # Show results
        console.print()
        console.print(create_ingest_result_panel(response.data or {}))
        
        return True
    
    def _scan_directory(self, directory: Path) -> list[Path]:
        """Scan a directory for supported files."""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"**/*{ext}"))
        return list(set(files))  # Remove duplicates
    
    def _prompt_for_files(self) -> list[Path]:
        """Prompt user for file paths."""
        console.print()
        console.print(Panel(
            Text.from_markup(
                "[primary]Document Ingestion[/primary]\n\n"
                "Enter file paths (one per line) or drag files here.\n"
                "Supported: [tertiary].pdf .txt .md .json .csv[/tertiary]\n"
                "Type [command]done[/command] when finished."
            ),
            border_style="primary",
            padding=(1, 2),
        ))
        
        files = []
        while True:
            path_str = Prompt.ask(
                "[primary]❯[/primary] [muted]File path[/muted]",
                console=console,
                default="done",
            ).strip()
            
            if path_str.lower() == "done":
                break
            
            # Handle quoted paths
            path_str = path_str.strip("'\"")
            
            path = Path(path_str)
            if path.is_file():
                files.append(path)
                console.print(f"  [success]✔[/success] Added: {path.name}")
            elif path.is_dir():
                dir_files = self._scan_directory(path)
                files.extend(dir_files)
                console.print(f"  [success]✔[/success] Added {len(dir_files)} files from {path}")
            else:
                console.print(f"  [error]✖[/error] Not found: {path_str}")
        
        return files
    
    def _show_file_summary(self, files: list[Path]) -> None:
        """Show summary of files to be ingested."""
        console.print()
        
        table = Table(
            title="[primary]📁 Files to Ingest[/primary]",
            show_header=True,
            header_style="primary",
            border_style="muted",
        )
        table.add_column("#", style="number", width=4)
        table.add_column("File", style="path")
        table.add_column("Type", style="tertiary", width=8)
        table.add_column("Size", style="number", width=12, justify="right")
        
        total_size = 0
        for i, f in enumerate(files[:10], 1):
            size = f.stat().st_size
            total_size += size
            table.add_row(
                str(i),
                f.name,
                f.suffix[1:].upper(),
                self._format_size(size),
            )
        
        if len(files) > 10:
            table.add_row(
                "...",
                f"+{len(files) - 10} more files",
                "",
                "",
            )
        
        console.print(table)
        console.print(f"\n  [muted]Total:[/muted] [number]{len(files)}[/number] files, [number]{self._format_size(total_size)}[/number]")
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
