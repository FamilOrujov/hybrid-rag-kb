"""Chunks command - view and inspect chunks."""

from __future__ import annotations

from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error
from cli.ui.spinners import create_spinner


class ChunksCommand(BaseCommand):
    """View and inspect chunks from the knowledge base."""
    
    name = "chunk"
    description = "View a specific chunk by ID"
    usage = "/chunk <chunk_id>"
    aliases = ["chunks", "c"]
    
    def execute(self, args: list[str]) -> bool:
        """View a chunk."""
        flags, remaining = self.parse_flags(args)
        
        # Get chunk ID
        if remaining:
            try:
                chunk_id = int(remaining[0])
            except ValueError:
                print_error(f"Invalid chunk ID: {remaining[0]}")
                return False
        else:
            chunk_id_str = Prompt.ask(
                "[primary]❯[/primary] [text]Enter chunk ID[/text]",
                console=console,
            ).strip()
            
            if not chunk_id_str:
                return True
            
            try:
                chunk_id = int(chunk_id_str)
            except ValueError:
                print_error(f"Invalid chunk ID: {chunk_id_str}")
                return False
        
        # Fetch chunk
        with create_spinner(f"Fetching chunk {chunk_id}...", style="loading"):
            response = self.api.get_chunk(chunk_id)
        
        if not response.success:
            print_error(response.error or f"Chunk {chunk_id} not found")
            return False
        
        data = response.data or {}
        
        console.print()
        self._display_chunk(data)
        
        return True
    
    def _display_chunk(self, data: dict) -> None:
        """Display chunk details."""
        # Metadata panel
        meta_text = Text()
        meta_text.append("Chunk ID:     ", style="muted")
        meta_text.append(str(data.get("chunk_id", "?")), style="number")
        meta_text.append("\n")
        meta_text.append("Document ID:  ", style="muted")
        meta_text.append(str(data.get("document_id", "?")), style="number")
        meta_text.append("\n")
        meta_text.append("Chunk Index:  ", style="muted")
        meta_text.append(str(data.get("chunk_index", "?")), style="number")
        meta_text.append("\n")
        meta_text.append("Filename:     ", style="muted")
        meta_text.append(str(data.get("filename", "?")), style="path")
        
        # Additional metadata
        metadata = data.get("metadata", {})
        if metadata:
            meta_text.append("\n\n")
            meta_text.append("Metadata:\n", style="muted")
            for key, value in metadata.items():
                meta_text.append(f"  {key}: ", style="muted")
                meta_text.append(str(value), style="text")
                meta_text.append("\n")
        
        console.print(Panel(
            meta_text,
            title=f"[primary]📄 Chunk #{data.get('chunk_id', '?')}[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))
        
        # Text content panel
        text = data.get("text", "")
        console.print()
        console.print(Panel(
            Text(text, style="text"),
            title="[tertiary]📝 Content[/tertiary]",
            subtitle=f"[muted]{len(text)} characters[/muted]",
            border_style="tertiary",
            padding=(1, 2),
        ))
