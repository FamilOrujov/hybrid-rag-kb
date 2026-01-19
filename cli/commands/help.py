"""Help command - display CLI help and documentation."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.commands.base import BaseCommand
from cli.ui.console import console
from cli.ui.logo import print_logo

COMMANDS = [
    {
        "name": "/start",
        "aliases": "/serve, /run",
        "description": "Start the FastAPI server (background mode)",
        "usage": "/start [--host HOST] [--port PORT] [--foreground]",
    },
    {
        "name": "/stop",
        "aliases": "",
        "description": "Stop the running server",
        "usage": "/stop",
    },
    {
        "name": "/restart",
        "aliases": "/reboot",
        "description": "Restart the server (stop + start)",
        "usage": "/restart",
    },
    {
        "name": "/query",
        "aliases": "/ask, /q",
        "description": "Ask questions and get answers with citations",
        "usage": "/query [question] [--session ID] [--top_k N] [--show-sources]",
    },
    {
        "name": "/chat",
        "aliases": "/interactive",
        "description": "Start interactive chat mode",
        "usage": "/chat [--session ID]",
    },
    {
        "name": "/ingest",
        "aliases": "/upload, /add",
        "description": "Upload documents to the knowledge base",
        "usage": "/ingest <file1> [file2] ... [--dir DIRECTORY]",
    },
    {
        "name": "/stats",
        "aliases": "/status, /info",
        "description": "Show system statistics",
        "usage": "/stats [--json]",
    },
    {
        "name": "/doctor",
        "aliases": "/health, /check",
        "description": "Run comprehensive system health checks",
        "usage": "/doctor [--verbose] [--fix]",
    },
    {
        "name": "/reset",
        "aliases": "/clean, /wipe",
        "description": "Remove database and index (start fresh)",
        "usage": "/reset [--force] [--db-only] [--index-only]",
    },
    {
        "name": "/model",
        "aliases": "/models, /llm",
        "description": "List and select Ollama models for chat/embeddings",
        "usage": "/model [list|set|info] [--chat MODEL] [--embed MODEL]",
    },
    {
        "name": "/debug",
        "aliases": "/dbg",
        "description": "Debug retrieval and citations",
        "usage": "/debug <retrieval|citations> [query] [--bm25_k N] [--vec_k N]",
    },
    {
        "name": "/chunk",
        "aliases": "/chunks, /c",
        "description": "View a specific chunk by ID",
        "usage": "/chunk <chunk_id>",
    },
    {
        "name": "/help",
        "aliases": "/h, /?",
        "description": "Show this help message",
        "usage": "/help [command]",
    },
    {
        "name": "/clear",
        "aliases": "/cls",
        "description": "Clear the terminal screen",
        "usage": "/clear",
    },
    {
        "name": "/quit",
        "aliases": "/exit",
        "description": "Exit the CLI",
        "usage": "/quit",
    },
]


class HelpCommand(BaseCommand):
    """Display help information."""

    name = "help"
    description = "Show help information"
    usage = "/help [command]"
    aliases = ["h", "?"]

    def execute(self, args: list[str]) -> bool:
        """Display help."""
        flags, remaining = self.parse_flags(args)

        if remaining:
            # Show help for specific command
            cmd_name = remaining[0].lower().lstrip("/")
            return self._show_command_help(cmd_name)
        else:
            # Show general help
            return self._show_general_help()

    def _show_general_help(self) -> bool:
        """Show general help with all commands."""
        print_logo(show_commands=False)

        # Commands table
        table = Table(
            show_header=True,
            header_style="primary",
            border_style="muted",
            padding=(0, 2),
        )
        table.add_column("Command", style="command", width=15)
        table.add_column("Aliases", style="muted", width=18)
        table.add_column("Description", style="text")

        for cmd in COMMANDS:
            table.add_row(
                cmd["name"],
                cmd["aliases"],
                cmd["description"],
            )

        console.print(Panel(
            table,
            title="[primary]Available Commands[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))

        # Tips
        console.print()
        tips = Text()
        tips.append("Tips:\n", style="primary")
        tips.append("  • ", style="muted")
        tips.append("Use ", style="text")
        tips.append("/help <command>", style="command")
        tips.append(" for detailed command help\n", style="text")
        tips.append("  • ", style="muted")
        tips.append("Commands can be typed without the ", style="text")
        tips.append("/", style="command")
        tips.append(" prefix\n", style="text")
        tips.append("  • ", style="muted")
        tips.append("Press ", style="text")
        tips.append("Tab", style="warning")
        tips.append(" for command completion\n", style="text")
        tips.append("  • ", style="muted")
        tips.append("Press ", style="text")
        tips.append("Ctrl+C", style="warning")
        tips.append(" to cancel current operation", style="text")

        console.print(tips)

        return True

    def _show_command_help(self, cmd_name: str) -> bool:
        """Show detailed help for a specific command."""
        # Find command
        cmd = None
        for c in COMMANDS:
            name = c["name"].lstrip("/")
            aliases = [a.strip().lstrip("/") for a in c["aliases"].split(",")]
            if cmd_name == name or cmd_name in aliases:
                cmd = c
                break

        if not cmd:
            console.print(f"[error]Unknown command: {cmd_name}[/error]")
            console.print("[muted]Use /help to see available commands[/muted]")
            return False

        console.print()

        # Command details
        text = Text()
        text.append(f"{cmd['name']}\n\n", style="primary.bold")
        text.append(f"{cmd['description']}\n\n", style="text")
        text.append("Usage:\n", style="muted")
        text.append(f"  {cmd['usage']}\n\n", style="command")
        text.append("Aliases:\n", style="muted")
        text.append(f"  {cmd['aliases']}", style="tertiary")

        console.print(Panel(
            text,
            title=f"[primary]📖 {cmd['name']}[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))

        return True
