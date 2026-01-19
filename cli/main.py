"""Main CLI entry point - Interactive REPL for Hybrid RAG."""

from __future__ import annotations

import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from cli import __version__
from cli.commands.chunks import ChunksCommand
from cli.commands.debug import DebugCommand
from cli.commands.doctor import DoctorCommand
from cli.commands.help import HelpCommand
from cli.commands.ingest import IngestCommand
from cli.commands.model import ModelCommand
from cli.commands.query import InteractiveQueryCommand, QueryCommand
from cli.commands.reset import ResetCommand
from cli.commands.start import RestartCommand, StartCommand, StopCommand
from cli.commands.stats import StatsCommand
from cli.core.api_client import APIClient
from cli.core.config import CLIConfig, get_config, set_config
from cli.ui.console import console, print_error
from cli.ui.logo import print_logo, print_logo_animated
from cli.utils.completions import CommandCompleter
from cli.utils.history import CommandHistory

# Prompt styling
PROMPT_STYLE = Style.from_dict({
    "prompt": "#FF8C42 bold",
    # Completion menu styling
    "completion-menu": "bg:#1a1625 #e8e8e8",
    "completion-menu.completion": "bg:#1a1625 #c77dff",
    "completion-menu.completion.current": "bg:#9d4edd #ffffff bold",
    "completion-menu.meta.completion": "bg:#1a1625 #888888",
    "completion-menu.meta.completion.current": "bg:#9d4edd #e8e8e8",
    # Scrollbar
    "scrollbar.background": "bg:#1a1625",
    "scrollbar.button": "bg:#9d4edd",
})


class HybridRAGCLI:
    """Main CLI application."""

    def __init__(self, config: CLIConfig | None = None):
        self.config = config or get_config()
        set_config(self.config)

        # Initialize components
        self.history = CommandHistory(
            Path.home() / ".hybrid-rag" / "history"
        )
        self.completer = CommandCompleter()

        # Command registry
        self.commands = {
            "start": StartCommand(self.config),
            "serve": StartCommand(self.config),
            "run": StartCommand(self.config),
            "stop": StopCommand(self.config),
            "restart": RestartCommand(self.config),
            "reboot": RestartCommand(self.config),
            "query": QueryCommand(self.config),
            "ask": QueryCommand(self.config),
            "q": QueryCommand(self.config),
            "chat": InteractiveQueryCommand(self.config),
            "interactive": InteractiveQueryCommand(self.config),
            "ingest": IngestCommand(self.config),
            "upload": IngestCommand(self.config),
            "add": IngestCommand(self.config),
            "stats": StatsCommand(self.config),
            "status": StatsCommand(self.config),
            "info": StatsCommand(self.config),
            "debug": DebugCommand(self.config),
            "dbg": DebugCommand(self.config),
            "chunk": ChunksCommand(self.config),
            "chunks": ChunksCommand(self.config),
            "c": ChunksCommand(self.config),
            "doctor": DoctorCommand(self.config),
            "health": DoctorCommand(self.config),
            "check": DoctorCommand(self.config),
            "reset": ResetCommand(self.config),
            "clean": ResetCommand(self.config),
            "wipe": ResetCommand(self.config),
            "model": ModelCommand(self.config),
            "models": ModelCommand(self.config),
            "llm": ModelCommand(self.config),
            "help": HelpCommand(self.config),
            "h": HelpCommand(self.config),
            "?": HelpCommand(self.config),
        }

        # Create prompt session
        self.session = PromptSession(
            history=self.history.history,
            completer=self.completer,
            style=PROMPT_STYLE,
            complete_while_typing=True,
            complete_in_thread=True,
            mouse_support=False,
        )

    def get_prompt(self) -> HTML:
        """Generate the prompt."""
        return HTML('<prompt>❯</prompt> ')

    def run(self) -> int:
        """Run the CLI REPL."""
        # Show logo on startup with animation
        console.clear()
        print_logo_animated()
        console.print()

        # Check server status
        self._check_server_status()

        # Main REPL loop
        while True:
            try:
                # Get input
                user_input = self.session.prompt(self.get_prompt()).strip()

                if not user_input:
                    continue

                # Add to history
                self.history.add(user_input)

                # Parse command
                cmd_name, args = self._parse_input(user_input)

                # Handle built-in commands
                if cmd_name in ["quit", "exit"]:
                    console.print("[muted]Goodbye.[/muted]")
                    return 0

                if cmd_name == "clear":
                    console.clear()
                    print_logo(show_tagline=False, show_commands=False)
                    continue

                # Execute command
                if cmd_name in self.commands:
                    try:
                        self.commands[cmd_name].execute(args)
                    except KeyboardInterrupt:
                        console.print("\n[warning]Interrupted[/warning]")
                    except Exception as e:
                        print_error(f"Command failed: {e}")
                else:
                    # Try as a query
                    if not cmd_name.startswith("/"):
                        self.commands["query"].execute([user_input])
                    else:
                        print_error(f"Unknown command: {cmd_name}")
                        console.print("[muted]Type /help for available commands[/muted]")

                console.print()

            except KeyboardInterrupt:
                console.print("\n[muted]Type /quit to exit[/muted]")
            except EOFError:
                console.print("\n[muted]Goodbye.[/muted]")
                return 0
            except Exception as e:
                print_error(f"Unexpected error: {e}")

        return 0

    def _parse_input(self, user_input: str) -> tuple[str, list[str]]:
        """Parse user input into command and arguments."""
        parts = user_input.split()

        if not parts:
            return "", []

        cmd = parts[0].lower().lstrip("/")
        args = parts[1:]

        return cmd, args

    def _check_server_status(self) -> None:
        """Check if the server is running and show status."""
        api = APIClient(self.config.base_url)

        try:
            response = api.health()
            if response.success:
                console.print(f"  [success]●[/success] Server running at [primary]{self.config.base_url}[/primary]")
            else:
                console.print("  [warning]○[/warning] Server not running. Use [command]/start[/command] to launch.")
        except Exception:
            console.print("  [warning]○[/warning] Server not running. Use [command]/start[/command] to launch.")

        console.print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid RAG CLI - Modern interface for Hybrid RAG KB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Hybrid RAG CLI {__version__}",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Command to execute (optional, starts REPL if not provided)",
    )

    # Use parse_known_args to allow command-specific flags to pass through
    args, remaining = parser.parse_known_args()
    args.args = remaining

    # Create config
    config = CLIConfig(
        host=args.host,
        port=args.port,
        project_root=Path.cwd(),
    )

    # Create CLI
    cli = HybridRAGCLI(config)

    # If command provided, execute it and exit
    if args.command:
        cmd_name = args.command.lower().lstrip("/")
        if cmd_name in cli.commands:
            success = cli.commands[cmd_name].execute(args.args)
            sys.exit(0 if success else 1)
        else:
            print_error(f"Unknown command: {args.command}")
            sys.exit(1)

    # Otherwise, run interactive REPL
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
