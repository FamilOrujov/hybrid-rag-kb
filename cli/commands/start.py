"""Start command - launches the FastAPI server in background with status monitoring."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error, print_success, print_warning
from cli.ui.logo import print_mini_logo


class StartCommand(BaseCommand):
    """Start the Hybrid RAG FastAPI server in background."""

    name = "start"
    description = "Start the FastAPI server (runs in background)"
    usage = "/start [--host HOST] [--port PORT] [--reload] [--foreground]"
    aliases = ["serve", "run"]

    def __init__(self, config):
        super().__init__(config)
        self.process: subprocess.Popen | None = None

    def execute(self, args: list[str]) -> bool:
        """Start the server."""
        flags, _ = self.parse_flags(args)

        host = flags.get("host", self.config.host)
        port = int(flags.get("port", self.config.port))
        reload = flags.get("reload", True)
        foreground = flags.get("foreground", flags.get("fg", False))

        # Check if server is already running
        health_check = self.api.health()
        if health_check.success:
            print_warning(f"Server is already running at {self.config.base_url}")
            console.print("  [muted]Use the existing server or stop it first.[/muted]")
            return True

        # Build the command
        cmd = self._build_command(host, port, reload)

        console.print()
        print_mini_logo()
        console.print()

        if foreground:
            return self._start_foreground(cmd, host, port)
        else:
            return self._start_background(cmd, host, port)

    def _build_command(self, host: str, port: int, reload: bool) -> list[str]:
        """Build the uvicorn command."""
        # Check if we're in a uv project
        pyproject = Path(self.config.project_root) / "pyproject.toml"

        if pyproject.exists():
            # Use uv run
            cmd = ["uv", "run", "uvicorn", "src.main:app"]
        else:
            # Use python directly
            cmd = [sys.executable, "-m", "uvicorn", "src.main:app"]

        cmd.extend(["--host", host, "--port", str(port)])

        if reload:
            cmd.append("--reload")

        return cmd

    def _start_background(self, cmd: list[str], host: str, port: int) -> bool:
        """Start server in background and return to CLI."""
        console.print(
            Panel(
                self._create_startup_text(host, port, background=True),
                border_style="primary",
                padding=(1, 2),
            )
        )

        # Animation frames
        frames = ["◐", "◓", "◑", "◒"]

        try:
            # Start process in background with output redirected
            log_file = Path(self.config.project_root) / "data" / "server.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "w") as log:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=self.config.project_root,
                )

            # Wait for server to start with animated progress
            console.print()
            max_wait = 30
            start_time = time.time()

            with Live(console=console, refresh_per_second=10) as live:
                frame_idx = 0
                while time.time() - start_time < max_wait:
                    # Check if server is up
                    health = self.api.health()
                    if health.success:
                        break

                    # Check if process died
                    if self.process.poll() is not None:
                        console.print()
                        print_error("Server failed to start. Check logs at data/server.log")
                        return False

                    # Update animation
                    elapsed = time.time() - start_time
                    frame = frames[frame_idx % len(frames)]
                    text = Text()
                    text.append(f"  {frame} ", style="#C77DFF")
                    text.append("Starting server", style="text")
                    text.append(f" ({elapsed:.1f}s)", style="muted")
                    live.update(text)

                    frame_idx += 1
                    time.sleep(0.1)

            # Check final status
            health = self.api.health()
            if health.success:
                console.print()
                print_success("Server started successfully!")
                console.print()

                # Show status table
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Label", style="muted")
                table.add_column("Value", style="text")
                table.add_row("URL", f"[primary]http://{host}:{port}[/primary]")
                table.add_row("API Docs", f"[secondary]http://{host}:{port}/docs[/secondary]")
                table.add_row("PID", f"[muted]{self.process.pid}[/muted]")
                table.add_row("Log", "[muted]data/server.log[/muted]")
                console.print(table)
                console.print()
                console.print(
                    "  [muted]Server is running in background. Use [/muted][command]/stop[/command][muted] to stop.[/muted]"
                )

                return True
            else:
                print_error("Server failed to respond. Check logs at data/server.log")
                return False

        except Exception as e:
            print_error(f"Failed to start server: {e}")
            return False

    def _start_foreground(self, cmd: list[str], host: str, port: int) -> bool:
        """Start server in foreground with live output (blocks CLI)."""
        console.print(
            Panel(
                self._create_startup_text(host, port, background=False),
                border_style="primary",
                padding=(1, 2),
            )
        )

        console.print()
        console.print("  [warning]⚠ Running in foreground mode. Press Ctrl+C to stop.[/warning]")
        console.print()

        try:
            # Run in foreground
            self.process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                cwd=self.config.project_root,
            )

            # Wait for process
            self.process.wait()
            return True

        except KeyboardInterrupt:
            console.print("\n")
            print_warning("Shutting down server...")
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=5)
            print_success("Server stopped")
            return True
        except Exception as e:
            print_error(f"Server error: {e}")
            return False

    def _create_startup_text(self, host: str, port: int, background: bool) -> Text:
        """Create the startup info text."""
        text = Text()
        text.append("Starting Server\n\n", style="#FF8C42 bold")
        text.append("  Host:   ", style="muted")
        text.append(f"{host}\n", style="text")
        text.append("  Port:   ", style="muted")
        text.append(f"{port}\n", style="#FFB347")
        text.append("  Mode:   ", style="muted")
        if background:
            text.append("Background", style="#C77DFF")
            text.append(" (CLI remains interactive)\n", style="muted")
        else:
            text.append("Foreground", style="warning")
            text.append(" (blocks CLI)\n", style="muted")
        text.append("  URL:    ", style="muted")
        text.append(f"http://{host}:{port}\n", style="primary")
        text.append("  Docs:   ", style="muted")
        text.append(f"http://{host}:{port}/docs", style="secondary")

        return text


class StopCommand(BaseCommand):
    """Stop the running server."""

    name = "stop"
    description = "Stop the running FastAPI server"
    usage = "/stop"
    aliases = []

    def execute(self, args: list[str]) -> bool:
        """Stop the server."""
        return self._stop_server()

    def _stop_server(self, silent: bool = False) -> bool:
        """Stop the server. Returns True if stopped successfully.

        Always searches for and kills orphan processes, even if health check fails.
        This handles cases where the server crashed but processes are still lingering.
        """
        # Find and kill the server process
        try:
            import psutil
        except ImportError:
            if not silent:
                print_warning("psutil not installed. Please stop the server manually.")
            return False

        if not silent:
            console.print()

        # Find uvicorn processes (always search, even if health check fails)
        killed = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if (
                    cmdline
                    and "uvicorn" in " ".join(cmdline)
                    and "src.main:app" in " ".join(cmdline)
                ):
                    proc.terminate()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed > 0:
            # Wait for processes to terminate gracefully
            time.sleep(1)

            # Force kill any remaining processes
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline", [])
                    if (
                        cmdline
                        and "uvicorn" in " ".join(cmdline)
                        and "src.main:app" in " ".join(cmdline)
                    ):
                        proc.kill()  # Force kill if still running
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if not silent:
                print_success(f"Stopped {killed} server process(es)")
            return True
        else:
            if not silent:
                print_warning("Server is not running")
            return True  # Not an error, just nothing to stop


class RestartCommand(BaseCommand):
    """Restart the server."""

    name = "restart"
    description = "Restart the FastAPI server"
    usage = "/restart"
    aliases = ["reboot"]

    def execute(self, args: list[str]) -> bool:
        """Restart the server."""
        flags, _ = self.parse_flags(args)

        console.print()
        print_mini_logo()
        console.print()

        # Stop the server if running
        stop_cmd = StopCommand(self.config)
        was_running = self.api.health().success

        if was_running:
            console.print("  [warning]Stopping server...[/warning]")
            stop_cmd._stop_server(silent=True)
            time.sleep(1)
            console.print("  [success]✔[/success] Server stopped")

        # Start the server
        console.print("  [primary]Starting server...[/primary]")
        start_cmd = StartCommand(self.config)
        return start_cmd._start_background(
            start_cmd._build_command(self.config.host, self.config.port, reload=True),
            self.config.host,
            self.config.port,
        )
