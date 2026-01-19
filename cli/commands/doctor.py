"""Doctor command - comprehensive system health check with animations."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from rich.align import Align
from rich.panel import Panel
from rich.text import Text

from cli.commands.base import BaseCommand
from cli.ui.console import console
from cli.ui.spinners import DoctorAnimation


class DoctorCommand(BaseCommand):
    """Comprehensive system health check."""

    name = "doctor"
    description = "Run comprehensive system health checks"
    usage = "/doctor [--fix] [--verbose]"
    aliases = ["health", "check"]

    def execute(self, args: list[str]) -> bool:
        """Run all health checks."""
        flags, _ = self.parse_flags(args)
        verbose = flags.get("verbose", flags.get("v", False))
        flags.get("fix", False)

        console.print()

        # Create animated doctor display
        doctor = DoctorAnimation("Hybrid RAG System Check")

        # Define all checks
        checks = [
            ("Python Environment", self._check_python),
            ("Dependencies", self._check_dependencies),
            ("Ollama Server", self._check_ollama),
            ("Ollama Models", self._check_ollama_models),
            ("SQLite Database", self._check_sqlite),
            ("FAISS Index", self._check_faiss),
            ("API Server", self._check_api_server),
            ("GPU Support", self._check_gpu),
        ]

        # Add checks to display
        check_items = []
        for name, _ in checks:
            check_items.append(doctor.add_check(name))

        results = {
            "ok": 0,
            "warn": 0,
            "fail": 0,
        }

        # Check if we're in a TTY for animations
        is_tty = sys.stdout.isatty()

        if is_tty:
            with doctor.live_display():
                for i, (name, check_fn) in enumerate(checks):
                    check = check_items[i]

                    # Set to checking state
                    check.status = "checking"
                    check.message = "Checking..."
                    doctor.update()

                    # Animate a bit
                    for _ in range(6):
                        time.sleep(0.08)
                        doctor.update()

                    # Run the actual check
                    try:
                        status, message, details = check_fn(verbose)
                        check.status = status
                        check.message = message
                        check.details = details if verbose else ""
                        results[status] = results.get(status, 0) + 1
                    except Exception as e:
                        check.status = "fail"
                        check.message = f"Check failed: {e}"
                        results["fail"] += 1

                    doctor.update()
                    time.sleep(0.1)
        else:
            # Non-TTY mode - run checks without animation
            for i, (name, check_fn) in enumerate(checks):
                check = check_items[i]

                try:
                    status, message, details = check_fn(verbose)
                    check.status = status
                    check.message = message
                    check.details = details if verbose else ""
                    results[status] = results.get(status, 0) + 1
                except Exception as e:
                    check.status = "fail"
                    check.message = f"Check failed: {e}"
                    results["fail"] += 1

            # Print the final panel
            console.print(doctor.render())

        # Print summary
        console.print()
        self._print_summary(results)

        return results["fail"] == 0

    def _check_python(self, verbose: bool) -> tuple[str, str, str]:
        """Check Python version and environment."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 12:
            return "ok", f"Python {version_str}", f"Executable: {sys.executable}"
        elif version.major == 3 and version.minor >= 10:
            return "warn", f"Python {version_str} (3.12+ recommended)", ""
        else:
            return "fail", f"Python {version_str} (requires 3.12+)", ""

    def _check_dependencies(self, verbose: bool) -> tuple[str, str, str]:
        """Check if key dependencies are installed."""
        required = ["fastapi", "uvicorn", "langchain", "faiss", "rich", "httpx"]
        missing = []

        for pkg in required:
            try:
                if pkg == "faiss":
                    import faiss  # noqa: F401
                else:
                    __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)

        if not missing:
            return "ok", "All dependencies installed", ""
        else:
            return "fail", f"Missing: {', '.join(missing)}", "Run: uv sync"

    def _check_ollama(self, verbose: bool) -> tuple[str, str, str]:
        """Check if Ollama server is running."""
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                return "ok", "Ollama server running", "http://localhost:11434"
            else:
                return "warn", f"Ollama returned {response.status_code}", ""
        except httpx.ConnectError:
            return "fail", "Ollama not running", "Run: ollama serve"
        except Exception as e:
            return "fail", f"Cannot connect: {e}", ""

    def _check_ollama_models(self, verbose: bool) -> tuple[str, str, str]:
        """Check if required Ollama models are available."""
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                return "warn", "Cannot check models", "Ollama not responding"

            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]

            # Check for required models (from config)
            required = ["gemma3", "mxbai-embed-large"]
            found = []
            missing = []

            for req in required:
                # Check if any model starts with the required name
                if any(m.startswith(req.split(":")[0]) for m in models):
                    found.append(req)
                else:
                    missing.append(req)

            if not missing:
                return "ok", f"{len(found)} models available", ", ".join(found)
            else:
                return (
                    "warn",
                    f"Missing models: {', '.join(missing)}",
                    f"Run: ollama pull {missing[0]}",
                )
        except Exception as e:
            return "warn", "Cannot check models", str(e)

    def _check_sqlite(self, verbose: bool) -> tuple[str, str, str]:
        """Check SQLite database status."""
        db_path = Path(self.config.project_root) / "data" / "db" / "app.db"

        if not db_path.exists():
            return "warn", "Database not created", "Restart server to initialize: /stop then /start"

        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Count documents and chunks
            cursor.execute("SELECT COUNT(*) FROM documents")
            docs = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunks = cursor.fetchone()[0]

            conn.close()

            return "ok", f"{docs} docs, {chunks} chunks", str(db_path)
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return "fail", "Schema not initialized", "Restart server: /stop then /start"
            return "fail", f"Database error: {e}", ""
        except Exception as e:
            return "fail", f"Database error: {e}", ""

    def _check_faiss(self, verbose: bool) -> tuple[str, str, str]:
        """Check FAISS index status."""
        index_path = Path(self.config.project_root) / "data" / "index" / "faiss" / "index.faiss"

        if not index_path.exists():
            return "warn", "Index not created", "Will be created on first ingest"

        try:
            import faiss

            index = faiss.read_index(str(index_path))
            ntotal = index.ntotal
            d = getattr(index, "d", None)

            details = f"Vectors: {ntotal}"
            if d:
                details += f", Dim: {d}"

            return "ok", f"{ntotal} vectors indexed", details
        except Exception as e:
            return "fail", f"Index error: {e}", ""

    def _check_api_server(self, verbose: bool) -> tuple[str, str, str]:
        """Check if the API server is running."""
        response = self.api.health()

        if response.success:
            return "ok", "Server running", self.config.base_url
        else:
            return "warn", "Server not running", "Use /start to launch"

    def _check_gpu(self, verbose: bool) -> tuple[str, str, str]:
        """Check GPU availability for FAISS."""
        try:
            import faiss

            if hasattr(faiss, "get_num_gpus"):
                num_gpus = faiss.get_num_gpus()
                if num_gpus > 0:
                    return "ok", f"{num_gpus} GPU(s) available", "FAISS GPU acceleration enabled"
                else:
                    return "warn", "No GPUs detected", "Using CPU mode"
            else:
                return "warn", "FAISS CPU build", "Install faiss-gpu for GPU support"
        except ImportError:
            return "fail", "FAISS not installed", ""
        except Exception as e:
            return "warn", f"GPU check failed: {e}", ""

    def _print_summary(self, results: dict[str, int]) -> None:
        """Print the summary panel."""
        sum(results.values())

        # Determine overall status
        if results["fail"] > 0:
            status_icon = "✖"
            status_text = "Issues Found"
            status_style = "error"
            border_style = "red"
        elif results["warn"] > 0:
            status_icon = "⚠"
            status_text = "Warnings"
            status_style = "warning"
            border_style = "yellow"
        else:
            status_icon = "✔"
            status_text = "All Checks Passed"
            status_style = "success"
            border_style = "green"

        # Build summary text
        text = Text()
        text.append(f"{status_icon} {status_text}\n\n", style=f"{status_style} bold")

        text.append("  ● ", style="doctor.ok")
        text.append(f"{results['ok']} passed", style="doctor.ok")
        text.append("   ", style="")

        if results["warn"] > 0:
            text.append("● ", style="doctor.warn")
            text.append(f"{results['warn']} warnings", style="doctor.warn")
            text.append("   ", style="")

        if results["fail"] > 0:
            text.append("● ", style="doctor.fail")
            text.append(f"{results['fail']} failed", style="doctor.fail")

        console.print(
            Panel(
                Align.center(text),
                border_style=border_style,
                padding=(1, 4),
            )
        )

        # Suggestions
        if results["fail"] > 0 or results["warn"] > 0:
            console.print()
            console.print(
                "  [muted]Run [/muted][command]/doctor --verbose[/command][muted] for more details[/muted]"
            )
