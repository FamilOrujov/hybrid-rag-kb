"""Spinner and progress indicators for the CLI with smooth animations."""

from __future__ import annotations

import time
import sys
from contextlib import contextmanager
from typing import Generator, Callable, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.console import Group
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from cli.ui.console import console


# Custom spinner styles with better visuals
SPINNER_STYLES = {
    "default": "dots",
    "loading": "dots12",
    "processing": "arc",
    "server": "bouncingBar",
    "thinking": "moon",
    "doctor": "dots8Bit",
    "pulse": "point",
    "bounce": "bouncingBall",
}

# Animated frames for custom animations
LOADING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PULSE_FRAMES = ["●", "◉", "○", "◉"]
CHECK_FRAMES = ["◯", "◔", "◑", "◕", "●"]
PROGRESS_CHARS = "▏▎▍▌▋▊▉█"


@contextmanager
def create_spinner(
    message: str,
    style: str = "default",
    success_message: str | None = None,
    error_message: str | None = None,
) -> Generator[None, None, None]:
    """Context manager for showing a spinner during an operation."""
    
    spinner_type = SPINNER_STYLES.get(style, "dots")
    
    with console.status(
        f"[primary]{message}[/primary]",
        spinner=spinner_type,
        spinner_style="primary",
    ) as status:
        try:
            yield
            if success_message:
                console.print(f"[success]✔ {success_message}[/success]")
        except Exception as e:
            if error_message:
                console.print(f"[error]✖ {error_message}: {e}[/error]")
            raise


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a styled progress bar."""
    return Progress(
        SpinnerColumn(spinner_name="dots", style="primary"),
        TextColumn("[primary]{task.description}[/primary]"),
        BarColumn(bar_width=40, style="muted", complete_style="primary"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def animate_text_reveal(text: str, style: str = "primary", delay: float = 0.02) -> None:
    """Animate text appearing character by character."""
    if not sys.stdout.isatty():
        console.print(f"[{style}]{text}[/{style}]")
        return
    
    with Live(console=console, refresh_per_second=60, transient=True) as live:
        revealed = ""
        for char in text:
            revealed += char
            live.update(Text(revealed, style=style))
            time.sleep(delay)
    console.print(f"[{style}]{text}[/{style}]")


def animate_progress_bar(
    total: int,
    description: str = "Processing",
    callback: Callable[[int], Any] | None = None,
) -> None:
    """Animate a progress bar with optional callback per step."""
    with create_progress_bar(description) as progress:
        task = progress.add_task(description, total=total)
        for i in range(total):
            if callback:
                callback(i)
            progress.update(task, advance=1)
            time.sleep(0.05)


class AnimatedCheck:
    """Animated checkmark for doctor-style checks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status: str = "pending"  # pending, checking, ok, warn, fail
        self.message: str = ""
        self.details: str = ""
    
    def render(self, frame: int = 0) -> Text:
        """Render the check item."""
        text = Text()
        
        if self.status == "pending":
            text.append("  ○ ", style="#555555")
        elif self.status == "checking":
            spinner = LOADING_FRAMES[frame % len(LOADING_FRAMES)]
            text.append(f"  {spinner} ", style="#C77DFF")
        elif self.status == "ok":
            text.append("  ✔ ", style="doctor.ok")
        elif self.status == "warn":
            text.append("  ⚠ ", style="doctor.warn")
        elif self.status == "fail":
            text.append("  ✖ ", style="doctor.fail")
        
        # Name
        name_style = "text" if self.status != "checking" else "primary"
        text.append(self.name, style=name_style)
        
        # Message
        if self.message:
            text.append(" — ", style="muted")
            msg_style = {
                "ok": "doctor.ok",
                "warn": "doctor.warn", 
                "fail": "doctor.fail",
                "checking": "muted",
            }.get(self.status, "muted")
            text.append(self.message, style=msg_style)
        
        # Details on new line
        if self.details:
            text.append(f"\n      {self.details}", style="muted")
        
        return text


class DoctorAnimation:
    """Animated doctor check display."""
    
    def __init__(self, title: str = "System Check"):
        self.title = title
        self.checks: list[AnimatedCheck] = []
        self.frame = 0
        self.live: Live | None = None
    
    def add_check(self, name: str, description: str = "") -> AnimatedCheck:
        """Add a check to the list."""
        check = AnimatedCheck(name, description)
        self.checks.append(check)
        return check
    
    def render(self) -> Panel:
        """Render all checks."""
        elements = []
        
        for check in self.checks:
            elements.append(check.render(self.frame))
        
        content = Group(*elements) if elements else Text("No checks", style="muted")
        
        return Panel(
            content,
            title=f"[primary]🩺 {self.title}[/primary]",
            border_style="primary",
            padding=(1, 2),
        )
    
    def update(self) -> None:
        """Update the display."""
        self.frame += 1
        if self.live:
            self.live.update(self.render())
    
    @contextmanager
    def live_display(self) -> Generator["DoctorAnimation", None, None]:
        """Context manager for live display."""
        if not sys.stdout.isatty():
            yield self
            return
        
        with Live(self.render(), console=console, refresh_per_second=12) as live:
            self.live = live
            yield self
            self.live = None


class ServerOutputDisplay:
    """Live display for server output with styled formatting."""
    
    def __init__(self):
        self.lines: list[Text] = []
        self.max_lines = 20
        self.live: Live | None = None
    
    def add_line(self, line: str) -> None:
        """Add a line to the display."""
        styled = self._style_line(line)
        self.lines.append(styled)
        
        # Keep only last N lines
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]
        
        if self.live:
            self.live.update(self._render())
    
    def _style_line(self, line: str) -> Text:
        """Apply styling to a log line."""
        text = Text()
        
        # Color based on content
        if "ERROR" in line or "error" in line.lower():
            text.append(line, style="error")
        elif "WARNING" in line or "warning" in line.lower():
            text.append(line, style="warning")
        elif "INFO" in line:
            text.append(line, style="info")
        elif "Started" in line or "Uvicorn running" in line:
            text.append(line, style="success")
        elif "GET" in line or "POST" in line:
            # HTTP request lines
            text.append(line, style="tertiary")
        else:
            text.append(line, style="muted")
        
        return text
    
    def _render(self) -> Group:
        """Render all lines."""
        return Group(*self.lines)
    
    @contextmanager
    def live_display(self) -> Generator["ServerOutputDisplay", None, None]:
        """Context manager for live display."""
        with Live(self._render(), console=console, refresh_per_second=10) as live:
            self.live = live
            yield self
            self.live = None
