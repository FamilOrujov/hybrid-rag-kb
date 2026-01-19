"""Modern animations and visual effects for the CLI."""

from __future__ import annotations

import random
import sys
import time

from rich.align import Align
from rich.live import Live
from rich.text import Text

from cli.ui.console import console

# Animation frame sets
SPINNER_DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_BOUNCE = ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]
SPINNER_PULSE = ["◜", "◠", "◝", "◞", "◡", "◟"]
SPINNER_ORBIT = ["◐", "◓", "◑", "◒"]
SPINNER_BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]
SPINNER_ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
SPARKLE_FRAMES = ["✦", "✧", "★", "✧"]
WAVE_FRAMES = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂", "▁"]

# Color gradients
GRADIENT_PURPLE_ORANGE = ["#FF6B35", "#FF8C42", "#FFB347", "#C77DFF", "#9D4EDD", "#7B2CBF"]
GRADIENT_CYAN = ["#00CED1", "#00E5EE", "#00FFFF", "#00E5EE", "#00CED1"]


def animate_command_start(command: str) -> None:
    """Subtle animation when a command starts executing."""
    # Disabled - no animation needed
    pass


def animate_status_check(api, base_url: str) -> bool:
    """Check server status."""
    try:
        response = api.health()
        return response.success
    except Exception:
        return False


def print_welcome_tip() -> None:
    """Print a random helpful tip on startup."""
    tips = [
        "Type any question directly to query the knowledge base",
        "Use /start to launch the server in background mode",
        "Use /ingest <file> to add documents to your knowledge base",
        "Use /debug retrieval <query> to analyze search results",
        "Press Tab for command auto-completion",
        "Run /doctor to check system health",
        "Use /model set to switch LLM models",
    ]

    tip = random.choice(tips)

    text = Text()
    text.append("  * ", style="#C77DFF")
    text.append(tip, style="#888888")
    console.print(text)
    console.print()


def animate_processing(message: str, duration: float = 1.0) -> None:
    """Show an animated processing indicator."""
    if not sys.stdout.isatty():
        console.print(f"  ● {message}")
        return

    frames_count = int(duration * 15)

    with Live(console=console, refresh_per_second=15, transient=True) as live:
        for i in range(frames_count):
            frame = SPINNER_DOTS[i % len(SPINNER_DOTS)]
            color_idx = i % len(GRADIENT_PURPLE_ORANGE)
            color = GRADIENT_PURPLE_ORANGE[color_idx]

            text = Text()
            text.append(f"  {frame} ", style=color)
            text.append(message, style="#e8e8e8")
            live.update(text)
            time.sleep(1 / 15)


def animate_success(message: str) -> None:
    """Animated success message."""
    if not sys.stdout.isatty():
        console.print(f"  [success]✔[/success] {message}")
        return

    # Quick success flash
    with Live(console=console, refresh_per_second=20, transient=True) as live:
        for char in ["○", "◔", "◑", "◕", "●", "✔"]:
            style = "#00E676" if char == "✔" else "#555555"
            text = Text()
            text.append(f"  {char} ", style=style)
            text.append(message, style="#e8e8e8" if char != "✔" else "#00E676")
            live.update(text)
            time.sleep(0.04)

    console.print(f"  [success]✔[/success] {message}")


def animate_error(message: str) -> None:
    """Animated error message."""
    if not sys.stdout.isatty():
        console.print(f"  [error]✖[/error] {message}")
        return

    # Quick error flash
    with Live(console=console, refresh_per_second=20, transient=True) as live:
        for i in range(3):
            text = Text()
            text.append("  ✖ ", style="#FF5252" if i % 2 == 0 else "#AA3333")
            text.append(message, style="#FF5252" if i % 2 == 0 else "#AA3333")
            live.update(text)
            time.sleep(0.08)

    console.print(f"  [error]✖[/error] {message}")


def create_gradient_bar(progress: float, width: int = 30) -> Text:
    """Create a gradient progress bar."""
    filled = int(progress * width)
    empty = width - filled

    text = Text()
    text.append("│", style="#555555")

    for i in range(filled):
        color_idx = int((i / width) * len(GRADIENT_PURPLE_ORANGE))
        color_idx = min(color_idx, len(GRADIENT_PURPLE_ORANGE) - 1)
        text.append("█", style=GRADIENT_PURPLE_ORANGE[color_idx])

    text.append("░" * empty, style="#333333")
    text.append("│", style="#555555")

    return text


def animate_wave_text(message: str, style: str = "#C77DFF") -> None:
    """Display text with a wave animation effect."""
    if not sys.stdout.isatty():
        console.print(f"[{style}]{message}[/{style}]")
        return

    with Live(console=console, refresh_per_second=20, transient=True) as live:
        for offset in range(len(message) + 5):
            text = Text()
            for i, char in enumerate(message):
                wave_pos = (i - offset) % 8
                if wave_pos < 4:
                    char_style = style
                else:
                    char_style = "#888888"
                text.append(char, style=char_style)
            live.update(Align.center(text))
            time.sleep(0.05)

    console.print(Align.center(Text(message, style=style)))


class ModernSpinner:
    """A modern animated spinner with multiple styles."""

    STYLES = {
        "dots": SPINNER_DOTS,
        "bounce": SPINNER_BOUNCE,
        "pulse": SPINNER_PULSE,
        "orbit": SPINNER_ORBIT,
        "blocks": SPINNER_BLOCKS,
        "arrows": SPINNER_ARROWS,
    }

    def __init__(
        self,
        message: str,
        style: str = "dots",
        color: str = "#C77DFF",
    ):
        self.message = message
        self.frames = self.STYLES.get(style, SPINNER_DOTS)
        self.color = color
        self.frame_idx = 0
        self.live: Live | None = None
        self.start_time = 0.0

    def __enter__(self) -> ModernSpinner:
        if sys.stdout.isatty():
            self.start_time = time.time()
            self.live = Live(
                self._render(),
                console=console,
                refresh_per_second=12,
                transient=True,
            )
            self.live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self.live:
            self.live.__exit__(*args)

    def _render(self) -> Text:
        frame = self.frames[self.frame_idx % len(self.frames)]
        elapsed = time.time() - self.start_time

        text = Text()
        text.append(f"  {frame} ", style=self.color)
        text.append(self.message, style="#e8e8e8")
        text.append(f" ({elapsed:.1f}s)", style="#555555")

        return text

    def update(self, message: str | None = None) -> None:
        """Update the spinner display."""
        if message:
            self.message = message
        self.frame_idx += 1
        if self.live:
            self.live.update(self._render())


class TypewriterText:
    """Display text with a typewriter animation effect."""

    def __init__(
        self,
        text: str,
        style: str = "#e8e8e8",
        speed: float = 0.03,
        cursor: str = "▌",
    ):
        self.text = text
        self.style = style
        self.speed = speed
        self.cursor = cursor

    def display(self) -> None:
        """Display the text with typewriter effect."""
        if not sys.stdout.isatty():
            console.print(f"[{self.style}]{self.text}[/{self.style}]")
            return

        with Live(console=console, refresh_per_second=60, transient=True) as live:
            revealed = ""
            for i, char in enumerate(self.text):
                revealed += char
                display_text = Text()
                display_text.append(revealed, style=self.style)

                # Blinking cursor
                if i < len(self.text) - 1:
                    cursor_style = self.style if (i % 2 == 0) else "#555555"
                    display_text.append(self.cursor, style=cursor_style)

                live.update(display_text)
                time.sleep(self.speed)

        console.print(f"[{self.style}]{self.text}[/{self.style}]")


def shimmer_text(text: str, base_color: str = "#888888", highlight_color: str = "#C77DFF") -> None:
    """Display text with a shimmer/highlight sweep effect."""
    if not sys.stdout.isatty():
        console.print(f"[{highlight_color}]{text}[/{highlight_color}]")
        return

    text_len = len(text)
    sweep_width = 3

    with Live(console=console, refresh_per_second=30, transient=True) as live:
        for pos in range(-sweep_width, text_len + sweep_width):
            display = Text()
            for i, char in enumerate(text):
                if pos <= i < pos + sweep_width:
                    display.append(char, style=highlight_color)
                else:
                    display.append(char, style=base_color)
            live.update(Align.center(display))
            time.sleep(0.02)

    console.print(Align.center(Text(text, style=highlight_color)))
