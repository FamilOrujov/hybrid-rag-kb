"""Rich console instance and helper functions."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from cli.ui.theme import get_theme


# Create the global console with our theme
console = Console(theme=get_theme().to_rich_theme(), highlight=True)


def print_error(message: str, title: str = "Error") -> None:
    """Print an error message."""
    console.print(Panel(
        Text(message, style="error"),
        title=f"[error]✖ {title}[/error]",
        border_style="red",
        padding=(0, 1),
    ))


def print_success(message: str, title: str = "Success") -> None:
    """Print a success message."""
    console.print(Panel(
        Text(message, style="success"),
        title=f"[success]✔ {title}[/success]",
        border_style="green",
        padding=(0, 1),
    ))


def print_warning(message: str, title: str = "Warning") -> None:
    """Print a warning message."""
    console.print(Panel(
        Text(message, style="warning"),
        title=f"[warning]⚠ {title}[/warning]",
        border_style="yellow",
        padding=(0, 1),
    ))


def print_info(message: str, title: str = "Info") -> None:
    """Print an info message."""
    console.print(Panel(
        Text(message, style="info"),
        title=f"[info]ℹ {title}[/info]",
        border_style="blue",
        padding=(0, 1),
    ))
