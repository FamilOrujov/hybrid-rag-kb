"""Rich console instance and helper functions with modern styling."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich import box
from cli.ui.theme import get_theme


# Create the global console with our theme
console = Console(theme=get_theme().to_rich_theme(), highlight=True)


def print_error(message: str, title: str = "Error") -> None:
    """Print an error message with modern styling."""
    content = Text()
    content.append(message, style="#FF5252")
    
    console.print(Panel(
        content,
        title=f"[#FF5252 bold]✖ {title}[/#FF5252 bold]",
        border_style="#FF5252",
        box=box.ROUNDED,
        padding=(0, 2),
    ))


def print_success(message: str, title: str = "Success") -> None:
    """Print a success message with modern styling."""
    content = Text()
    content.append(message, style="#00E676")
    
    console.print(Panel(
        content,
        title=f"[#00E676 bold]✔ {title}[/#00E676 bold]",
        border_style="#00E676",
        box=box.ROUNDED,
        padding=(0, 2),
    ))


def print_warning(message: str, title: str = "Warning") -> None:
    """Print a warning message with modern styling."""
    content = Text()
    content.append(message, style="#FFB347")
    
    console.print(Panel(
        content,
        title=f"[#FFB347 bold]⚠ {title}[/#FFB347 bold]",
        border_style="#FFB347",
        box=box.ROUNDED,
        padding=(0, 2),
    ))


def print_info(message: str, title: str = "Info") -> None:
    """Print an info message with modern styling."""
    content = Text()
    content.append(message, style="#B388FF")
    
    console.print(Panel(
        content,
        title=f"[#B388FF bold]ℹ {title}[/#B388FF bold]",
        border_style="#B388FF",
        box=box.ROUNDED,
        padding=(0, 2),
    ))


def print_accent(message: str, title: str = "") -> None:
    """Print an accent message with cyan styling."""
    content = Text()
    content.append(message, style="#00CED1")
    
    if title:
        console.print(Panel(
            content,
            title=f"[#00CED1 bold]{title}[/#00CED1 bold]",
            border_style="#00CED1",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
    else:
        console.print(content)
