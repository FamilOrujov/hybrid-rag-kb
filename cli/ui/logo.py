"""ASCII art logo for Hybrid RAG CLI - Minecraft/Keploy style with animations."""

from __future__ import annotations

import time
import sys
from rich.text import Text
from rich.align import Align
from rich.panel import Panel
from rich.console import Group
from rich.live import Live
from cli.ui.console import console


LOGO_LINES = [
    "██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗     ██████╗  █████╗  ██████╗ ",
    "██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗    ██╔══██╗██╔══██╗██╔════╝ ",
    "███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║    ██████╔╝███████║██║  ███╗",
    "██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║    ██╔══██╗██╔══██║██║   ██║",
    "██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝    ██║  ██║██║  ██║╚██████╔╝",
    "╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ",
]

# Smaller subtitle - Knowledge Base
SUBTITLE_LINES = [
    "╦╔═╔╗╔╔═╗╦ ╦╦  ╔═╗╔╦╗╔═╗╔═╗  ╔╗ ╔═╗╔═╗╔═╗",
    "╠╩╗║║║║ ║║║║║  ║╣  ║║║ ╦║╣   ╠╩╗╠═╣╚═╗║╣ ",
    "╩ ╩╝╚╝╚═╝╚╩╝╩═╝╚═╝═╩╝╚═╝╚═╝  ╚═╝╩ ╩╚═╝╚═╝",
]

# Purple-Orange gradient colors for logo (top to bottom)
GRADIENT_COLORS = [
    "#FF6B35",  # Bright Orange
    "#FF8C42",  # Orange
    "#FFB347",  # Light Orange
    "#C77DFF",  # Light Purple
    "#9D4EDD",  # Purple
    "#7B2CBF",  # Deep Purple
]

# Subtitle colors (more muted purple-orange)
SUBTITLE_COLORS = [
    "#E07B53",  # Muted Orange
    "#B57EDC",  # Muted Purple
    "#9D6BC7",  # Deeper Muted Purple
]

VERSION = "v0.1.0"


def create_logo_text(animate_index: int = -1) -> Text:
    """Create the styled logo text with purple-orange gradient."""
    text = Text()
    
    for i, line in enumerate(LOGO_LINES):
        if animate_index >= 0 and i > animate_index:
            # Don't show lines not yet animated
            continue
        color = GRADIENT_COLORS[i % len(GRADIENT_COLORS)]
        text.append(line, style=f"bold {color}")
        if i < len(LOGO_LINES) - 1 and (animate_index < 0 or i < animate_index):
            text.append("\n")
        elif animate_index >= 0 and i == animate_index:
            text.append("\n")
    
    return text


def create_subtitle_text(animate_index: int = -1) -> Text:
    """Create the styled subtitle text."""
    text = Text()
    
    for i, line in enumerate(SUBTITLE_LINES):
        if animate_index >= 0 and i > animate_index:
            continue
        color = SUBTITLE_COLORS[i % len(SUBTITLE_COLORS)]
        text.append(line, style=f"{color}")
        if i < len(SUBTITLE_LINES) - 1:
            text.append("\n")
    
    return text


def create_full_logo(show_tagline: bool = True, show_commands: bool = True) -> Group:
    """Create the full logo with all elements."""
    elements = []
    
    # Main logo
    logo_text = create_logo_text()
    elements.append(Align.center(logo_text))
    elements.append(Text())
    
    # Subtitle
    subtitle_text = create_subtitle_text()
    elements.append(Align.center(subtitle_text))
    
    if show_tagline:
        elements.append(Text())
        
        # Tagline with gradient
        tagline = Text()
        tagline.append("◆ ", style="#FF8C42")
        tagline.append("BM25 + FAISS Hybrid Retrieval", style="#FFB347")
        tagline.append(" │ ", style="#888888")
        tagline.append("Local LLM", style="#C77DFF")
        tagline.append(" │ ", style="#888888")
        tagline.append("Citation Enforcement", style="#9D4EDD")
        tagline.append(" ◆", style="#FF8C42")
        elements.append(Align.center(tagline))
        
        # Version line
        elements.append(Text())
        version = Text()
        version.append("─" * 15, style="#555555")
        version.append(f" {VERSION} ", style="#888888")
        version.append("─" * 15, style="#555555")
        elements.append(Align.center(version))
    
    if show_commands:
        elements.append(Text())
        
        # Command hints with better styling
        commands_text = Text()
        commands_text.append("Commands: ", style="#888888")
        
        cmds = ["/start", "/query", "/ingest", "/stats", "/doctor", "/help", "/quit"]
        for i, cmd in enumerate(cmds):
            if i > 0:
                commands_text.append(" • ", style="#555555")
            commands_text.append(cmd, style="#C77DFF bold")
        
        elements.append(Align.center(commands_text))
    
    return Group(*elements)


def print_logo_animated(show_tagline: bool = True, show_commands: bool = True) -> None:
    """Print the logo with a smooth reveal animation."""
    # Check if we're in a real terminal
    if not sys.stdout.isatty():
        # No animation, just print
        print_logo(show_tagline, show_commands)
        return
    
    console.print()
    
    # Animate main logo lines
    with Live(console=console, refresh_per_second=30, transient=True) as live:
        # Reveal each line of main logo
        for i in range(len(LOGO_LINES)):
            text = create_logo_text(animate_index=i)
            live.update(Align.center(text))
            time.sleep(0.06)
        
        time.sleep(0.1)
    
    # Print final static version
    console.print(create_full_logo(show_tagline, show_commands))
    console.print()


def print_logo(show_tagline: bool = True, show_commands: bool = True) -> None:
    """Print the Hybrid RAG logo centered on screen (no animation)."""
    console.print()
    console.print(create_full_logo(show_tagline, show_commands))
    console.print()


def print_mini_logo() -> None:
    """Print a smaller inline logo."""
    text = Text()
    text.append("◆ ", style="#FF8C42")
    text.append("HYBRID", style="#FF6B35 bold")
    text.append(" ", style="")
    text.append("RAG", style="#9D4EDD bold")
    text.append(" ◆", style="#FF8C42")
    console.print(Align.center(text))


def print_minimal_logo() -> None:
    """Print minimal logo for clear screen - just the main logo."""
    console.print()
    logo_text = create_logo_text()
    console.print(Align.center(logo_text))
    console.print()
