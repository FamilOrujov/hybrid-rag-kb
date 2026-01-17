"""Theme and color definitions for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from rich.style import Style
from rich.theme import Theme as RichTheme


@dataclass
class Theme:
    """Color theme for the CLI - Purple & Orange gradient palette."""
    
    # Primary colors (Purple-Orange theme)
    primary: str = "#C77DFF"      # Light Purple - main accent
    secondary: str = "#FF8C42"    # Orange - secondary accent
    tertiary: str = "#9D4EDD"     # Purple - tertiary accent
    
    # Status colors
    success: str = "#00E676"      # Bright green
    error: str = "#FF5252"        # Red
    warning: str = "#FFB347"      # Orange-yellow
    info: str = "#B388FF"         # Light purple
    
    # Text colors
    text: str = "#E8E8E8"         # Light gray
    muted: str = "#888888"        # Muted gray
    highlight: str = "#FFFFFF"    # White
    
    # UI colors
    border: str = "#5D4E6D"       # Purple-gray border
    panel_bg: str = "#1A1625"     # Dark purple background
    
    # Logo gradient colors (Purple to Orange)
    logo_1: str = "#FF6B35"       # Bright Orange
    logo_2: str = "#FF8C42"       # Orange
    logo_3: str = "#FFB347"       # Light Orange
    logo_4: str = "#C77DFF"       # Light Purple
    logo_5: str = "#9D4EDD"       # Purple
    logo_6: str = "#7B2CBF"       # Deep Purple
    
    def to_rich_theme(self) -> RichTheme:
        """Convert to Rich theme."""
        return RichTheme({
            "primary": Style(color=self.primary),
            "secondary": Style(color=self.secondary),
            "tertiary": Style(color=self.tertiary),
            "primary.bold": Style(color=self.primary, bold=True),
            "secondary.bold": Style(color=self.secondary, bold=True),
            "tertiary.bold": Style(color=self.tertiary, bold=True),
            "success": Style(color=self.success, bold=True),
            "error": Style(color=self.error, bold=True),
            "warning": Style(color=self.warning),
            "info": Style(color=self.info),
            "text": Style(color=self.text),
            "muted": Style(color=self.muted),
            "highlight": Style(color=self.highlight, bold=True),
            "border": Style(color=self.border),
            "logo.1": Style(color=self.logo_1, bold=True),
            "logo.2": Style(color=self.logo_2, bold=True),
            "logo.3": Style(color=self.logo_3, bold=True),
            "logo.4": Style(color=self.logo_4, bold=True),
            "logo.5": Style(color=self.logo_5, bold=True),
            "logo.6": Style(color=self.logo_6, bold=True),
            "command": Style(color=self.primary, bold=True),
            "path": Style(color=self.secondary),
            "number": Style(color=self.warning),
            "citation": Style(color="#00CED1", bold=True),  # Dark Cyan - distinct from text
            "orange": Style(color="#FF8C42"),
            "purple": Style(color="#9D4EDD"),
            "doctor.ok": Style(color="#00E676", bold=True),
            "doctor.warn": Style(color="#FFB347", bold=True),
            "doctor.fail": Style(color="#FF5252", bold=True),
        })


# Default theme instance
_theme = Theme()


def get_theme() -> Theme:
    """Get the current theme."""
    return _theme


def set_theme(theme: Theme) -> None:
    """Set the current theme."""
    global _theme
    _theme = theme
