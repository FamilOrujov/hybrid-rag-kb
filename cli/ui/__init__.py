"""UI components for the Hybrid RAG CLI."""

from cli.ui.animations import (
    ModernSpinner,
    TypewriterText,
    animate_command_start,
    animate_error,
    animate_processing,
    animate_status_check,
    animate_success,
    print_welcome_tip,
)
from cli.ui.console import (
    console,
    print_accent,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from cli.ui.logo import print_logo, print_logo_animated, print_mini_logo, print_minimal_logo
from cli.ui.panels import (
    create_debug_panel,
    create_query_result_panel,
    create_sources_panel,
    create_stats_panel,
)
from cli.ui.spinners import AnimatedCheck, DoctorAnimation, create_spinner
from cli.ui.theme import Theme, get_theme

__all__ = [
    # Theme
    "Theme",
    "get_theme",
    # Console
    "console",
    "print_error",
    "print_success",
    "print_warning",
    "print_info",
    "print_accent",
    # Logo
    "print_logo",
    "print_logo_animated",
    "print_mini_logo",
    "print_minimal_logo",
    # Panels
    "create_stats_panel",
    "create_query_result_panel",
    "create_sources_panel",
    "create_debug_panel",
    # Spinners
    "create_spinner",
    "DoctorAnimation",
    "AnimatedCheck",
    # Animations
    "animate_command_start",
    "animate_status_check",
    "animate_processing",
    "animate_success",
    "animate_error",
    "print_welcome_tip",
    "ModernSpinner",
    "TypewriterText",
]
