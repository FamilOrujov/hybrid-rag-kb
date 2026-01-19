"""UI components for the Hybrid RAG CLI."""

from cli.ui.theme import Theme, get_theme
from cli.ui.console import console, print_error, print_success, print_warning, print_info, print_accent
from cli.ui.logo import print_logo, print_logo_animated, print_mini_logo, print_minimal_logo
from cli.ui.panels import (
    create_stats_panel,
    create_query_result_panel,
    create_sources_panel,
    create_debug_panel,
)
from cli.ui.spinners import create_spinner, DoctorAnimation, AnimatedCheck
from cli.ui.animations import (
    animate_command_start,
    animate_status_check,
    animate_processing,
    animate_success,
    animate_error,
    print_welcome_tip,
    ModernSpinner,
    TypewriterText,
)

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
