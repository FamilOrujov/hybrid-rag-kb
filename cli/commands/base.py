"""Base command class for CLI commands."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from cli.core.api_client import APIClient
from cli.core.config import CLIConfig


class BaseCommand(ABC):
    """Base class for all CLI commands."""
    
    name: str = "base"
    description: str = "Base command"
    usage: str = ""
    aliases: list[str] = []
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.api = APIClient(config.base_url)
    
    @abstractmethod
    def execute(self, args: list[str]) -> bool:
        """
        Execute the command.
        
        Args:
            args: Command arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def parse_flags(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        """Parse command-line flags from arguments."""
        flags = {}
        remaining = []
        
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                key = arg[2:]
                if "=" in key:
                    key, value = key.split("=", 1)
                    flags[key] = value
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    flags[key] = args[i + 1]
                    i += 1
                else:
                    flags[key] = True
            elif arg.startswith("-") and len(arg) == 2:
                key = arg[1]
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    flags[key] = args[i + 1]
                    i += 1
                else:
                    flags[key] = True
            else:
                remaining.append(arg)
            i += 1
        
        return flags, remaining
