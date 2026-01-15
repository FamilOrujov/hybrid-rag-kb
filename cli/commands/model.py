"""Model management command - list and select Ollama models."""

from __future__ import annotations

import httpx
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.console import Group

from cli.commands.base import BaseCommand
from cli.ui.console import console, print_error, print_success, print_warning
from cli.ui.spinners import create_spinner


# Longer timeout for model operations (model loading can be slow)
MODEL_TIMEOUT = 300.0  # 5 minutes


class ModelCommand(BaseCommand):
    """Manage LLM models for chat and embeddings."""
    
    name = "model"
    description = "List and select Ollama models for chat and embeddings"
    usage = "/model [list|set|info] [--chat MODEL] [--embed MODEL]"
    aliases = ["models", "llm"]
    
    def execute(self, args: list[str]) -> bool:
        """Execute model command."""
        flags, remaining = self.parse_flags(args)
        
        # Direct model setting via flags
        if flags.get("chat") or flags.get("embed"):
            return self._set_models(
                chat_model=flags.get("chat"),
                embed_model=flags.get("embed"),
            )
        
        if not remaining:
            # Default: show current models and available options
            return self._show_models()
        
        subcommand = remaining[0].lower()
        
        if subcommand in ["list", "ls", "l"]:
            return self._list_models()
        elif subcommand in ["set", "use", "select"]:
            return self._interactive_select()
        elif subcommand in ["info", "current", "status"]:
            return self._show_current()
        else:
            # Treat as model name for quick set
            self._show_model_help()
            return True
    
    def _show_models(self) -> bool:
        """Show current models and available options."""
        with create_spinner("Fetching models from Ollama...", style="processing"):
            response = self.api._request("GET", "/models")
        
        if not response.success:
            # Try querying Ollama directly
            return self._list_models_direct()
        
        data = response.data or {}
        current = data.get("current", {})
        available = data.get("available", {})
        error = data.get("error")
        
        console.print()
        
        # Current configuration
        current_text = Text()
        current_text.append("Chat Model:  ", style="muted")
        current_text.append(current.get("chat_model", "N/A"), style="primary.bold")
        current_text.append("\n")
        current_text.append("Embed Model: ", style="muted")
        current_text.append(current.get("embed_model", "N/A"), style="secondary.bold")
        current_text.append("\n")
        current_text.append("Ollama URL:  ", style="muted")
        current_text.append(current.get("ollama_base_url", "N/A"), style="tertiary")
        
        console.print(Panel(
            current_text,
            title="[primary]Current Models[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))
        
        if error:
            console.print()
            print_warning(f"Ollama: {error}")
        
        # Available models (separate tables)
        chat_models = available.get("chat_models", [])
        embed_models = available.get("embed_models", [])
        
        if chat_models or embed_models:
            console.print()
            self._show_available_models_separate(chat_models, embed_models)
        
        # Usage tips
        console.print()
        tips = Text()
        tips.append("Usage:\n", style="muted")
        tips.append("  /model set", style="command")
        tips.append("                    Interactive model selection\n", style="text")
        tips.append("  /model --chat ", style="command")
        tips.append("MODEL", style="warning")
        tips.append("         Set chat model\n", style="text")
        tips.append("  /model --embed ", style="command")
        tips.append("MODEL", style="warning")
        tips.append("        Set embedding model\n", style="text")
        console.print(tips)
        
        return True
    
    def _show_available_models_separate(self, chat_models: list, embed_models: list) -> None:
        """Display available models in separate tables for chat and embed."""
        # Chat models table
        if chat_models:
            chat_table = Table(
                show_header=True,
                header_style="primary",
                border_style="muted",
                title="[primary]Chat Models[/primary]",
            )
            chat_table.add_column("#", style="muted", width=4)
            chat_table.add_column("Model Name", style="command", width=30)
            chat_table.add_column("Size", style="number", width=10)
            
            for i, model in enumerate(chat_models, 1):
                chat_table.add_row(
                    str(i),
                    model.get("name", "?"),
                    f"{model.get('size_gb', 0):.1f} GB",
                )
            
            console.print(chat_table)
        
        # Embedding models table
        if embed_models:
            console.print()
            embed_table = Table(
                show_header=True,
                header_style="secondary",
                border_style="muted",
                title="[secondary]Embedding Models[/secondary]",
            )
            embed_table.add_column("#", style="muted", width=4)
            embed_table.add_column("Model Name", style="command", width=30)
            embed_table.add_column("Size", style="number", width=10)
            
            for i, model in enumerate(embed_models, 1):
                embed_table.add_row(
                    str(i),
                    model.get("name", "?"),
                    f"{model.get('size_gb', 0):.1f} GB",
                )
            
            console.print(embed_table)
    
    def _list_models(self) -> bool:
        """List all available models from Ollama."""
        return self._list_models_direct()
    
    def _list_models_direct(self) -> bool:
        """Query Ollama directly for models."""
        with create_spinner("Connecting to Ollama...", style="processing"):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get("http://localhost:11434/api/tags")
                    if response.status_code != 200:
                        print_error(f"Ollama returned status {response.status_code}")
                        return False
                    data = response.json()
            except httpx.ConnectError:
                print_error("Cannot connect to Ollama. Is it running?")
                console.print("[muted]Try: ollama serve[/muted]")
                return False
            except Exception as e:
                print_error(f"Failed to query Ollama: {e}")
                return False
        
        models = data.get("models", [])
        
        if not models:
            print_warning("No models found in Ollama.")
            console.print("[muted]Try: ollama pull gemma3:1b[/muted]")
            return True
        
        console.print()
        
        # Categorize models
        chat_models = []
        embed_models = []
        
        for model in models:
            name = model.get("name", "")
            size = model.get("size", 0)
            modified = model.get("modified_at", "")[:10] if model.get("modified_at") else ""
            
            model_info = {
                "name": name,
                "size_gb": round(size / (1024**3), 2) if size else 0,
                "modified": modified,
            }
            
            if "embed" in name.lower():
                embed_models.append(model_info)
            else:
                chat_models.append(model_info)
        
        # Display tables
        if chat_models:
            chat_table = Table(
                title="[primary]Chat Models[/primary]",
                show_header=True,
                header_style="primary",
                border_style="muted",
            )
            chat_table.add_column("#", style="muted", width=4)
            chat_table.add_column("Model Name", style="command", width=35)
            chat_table.add_column("Size", style="number", width=10)
            chat_table.add_column("Modified", style="muted", width=12)
            
            for i, model in enumerate(chat_models, 1):
                chat_table.add_row(
                    str(i),
                    model["name"],
                    f"{model['size_gb']:.1f} GB",
                    model["modified"],
                )
            
            console.print(chat_table)
        
        if embed_models:
            console.print()
            embed_table = Table(
                title="[secondary]Embedding Models[/secondary]",
                show_header=True,
                header_style="secondary",
                border_style="muted",
            )
            embed_table.add_column("#", style="muted", width=4)
            embed_table.add_column("Model Name", style="command", width=35)
            embed_table.add_column("Size", style="number", width=10)
            embed_table.add_column("Modified", style="muted", width=12)
            
            for i, model in enumerate(embed_models, 1):
                embed_table.add_row(
                    str(i),
                    model["name"],
                    f"{model['size_gb']:.1f} GB",
                    model["modified"],
                )
            
            console.print(embed_table)
        
        console.print()
        console.print("[muted]Use [command]/model set[/command] to change models interactively[/muted]")
        
        return True
    
    def _interactive_select(self) -> bool:
        """Interactive model selection."""
        # First get available models
        with create_spinner("Fetching available models...", style="processing"):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get("http://localhost:11434/api/tags")
                    if response.status_code != 200:
                        print_error(f"Ollama returned status {response.status_code}")
                        return False
                    data = response.json()
            except httpx.ConnectError:
                print_error("Cannot connect to Ollama. Is it running?")
                return False
            except Exception as e:
                print_error(f"Failed to query Ollama: {e}")
                return False
        
        models = data.get("models", [])
        if not models:
            print_warning("No models available in Ollama.")
            return False
        
        # Categorize
        all_names = [m.get("name", "") for m in models]
        chat_names = [n for n in all_names if "embed" not in n.lower()]
        embed_names = [n for n in all_names if "embed" in n.lower()]
        
        # Get current from API
        current_chat = "unknown"
        current_embed = "unknown"
        try:
            resp = self.api._request("GET", "/models")
            if resp.success and resp.data:
                current_chat = resp.data.get("current", {}).get("chat_model", "unknown")
                current_embed = resp.data.get("current", {}).get("embed_model", "unknown")
        except Exception:
            pass
        
        console.print()
        console.print(Panel(
            f"[muted]Current chat model:[/muted] [primary]{current_chat}[/primary]\n"
            f"[muted]Current embed model:[/muted] [secondary]{current_embed}[/secondary]",
            title="[primary]Model Selection[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))
        
        # Select chat model
        console.print()
        console.print("[primary]Available Chat Models:[/primary]")
        for i, name in enumerate(chat_names, 1):
            marker = " [success]current[/success]" if name == current_chat else ""
            console.print(f"  [muted]{i}.[/muted] [command]{name}[/command]{marker}")
        
        console.print()
        chat_input = Prompt.ask(
            "[primary]>[/primary] [text]Select chat model (number, name, or Enter to skip)[/text]",
            console=console,
            default="",
        ).strip()
        
        new_chat = None
        if chat_input:
            if chat_input.isdigit():
                idx = int(chat_input) - 1
                if 0 <= idx < len(chat_names):
                    new_chat = chat_names[idx]
            elif chat_input in all_names:
                new_chat = chat_input
            else:
                # Fuzzy match
                matches = [n for n in all_names if chat_input.lower() in n.lower()]
                if matches:
                    new_chat = matches[0]
        
        # Select embed model
        if embed_names:
            console.print()
            console.print("[secondary]Available Embedding Models:[/secondary]")
            for i, name in enumerate(embed_names, 1):
                marker = " [success]current[/success]" if name == current_embed else ""
                console.print(f"  [muted]{i}.[/muted] [command]{name}[/command]{marker}")
            
            console.print()
            embed_input = Prompt.ask(
                "[primary]>[/primary] [text]Select embed model (number, name, or Enter to skip)[/text]",
                console=console,
                default="",
            ).strip()
            
            new_embed = None
            if embed_input:
                if embed_input.isdigit():
                    idx = int(embed_input) - 1
                    if 0 <= idx < len(embed_names):
                        new_embed = embed_names[idx]
                elif embed_input in all_names:
                    new_embed = embed_input
                else:
                    matches = [n for n in embed_names if embed_input.lower() in n.lower()]
                    if matches:
                        new_embed = matches[0]
        else:
            new_embed = None
        
        # Apply changes
        if new_chat or new_embed:
            return self._set_models(chat_model=new_chat, embed_model=new_embed)
        else:
            console.print("[muted]No changes made.[/muted]")
            return True
    
    def _set_models(self, chat_model: str | None = None, embed_model: str | None = None) -> bool:
        """Set the active models via API."""
        if not chat_model and not embed_model:
            print_warning("No model specified. Use --chat MODEL or --embed MODEL")
            return False
        
        payload = {}
        if chat_model:
            payload["chat_model"] = chat_model
        if embed_model:
            payload["embed_model"] = embed_model
        
        # Show what we're doing
        console.print()
        if chat_model:
            console.print(f"[muted]Loading chat model:[/muted] [primary]{chat_model}[/primary]")
        if embed_model:
            console.print(f"[muted]Loading embed model:[/muted] [secondary]{embed_model}[/secondary]")
        console.print("[muted]This may take a moment for large models...[/muted]")
        console.print()
        
        # Use longer timeout for model operations
        try:
            with create_spinner("Updating models (this may take a while for large models)...", style="processing"):
                with httpx.Client(timeout=MODEL_TIMEOUT) as client:
                    response = client.post(
                        f"{self.api.base_url}/models",
                        json=payload,
                    )
                    
                    if response.status_code >= 400:
                        try:
                            error_data = response.json()
                            if "detail" in error_data:
                                detail = error_data["detail"]
                                if isinstance(detail, dict) and "errors" in detail:
                                    error_msg = ", ".join(detail["errors"])
                                else:
                                    error_msg = str(detail)
                            else:
                                error_msg = response.text
                        except Exception:
                            error_msg = response.text
                        print_error(f"Failed to update models: {error_msg}")
                        return False
                    
                    data = response.json()
        
        except httpx.TimeoutException:
            print_error(
                "Model loading timed out. This can happen with very large models.\n"
                "The model may still be loading in the background.\n"
                "Try running /model to check the current status."
            )
            return False
        except httpx.ConnectError:
            print_error("Cannot connect to server. Is it running? Try '/start' first.")
            return False
        except Exception as e:
            print_error(f"Request failed: {type(e).__name__}: {e}")
            return False
        
        changes = data.get("changes", {})
        
        if changes:
            for key, change in changes.items():
                label = "Chat Model" if key == "chat_model" else "Embed Model"
                console.print(
                    f"[success]OK[/success] {label}: "
                    f"[muted]{change.get('from')}[/muted] -> "
                    f"[primary]{change.get('to')}[/primary]"
                )
            print_success("Models updated successfully!")
            
            # Check for dimension warning (embedding model change with existing index)
            embed_change = changes.get("embed_model", {})
            dimension_warning = embed_change.get("dimension_warning")
            if dimension_warning:
                console.print()
                console.print(Panel(
                    f"[error bold]⚠ Dimension Mismatch Detected![/error bold]\n\n"
                    f"[warning]{dimension_warning}[/warning]\n\n"
                    f"[text]To fix this, run:[/text]\n"
                    f"  [command]/reset[/command]   [muted]Clear database and FAISS index[/muted]\n"
                    f"  [command]/restart[/command] [muted]Reinitialize server[/muted]\n"
                    f"  [command]/ingest[/command]  [muted]Re-embed documents with new model[/muted]",
                    title="[error]Action Required[/error]",
                    border_style="error",
                    padding=(1, 2),
                ))
        else:
            console.print("[muted]No changes were made.[/muted]")
        
        # Warning about persistence
        console.print()
        env_lines = []
        if chat_model:
            env_lines.append(f"  OLLAMA_CHAT_MODEL={chat_model}")
        if embed_model:
            env_lines.append(f"  OLLAMA_EMBED_MODEL={embed_model}")
        
        print_warning(
            "Note: This change is temporary. To persist, update your .env file:\n" +
            "\n".join(env_lines)
        )
        
        return True
    
    def _show_current(self) -> bool:
        """Show current model configuration."""
        with create_spinner("Fetching configuration...", style="processing"):
            response = self.api._request("GET", "/models")
        
        if not response.success:
            print_error(response.error or "Failed to get model info")
            return False
        
        data = response.data or {}
        current = data.get("current", {})
        
        console.print()
        console.print(Panel(
            f"[muted]Chat Model:[/muted]   [primary.bold]{current.get('chat_model', 'N/A')}[/primary.bold]\n"
            f"[muted]Embed Model:[/muted]  [secondary.bold]{current.get('embed_model', 'N/A')}[/secondary.bold]\n"
            f"[muted]Ollama URL:[/muted]   [tertiary]{current.get('ollama_base_url', 'N/A')}[/tertiary]",
            title="[primary]Current Configuration[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))
        
        return True
    
    def _show_model_help(self) -> None:
        """Show model command help."""
        console.print()
        console.print(Panel(
            Text.from_markup(
                "[primary]Model Management Commands[/primary]\n\n"
                "[command]/model[/command]\n"
                "  Show current models and available options\n\n"
                "[command]/model list[/command]\n"
                "  List all models from Ollama\n\n"
                "[command]/model set[/command]\n"
                "  Interactive model selection\n\n"
                "[command]/model --chat[/command] [warning]MODEL[/warning]\n"
                "  Set the chat/LLM model\n\n"
                "[command]/model --embed[/command] [warning]MODEL[/warning]\n"
                "  Set the embedding model\n\n"
                "[muted]Examples:[/muted]\n"
                "  /model --chat llama3.2:3b\n"
                "  /model --embed nomic-embed-text\n"
                "  /model --chat gemma3:1b --embed mxbai-embed-large"
            ),
            title="[primary]Model Commands[/primary]",
            border_style="primary",
            padding=(1, 2),
        ))
