"""Main TUI application for the dive-clip AI curation pipeline.

Provides an interactive terminal interface for browsing footage,
reviewing AI detections, and running pipeline steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from tui.screens.browse import BrowseScreen
from tui.screens.pipeline import PipelineScreen
from tui.screens.review import ReviewScreen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class DiveClipTUI(App[None]):
    """Main TUI application for the dive-clip pipeline."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #title {
        dock: top;
        height: 3;
        content-align: center middle;
        text-style: bold;
        color: $text;
    }

    #content {
        height: 1fr;
        padding: 1 2;
    }

    #button-bar {
        dock: bottom;
        height: 3;
        layout: horizontal;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    TITLE = "Dive Clip AI — TUI"

    SCREENS = {
        "browse": BrowseScreen,
        "review": ReviewScreen,
        "pipeline": PipelineScreen,
    }

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Dive Clip AI — Local footage curation pipeline", id="title")
        yield Container(id="content")
        yield Horizontal(
            Button("Quit", id="quit", variant="error"),
            id="button-bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Push the browse screen as the initial screen."""
        self.push_screen("browse")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle global button presses."""
        if event.button.id == "quit":
            self.exit()


def main() -> None:
    """Entry point for the dive-clip-tui console script."""
    app = DiveClipTUI()
    app.run()


if __name__ == "__main__":
    main()
