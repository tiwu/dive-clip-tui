"""Browse screen: select footage folder and view manifest."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static, DataTable


class BrowseScreen(Screen[None]):
    """Screen for browsing a footage folder and viewing the manifest."""

    def compose(self) -> ComposeResult:
        yield Label("Footage folder path:", id="folder-label")
        yield Input(placeholder="/path/to/footage", id="folder-input")
        yield Button("Load manifest", id="load-manifest", variant="primary")
        yield Static(id="manifest-status")
        yield ScrollableContainer(DataTable(id="manifest-table"), id="manifest-scroll")
        yield Button("Review detections", id="go-review", variant="success", disabled=True)
        yield Button("Back", id="go-back", variant="default")

    def on_mount(self) -> None:
        """Set up the manifest data table columns."""
        table = self.query_one("#manifest-table", DataTable)
        table.add_columns("File", "Duration", "Resolution", "FPS", "Tags")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "load-manifest":
            self._load_manifest()
        elif button_id == "go-review":
            self.app.push_screen("review")
        elif button_id == "go-back":
            self.app.pop_screen()

    def _load_manifest(self) -> None:
        """Load the manifest from the footage folder."""
        input_widget = self.query_one("#folder-input", Input)
        folder_path = Path(input_widget.value.strip()) if input_widget.value else None

        if not folder_path or not folder_path.exists():
            status = self.query_one("#manifest-status", Static)
            status.update("ERROR: Folder does not exist or no path provided.")
            return

        manifest_path = folder_path / "manifest_raw.json"
        outputs_manifest = Path("outputs") / "manifest_raw.json"

        if manifest_path.exists():
            manifest_file = manifest_path
        elif outputs_manifest.exists():
            manifest_file = outputs_manifest
        else:
            status = self.query_one("#manifest-status", Static)
            status.update("ERROR: No manifest_raw.json found in folder or outputs/.")
            return

        try:
            with open(manifest_file) as f:
                data = json.load(f)

            table = self.query_one("#manifest-table", DataTable)
            table.clear()

            clips = data.get("clips", [])
            for clip in clips:
                tags = ", ".join(clip.get("tags", []))
                table.add_row(
                    clip.get("file_name", ""),
                    clip.get("duration", ""),
                    clip.get("resolution", ""),
                    clip.get("fps", ""),
                    tags,
                )

            status = self.query_one("#manifest-status", Static)
            status.update(f"Loaded {len(clips)} clips from {manifest_file}")

            review_btn = self.query_one("#go-review", Button)
            review_btn.disabled = False

        except (json.JSONDecodeError, OSError) as exc:
            status = self.query_one("#manifest-status", Static)
            status.update(f"ERROR: Failed to load manifest: {exc}")

