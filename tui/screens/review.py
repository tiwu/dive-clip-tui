"""Review screen: review AI detections before rendering."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, DataTable


class ReviewScreen(Screen[None]):
    """Screen for reviewing AI detections and accepting/rejecting them."""

    def compose(self) -> ComposeResult:
        yield Label("Detections review", id="review-title")
        yield Input(placeholder="/path/to/detections_refined.json", id="detections-input")
        yield Button("Load detections", id="load-detections", variant="primary")
        yield Static(id="detections-status")

        yield Label("Filter by quality score (1-10):", id="quality-label")
        yield Select(
            options=[("All", "all"), ("7+", "7"), ("8+", "8"), ("9+", "9")],
            value="all",
            id="quality-filter",
        )

        yield ScrollableContainer(
            DataTable(id="detections-table"), id="detections-scroll"
        )

        yield Button("Accept selected", id="accept-selected", variant="success", disabled=True)
        yield Button("Render accepted clips", id="render-clips", variant="warning", disabled=True)
        yield Button("Back", id="go-back", variant="default")

        self._accepted: set[int] = set()
        self._detections: list[dict] = []
        self._filtered: list[dict] = []

    def on_mount(self) -> None:
        """Set up the detections data table columns."""
        table = self.query_one("#detections-table", DataTable)
        table.add_columns(
            "Accept", "File", "Start", "End", "Subject", "Quality", "Confidence", "Description"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "load-detections":
            self._load_detections()
        elif button_id == "accept-selected":
            self._toggle_accept()
        elif button_id == "render-clips":
            self._render_clips()
        elif button_id == "go-back":
            self.app.pop_screen()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle quality filter changes."""
        if event.select.id == "quality-filter":
            self._apply_filter()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle accept checkbox toggles."""
        row_index = int(event.checkbox.id.split("-")[1]) if event.checkbox.id else -1
        if row_index >= 0:
            if event.checkbox.value:
                self._accepted.add(row_index)
            else:
                self._accepted.discard(row_index)

        accept_btn = self.query_one("#accept-selected", Button)
        render_btn = self.query_one("#render-clips", Button)
        has_accepted = len(self._accepted) > 0
        accept_btn.disabled = not has_accepted
        render_btn.disabled = not has_accepted

    def _load_detections(self) -> None:
        """Load detections from the JSON file."""
        input_widget = self.query_one("#detections-input", Input)
        detections_path = Path(input_widget.value.strip()) if input_widget.value else None

        if not detections_path or not detections_path.exists():
            detections_path = Path("outputs") / "detections_refined.json"

        if not detections_path.exists():
            status = self.query_one("#detections-status", Static)
            status.update("ERROR: No detections_refined.json found.")
            return

        try:
            with open(detections_path) as f:
                data = json.load(f)

            self._detections = data.get("detections", [])
            self._filtered = list(self._detections)
            self._accepted.clear()

            table = self.query_one("#detections-table", DataTable)
            table.clear()

            for i, det in enumerate(self._filtered):
                checkbox = Checkbox(id=f"cb-{i}")
                table.add_row(
                    checkbox,
                    det.get("file_name", ""),
                    f"{det.get('start_time', 0):.1f}s",
                    f"{det.get('end_time', 0):.1f}s",
                    det.get("subject", ""),
                    str(det.get("quality_score", "")),
                    f"{det.get('confidence', 0):.2f}",
                    det.get("description", ""),
                )

            status = self.query_one("#detections-status", Static)
            status.update(f"Loaded {len(self._detections)} detections from {detections_path}")

        except (json.JSONDecodeError, OSError) as exc:
            status = self.query_one("#detections-status", Static)
            status.update(f"ERROR: Failed to load detections: {exc}")

    def _apply_filter(self) -> None:
        """Apply the quality score filter to the table."""
        select = self.query_one("#quality-filter", Select)
        threshold = select.value

        table = self.query_one("#detections-table", DataTable)
        table.clear()

        if threshold == "all":
            filtered = self._detections
        else:
            min_score = int(threshold)
            filtered = [d for d in self._detections if d.get("quality_score", 0) >= min_score]

        self._filtered = filtered
        self._accepted.clear()

        for i, det in enumerate(filtered):
            checkbox = Checkbox(id=f"cb-{i}")
            table.add_row(
                checkbox,
                det.get("file_name", ""),
                f"{det.get('start_time', 0):.1f}s",
                f"{det.get('end_time', 0):.1f}s",
                det.get("subject", ""),
                str(det.get("quality_score", "")),
                f"{det.get('confidence', 0):.2f}",
                det.get("description", ""),
            )

        status = self.query_one("#detections-status", Static)
        status.update(f"Showing {len(filtered)} detections (quality >= {threshold})")

    def _toggle_accept(self) -> None:
        """Toggle accept state for all visible detections."""
        # Implementation would toggle all visible rows
        pass

    def _render_clips(self) -> None:
        """Render accepted clips via FFmpeg."""
        status = self.query_one("#detections-status", Static)
        status.update(f"Rendering {len(self._accepted)} accepted clips... (not yet implemented)")