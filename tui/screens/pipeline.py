"""Pipeline screen: run pipeline steps with live log output."""

from __future__ import annotations

import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, Label, Log, Static


class PipelineScreen(Screen[None]):
    """Screen for running pipeline steps with live log output."""

    def compose(self) -> ComposeResult:
        yield Label("Pipeline runner", id="pipeline-title")

        yield Button("1. Ingest media (manifest)", id="step-ingest", variant="primary")
        yield Button("2. Scan footage (AI detections)", id="step-scan", variant="primary")
        yield Button("3. Generate EDL/CSV markers", id="step-edl", variant="primary")
        yield Button("4. Plan storyline (cloud LLM)", id="step-storyline", variant="primary")

        yield Static(id="pipeline-status")
        yield ScrollableContainer(Log(id="pipeline-log", highlight=True), id="pipeline-log-scroll")
        yield Button("Back", id="go-back", variant="default")

        self._process: subprocess.Popen | None = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "step-ingest":
            self._run_step("ingest", ["uv", "run", "scripts/00_ingest_media.py"])
        elif button_id == "step-scan":
            self._run_step("scan", ["uv", "run", "scripts/02_scan_footage.py", "--config", "config.yaml"])
        elif button_id == "step-edl":
            self._run_step("edl", ["uv", "run", "scripts/03_generate_edl.py", "--config", "config.yaml"])
        elif button_id == "step-storyline":
            self._run_step("storyline", ["uv", "run", "scripts/04_plan_storyline.py", "--config", "config.yaml"])
        elif button_id == "go-back":
            self.app.pop_screen()

    def _run_step(self, name: str, cmd: list[str]) -> None:
        """Run a pipeline step as a subprocess and stream output to the log."""
        log = self.query_one("#pipeline-log", Log)
        status = self.query_one("#pipeline-status", Static)

        log.clear()
        log.write_line(f"$ {' '.join(cmd)}")
        status.update(f"Running {name}...")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert self._process.stdout is not None
            for line in self._process.stdout:
                log.write_line(line.rstrip("\n"))

            self._process.wait()
            if self._process.returncode == 0:
                status.update(f"{name} completed successfully.")
            else:
                status.update(f"{name} failed with exit code {self._process.returncode}.")

        except FileNotFoundError:
            status.update(f"ERROR: Script not found for {name} step.")
        except Exception as exc:
            status.update(f"ERROR: {exc}")
        finally:
            self._process = None