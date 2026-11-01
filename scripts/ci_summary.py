"""Helper script to format and write GitHub Actions job summaries ($GITHUB_STEP_SUMMARY)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_summary_path(explicit_path: Path | str | None = None) -> Path | None:
    """Resolve destination summary file from CLI argument or GITHUB_STEP_SUMMARY env var."""
    if explicit_path:
        return Path(explicit_path)
    env_target = os.environ.get("GITHUB_STEP_SUMMARY")
    if env_target:
        return Path(env_target)
    return None


def _append_summary(text: str, target_path: Path | None) -> None:
    """Append text to target summary path if resolved, or stdout if running locally."""
    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")
    else:
        print(text.rstrip() + "\n")


def summarize_coverage(coverage_json: Path, output_file: Path | None = None) -> str:
    """Read coverage.json and generate Markdown summary of test results and coverage."""
    with coverage_json.open(encoding="utf-8") as f:
        cov = json.load(f)

    totals = cov.get("totals", {})
    files = cov.get("files", {})

    pct = totals.get("percent_covered", 0.0)
    covered = totals.get("covered_lines", 0)
    total_statements = totals.get("num_statements", 0)
    missing = totals.get("missing_lines", 0)

    dirs: dict[str, list[int]] = {"scripts": [0, 0], "utils": [0, 0], "tui": [0, 0]}
    for path, data in files.items():
        prefix = path.split("/")[0] if "/" in path else "other"
        if prefix in dirs:
            dirs[prefix][0] += data.get("summary", {}).get("covered_lines", 0)
            dirs[prefix][1] += data.get("summary", {}).get("num_statements", 0)

    lines = [
        "## 🧪 Test Results & Coverage",
        "",
        f"- **Overall Coverage:** {pct:.1f}% ({covered}/{total_statements} statements)",
        f"- **Missing Statements:** {missing}",
        "",
        "| Component | Covered Statements | Total Statements | Coverage |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for name, (cov_lines, tot_lines) in dirs.items():
        comp_pct = (cov_lines / tot_lines * 100.0) if tot_lines else 0.0
        lines.append(f"| `{name}/` | {cov_lines} | {tot_lines} | {comp_pct:.1f}% |")

    content = "\n".join(lines)
    _append_summary(content, output_file)
    return content


def summarize_lint(file_count: int, output_file: Path | None = None) -> str:
    """Generate Markdown summary of Ruff lint and formatting results."""
    scope_desc = (
        f"{file_count} changed Python file(s) checked"
        if file_count > 0
        else "No changed Python files to check"
    )
    lines = [
        "## 🎨 Code Quality & Linting",
        "",
        "- **Ruff Lint:** Passed ✅",
        "- **Ruff Formatting:** Passed ✅",
        f"- **Scope:** {scope_desc}",
    ]
    content = "\n".join(lines)
    _append_summary(content, output_file)
    return content


def summarize_release(labels_json: str, output_file: Path | None = None) -> str:
    """Generate Markdown summary of SemVer impact and changelog entry validation."""
    try:
        labels = json.loads(labels_json)
        if not isinstance(labels, list):
            labels = []
    except json.JSONDecodeError, TypeError:
        labels = []

    semver_labels = [
        label for label in labels if isinstance(label, str) and label.startswith("semver:")
    ]
    semver = semver_labels[0] if semver_labels else "semver:minor (default)"

    lines = [
        "## 📝 Release & Changelog",
        "",
        f"- **Proposed SemVer Impact:** `{semver}`",
        "- **Changelog Validation:** Valid ✅",
    ]
    content = "\n".join(lines)
    _append_summary(content, output_file)
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cov_parser = subparsers.add_parser("test-coverage")
    cov_parser.add_argument("--coverage-json", type=Path, required=True)
    cov_parser.add_argument("--output-file", type=Path, default=None)

    lint_parser = subparsers.add_parser("lint-format")
    lint_parser.add_argument("--file-count", type=int, default=0)
    lint_parser.add_argument("--output-file", type=Path, default=None)

    release_parser = subparsers.add_parser("release-pr")
    release_parser.add_argument("--labels-json", default="[]")
    release_parser.add_argument("--output-file", type=Path, default=None)

    args = parser.parse_args()
    dest = _resolve_summary_path(args.output_file)

    if args.command == "test-coverage":
        summarize_coverage(args.coverage_json, dest)
    elif args.command == "lint-format":
        summarize_lint(args.file_count, dest)
    elif args.command == "release-pr":
        summarize_release(args.labels_json, dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
