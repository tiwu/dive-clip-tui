"""Tests for scripts/ci_summary.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci_summary import (
    main,
    summarize_coverage,
    summarize_lint,
    summarize_release,
)


def test_summarize_coverage(tmp_path: Path) -> None:
    """summarize_coverage should parse coverage JSON and write Markdown table."""
    cov_data = {
        "totals": {
            "percent_covered": 85.5,
            "covered_lines": 171,
            "num_statements": 200,
            "missing_lines": 29,
        },
        "files": {
            "scripts/test.py": {"summary": {"covered_lines": 80, "num_statements": 100}},
            "utils/test.py": {"summary": {"covered_lines": 91, "num_statements": 100}},
        },
    }
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps(cov_data), encoding="utf-8")

    out_file = tmp_path / "summary.md"
    result = summarize_coverage(cov_json, out_file)

    assert "## 🧪 Test Results & Coverage" in result
    assert "85.5%" in result
    assert "| `scripts/` | 80 | 100 | 80.0% |" in result
    assert "| `utils/` | 91 | 100 | 91.0% |" in result
    assert out_file.read_text(encoding="utf-8").strip() == result.strip()


def test_summarize_lint_with_files(tmp_path: Path) -> None:
    """summarize_lint should report passed checks and file count."""
    out_file = tmp_path / "summary.md"
    result = summarize_lint(3, out_file)

    assert "## 🎨 Code Quality & Linting" in result
    assert "3 changed Python file(s) checked" in result
    assert out_file.read_text(encoding="utf-8").strip() == result.strip()


def test_summarize_lint_zero_files(tmp_path: Path) -> None:
    """summarize_lint should report when no files were changed."""
    result = summarize_lint(0)
    assert "No changed Python files to check" in result


def test_summarize_release_with_label(tmp_path: Path) -> None:
    """summarize_release should extract semver label."""
    out_file = tmp_path / "summary.md"
    result = summarize_release('["feature", "semver:minor"]', out_file)

    assert "## 📝 Release & Changelog" in result
    assert "`semver:minor`" in result
    assert "Changelog Validation:** Valid ✅" in result


def test_summarize_release_default_unlabeled() -> None:
    """summarize_release should default to minor bump when unlabeled."""
    result = summarize_release("[]")
    assert "`semver:minor (default)`" in result


def test_main_cli(tmp_path: Path, monkeypatch) -> None:
    """main CLI should execute subcommands."""
    import sys

    cov_data = {"totals": {}, "files": {}}
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps(cov_data), encoding="utf-8")
    out_file = tmp_path / "summary.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_summary.py",
            "test-coverage",
            "--coverage-json",
            str(cov_json),
            "--output-file",
            str(out_file),
        ],
    )
    assert main() == 0
    assert out_file.exists()
