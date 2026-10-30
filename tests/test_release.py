from __future__ import annotations

import subprocess

import pytest

from scripts.release import (
    ReleaseError,
    bump_for_labels,
    bump_version,
    highest_bump,
    prepare_release,
    project_version,
    release_changelog,
    update_project_version,
    validate_changelog_entry,
)


def test_bump_version_applies_semver_including_zero_major() -> None:
    assert bump_version("0.1.0", "patch") == "0.1.1"
    assert bump_version("0.1.0", "minor") == "0.2.0"
    assert bump_version("0.9.8", "major") == "1.0.0"


def test_bump_for_labels_defaults_to_minor_and_honors_none() -> None:
    assert bump_for_labels(["feature"]) == "minor"
    assert bump_for_labels(["semver:patch"]) == "patch"
    assert bump_for_labels(["semver:none"]) is None


def test_bump_for_labels_rejects_conflicting_or_unknown_labels() -> None:
    with pytest.raises(ReleaseError, match="only one"):
        bump_for_labels(["semver:patch", "semver:minor"])
    with pytest.raises(ReleaseError, match="Unknown SemVer"):
        bump_for_labels(["semver:critical"])


def test_highest_bump_uses_highest_releasable_pr_bump() -> None:
    assert (
        highest_bump([["semver:patch"], ["semver:none"], ["semver:major"], ["semver:minor"]])
        == "major"
    )
    assert highest_bump([["semver:none"]]) is None


def test_prepare_release_skips_semver_none(tmp_path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")
    output = tmp_path / "github-output"
    monkeypatch.setattr("scripts.release._new_pull_request_labels", lambda *_: [["semver:none"]])

    version = prepare_release(tmp_path, [], "2026-10-30", github_output=output)

    assert version is None
    assert output.read_text(encoding="utf-8") == "release=false\n"


def test_release_changelog_moves_entries_and_keeps_unreleased_heading() -> None:
    changelog = """# Changelog

## [Unreleased]

### Added
- Add clip export.

## [0.1.0] - 2025-01-01

### Fixed
- Fix an old issue.
"""

    updated, notes = release_changelog(changelog, "0.2.0", "2026-10-30")

    assert updated.startswith("# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-10-30")
    assert "### Added\n- Add clip export." in updated
    assert "## [0.1.0] - 2025-01-01\n\n### Fixed" in updated
    assert notes == "## [0.2.0] - 2026-10-30\n\n### Added\n- Add clip export."


def test_release_changelog_rejects_empty_unreleased_section() -> None:
    with pytest.raises(ReleaseError, match="no entries"):
        release_changelog("# Changelog\n\n## [Unreleased]\n", "0.2.0", "2026-10-30")


def test_validate_changelog_entry_requires_new_unreleased_bullet() -> None:
    base = "# Changelog\n\n## [Unreleased]\n\n### Added\n- Existing note.\n"
    updated = base.replace("## [Unreleased]", "## [Unreleased]\n\n### Fixed\n- Fix a bug.")
    validate_changelog_entry(base, updated, ["semver:patch"])

    with pytest.raises(ReleaseError, match="must add a bullet"):
        validate_changelog_entry(base, base, ["semver:minor"])


def test_validate_changelog_entry_allows_a_new_changelog_file() -> None:
    head = "# Changelog\n\n## [Unreleased]\n\n### Added\n- Add the changelog.\n"
    validate_changelog_entry("", head, ["semver:minor"])


def test_semver_none_does_not_require_a_changelog_entry() -> None:
    changelog = "# Changelog\n\n## [Unreleased]\n"
    validate_changelog_entry(changelog, changelog, ["semver:none"])


def test_project_version_updates_only_project_metadata() -> None:
    pyproject = (
        '[project]\nname = "demo"\nversion = "0.1.0"\n\n[tool.other]\nversion = "unchanged"\n'
    )

    updated = update_project_version(pyproject, "0.2.0")

    assert project_version(updated) == "0.2.0"
    assert '[tool.other]\nversion = "unchanged"' in updated


def test_prepare_release_updates_version_and_archives_unreleased(
    monkeypatch, tmp_path_factory
) -> None:
    repo = tmp_path_factory.mktemp("release-repo")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "baseline"], check=True)
    subprocess.run(["git", "-C", str(repo), "tag", "v0.1.0", "-m", "v0.1.0"], check=True)
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n- Add release automation.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "CHANGELOG.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "add release note"], check=True)
    merge_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(repo)
    output = repo / "github-output"
    notes = repo / "release-notes.md"

    version = prepare_release(
        repo,
        [
            {
                "mergeCommit": {"oid": merge_commit},
                "labels": [{"name": "semver:patch"}],
            }
        ],
        "2026-10-30",
        github_output=output,
        notes_file=notes,
    )

    assert version == "0.1.1"
    assert project_version((repo / "pyproject.toml").read_text(encoding="utf-8")) == "0.1.1"
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    assert "## [0.1.1] - 2026-10-30" in changelog
    assert notes.read_text(encoding="utf-8").endswith("- Add release automation.\n")
    assert output.read_text(encoding="utf-8") == "release=true\nversion=0.1.1\n"
