"""Prepare Keep a Changelog releases from merged pull-request metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SEMVER_LABELS = {
    "semver:patch": "patch",
    "semver:minor": "minor",
    "semver:major": "major",
    "semver:none": None,
}
_VERSION_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
_UNRELEASED_RE = re.compile(r"^## \[Unreleased\]\s*$")
_RELEASE_HEADING_RE = re.compile(r"^## \[")
_BULLET_RE = re.compile(r"^\s*[-*]\s+\S")


class ReleaseError(ValueError):
    """Raised when release metadata or changelog contents are invalid."""


def parse_version(version: str) -> tuple[int, int, int]:
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ReleaseError(f"Expected a stable SemVer version, got {version!r}")
    return tuple(int(part) for part in match.groups())


def bump_version(version: str, bump: str) -> str:
    major, minor, patch = parse_version(version)
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    elif bump == "patch":
        patch += 1
    else:
        raise ReleaseError(f"Unknown SemVer bump: {bump!r}")
    return f"{major}.{minor}.{patch}"


def bump_for_labels(labels: list[str]) -> str | None:
    semver_labels = [label for label in labels if label.startswith("semver:")]
    unknown = [label for label in semver_labels if label not in SEMVER_LABELS]
    if unknown:
        raise ReleaseError(f"Unknown SemVer label(s): {', '.join(unknown)}")
    if len(semver_labels) > 1:
        raise ReleaseError("A pull request may have only one semver:* label")
    if not semver_labels:
        return "minor"
    return SEMVER_LABELS[semver_labels[0]]


def highest_bump(label_sets: list[list[str]]) -> str | None:
    bumps = [bump for labels in label_sets if (bump := bump_for_labels(labels)) is not None]
    if not bumps:
        return None
    return max(bumps, key=("patch", "minor", "major").index)


def _unreleased_bounds(lines: list[str]) -> tuple[int, int]:
    headings = [index for index, line in enumerate(lines) if _UNRELEASED_RE.fullmatch(line)]
    if len(headings) != 1:
        raise ReleaseError("CHANGELOG.md must contain exactly one '## [Unreleased]' heading")
    start = headings[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if _RELEASE_HEADING_RE.match(lines[index])),
        len(lines),
    )
    return start, end


def _bullet_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if _BULLET_RE.match(line)]


def validate_changelog_entry(base: str, head: str, labels: list[str]) -> None:
    if bump_for_labels(labels) is None:
        return

    head_lines = head.splitlines()
    head_start, head_end = _unreleased_bounds(head_lines)
    if base.strip():
        base_lines = base.splitlines()
        base_start, base_end = _unreleased_bounds(base_lines)
        previous = Counter(_bullet_lines(base_lines[base_start:base_end]))
    else:
        previous = Counter()
    current = Counter(_bullet_lines(head_lines[head_start:head_end]))
    if not (current - previous):
        raise ReleaseError(
            "A releasable PR must add a bullet entry under CHANGELOG.md's [Unreleased] section"
        )


def release_changelog(changelog: str, version: str, release_date: str) -> tuple[str, str]:
    lines = changelog.splitlines()
    start, end = _unreleased_bounds(lines)
    entries = lines[start:end]
    if not _bullet_lines(entries):
        raise ReleaseError("Cannot release: CHANGELOG.md has no entries under [Unreleased]")

    body = "\n".join(entries).strip()
    if not body:
        raise ReleaseError("Cannot release: CHANGELOG.md has no entries under [Unreleased]")

    released_section = f"## [{version}] - {release_date}\n\n{body}"
    prefix = "\n".join(lines[:start])
    suffix = "\n".join(lines[end:])
    result = f"{prefix}\n\n{released_section}"
    if suffix.strip():
        result += f"\n\n{suffix.lstrip()}"
    return f"{result.rstrip()}\n", released_section


def project_version(pyproject: str) -> str:
    section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", pyproject)
    if section is None:
        raise ReleaseError("pyproject.toml has no [project] section")
    match = re.search(r'(?m)^version = "([^"]+)"[ \t]*$', section.group(1))
    if match is None:
        raise ReleaseError("pyproject.toml [project] section has no version")
    parse_version(match.group(1))
    return match.group(1)


def update_project_version(pyproject: str, version: str) -> str:
    parse_version(version)
    section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", pyproject)
    if section is None:
        raise ReleaseError("pyproject.toml has no [project] section")
    body, replacements = re.subn(
        r'(?m)^version = "[^"]+"[ \t]*$',
        f'version = "{version}"',
        section.group(1),
    )
    if replacements != 1:
        raise ReleaseError("Expected exactly one version in pyproject.toml [project] section")
    return pyproject[: section.start(1)] + body + pyproject[section.end(1) :]


def _new_pull_request_labels(
    pull_requests: list[dict[str, Any]], head: str = "HEAD"
) -> list[list[str]]:
    try:
        latest_tag = subprocess.run(
            [
                "git",
                "describe",
                "--tags",
                "--abbrev=0",
                "--match",
                "v[0-9]*",
                head,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        latest_tag = ""

    revision = f"{latest_tag}..{head}" if latest_tag else head
    commits = set(
        subprocess.run(
            ["git", "rev-list", revision],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    matching = []
    for pull_request in pull_requests:
        merge_commit = pull_request.get("mergeCommit") or {}
        commit = merge_commit.get("oid")
        if commit not in commits:
            continue
        matching.append([label["name"] for label in pull_request.get("labels", [])])
    return matching


def _write_github_output(path: Path | None, *, release: bool, version: str = "") -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output:
        output.write(f"release={'true' if release else 'false'}\n")
        if version:
            output.write(f"version={version}\n")


def prepare_release(
    repo: Path,
    pull_requests: list[dict[str, Any]],
    release_date: str,
    *,
    head: str = "HEAD",
    github_output: Path | None = None,
    notes_file: Path | None = None,
) -> str | None:
    label_sets = _new_pull_request_labels(pull_requests, head)
    if not label_sets:
        changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
        lines = changelog.splitlines()
        start, end = _unreleased_bounds(lines)
        if _bullet_lines(lines[start:end]):
            raise ReleaseError("Unreleased changelog entries have no matching merged pull request")
        _write_github_output(github_output, release=False)
        return None

    bump = highest_bump(label_sets)
    if bump is None:
        _write_github_output(github_output, release=False)
        return None

    pyproject_path = repo / "pyproject.toml"
    changelog_path = repo / "CHANGELOG.md"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    changelog = changelog_path.read_text(encoding="utf-8")
    current_version = project_version(pyproject)

    latest_tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", head],
        capture_output=True,
        text=True,
    )
    if latest_tag.returncode == 0:
        tagged_version = latest_tag.stdout.strip().removeprefix("v")
        if parse_version(tagged_version) != parse_version(current_version):
            raise ReleaseError(
                f"Latest tag v{tagged_version} does not match pyproject version {current_version}"
            )

    version = bump_version(current_version, bump)
    new_changelog, release_notes = release_changelog(changelog, version, release_date)
    new_pyproject = update_project_version(pyproject, version)

    changelog_path.write_text(new_changelog, encoding="utf-8")
    pyproject_path.write_text(new_pyproject, encoding="utf-8")
    if notes_file is not None:
        notes_file.write_text(f"{release_notes}\n", encoding="utf-8")
    _write_github_output(github_output, release=True, version=version)
    return version


def _read_labels(raw_labels: str) -> list[str]:
    labels = json.loads(raw_labels)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ReleaseError("PR labels must be supplied as a JSON array of strings")
    return labels


def _validate_pr(args: argparse.Namespace) -> None:
    labels = _read_labels(args.labels_json)
    try:
        base = subprocess.run(
            ["git", "show", f"{args.base}:CHANGELOG.md"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        base = ""
    head = (Path.cwd() / "CHANGELOG.md").read_text(encoding="utf-8")
    validate_changelog_entry(base, head, labels)
    print("Changelog entry is valid.")


def _prepare(args: argparse.Namespace) -> None:
    pull_requests = json.loads(args.pull_requests_json.read_text(encoding="utf-8"))
    if not isinstance(pull_requests, list):
        raise ReleaseError("Merged pull-request metadata must be a JSON array")
    version = prepare_release(
        Path.cwd(),
        pull_requests,
        args.date,
        github_output=args.github_output,
        notes_file=args.notes_file,
    )
    if version is None:
        print("No releasable merged pull requests; skipping release.")
    else:
        print(f"Prepared release v{version}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-pr")
    validate_parser.add_argument("--base", required=True)
    validate_parser.add_argument("--labels-json", required=True)
    validate_parser.set_defaults(run=_validate_pr)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--pull-requests-json", type=Path, required=True)
    prepare_parser.add_argument("--date", required=True)
    prepare_parser.add_argument("--github-output", type=Path)
    prepare_parser.add_argument("--notes-file", type=Path)
    prepare_parser.set_defaults(run=_prepare)

    args = parser.parse_args()
    try:
        args.run(args)
    except ReleaseError as error:
        print(f"Release error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
