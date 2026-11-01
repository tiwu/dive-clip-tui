# Architecture & Feature Plans

This directory contains version-controlled architectural specifications and implementation plans.

## Two-Stage Issue and PR Lifecycle

```text
Stage 1 — Plan the change
1. User queues an issue with 'agent:plan' or initiates planning in chat (e.g. '/plan').
2. If no issue exists yet, agent creates one via 'gh issue create --label "agent:plan-review"'.
3. Agent writes docs/plans/PLAN-<id>-<slug>.md, embeds/links it in the issue body, and sets 'agent:plan-review'.
4. The plan records the change category, proposed SemVer impact, and a draft changelog entry.
5. Agent stops and waits for the user to approve the plan ('agent:approved' or explicit chat approval).

Stage 2 — Implement and release
6. Agent sets 'agent:in-progress', creates feature/issue-<id>-<slug>, implements the approved scope, and opens a linked PR.
7. PR adds a user-facing CHANGELOG.md entry under [Unreleased] and may set one SemVer label.
8. CI runs tests, lint, formatting, and changelog validation; agent marks passing PR 'agent:review-ready'.
9. User reviews and merges the PR; the issue closes and the release workflow processes the merge.
```

### GitHub CLI Guidelines
- **Issue/PR Viewing**: Always pass `--json title,body,labels,comments` to `gh issue view` and `gh pr view` to avoid GraphQL `projectCards` deprecation errors.
- **PR Bodies**: Write PR descriptions containing Markdown, diagrams, or code blocks to a file and pass `--body-file <path>` to `gh pr create` / `gh pr edit` to avoid shell command substitution and redirection syntax errors.
- **PR Update Fallback**: If `gh pr edit` errors on `projectCards`, use `gh api repos/:owner/:repo/pulls/:number -X PATCH -F body=@<file>`.

### Label Taxonomy

| Label | Meaning | Who Applies |
| :--- | :--- | :--- |
| `agent:plan` | Ready for agent to formulate architecture plan | Human |
| `agent:plan-review` | Plan drafted in `docs/plans/`; awaiting human review | Agent |
| `agent:approved` | Plan approved; ready for implementation | Human |
| `agent:in-progress` | Agent is actively implementing code & tests | Agent |
| `agent:review-ready` | PR opened with passing tests; awaiting human review | Agent |
| `agent:blocked` | Agent blocked on user input or credentials | Agent |
| `agent:manual` | Human-only task; agent ignores | Human |

### Changelog and SemVer policy

Follow [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Every
releasable PR adds a concise, user-facing bullet under `CHANGELOG.md`'s
`[Unreleased]` heading, using the applicable `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, or `Security` category. Documentation-only or other changes
that should not trigger a release use `semver:none` and may omit a changelog entry.

Each PR may use at most one of these labels:

| PR label | Version impact |
| :--- | :--- |
| `semver:major` | Backward-incompatible change; increments the major version |
| `semver:minor` | Backward-compatible feature; increments the minor version |
| `semver:patch` | Backward-compatible fix; increments the patch version |
| `semver:none` | No release for this change |

If a PR has no `semver:*` label, automation defaults to a minor bump. Conflicting
or unknown `semver:*` labels fail validation. When multiple PRs have merged since
the previous release, the release uses the highest applicable bump; `semver:none`
PRs do not contribute a bump. The current `0.1.0` package version is the initial
baseline. Releases update `pyproject.toml` and `uv.lock`, move all Unreleased
entries into a dated version section, and create a `vX.Y.Z` tag and GitHub Release.

The PR CI workflow runs `uv run python -m pytest` and Ruff lint/format checks on
changed Python files. Releasable PRs must add a new Unreleased bullet.
After CI is active, repository maintainers should require the CI check in branch
protection. The release workflow needs Actions `contents: write` permission and
branch rules that allow `github-actions[bot]` to commit version/changelog updates
directly to `main`.

## Naming Standards

Plan files follow the naming convention:
```text
docs/plans/PLAN-<issue_number>-<slug>.md
```

Examples:
- `docs/plans/PLAN-3-option-to-make-it-better.md`
- `docs/plans/PLAN-30-security-findings-resolution.md`

Every non-trivial architectural change must adhere to [`PLAN-TEMPLATE.md`](PLAN-TEMPLATE.md) and specify architecture layer boundaries, TDD verification strategies, and statutory citations where applicable.
