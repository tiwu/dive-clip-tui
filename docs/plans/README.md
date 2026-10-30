# Architecture & Feature Plans

This directory contains version-controlled architectural specifications and implementation plans.

## Two-Stage Issue & Plan Lifecycle

```text
1. User queues issue with label 'agent:plan'
2. Agent inspects issue -> writes docs/plans/PLAN-<id>-<slug>.md -> comments on issue, attaches plan -> sets 'agent:plan-review'
3. User reviews plan -> sets label 'agent:approved'
4. Agent creates branch feature/issue-<id>-<slug> -> implements with TDD (>=90% cov) -> opens PR -> sets 'agent:review-ready'
5. User reviews PR -> merges to main -> issue closes
```

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

## Naming Standards

Plan files follow the naming convention:
```text
docs/plans/PLAN-<issue_number>-<slug>.md
```

Examples:
- `docs/plans/PLAN-3-option-to-make-it-better.md`
- `docs/plans/PLAN-30-security-findings-resolution.md`

Every non-trivial architectural change must adhere to [`PLAN-TEMPLATE.md`](PLAN-TEMPLATE.md) and specify architecture layer boundaries, TDD verification strategies, and statutory citations where applicable.
