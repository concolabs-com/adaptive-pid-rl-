# Contributing Guide

## Branch Naming

Use `{name}/{kebab-description}` — your first name or GitHub handle, then a short
description in lowercase kebab-case.

```
sahan/add-training-script
sahan/fix-reward-calculation
sahan/chapter-7-conclusion
intern/refactor-eval-loop
```

No type prefix in the branch name. Keep the description short (3–5 words max).

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<optional scope>): <short description>
```

**Types:**

| Type       | When to use                                      |
|------------|--------------------------------------------------|
| `feat`     | new feature or script                            |
| `fix`      | bug fix                                          |
| `docs`     | documentation, thesis chapters, markdown files   |
| `refactor` | code restructure with no behaviour change        |
| `test`     | adding or fixing tests                           |
| `chore`    | deps, config, tooling                            |
| `ci`       | GitHub Actions or CI/CD changes                  |
| `perf`     | performance improvement                          |

**Rules:**
- lowercase after the colon
- imperative mood: "add" not "added", "fix" not "fixed"
- no period at the end
- keep subject under 72 characters
- scope is optional but useful (e.g. `docs(chapter6)`, `fix(eval)`)

**Examples:**
```
feat(training): add curriculum phase scheduler
fix(eval): correct settling time dt from 0.002 to 0.02
docs(chapter6): add discussion and limitations chapter
chore(deps): pin mujoco to 3.2.0
ci: replace npm workflow with python lint
```

## Pull Requests

**PR title must follow Conventional Commits** — this is enforced by CI. The PR
title becomes the squash-merge commit message on main, so it must be valid.

- One PR per logical change
- Keep PRs small and focused
- Fill the PR template fully
- Add at least one label (enforced by CI)
- Do not commit secrets, credentials, or `.env` files

## Review Rules

- Code must be readable without inline explanation
- All CI checks must pass before merge
- Use squash merge — the PR title is the commit that lands on main
- Update documentation when behaviour changes
