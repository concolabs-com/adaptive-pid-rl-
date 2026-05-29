# Contributing Guide

## Workflow: Issue → Branch → PR

Every change must start with a GitHub issue. The issue number links the branch,
commits, and PR together so changes are traceable and reversible.

```
1. Create or pick an issue (e.g. #42)
2. Create a branch: {name}/{issue-number}-{description}
3. Make commits following Conventional Commits
4. Open a PR with "Closes #42" in the description
5. PR is squash-merged — the PR title lands on main as a single commit
```

---

## 1. Issues

Open an issue before starting any work. Use the appropriate template:

| Template | When to use |
|----------|-------------|
| **Bug Report** | Something is broken or produces wrong output |
| **Feature Request** | New script, model, training config, or capability |
| **Task** | Refactor, cleanup, or maintenance work |
| **Documentation** | Thesis chapters, README, or guide updates |
| **Security Report** | Security concern — do not include exploit details |

Issue titles should follow the same Conventional Commits format as commits:
```
fix: settling time dt uses wrong timestep
feat: add recurrent GRU policy option
docs: write chapter 7 conclusion
```

---

## 2. Branch Naming

```
{name}/{issue-number}-{kebab-description}
```

- `{name}` — your first name or GitHub handle
- `{issue-number}` — the GitHub issue number this branch addresses
- `{kebab-description}` — short description in lowercase kebab-case

**Examples:**
```
sahan/42-fix-settling-time-dt
sahan/17-add-curriculum-scheduler
intern/55-chapter-6-discussion
```

No type prefix in the branch name — the issue number provides the link.

---

## 3. Commit Messages

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
- Lowercase after the colon
- Imperative mood: "add" not "added", "fix" not "fixed"
- No period at the end
- Subject under 72 characters
- Scope is optional but useful: `docs(chapter6)`, `fix(eval)`

**Examples:**
```
feat(training): add curriculum phase scheduler
fix(eval): correct settling time dt from 0.002 to 0.02
docs(chapter6): add discussion and limitations chapter
chore(deps): pin mujoco to 3.2.0
ci: replace npm workflow with python lint
```

---

## 4. Pull Requests

**PR title must follow Conventional Commits** — enforced by CI. The PR title
becomes the squash-merge commit on `main`, so it must be valid.

**Always include `Closes #N`** in the PR description. This:
- Links the PR to the issue on GitHub
- Auto-closes the issue when the PR is merged
- Makes revert traceable — `git revert <commit>` on main maps back to the issue

**Labels are applied automatically** from the PR title type prefix. You do not
need to add labels manually.

**PR rules:**
- One PR per issue
- Keep PRs small and focused
- Fill the PR template fully
- Do not commit secrets, credentials, or `.env` files
- All CI checks must pass before requesting review

---

## 5. Review and Merge

- Use **squash merge** — one commit per PR lands on `main`
- The PR title is the commit message — make it meaningful
- Reviewer approves before merge
- Delete the branch after merge

---

## Reverting a Change

Because every PR maps to one issue and one squash commit:

```bash
# Find the commit for a feature
git log --oneline main

# Revert it
git revert <commit-hash>

# Open a PR for the revert referencing the original issue
# Title: revert: feat(training) add curriculum phase scheduler (#42)
```
