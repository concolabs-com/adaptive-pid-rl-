# Contributing Guide

## Branch Naming
Use lowercase words separated by hyphens.

Examples:
```txt
feature/login-page
fix/auth-token-refresh
docs/update-readme
chore/update-dependencies
```

## Commit Naming
Use Conventional Commits.

Examples:
```txt
feat(auth): add Google login
fix(api): handle expired token
docs(readme): update setup guide
chore(deps): update packages
```

## Pull Requests
- Create one PR per logical change.
- Keep PRs small.
- Fill the PR template properly.
- Add screenshots for UI changes.
- Do not commit secrets or `.env` files.

## Review Rules
- Code should be readable and simple.
- Tests should pass before merge.
- Documentation should be updated when needed.
- Use squash merge for clean history.
