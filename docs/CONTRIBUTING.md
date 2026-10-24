# Contributing to FastAPI

Thank you for your interest in contributing to FastAPI! As an open-source community project, contributions from developers of all skill levels are warmly welcome.

## Ways to Contribute
You do not have to write code to contribute. You can:
1. **Help others with questions** on GitHub Discussions and Discord.
2. **Translate documentation** into your native language (check the `docs/<lang>` directory).
3. **Report bugs** with minimal reproducible examples.
4. **Fix open issues** tagged with `good first issue` or `help wanted`.
5. **Improve existing documentation** and fix typos.

## Finding an Issue to Work On
- Start by checking issues labeled with `good first issue` on GitHub. These are vetted tasks suitable for newcomers.
- Comment on the issue asking to be assigned before starting work to avoid duplicate efforts.
- If you have an idea for a brand new feature, create a GitHub Discussion first to get feedback from core maintainers before writing a large pull request.

## Pull Request Workflow
1. **Fork the repository** on GitHub to your personal account.
2. **Clone your fork** locally to your machine.
3. **Create a descriptive feature branch** from `master`:
   ```bash
   git checkout -b fix/issue-1234-query-param-validation
   ```
4. **Make your changes**: Keep changes focused and atomic. Avoid mixing unrelated refactors into a bug fix.
5. **Write tests**: Every bug fix or feature must include new or updated tests under `tests/`. PRs with reduced test coverage will not be merged.
6. **Run the test suite locally** and verify all tests pass:
   ```bash
   bash scripts/test.sh
   # or
   pytest
   ```
7. **Format and lint your code**:
   ```bash
   bash scripts/format.sh
   bash scripts/lint.sh
   ```
8. **Push to your fork** and submit a Pull Request to the `master` branch.
9. **PR Title and Description**:
   - Use conventional commit title style (e.g., `fix: resolve query parameter type coercion`).
   - Fill out the PR template completely.
   - Reference the issue number (e.g., `Fixes #1234`).
10. **Review Process**: At least one core maintainer approval is required. Be prepared to address feedback and review comments politely.
