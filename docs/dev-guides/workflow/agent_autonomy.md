# Agent Autonomy Boundaries

This guide lists actions that an automated agent must **not** perform without an explicit, in-context instruction from the user. Local, reversible work (editing files, running tests, formatting) needs no special permission. The actions below are different: they are irreversible, externally visible, or scientifically consequential.

## Always require explicit user approval

### Git / GitHub

- `git push` to any remote, including the agent's own branch.
- `git push --force` (always — including on agent-owned branches).
- `git commit --amend` on a commit that has already been pushed.
- `git rebase` onto a different base, or any interactive rebase.
- `git reset --hard`, `git checkout --`, `git restore .`, or `git clean -fd` when uncommitted changes exist.
- Deleting branches (local or remote).
- Creating, merging, or closing pull requests.
- Adding labels, assignees, or reviewers; commenting on issues or PRs.
- Tagging a release, or pushing tags.

### Versioning and dependencies

- Bumping `version` in `Project.toml`. Version bumps belong to release PRs and are owned by maintainers.
- Editing `Manifest.toml`. Treat the manifest as derived state. Most CliMA repos do not commit it; if a repo does, do not modify it without explicit instruction.
- Adding, removing, or relaxing `[compat]` entries in `Project.toml`.
- Adding new entries to `[deps]` for an unrelated capability.
- `Pkg.update()` against the project environment.

### Reproducibility and reference data

- Editing reference counters or reference data in any reproducibility-test directory (e.g., `reproducibility_tests/` in ClimaAtmos). These reference values gate scientific reproducibility; only update them in response to a deliberate, user-confirmed output change.
- Editing MSE tolerance thresholds.
- Editing recorded checksums or golden output files.

### CI and infrastructure

- Editing `.buildkite/pipeline.yml` job definitions or job names.
- Editing `.github/workflows/*` or any CI configuration that affects what runs on every PR.
- Adding or removing CI jobs.
- Disabling tests (`@test_skip`, commenting out a `@testset`, removing a file from `runtests.jl`).
- Skipping pre-commit or git hooks (`--no-verify`).

### Public API and user-visible behavior

- Renaming or removing exported symbols.
- Renaming, removing, or changing the units of a diagnostic.
- Changing default values of user-visible config keys.
- Removing config keys, even if they appear unused.

## Allowed without prior approval

For contrast, the following are routine and do not require permission:

- Reading any file in the repository.
- Editing source under `src/`, `test/`, `docs/`, `ext/`.
- Running `julia` to type-check, run tests, or reproduce a bug.
- Running the formatter (`JuliaFormatter`) over changed files.
- Creating new local commits on the current working branch.
- Adding new tests that exercise a bug or new feature.
- Adding `NEWS.md` entries for changes the agent has made.

## When in doubt

If an action could be visible to other contributors, change scientific output, or be hard to reverse, ask. The cost of a one-line confirmation is small; the cost of an unwanted force-push or a silently shifted reference value is large.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
