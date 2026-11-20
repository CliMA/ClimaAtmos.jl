# Agent Autonomy Boundaries

This guide lists actions that an automated agent performs only on an explicit, in-context instruction from the user: actions that are irreversible, externally visible, or scientifically consequential. Local, reversible work (editing files, running tests, formatting) needs no permission.

## Always require explicit user approval

### Git / GitHub

- `git push` to any remote, including the agent's own branch.
- `git push --force` (always, including on agent-owned branches).
- `git commit --amend` on a commit that has already been pushed.
- `git rebase` onto a different base, or any interactive rebase.
- `git reset --hard`, `git checkout --`, `git restore .`, or `git clean -fd` when uncommitted changes exist.
- Deleting branches (local or remote).
- Creating, merging, or closing pull requests.
- Adding labels, assignees, or reviewers; commenting on issues or PRs.
- Tagging a release, or pushing tags.

### Versioning and dependencies

- Bumping `version` in `Project.toml`. Version bumps belong to release PRs and are owned by maintainers.
- Renaming the `main` section of `NEWS.md` to a version number (part of cutting a release; see [changelogs_and_versions.md §3](../code-quality/changelogs_and_versions.md)).
- Editing `Manifest.toml` in a repo that commits it. Treat the manifest as derived state.
- Adding, removing, or relaxing `[compat]` entries in `Project.toml`.
- Adding new entries to `[deps]` for an unrelated capability.
- `Pkg.update()` against the project environment.

### Reproducibility and reference data

- Editing reference counters or reference data in any reproducibility-test directory (e.g., `reproducibility_tests/` in ClimaAtmos). These values gate scientific reproducibility; update them only for a user-confirmed output change.
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

The following are routine:

- Reading any file in the repository.
- Editing source under `src/`, `test/`, `docs/`, `ext/`.
- Running `julia` to type-check, run tests, or reproduce a bug.
- Running the formatter (`JuliaFormatter`) over changed files.
- Creating new local commits on the current working branch.
- Adding new tests that exercise a bug or new feature.
- Adding `NEWS.md` entries for changes the agent has made.

## When in doubt

If an action could be visible to other contributors, change scientific output, or be hard to reverse, ask.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
