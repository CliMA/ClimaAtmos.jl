# Changelog Hygiene

This guide defines when and how to write `NEWS.md` entries across CliMA repositories.

## When to write a `NEWS.md` entry

Write an entry when any of the following changes:

- A new version of the package is released.
- A user-visible config key or CLI flag is added, renamed, or removed.
- A diagnostic output name or units change.
- A public API in the package's main module is added, changed, or removed.
- A Buildkite job name or config flag changes.

## When not to write an entry

Internal refactors with zero user-visible effect do not need a `NEWS.md` entry.

## Format

### Unreleased entries

During development, group entries under a `main` section at the top of `NEWS.md`. One bullet per change, plain English, include the PR number as a link.

### Cutting a release

When a new version is tagged, rename the `main` section to the version number with a `-------` underline, then add a new empty `main` section above it. The `main` section must always exist at the top of `NEWS.md` after a release cut, even if it has no entries yet — subsequent PRs append to it. Entries under a version are final and should not be modified after release.

Note: cutting a release is a maintainer action. Agents should add entries under `main`, not rename `main` to a version. See [agent_autonomy.md](agent_autonomy.md).

### Badges

Prefix entries with badge references to classify the change. The badge definitions live at the bottom of `NEWS.md` and must not be removed:

| Badge | When to use |
|:---|:---|
| `![][badge-💥breaking]` | Breaking changes: removed functions/types, API changes |
| `![][badge-🔥behavioralΔ]` | Behavioral changes: new model, different defaults |
| `![][badge-🤖precisionΔ]` | Machine-precision changes: reordered arithmetic |
| `![][badge-🚀performance]` | Performance improvements: fewer allocations, better inference |
| `![][badge-✨feature/enhancement]` | New features |
| `![][badge-🐛bugfix]` | Bug fixes |

### Example

```markdown
main
----

v0.39.0
-------
- ![][badge-💥breaking] Removed deprecated `old_config_key`. Use `new_config_key` instead.
  PR [#1234](https://github.com/CliMA/MyPackage.jl/pull/1234).
- ![][badge-✨feature/enhancement] Added `my_new_config_key` to control feature X.
  PR [#1235](https://github.com/CliMA/MyPackage.jl/pull/1235).

v0.38.4
-------
- ![][badge-🐛bugfix] Fixed incorrect surface flux calculation.
  PR [#1230](https://github.com/CliMA/MyPackage.jl/pull/1230).
```

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
