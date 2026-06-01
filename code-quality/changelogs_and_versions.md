# Changelogs and Versioning

This guide covers two related but separate concerns:

- **The changelog (`NEWS.md`)** — a human-readable log of what changed in each release. Updated continuously, on every user-visible PR.
- **The package version (`Project.toml`)** — a single numeric string that signals *how breaking* a change is to downstream consumers. Bumped only when a release is cut.

Most PRs touch only `NEWS.md`. Only release PRs touch the version. The two come together when a release is *cut* (Part 3 below).

---

## Part 1 — The changelog (`NEWS.md`)

### 1.1 What goes in

Add an entry to `NEWS.md` when any of the following changes:

- A user-visible config key or CLI flag is added, renamed, or removed.
- A diagnostic output name or units change.
- A public API in the package's main module is added, changed, or removed.
- A Buildkite job name or config flag changes.
- A new version of the package is released (the release PR itself adds the section header — see Part 3).

### 1.2 What doesn't

Internal refactors with zero user-visible effect do not need a `NEWS.md` entry. Performance improvements and bug fixes generally *do* deserve an entry, because users care about both.

### 1.3 Layout

At the top of `NEWS.md` is an always-open `main` section that accumulates unreleased entries; below it are frozen sections, one per released version, in reverse chronological order. The `main` section must always exist, even immediately after a release cut.

```markdown
main
----

- One bullet per change, plain English, past tense.
- Include the PR number as a link: PR [#1234](https://github.com/CliMA/MyPackage.jl/pull/1234).

v0.39.0
-------

- ...entries from before the last release...
```

One bullet per change. Entries under a released version are final and should not be modified after release.

### 1.4 Badges

Some repos (notably `ClimaAtmos`, `ClimaCore`, `ClimaTimeSteppers`) prefix entries with a badge that classifies the change. The badge definitions live at the bottom of `NEWS.md` and must not be removed.

| Badge                              | When to use                                                   |
|:-----------------------------------|:--------------------------------------------------------------|
| `![][badge-💥breaking]`            | Breaking changes: removed functions/types, API changes        |
| `![][badge-🔥behavioralΔ]`         | Behavioral changes: new model, different defaults             |
| `![][badge-🤖precisionΔ]`          | Machine-precision changes: reordered arithmetic               |
| `![][badge-🚀performance]`         | Performance improvements: fewer allocations, better inference |
| `![][badge-✨feature/enhancement]` | New features                                                  |
| `![][badge-🐛bugfix]`              | Bug fixes                                                     |

Smaller library repos (`CloudMicrophysics`, `SurfaceFluxes`) use plain-text entries without badges. Match the convention already in the repo's `NEWS.md`.

### 1.5 Example

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

---

## Part 2 — The package version (`Project.toml`)

### 2.1 Where it lives

The authoritative version string is the `version = "..."` line near the top of the package's root `Project.toml`. There is exactly one such string per package; it is what the Julia General registry records and what `Pkg.status` reports to downstream users. The version is *not* derived from `NEWS.md` headers or from Git tags; those are downstream of `Project.toml`.

```toml
name = "MyPackage"
uuid = "..."
authors = ["Climate Modeling Alliance"]
version = "0.38.4"
```

### 2.2 Bump rules: Julia's modified SemVer

CliMA packages follow [Julia's modified Semantic Versioning](https://pkgdocs.julialang.org/v1/compatibility/) — the same scheme the General registry enforces. The rules differ for pre-1.0 and post-1.0 packages.

In practice, some CliMA packages deliberately diverge from strict SemVer. ClimaLand, despite being post-1.0, continues to treat the MINOR slot as the breaking slot and reserves a major bump (2.0) for a major milestone (e.g., aligned with a paper submission). Release cadence is also driven by downstream-coupling needs rather than API-stability milestones. Don't infer "no breaking changes" from the absence of a major bump — consult `NEWS.md` for the authoritative record.

| Package state | Bump                          | Format              | Meaning                                                                                                              |
|:--------------|:------------------------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------|
| **Post-1.0**  | Major (`1.x.y` → `2.0.0`)     | `MAJOR.MINOR.PATCH` | Breaking change to the public API (removed/renamed symbol, changed signature, changed default that affects results). |
|               | Minor (`1.2.x` → `1.3.0`)     | `MAJOR.MINOR.PATCH` | Non-breaking new feature or additive API surface.                                                                    |
|               | Patch (`1.2.0` → `1.2.1`)     | `MAJOR.MINOR.PATCH` | Bug fix or internal change with no API or behavioral effect on callers.                                              |
| **Pre-1.0**   | "Major" (`0.14.x` → `0.15.0`) | `0.MINOR.PATCH`     | Breaking change — when `MAJOR == 0`, the `MINOR` slot is the breaking slot.                                          |
|               | "Patch" (`0.14.5` → `0.14.6`) | `0.MINOR.PATCH`     | Anything non-breaking — both new features and bug fixes share this slot pre-1.0.                                     |

Most CliMA packages are still pre-1.0, so for them a `0.X.0` bump signals a breaking change. A handful of packages use a two-component `0.X` form, in which case `X` is the breaking slot. Check the existing `version` line in the package's `Project.toml` before bumping to confirm which regime applies.

### 2.3 What counts as "breaking"

A change is breaking if it alters the **public surface** of the package:

- Exported symbols (renames, removals, signature changes).
- Documented function signatures and default values.
- Documented config keys and CLI flags.
- Diagnostic names and units.
- Output recorded in reproducibility tests.

A change is *not* breaking just because it shifts bit-level results — performance refactors, fused broadcasts, and reordered arithmetic are flagged with `🤖precisionΔ` in `NEWS.md` (see §1.4), not by bumping the breaking slot.

Test for breaking-ness: *can a downstream package, pinned to the current compat, keep working without modification?* If the answer is no, bump the breaking slot.

### 2.4 Merging a PR with breaking changes

When merging a PR that contains breaking changes, ensure that the `NEWS.md` entry for the change is properly tagged with `![][badge-💥breaking]`. It is your responsibility to open PRs in downstream repositiories that will be affected when the breaking release is made. Ideally, the PRs in the downstream repositories should be opened before the breaking release is made. They can be tested with the new changes by `dev`ing the branch of the upstream package's repo.

---

## Part 3 — Cutting a release

A "release" is the moment the version in `Project.toml` becomes immutable in the Julia General registry. The mechanics tie Parts 1 and 2 together.

### 3.1 The release PR

Cutting a release is a maintainer action and should land in **its own PR**, distinct from the changes it releases. The release PR contains exactly:

1. The new version string in `Project.toml`.
2. In `NEWS.md`: the `main` section header renamed to the new version (with `-------` underline), and a new empty `main` section opened above it.
3. Any `[compat]` updates the release requires.

That's it — no functional source changes belong in the release PR.

### 3.2 Tagging mechanism

CliMA packages register through the Julia General registry; GitHub tags are created automatically afterwards. The flow:

1. Maintainer merges the release PR (Step 3.1).
2. Maintainer comments `@JuliaRegistrator register` on the release commit on GitHub. This opens a PR in `JuliaRegistries/General` recording the new version.
3. Once the registry PR auto-merges (typically within 15 minutes if compat checks pass), Julia's TagBot reacts by creating the GitHub tag `v<version>` and a GitHub Release pointing at the merge commit.

The `.github/workflows/TagBot.yml` workflow that ships in every CliMA repo wires this last step — it triggers on the comment from `JuliaTagBot` and runs `JuliaRegistries/TagBot@v1`. Maintainers do not manually push tags; pushing a tag by hand bypasses the registry and breaks downstream resolvers.

### 3.3 Agent autonomy

Cutting a release is a maintainer action. Agents must not:

- Bump `version` in `Project.toml`.
- Rename `NEWS.md`'s `main` section to a version number.
- Comment `@JuliaRegistrator register` or push tags.

Agents *should* add `NEWS.md` entries under `main` for their own changes. See [agent_autonomy.md](../workflow/agent_autonomy.md).

---

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
