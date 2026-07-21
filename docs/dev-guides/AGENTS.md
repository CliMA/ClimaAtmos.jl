# CliMA Developer Guides: Agent Entry Point

Read this file first. It is the agent entry point for the shared engineering guidelines. Each guide applies across the CliMA ecosystem unless stated otherwise.

In consumer repos, these guides live at `docs/dev-guides/` and are supplied by a git subtree from the canonical source <https://github.com/CliMA/DeveloperGuides>. The consumer's root `AGENTS.md` references this file and the repo-specific guide. Edit shared guides in the canonical repo, not in the subtree copy.

## Before you act: agent autonomy

Local, reversible work (editing files, running tests, formatting, committing to the current branch) needs no permission. Get explicit user approval before any irreversible, externally visible, or scientifically consequential action:

- **Git/GitHub**: `git push`, force-push, rebasing or amending pushed commits, `git reset --hard` or `git clean` with uncommitted changes, deleting branches, opening/merging/closing PRs, commenting on issues or PRs, tagging or pushing releases.
- **Versioning and dependencies**: bumping `version` in `Project.toml`, editing `Manifest.toml`, changing `[compat]` or `[deps]`, `Pkg.update()`.
- **Reproducibility data**: editing reference counters, MSE tolerances, or checksum/golden files in reproducibility-test directories. Change these only for a user-confirmed output change.
- **CI and infrastructure**: editing `.buildkite/pipeline.yml` or `.github/workflows/*`, adding or removing CI jobs, disabling tests, skipping hooks (`--no-verify`).
- **Public API and user-visible behavior**: renaming or removing exported symbols, renaming diagnostics or changing their units, changing or removing user-visible config keys.

The full enumeration and the allowed-without-approval list are in [workflow/agent_autonomy.md](workflow/agent_autonomy.md). When in doubt, ask.

## Guides

### Architecture

- [repo_structure.md](architecture/repo_structure.md): how to navigate any CliMA Julia package.
- [ecosystem_conventions.md](architecture/ecosystem_conventions.md): module aliases, `Y`/`Yₜ`/`p` state layout, `ᶜ`/`ᶠ` notation, CI structure, reproducibility, diagnostics.
- [architectural_boundaries.md](architecture/architectural_boundaries.md): layered architecture and boundary rules.
- [cross_repo_contracts.md](architecture/cross_repo_contracts.md): call-site conventions for ecosystem packages.
- [dependency_management.md](architecture/dependency_management.md): runtime vs dev dependencies, compat bounds.

### Performance

- [gpu_performance.md](performance/gpu_performance.md): GPU kernel rules, broadcast patterns, allocation avoidance.
- [branchless_code.md](performance/branchless_code.md): avoiding warp divergence with `ifelse`, evaluate-both-cases splits, and fixed-iteration solvers chosen by offline tests.
- [type_stability.md](performance/type_stability.md): Float32 compatibility, inference checks, struct field rules.
- [numerical_robustness.md](performance/numerical_robustness.md): denominator regularization, clamping, NaN/Inf avoidance.
- [ad_compatibility.md](performance/ad_compatibility.md): AD-safe patterns for ForwardDiff and Enzyme.
- [allocation_debugging.md](performance/allocation_debugging.md): locating heap allocations with `Profile.Allocs`, JET, `@code_warntype`, flame graphs.

### Code Quality

- [getting_started.md](code-quality/getting_started.md): orienting newcomers to writing pointwise code compatible with ClimaCore `Field`s and broadcasting.
- [code_style.md](code-quality/code_style.md): formatting, variable locality, Git workflow, feature removal, naming conventions.
- [documentation_policy.md](code-quality/documentation_policy.md): docstrings, repository-level docs, minimally viable documentation.
- [changelogs_and_versions.md](code-quality/changelogs_and_versions.md): `NEWS.md` format, SemVer rules, and the release/tagging flow.
- [variable_list.md](code-quality/variable_list.md): standardized CliMA variable naming conventions.
- [glossary.md](code-quality/glossary.md): general CliMA software and simulation terminology.
- [software_design_patterns.md](code-quality/software_design_patterns.md): numbered SDPs for branchless logic, functors, parameter extraction, and more.

### Infrastructure

- [testing_and_validation.md](infrastructure/testing_and_validation.md): type-stability checks, Aqua.jl, allocation regression, AD tests.
- [clima_comms.md](infrastructure/clima_comms.md): device-agnostic and MPI-distributed code patterns.

### Workflow

- [onboarding.md](workflow/onboarding.md): install Julia, clone a CliMA repo, set up Revise/Infiltrator/JuliaFormatter, first PR loop.
- [running_on_gpu.md](workflow/running_on_gpu.md): run a model on GPU — install Julia, add `CUDA.jl`, CUDA runtime compatibility, `CLIMACOMMS_DEVICE`, verify the device.
- [agent_autonomy.md](workflow/agent_autonomy.md): actions that require explicit user approval.
- [debugging.md](workflow/debugging.md): interactive debugging recipes for numerical instabilities, dispatch, and `Field` plotting.
- [review.md](workflow/review.md): PR review instructions and checklist.
- [ci_triage.md](workflow/ci_triage.md): checklist for "passes locally, fails on CI" failure modes.
- [cross_repo_issue_pr_search.md](workflow/cross_repo_issue_pr_search.md): org-scoped GitHub search to find and filter issues/PRs across CliMA.

## Self-correction

If this index is stale or missing a guide, update it here and in the matching [README.md](README.md#guides) overview.
