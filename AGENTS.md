# CliMA Developer Guides — Agent Index

Read this file first. It is the main index for all shared engineering guidelines. Each guide applies across the CliMA ecosystem unless stated otherwise.

In consumer repos, these guides live at `docs/dev-guides/` and are supplied by a git subtree from the canonical source <https://github.com/CliMA/DeveloperGuides>. The consumer's root `AGENTS.md` references this file and the repo-specific guide. Edit shared guides in the canonical repo, not in the subtree copy.

## Before you act: agent autonomy

Before making changes that are externally visible or scientifically consequential (`git push`, version bumps, reproducibility-test edits, CI config changes, public API renames), check [workflow/agent_autonomy.md](workflow/agent_autonomy.md). The boundaries listed there require explicit user approval.

## Architecture

1. [repo_structure.md](architecture/repo_structure.md) — how to navigate any CliMA Julia package.
2. [ecosystem_conventions.md](architecture/ecosystem_conventions.md) — module aliases, `Y`/`Yₜ`/`p` state layout, `ᶜ`/`ᶠ` notation, CI structure, reproducibility, diagnostics.
3. [architectural_boundaries.md](architecture/architectural_boundaries.md) — layered architecture and boundary rules.
4. [cross_repo_contracts.md](architecture/cross_repo_contracts.md) — call-site conventions for ecosystem packages.
5. [dependency_management.md](architecture/dependency_management.md) — runtime vs dev deps, compat bounds.

## Performance

1. [gpu_performance.md](performance/gpu_performance.md) — GPU kernel rules, broadcast patterns, allocation avoidance.
2. [type_stability.md](performance/type_stability.md) — Float32 compatibility, inference checks, struct field rules.
3. [numerical_robustness.md](performance/numerical_robustness.md) — denominator regularization, clamping, NaN/Inf avoidance.
4. [ad_compatibility.md](performance/ad_compatibility.md) — AD-safe patterns for ForwardDiff and Enzyme.
5. [allocation_debugging.md](performance/allocation_debugging.md) — locating heap allocations with `Profile.Allocs`, JET, `@code_warntype`, flame graphs.

## Code Quality

1. [code_style.md](code-quality/code_style.md) — formatting, variable locality, Git workflow, feature removal, naming conventions.
2. [documentation_policy.md](code-quality/documentation_policy.md) — docstrings, repository-level docs, minimally viable documentation.
3. [changelogs_and_versions.md](code-quality/changelogs_and_versions.md) — `NEWS.md` format, SemVer rules, and the release/tagging flow.
4. [variable_list.md](code-quality/variable_list.md) — standardized CliMA variable naming conventions.
5. [software_design_patterns.md](code-quality/software_design_patterns.md) — numbered SDPs: branchless logic, functors, parameter extraction, etc.

## Infrastructure

1. [testing_and_validation.md](infrastructure/testing_and_validation.md) — type-stability checks, Aqua.jl, allocation regression, AD tests.
2. [clima_comms.md](infrastructure/clima_comms.md) — device-agnostic and MPI-distributed code patterns.

## Workflow

1. [onboarding.md](workflow/onboarding.md) — install Julia, clone a CliMA repo, set up Revise/Infiltrator/JuliaFormatter, first PR loop.
2. [agent_autonomy.md](workflow/agent_autonomy.md) — actions that require explicit user approval.
3. [debugging.md](workflow/debugging.md) — interactive debugging recipes: numerical instabilities, dispatch, `Field` plotting.
4. [review.md](workflow/review.md) — PR review instructions and checklist.
5. [ci_triage.md](workflow/ci_triage.md) — checklist for "passes locally, fails on CI" failure modes.
6. [cross_repo_issue_pr_search.md](workflow/cross_repo_issue_pr_search.md) — org-scoped GitHub search to find and filter issues/PRs across CliMA.

## Self-correction

If this index is discovered to be stale or missing a guide, update it.
