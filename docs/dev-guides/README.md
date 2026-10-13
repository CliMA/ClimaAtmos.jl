# DeveloperGuides

Shared engineering standards, architectural patterns, and development guidelines for human and AI developers across the [CliMA](https://clima.caltech.edu) ecosystem.

## Guides

Every guide applies across the CliMA ecosystem unless it says otherwise.

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

---

## About DeveloperGuides

These guides are maintained in [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides) and vendored into consumer repos as a Git subtree at the standardized path `docs/dev-guides/`. The material below is for maintaining and consuming that subtree; readers looking for engineering guidance want the [Guides](#guides) overview above.

### Using the guides in a consumer repo

A consumer repo keeps its own `AGENTS.md` at the root, which references `docs/dev-guides/AGENTS.md` (the agent entry point) plus a repo-specific guide (e.g. `docs/clima_atmos_specific.md`). See [`templates/`](templates/) for ready-to-copy starter files: a root `AGENTS.md`, a repo-specific guide skeleton, and the monthly sync workflow.

```bash
# Add the subtree to a new consumer repo
git subtree add --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash

# Pull the latest guides manually (most repos automate this monthly via update_dev_guides.yml)
git subtree pull --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash \
    -m "chore: sync dev guides from central repo"
```

> [!NOTE]
> **Subtree pitfalls.**
>
> - `git subtree add --prefix docs/dev-guides ...` nests all of DeveloperGuides, including its own `AGENTS.md`, `LICENSE`, and `README.md`, under that prefix. It does not touch the consumer's root files, so the initial add does not conflict with them.
> - The real risk is editing the vendored copy under `docs/dev-guides/` directly instead of upstream (see "Contributing" below). A later `git subtree pull` merges upstream changes into that path, so a local edit there can produce a genuine merge conflict. Resolve it like any merge conflict: fix the conflicting file, `git add`, `git commit`. Subtree operations use merge, not rebase, so `git rebase --continue` does not apply.
> - When there are no new upstream commits the monthly run is a clean no-op. The workflow now fails loudly on a genuine `git subtree pull` error instead of masking it, so a red run means something actually needs attention.

### Fixing a broken subtree sync

The monthly sync breaks in one of two ways: a dev-guides PR was squash-merged — which discards the `git subtree` metadata the next pull relies on — or the workflow lacks the write permissions it needs to open a PR. If a repo's sync stopped producing PRs, apply whichever fix below it needs; most repos only need the first.

**1. Update the workflow file (do this on every consumer repo).** Replace the old workflow with the current template rather than hand-editing it:

```bash
git checkout -b update-dev-guides-workflow
mkdir -p .github/workflows
curl -fsSL https://raw.githubusercontent.com/CliMA/DeveloperGuides/main/templates/update_dev_guides.yml.template \
    -o .github/workflows/update_dev_guides.yml
git add .github/workflows/update_dev_guides.yml
git commit -m "ci: refresh dev-guides sync workflow from template"
git push -u origin update-dev-guides-workflow
```

Then open a PR for that branch and merge it normally — this PR does not touch the subtree, so squash is fine. Also make sure **Settings → Actions → General → "Allow GitHub Actions to create and approve pull requests"** is enabled, or the workflow will fail when it tries to open the monthly sync PR.

**2. Repair broken subtree metadata (only if a sync PR was ever squash-merged).** Symptom: a manual `git subtree pull` (or the workflow log) fails with `fatal: can't squash-merge: 'docs/dev-guides' was never added.` Remove and re-add the subtree:

```bash
git checkout -b fix-dev-guides-subtree
git rm -r docs/dev-guides
git commit -m "chore: remove dev-guides subtree (re-adding to fix metadata)"
git subtree add --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash
```

Open a PR for this branch and **merge it with a merge commit, not squash** — squash-merging here immediately re-breaks the metadata. Any local edits to files under `docs/dev-guides/` are discarded, which is correct: that copy is vendored and should only be changed upstream.

### Contributing

Edits to shared guidelines belong in [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides), not in the vendored copy inside a consumer repo. Open PRs there; once merged, the next subtree pull propagates them to every consumer.

- Each guide has a **Self-correction** section: if you discover a guide is stale or missing a pattern, update it directly.
- New guides go in the appropriate category directory and are added to this overview and to [AGENTS.md](AGENTS.md).
- Cross-references between guides use relative paths (e.g. `../performance/gpu_performance.md`).

### Repository layout

```text
├── AGENTS.md                  # Agent entry point: autonomy gate + guide index
├── README.md                  # This file: guide overview + repo info
├── architecture/              # System design, layering, contracts
├── performance/               # GPU, type stability, numerics, AD
├── code-quality/              # Style, docstrings, changelogs, naming
├── infrastructure/            # Testing, device abstraction
├── workflow/                  # Onboarding, debugging, review, CI triage
└── templates/                 # Starter files for consumer repos
```

### CliMA ecosystem

These guides are the central source of engineering standards across [CliMA](https://github.com/CliMA), including:

- [ClimaAtmos](https://github.com/CliMA/ClimaAtmos.jl)
- [ClimaCore](https://github.com/CliMA/ClimaCore.jl)
- [ClimaLand](https://github.com/CliMA/ClimaLand.jl)
- [ClimaOcean](https://github.com/CliMA/ClimaOcean.jl)
- [ClimaCoupler](https://github.com/CliMA/ClimaCoupler.jl)
- [Thermodynamics](https://github.com/CliMA/Thermodynamics.jl)
- [CloudMicrophysics](https://github.com/CliMA/CloudMicrophysics.jl)
- [SurfaceFluxes](https://github.com/CliMA/SurfaceFluxes.jl)
- [ClimaTimeSteppers](https://github.com/CliMA/ClimaTimeSteppers.jl)

### License

[![license][license-img]][license-url] Apache 2.0; see [LICENSE](LICENSE).

[license-img]: https://img.shields.io/github/license/CliMA/DeveloperGuides
[license-url]: https://github.com/CliMA/DeveloperGuides/blob/main/LICENSE

### Getting help

For questions or suggestions, open an issue on [GitHub](https://github.com/CliMA/DeveloperGuides/issues).
