# User guide to reproducibility tests

This document outlines how reproducibility tests work and how to update PRs to pass reproducibility tests.

## The basic idea

When a job opts into reproducibility testing (via the `reproducibility_test` command-line option), we compare its final prognostic state against a reference dataset stored on Caltech's central cluster.

The comparison computes **root-mean-square (RMS) differences** for each prognostic variable, reports them in a table with physical labels and SI units, and applies a tolerance of `eps_factor * eps(Float32) * data_scale`, where `data_scale` is the mean absolute value of the reference field. This means tiny floating-point perturbations (e.g., from reordered reductions or dependency updates) pass automatically, while genuine physics changes are flagged.

### What the output looks like

```
── Reference: abc1234 ──
┌───────────────────────────┬────────┬──────────┬────────────┬──────────┬────────┐
│ Variable                  │ Unit   │ RMS Diff │ Data Scale │ Tolerance│ Status │
├───────────────────────────┼────────┼──────────┼────────────┼──────────┼────────┤
│ ρ (density)               │ kg/m³  │ 0.00e+00 │ 1.17e+00   │ 1.40e-06 │   🟢   │
│ ρe_tot (total energy)     │ J/m³   │ 0.00e+00 │ 2.50e+05   │ 2.98e-01 │   🟢   │
│ u₁ (covariant horiz. u₁)  │ m/s    │ 0.00e+00 │ 1.23e+01   │ 1.47e-05 │   🟢   │
│ u₃ (covariant vert. u₃)   │ m/s    │ 0.00e+00 │ 3.45e-02   │ 1.19e-06 │   🟢   │
└───────────────────────────┴────────┴──────────┴────────────┴──────────┴────────┘
```

## Failure modes

We cannot compare against existing references when:
 - A **new reproducibility test** is added (no reference exists).
 - A **new variable** is added to the prognostic state.
 - A **variable name** has changed.
 - The **grid resolution** has changed (comparison requires matching grids).

In any of these cases, the reference counter must be incremented.

## How to merge PRs that break reproducibility tests

### Case 1: Comparable references exist, but RMS exceeds tolerance

Your PR genuinely changes the physics. Follow these steps:
 1. Increment the reference counter in `reproducibility_tests/ref_counter.jl`.
 2. Add a brief changelog entry explaining what changed.
 3. Once merged, the PR's output automatically becomes the new reference.

### Case 2: No comparable references exist

This happens when you've added a new job, changed variable names, or changed grid resolution:
 1. Increment the reference counter in `reproducibility_tests/ref_counter.jl`.
 2. Add a brief changelog entry.

## Allowing flaky tests

Add `--test_broken_report_flakiness true` to the `test_reproducibility.jl` command in a Buildkite job. This uses `@test_broken` so the job passes even if not yet reproducible, and alerts you when it becomes reproducible.

## Adding a new reproducibility test

 1. Set `reproducibility_test: true` in your config YAML.
 2. Add `julia --project=.buildkite reproducibility_tests/test_reproducibility.jl --job_id [job_id] --out_dir [job_id]/output_active` as a separate command in the pipeline.

The job ID is discovered automatically from the presence of `[job_id]/output_active/reproducibility_bundle/prog_state.hdf5` after the run.

## Developer details

### Procedure

1. Run simulation to final time with a unique `job_id`.
2. Export the final prognostic `FieldVector` to `prog_state.hdf5`.
3. Compute RMS differences against reference datasets using `compute_rms`.
4. Write results to `computed_rms_<commit_hash>.dat` files.
5. Test that all RMS differences are within tolerance using `test_reproducibility`.
6. Publish the data to the central cluster as the new reference (see below).

### Publishing references on merge

The new reference is published from data the PR already computed, immediately
upon merge. It's a two-step **stage → publish** flow:

```
   PR build (any branch)          merge to main
   ────────────────────           ─────────────
   compute bundle                 publish_reference.jl
        │                              │
        │ stage_output.jl              │ move + prune
        │ → stage_pr_data              ▼
        ▼                          reference store
   staging area          ───────▶  climaatmos-main/<merge_sha>/
   climaatmos-main-staging/pr-<n>/   (what later PRs compare against)
```

1. **Stage**: on a green PR/feature-branch build, the final step
   (`stage_output.jl` → `stage_pr_data`) copies the bundle to a per-PR staging
   directory `/resnick/scratch/esm/slurm-buildkite/climaatmos-main-staging/pr-<n>/reproducibility_bundle/`.
   Overwrites on each push, so at most one bundle is kept per open PR. Nothing is
   published yet.
2. **Publish**: on merge, `climaatmos-ci` builds `main`.
   `.buildkite/pipeline.yml` dispatches by branch inline: on `main` every job
   step is skipped (`if: build.branch != "main"`) and only the "Move
   reproducibility results" step runs `publish_reference.jl`. That resolves the
   merged PR number (`discover_pr_number`), moves `pr-<n>/reproducibility_bundle`
   into the reference store under the merge commit, and prunes old references.

Splitting it in two lets the bundle be computed *before* merge but only become the
official reference *if/when* the PR actually merges. This relies on branch
protection requiring PRs to be up to date with `main` before merging, so the PR
build's state matches the merged state. References are discovered by folder name +
`ref_counter.jl` value rather than by git history (`sorted_dirs_with_matched_files`),
so a reference stored under the merge commit is found by later PRs exactly as before.

#### If a merge queue is enabled

The same code handles a GitHub merge queue with no changes. A merge-queue build
(`gh-readonly-queue/main/...`) re-runs the full suite and its commit is the merge
candidate, so `stage_output.jl` publishes directly there
(`move_data_to_save_dir`) and drops the redundant `pr-<n>/` staging. The
subsequent `main` build's `publish_reference.jl` is idempotent — it finds the
reference already published (or no staging) and does nothing — so there is no
double publish. `discover_pr_number` reads the PR number from the
`gh-readonly-queue/main/pr-<n>-...` branch. In that mode the PR-build staging is
just a harmless safety net.

### Reference tracking

The reference counter in `ref_counter.jl` partitions the commit history into "bins." Only commits with matching reference counters are compared. See the detailed documentation in `reproducibility_utils.jl` for the bin management algorithm.
