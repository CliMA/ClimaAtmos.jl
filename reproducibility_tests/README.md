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
 3. Add the job ID to `reproducibility_test_job_ids.jl`.

## Developer details

### Procedure

1. Run simulation to final time with a unique `job_id`.
2. Export the final prognostic `FieldVector` to `prog_state.hdf5`.
3. Compute RMS differences against reference datasets using `compute_rms`.
4. Write results to `computed_rms_<commit_hash>.dat` files.
5. Test that all RMS differences are within tolerance using `test_reproducibility`.
6. On merge-queue or main, move data to the central cluster as the new reference.

### Reference tracking

The reference counter in `ref_counter.jl` partitions the commit history into "bins." Only commits with matching reference counters are compared. See the detailed documentation in `reproducibility_utils.jl` for the bin management algorithm.
