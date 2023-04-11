# User guide to regression tests

This document outlines how regression tests work and how to update PRs to pass regression tests.

## The basic idea of how regression tests work

When a particular job opts-in to testing regressions (using the `regression_test` command line option and the `julia --project=examples regression_tests/test_mse.jl` command), we compare the solution dataset (the prognostic state at the last timestep) of that job with a reference dataset.

We don't always have a reference to compare against, due to what we'll call **failure modes**. For a full list of failure modes, see [Failure modes](#Failure-modes), but here are a few examples:

 - There is no reference to compare against when we add a new experiment / buildkite job.

 - There is no reference to compare against when we add a new variable to opt-in for regression tests.

Our solution to dealing with failure modes is by providing users with two workflows: one when a comparable reference dataset exists (non-failure mode case), and another when it does not (failure mode case).

 - A comparable reference dataset exists:
   - [Update mse tables](#How-to-update-mse-tables)

 - A comparable reference dataset does **not** exists:
   - Increment the reference counter in `regression_tests/ref_counter.jl`. This triggers a "self-reference".
   - [Update mse tables](#How-to-update-mse-tables) _all to zero values_

At this moment, it's crucial to mention several important points:

 - When a reference dataset does not exist, we still perform a regression test so that we continuously exercise the testing infrastructure. However, we compare the solution dataset with itself (which we call a "self-reference"). Therefore, _all regression tests for all jobs will pass_ (no matter what the results look like) when the reference counter is incremented. So, it is important to review the quality of the results when the reference counter is incremented.

 - Every time the reference counter is incremented, data from that PR is saved onto Caltech's central cluster. And that solution's dataset is the new reference dataset that all future PRs are compared against (until the reference counter is incremented again).

## How to update mse tables

To update the mse tables:

 - Click the *Print new mse tables* buildkite job
 - Click the *Running commands* entry in the *Log* tab
 - Copy this output until `-- DO NOT COPY --`
 - Paste these contents into `regression_tests/mse_tables.jl`
 - Add, commit, and push these changes.

## Adding a new regression test

To add a new regression test:

 - Set the command-line `regression_test` to true, and add `julia --color=yes --project=examples regression_tests/test_mse.jl --job_id [job_id] --out_dir [job_id]` as a separate command for the new (or existing) job
 - Copy the `all_best_mse` dict template from the job's log
 - Paste the `all_best_mse` dict template into `regression_test/mse_tables.jl`

<!-- TODO: improve names / mark off sections for all_best_mse dict -->

## Failure modes

Here are some situations where we cannot (or cannot easily) compare against [existing] reference datasets. For example, when

 - a new regression test is added: no reference to compare against

 - a new variable is added to an existing regression test: no reference to compare against

 - a variable name has changed: cannot (easily) compare variables with two different names

 - the grid resolution has changed: see [A note on grid resolution failure mode]

### A note on grid resolution failure mode

We cannot (easily) compare the output with a reference if we change the spatial resolution (without interpolation). Coupling the regression infrastructure with interpolation would introduce a great deal of complexity since this depends on the type of grid (e.g., column, box, sphere), and details of those grids (e.g., radius of earth). Using interpolation in the regression infrastructure is not impossible, but it tightly couples details of the model / configuration, and introduces a lot of software complexity that results in specialized and difficult to maintain code.

# Developer guide to regression tests

## A detailed procedure of how regression tests are performed

Regression tests are performed at the end of `examples/hybrid/driver.jl`, after a simulation completes, and relies on a unique job id (`job_id`). Here is an outline of the regression test procedure:

 0) Run a simulation, with a particular `job_id`, to the final time.
 1) Load a dictionary, `all_best_mse`, of previous "best" mean-squared errors from `mse_tables.jl` and extract the mean squared errors for the given `job_id` (store in job-specific dictionary, `best_mse`).
 2) Export the solution (a `FieldVector`) at the final simulation time to an `NCDataset` file.
 3) Compute the errors between the exported solution and the exported solution from the reference `NCDataset` file (which is saved in a dedicated folder on the Caltech Central cluster) and save into a dictionary, called `computed_mse`.
 4) Export this dictionary (`computed_mse`) to the output folder
 5) Test that `computed_mse` is no worse than `best_mse` (determines if regression test passes or not).

After these steps are performed at the end of the driver, additional jobs are run:

 1) Print `computed_mse` for all jobs to make updating `regression_tests/mse_tables.jl` easy
 2) If we're on the staging branch (all tests have passed, and the PR is effectively merging), move the `NCDataset`s from the scratch directory onto the dedicated folder on the Caltech Central cluster.

## How we track which dataset to compare against

To think about tracking which dataset to compare against, it's helpful to consider the history of 1) commits into main and 2) commits of the reference data. Below is a toy example which also includes a "Reference counter" column:

```
Reference            hash of          hash of
 counter             merged           reference
ref_counter.jl       commit            commit
   1             =>  "V50XdC"  =>    "V50XdC" # Self reference
   1             =>  "lBKsAn"  =>    "V50XdC"
   1             =>  "Eh2ToX"  =>    "V50XdC"
   2             =>  "bnMLxi"  =>    "bnMLxi" # Self reference
   2             =>  "Jjx16f"  =>    "bnMLxi"
   3             =>  "dHkJqc"  =>    "dHkJqc" # Self reference
   3             =>  "SIgf1i"  =>    "dHkJqc"
   3             =>  "vTsCoY"  =>    "dHkJqc"
   3             =>  "VvCzAH"  =>    "dHkJqc"
```

The way this works is:

 1) We start off with a self reference: print a new reference
    counter in the `print new reference counter` job.

 2) (PR author) copy-paste counter into `regression_tests/ref_counter.jl`

 3) Upon next CI run, before performing CI test,
    we check if the counter indicates a self-reference by
    checking if `regression_tests/ref_counter.jl` in the PR
    matches (e.g.,) `aRsVoY/ref_counter.jl` in the last
    merged commit (on central). If yes, then it's a self
    reference, if not, then we look-up the dataset based
    on the counter.
