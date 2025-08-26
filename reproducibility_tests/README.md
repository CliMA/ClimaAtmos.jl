# User guide to reproducibility tests

This document outlines how reproducibility tests work and how to update PRs to pass reproducibility tests.

## The basic idea of how reproducibility tests work

When a particular job opts-in to testing reproducibilitys (using the `reproducibility_test` command line option and the `julia --project=.buildkite reproducibility_tests/test_mse.jl` command), we compare the solution dataset (the prognostic state at the last timestep) of that job with a reference dataset.

We don't always have a reference to compare against, due to what we'll call **failure modes**. For a full list of failure modes, see [Failure modes](#Failure-modes), but here are a few examples:

 - There is no reference to compare against when we add a new experiment / buildkite job.

 - There is no reference to compare against when we add a new variable to opt-in for reproducibility tests.

Our solution to dealing with failure modes is by providing users with two workflows: one when a comparable reference dataset exists (non-failure mode case), and another when it does not (failure mode case).

 - A comparable reference dataset exists:
   - [Update mse tables](#How-to-update-mse-tables)

 - A comparable reference dataset does **not** exists:
   - Increment the reference counter in `reproducibility_tests/ref_counter.jl`.
   - [Update mse tables](#How-to-update-mse-tables) _all to zero values_

At this moment, it's crucial to mention several important points:

 - When a reference dataset does not exist, we still perform a reproducibility test so that we continuously exercise the testing infrastructure. However, we compare the solution dataset with itself. Therefore, _all reproducibility tests for all jobs will pass_ (no matter what the results look like) when the reference counter is incremented. So, it is important to review the quality of the results when the reference counter is incremented.

 - When a PR passes CI on buildkite while in the github merge queue, or when a PR lands on the main branch, data from the HEAD commit of that PR is saved onto Caltech's central cluster. And that solution's dataset is the new reference dataset that all future PRs are compared against (until the reference counter is incremented again). So, a PR will have some number of comparable references (including zero). For example, if we line up pull requests in the order that they are merged:

```
0186_73hasd ...

0187_73hasd # PR 1000 has 0 comparable references
0187_fgsae7 # PR 2309 has 1 comparable references
0187_sdf63a # PR 1412 has 2 comparable references

0188_73hasd # PR 2359 has 0 comparable references

0189_sdf63a # PR 9346 has 0 comparable references
0189_73hasd # PR 3523 has 1 comparable references
...
```

Note: We currently do not prepend the folder names by the reference counter, however, we will make this improvement soon.

## Allowing flaky tests

Users can add the flag `test_broken_report_flakiness` to the `test_mse.jl` script: `julia --project=.buildkite reproducibility_tests/test_mse.jl --test_broken_report_flakiness true`, which will have the following behavior:

 - If the test is not reproducible (i.e., flaky) when compared against `N` comparable references, then the test will pass and be reported as broken.
 - If the test is reproducible when compared against `N` comparable references, then the test will fail `@test_broken`, and users will be asked to fix the broken test. At which point you can remove the `--test_broken_report_flakiness true` flag from that particular job, reinforcing a strict reproducibility constraint.

## How to update mse tables

To update the mse tables:

 - Click the *Print new mse tables* buildkite job
 - Click the *Running commands* entry in the *Log* tab
 - Copy this output until `-- DO NOT COPY --`
 - Paste these contents into `reproducibility_tests/reproducibility_test_job_ids.jl`
 - Add, commit, and push these changes.

## Adding a new reproducibility test

To add a new reproducibility test:

 - Set the command-line `reproducibility_test` to true, and add `julia --color=yes --project=.buildkite reproducibility_tests/test_mse.jl --job_id [job_id] --out_dir [job_id]` as a separate command for the new (or existing) job
 - Copy the `all_best_mse` dict template from the job's log
 - Paste the `all_best_mse` dict template into `reproducibility_test/reproducibility_test_job_ids.jl`

<!-- TODO: improve names / mark off sections for all_best_mse dict -->

## Failure modes

Here are some situations where we cannot (or cannot easily) compare against [existing] reference datasets. For example, when

 - a new reproducibility test is added: no reference to compare against

 - a new variable is added to an existing reproducibility test: no reference to compare against

 - a variable name has changed: cannot (easily) compare variables with two different names

 - the grid resolution has changed: see [A note on grid resolution failure mode]

### A note on grid resolution failure mode

We cannot (easily) compare the output with a reference if we change the spatial resolution (without interpolation). Coupling the reproducibility infrastructure with interpolation would introduce a great deal of complexity since this depends on the type of grid (e.g., column, box, sphere), and details of those grids (e.g., radius of earth). Using interpolation in the reproducibility infrastructure is not impossible, but it tightly couples details of the model / configuration, and introduces a lot of software complexity that results in specialized and difficult to maintain code.

# Developer guide to reproducibility tests

## A detailed procedure of how reproducibility tests are performed

Reprodicibility tests are performed at the end of `.buildkite/ci_driver.jl`, after a simulation completes, and relies on a unique job id (`job_id`). Here is an outline of the reproducibility test procedure:

 0) Run a simulation, with a particular `job_id`, to the final time.
 1) Load a list of job IDs in `reproducibility_test_job_ids.jl`.
 2) Export the solution (a `FieldVector`) at the final simulation time to an `NCDataset` file.
 3) Compute the errors between the exported solution and the exported solution from the reference `NCDataset` files (which are saved in a dedicated folders on the Caltech Central cluster) and save into a dictionary, called `computed_mse`.
 4) Export this dictionary (`computed_mse`) to the output folder
 5) Test that `computed_mse` is no worse than `best_mse` (determines if reproducibility test passes or not).

After these steps are performed at the end of the driver, additional jobs are run:

 1) Print `computed_mse` for all jobs
 2) If we're on the github queue merging branch (all tests have passed, and the PR is effectively merging), move the `NCDataset`s from the scratch directory onto the dedicated folder on the Caltech Central cluster.

## How we track which dataset to compare against

To think about tracking which dataset to compare against, it's helpful to consider the history of 1) commits into main and 2) commits of the reference data. Below is a toy example which also includes a "Reference counter" column:

```
Reference            hash of          hash of
 counter             merged           reference
ref_counter.jl       commit            commit
   1             =>  "V50XdC"  =>    "V50XdC" # no comparable references
   1             =>  "lBKsAn"  =>    "V50XdC"
   1             =>  "Eh2ToX"  =>    "V50XdC"
   2             =>  "bnMLxi"  =>    "bnMLxi" # no comparable references
   2             =>  "Jjx16f"  =>    "bnMLxi"
   3             =>  "dHkJqc"  =>    "dHkJqc" # no comparable references
   3             =>  "SIgf1i"  =>    "dHkJqc"
   3             =>  "vTsCoY"  =>    "dHkJqc"
   3             =>  "VvCzAH"  =>    "dHkJqc"
```

The way this works is:

 1) We start off with no comparable references: print a new reference
    counter in the `print new reference counter` job.

 2) (PR author) copy-paste counter into `reproducibility_tests/ref_counter.jl`

 3) Upon next CI run, before performing CI test,
    we check if the counter indicates the existence of comparable
    references by checking if `reproducibility_tests/ref_counter.jl`
    in the PR matches (for example) `aRsVoY/ref_counter.jl` in the last
    merged commit (on central). If there are comparable references,
    we compare against them and require they pass our
    reproducibility tests, if not, then we throw a warning to let
    users know that they should visually verify the simulation results.
