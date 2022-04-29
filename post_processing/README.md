# Post-processing / Regression tests

This document outlines how the post-processing and regression
test infrastructure works, including

 - How regression tests work
 - How to approve pull requests that fail regression tests
 - Other failures

## How the regression tests work

Regression tests are performed at the end of `examples/hybrid/driver.jl`, after a simulation completes, and relies on a unique job id (`job_id`). Here is an outline of the regression test procedure:

 0) Run a simulation, with a particular `job_id`, to the final time.
 1) Load a dictionary, `all_best_mse`, of previous "best" mean-squared errors from `mse_tables.jl` and extract the mean squared errors for the given `job_id` (store in job-specific dictionary, `best_mse`).
 2) Export the solution (a `FieldVector`) at the final simulation time to an `NCDataset` file.
 3) Compute the errors between the exported solution (in step 2) and the exported solution from the reference `NCDataset` file (which is saved in a dedicated folder on the Caltech Central cluster) and save into a dictionary, called `computed_mse`.
 4) Export this dictionary (`computed_mse`) to the output folder
 5) Test that `computed_mse` is no worse than `best_mse` (determines if regression test passes or not).

After these steps are performed at the end of the driver, additional jobs are run:

 1) Print `computed_mse` for all jobs to make updating `post_processing/mse_tables.jl` easy
 2) If we're on the staging branch (all tests have passed, and the PR is effectively merging), move the `NCDataset`s from the scratch directory onto the dedicated folder on the Caltech Central cluster.

## How to merge pull requests (PR) that get approved but *fail* regression tests

To "approve" PRs, authors can:
 - Click the *Print new mse tables* buildkite job
 - Click the *Running commands* entry in the *Log* tab
 - Copy this output until `-- DO NOT COPY --`
 - Paste these contents into `post_processing/mse_tables.jl`
 - Add, commit, and push these changes and the updated PR should pass the regression test.

## How regression tests *break*

The regression test infrastructure has some built-in assumptions. Concretely, the compared `NCDataset`s must:

 - exist (i.e., we must have two datasets to compare against)
 - have matching keys (i.e., variable names) for each compared variable (which are specified via the keys of `best_mse`).
 - for each variable, have matching number of dimensions (i.e., cannot compare single column and sphere) for all compared variables
 - for each variable, have matching size along each dimension (i.e., must have the same spatial resolution)

If we fail to satisfy _any_ of these requirements, then the regression test infrastructure will *break* (fail to generate a computed mse dictionary).

The next section describes how to deal with this.

## How to merge pull requests (PR) that get approved but *break* regression tests

If your PR is breaking the regression test due to one of the following reasons:

 - Variable name has changed
 - A new regression test was added
 - Grid resolution has changed

Then you can merge your PR by copying the new reference counter, printed in the `Print new reference counter` job, and pasting that into `post_processing/ref_counter.jl` in your PR.

**Please be warned:** updating the counter effectively bypasses the regression tests and updates the dataset that CI compares future PRs against (to the datasets exported in the PR). So, if you are the PR author, please review the output results before merging.

## How we track which dataset to compare against

A simple view of how this can be done is by considering the following. Below is an toy example list of several commits merged into the main branch, identified by its commit SHA (left column), or hash. NCData from each of these commits are compared against NCData in a reference commit (possibly itself). The self references are needed when we hit a failure mode (described above).

```
hash of      hash of
 merged     reference
 commit      commit
"V50XdC" => "V50XdC" # Self reference
"lBKsAn" => "V50XdC"
"Eh2ToX" => "V50XdC"
"bnMLxi" => "bnMLxi" # Self reference
"Jjx16f" => "bnMLxi"
"dHkJqc" => "dHkJqc" # Self reference
"SIgf1i" => "dHkJqc"
"vTsCoY" => "dHkJqc"
"VvCzAH" => "dHkJqc"
```

Ideally, we could `git commit` the commit hash into our PRs in order to update the reference when we hit a failure mode. However, there are two issues with this:

 - Predicting the next commit hash is [cryptographically impossible](https://stackoverflow.com/questions/21942694/predict-git-commit-id-and-commit-a-file-which-contains-that-commit-id).
 - Merged commits are actually created by [bors](https://bors.tech/).

A simple way to circumvent this is to create a hash ourselves, and simply compare against old hashes:

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

Work-flow procedure:
 1) We start off with a self reference: print a new reference
    counter in the `print new reference counter` job.
 2) (PR author) copy-paste counter into `post_processing/ref_counter.jl`
 3) Upon next CI run, before performing CI test,
    we check if the counter indicates a self-reference by
    checking if `post_processing/ref_counter.jl` in the PR
    matches (e.g.,) `aRsVoY/ref_counter.jl` in the last
    merged commit (on central). If yes, then it's a self
    reference, if not, then we look-up the dataset based
    on the counter.
