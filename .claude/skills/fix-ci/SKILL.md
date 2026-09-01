---
name: fix-ci
description: Fix CI failures and performance regressions for a Julia package PR by iterating - triage the latest CI results, fix one root cause, verify locally, push. Use when a PR's GitHub Actions or Buildkite CI is failing, or when CI jobs run slower than they do on the main branch. Verifies CPU and GPU compilation locally before every push and benchmarks performance-relevant changes against main.
---

# Fix CI

The loop:

0. At session start: set up CI access (Section 1), verify the local tools
   (Section 4), and create or re-read the measurement ledger (Section 7).
1. Triage every failed job of the latest build — Buildkite `hard_failed`
   steps and failed GitHub Actions checks (Section 2). After the FIRST
   complete build of the PR, also run the job-time comparison against main
   (Section 5.2) — do not wait until correctness work is done.
2. Fix ONE root cause (Sections 3, 6, 8, 9).
3. Verify the fix locally (Section 4).
4. If the change is performance-relevant — it touches hot loops, kernels,
   broadcasting or indexing machinery, type parameters of core structs, or
   compiler annotations — benchmark against main (Section 5).
5. Format, commit, push (Section 2).
6. Repeat until CI is green or you reach a decision only the repository owner
   can make (Section 3).

## 1. Set up access to CI results before anything else

- **GitHub Actions**: raw job logs require authentication. Check
  `gh auth status`; if `gh` is missing or logged out, STOP and ask the user to
  install and authenticate it (`gh auth login`). Then:
  - Statuses: `gh api repos/<org>/<repo>/commits/<sha>/check-runs`
  - Logs: `gh api repos/<org>/<repo>/actions/jobs/<job id>/logs`
    (the check-run id is the job id)
- **Buildkite** (CliMA's GPU/MPI CI): public, no auth needed. All endpoints
  return JSON with the header `Accept: application/json`:
  - Build state: `https://buildkite.com/<org>/<pipeline>/builds/<N>.json`
  - Steps with outcomes and job ids:
    `https://buildkite.com/<org>/<pipeline>/builds/<N>/data/steps?exclude_group_steps=true`
    (each step has `label`, `outcome` — `passed`/`hard_failed`/`soft_failed` —
    `statistics.latest_job_id`, and `latest_job_started_at`/`finished_at`
    timestamps for wall-time comparisons)
  - Job log:
    `https://buildkite.com/organizations/<org>/pipelines/<pipeline>/builds/<N>/jobs/<job id>/log`
    (read the `output` field, strip HTML tags, unescape entities)
  - Job artifacts (e.g. flame graphs): GET
    `https://buildkite.com/organizations/<org>/pipelines/<pipeline>/builds/<N>/jobs/<job id>/artifacts`,
    then GET each artifact's `url`.
- Map a commit to all of its builds (including extra pipelines like an
  end-to-end performance pipeline) via
  `gh api repos/<org>/<repo>/commits/<sha>/status` — each status carries a
  `target_url`. To find a branch's latest build, resolve its head sha first
  (`gh api repos/<org>/<repo>/commits/<branch> --jq .sha`).

## 2. Run the loop

- Read logs for the *first* error and note which testsets completed; inner
  `@testset`s print no summary even when they pass — only the top-level one
  does — so a missing summary does not locate the crash. Cluster failures by
  root cause before fixing anything; one fix often clears many jobs.
- Format before every commit with the same JuliaFormatter major version the CI
  action uses — read the version from the formatting workflow under
  `.github/workflows/` or the repo's formatter environment.
- If the pipeline cancels in-flight builds when the branch is pushed (check
  the pipeline settings, or whether earlier builds show as canceled after
  pushes), never push while waiting on a diagnostic job's results.
- For failures that only manifest on CI hardware (runtime GPU errors,
  segfaults), get a full answer in ONE round with a temporary diagnostic step
  in `.buildkite/pipeline.yml` marked `soft_fail: true`: run each suspect case
  via `Distributed.remotecall_eval` on a worker process, catch
  `ProcessExitedException`, respawn, and print a PASS/CRASH map. Remove
  temporary steps once the cause is fixed.
- When new failures appear after your own push, suspect your latest commit
  first and bisect against the previous build's results. Confirm which commit
  a "known good" build actually ran (builds map to commits via Section 1)
  before treating its numbers as a baseline.

## 3. Decide what is a design question and what is a mechanical detail

- Never redesign or bypass a feature the PR exists to introduce just to make
  a test pass — assume a small bug underneath and find it. Stop and ask the
  user before changing public APIs, algorithmic behavior, or stated PR goals
  (including accepting a performance cost inherent to a design choice); do
  NOT stop for mechanical choices (test tolerances, comment wording, internal
  helper structure) — pick the option supported by measurements and keep
  moving.
- Fix problems at the level where the design lives, in their most general
  form:
  - If a validation helper rejects legitimate inputs, generalize the
    validation; do not switch call sites to an unchecked variant.
  - When a pattern is justified by measurement, apply it to every sibling
    call site, not just the one that was measured.
  - When an operation's result is fixed by its API contract (e.g. an output
    whose size the contract determines), encode the contract as a dedicated
    method instead of computing the result generically at runtime, which
    blocks constant folding.
  - Prefer facts derived from existing machinery over hand-maintained
    parallel tables of the same facts — but verify the derivation
    constant-folds wherever the fact is consumed, including inside kernels.
  - An `if` on runtime argument properties (which the compiler folds away
    when the answer is fixed by the argument types) often replaces several
    methods, eliminating whole classes of method ambiguities; conversely, a
    `Vararg` method that overlaps a typed method is an ambiguity waiting to
    happen (constrain the `Vararg` method's leading arguments so the
    signatures no longer overlap, or split it by argument type).
- When a test's expectation is wrong rather than the code (stale baselines,
  `@test_broken` that now passes), update the expectation — but only flip
  markers that pass in *every* invocation across at least two builds, and use
  `skip` instead of `broken` when the pass/fail set is unstable.

## 4. Verify locally before pushing; use CI only for what needs hardware

- **The tools.** This skill's folder ships two: `test_compilation.jl` (module
  `TestCompilation`: `@test_compilation` and `compilation_reports`, which
  check CPU and GPU compilation without needing a GPU) and `flame_diff.jl`
  (module `FlameDiff`, flame
  graph comparison — Section 5.4). Both need an environment with `Adapt`,
  `CUDA`, `JET`, and `ProfileCanvas`: use the repository's test or benchmark
  environment if it provides them, otherwise create one once with
  `julia -e 'using Pkg; Pkg.activate("<scratch>/ci_tools");
  Pkg.add(["Adapt", "CUDA", "JET", "ProfileCanvas"])'`
  (`<scratch>` is any writable directory outside the repository).
- **Verify the tools before trusting any conclusion drawn from them.** Run
  both test suites to completion —
  `julia --project=<env> <skill folder>/test_compilation_tests.jl` and
  `julia --project=<env> <skill folder>/flame_diff_tests.jl` — before the
  first use of either tool in a session (including any integration tests the
  suites run against the package). If a tool's own tests fail, fix that
  first. Separately, the checker's *results* go stale: it runs through the
  package's `Adapt` rules, broadcasting machinery, and array wrappers, so
  whenever a commit changes any of those, re-run the compilation checks
  themselves on the functions you touched — earlier passes no longer count.
- CPU: run the failing test files directly, matching CI's flags — read the
  exact `julia` invocation (project environment, `--check-bounds`, threads)
  from the job's entry in the CI configuration. Also run the relevant files
  with more than one thread (e.g. `--threads=4`): branches guarded by thread
  counts never execute in single-threaded runs.
- GPU without a device: almost everything except runtime errors and *novel*
  instruction-selection crashes can be checked locally (known crashers are
  caught by the checker's `:llvm_types` stage).
  `include("<skill folder>/test_compilation.jl")`
  and check every function you touched over every kind of argument CI can
  pass it — plain arrays, unmaterialized broadcast expressions
  (`Base.Broadcast.Broadcasted`), and any wrapper types the package defines.
  A fix validated on only one of these is not validated.
- If a CUDA device IS available locally, iterate on the actual failing GPU
  tests locally instead.
- Julia-version-specific failures (e.g. method-ambiguity counts from Aqua)
  must be reproduced under the same Julia minor version CI uses
  (`juliaup add <version>`, then `julia +<version> ...`), with the same set of
  loaded packages as the test process.

## 5. Compare against the main branch

### 5.1 Maintain a main-branch worktree

```
git worktree add <scratch>/main_wt origin/main
```

If the test/benchmark environment `Pkg.develop`s the package by relative path
(common: `path = ".."` in the environment's Manifest), copy that environment's
`Manifest.toml` into the same relative location inside the worktree (e.g.
`main_wt/.buildkite/Manifest.toml`) so it resolves against the worktree's own
source, then run `Pkg.instantiate()` in that environment. Expect a one-time
precompile.

### 5.2 Compare whole builds job-by-job

Fetch the steps JSON (Section 1) for the latest PR build and the latest main
build (resolve main's head sha, then its commit status), compute per-job wall
times from `latest_job_started_at`/`finished_at`, and rank by ratio.
Interpret with care:

- Job wall time conflates compile time and runtime. Extract logged
  measurements (`BenchmarkTools` tables, `@time` lines, per-testset times in
  `Test Summary` lines) to separate them: identical allocation counts with
  higher times mean pure compute; testset times include compilation.
- Jobs run with `--check-bounds=yes` amplify per-index overhead and are not
  comparable to unchecked runs of the same code.
- A benchmark job whose script changed on the PR is not comparable
  positionally — align by case labels and problem sizes first.
- Soft-failed jobs may be pre-existing: diff the soft-fail label sets of the
  two builds before investigating any of them.

### 5.3 Whole-workload A/B benchmarks, interleaved

Micro-benchmarks cannot catch the type-inference and inlining failures that
only appear at realistic problem complexity —
keep at least one whole-workload benchmark in the verification battery (a
full model step if the repo has one, otherwise its heaviest realistic
workload). Run it INTERLEAVED (branch, main, branch, main — via the worktree)
in the same session; never compare a fresh number against one recorded
earlier. Rerun at least twice; a single elevated run right after heavy
compilation is usually warmup.

### 5.4 Flame graphs

- Generate a profile as HTML with ProfileCanvas:
  `import Profile, ProfileCanvas; Profile.@profile <workload>;
  ProfileCanvas.html_file("flame.html")`. CI flame-graph jobs usually upload
  the same kind of file as an artifact (Section 1 shows how to download it).
- Diff two flame graphs with
  `julia --project=<env> <skill folder>/flame_diff.jl baseline.html
  candidate.html [TOP_N]` — it prints root sample counts, their ratio, and
  the frames with the largest self-sample increases and decreases; as a
  library (`include` + `using .FlameDiff`), `flame_diff` also returns every
  per-frame row sorted by delta.
- **Regenerate the flame pair locally before acting on CI frames**: a CI
  flame may come from an older commit than the code being debugged, and its
  line numbers or even mechanisms can be stale. Profile the same workload in
  the PR tree and the main worktree and treat the local pair as ground truth.
- Self-sample deltas below ~5 are noise. Self counts between same-duration
  runs are comparable in absolute terms; totals and fractions are not.
- Read flames mechanistically: self time on a loop-frame line means loop
  overhead (argument materialization, iteration); equal absolute samples in
  the data-access leaves with a higher total means the extra time is
  elsewhere; time under `similar`/allocation frames with unchanged allocation
  counts means slower construction, not more of it.

### 5.5 Hot-loop IR diagnostics (faster than any profile)

For a hot-loop throughput regression, two cheap checks localize the cause:

- Count vector instructions: `sprint(code_llvm, f, types)` and count matches
  of `r"<[0-9]+ x double>"` (or `float`). Compare against the equivalent
  plain-array loop, which should vectorize.
- Count non-cold `invoke`s in `Base.code_typed(f, types)[1][1].code`
  statements (ignore `throw`/error paths): any hot-path `invoke` means the
  inliner bailed and arguments are materialized per call.

Known traps:

- LLVM cannot vectorize a loop over a flattened `CartesianIndices` iterator,
  whose index increments branch at every dimension boundary. `@simd` fixes it
  by splitting the innermost dimension into a unit-stride loop — but test
  with realistic inner trip counts (a length-4 inner loop never vectorizes,
  which masks any improvement), and `@simd` requires an indexable iterator
  (`AbstractArray`/ranges) — over other iterators it degrades silently
  instead of erroring.
- The compiler silently stops inlining function arguments whose call
  signatures grow large (a call over a single broadcast inlines; one over a
  deeply nested broadcast expression may not). Force it with a call-site
  `@inline f(args...)` wherever the loop body must be fully inlined.
- Per-point wrapper construction (views, broadcast slices) is free only if it
  is fully elided; check the typed IR for `%new` of wrapper types surviving
  in the loop body.

### 5.6 End-to-end performance pipeline

Some CliMA repositories have an end-to-end performance pipeline (e.g. an AMIP
or flagship-model run reporting SYPD — simulated years per day) that is
triggered by appending `[perf]` to the commit message. To check whether the
current repo has one, check the commit status of a recent `[perf]` commit
(Section 1) or grep the CI configuration. Where it exists, it is the gold
standard for runtime performance:

- Trigger it EARLY on a performance-relevant PR — it costs one commit-message
  tag — rather than saving it as a final gate.
- The job log contains the SYPD measurement and a comparison to the latest
  stored reference value; other recent builds of the same pipeline show the
  acceptable range.

### 5.7 Performance-baseline hygiene

- When updating any stored perf baseline (latency thresholds, timing caps),
  record in the commit message which build/commit produced the new numbers
  AND how they compare to main. A baseline set mid-PR may have been measured
  on a state where the code path was not doing its full work — verify the
  path is exercised end-to-end before trusting a "best ever" number.
- Fixed absolute deltas across benchmarks of very different sizes indicate
  fixed per-call overhead (e.g. launch-time work); proportional deltas
  indicate per-element work. Diagnose accordingly before retuning baselines.

## 6. Isolate low-level issues and fix them robustly

- Chase symptoms down to the single lowest-level function responsible, using
  JET/`@test_compilation`, then fix that function, not the symptom — e.g.
  call a branch-free internal instead of an entry point whose unreachable
  throw branch JET flags, or construct a wrapper explicitly when a library's
  constructor returns a type inference cannot determine.
- Using un-exported internals is acceptable when the fix is small, guarded
  (`@static if isdefined(...)` with a public-API fallback), and documented
  with the reason.
- Rules that recur on GPUs: kernel-launched closures may capture only isbits
  values (derive types from argument types inside the closure, never capture
  a `Type`); error paths that build strings at runtime cannot compile in
  kernels (use static messages or move checks to the host); `Adapt` does not
  descend into `Base.Pair` or unregistered wrappers (write explicit
  `adapt_structure` rules); partial-rank views/reshapes of device arrays pull
  in `SignedMultiplicativeInverse` string-throwing constructors (index at
  full rank).

## 7. Keep hypotheses alive in a ledger; re-test them as the code evolves

- **Maintain a written ledger of measurements and hypotheses; never rely on
  memory or scrollback.** Keep one untracked Markdown file at the repository
  root (e.g. `ci_ledger.md` — NOT in a session-specific temp directory, so it
  survives across sessions; never commit it) with two tables:
  - *Measurements*: one row per (implementation option, metric) — the
    commit/diff identifier, the number, what it was measured INTERLEAVED
    against, and the machine state (load, same-session baseline). Update it
    after every benchmark or compilation check, and quote numbers only from
    the ledger.
  - *Hypotheses*: one row per hypothesis — status (open/confirmed/refuted),
    the discriminating experiment, and a pointer to the evidence. Mark
    refuted entries instead of deleting them, so a dead end is not re-derived
    in a later session. Re-read the ledger at session start and before every
    re-measurement.
- Keep several hypotheses alive when debugging: record them, design one
  experiment that discriminates between them (the CI diagnostic step from
  Section 2, an A/B worktree benchmark, a JET diff), and re-check the losers
  later — a wrong hypothesis about one failure may be right about another.
- After each significant change, re-run the checks that motivated earlier
  workarounds; delete workarounds whose cause is gone.
- Fixes stacked while another regression is still present are all suspect:
  every intermediate measurement is contaminated, so effects get attributed
  to the wrong edit. Redo the attribution from scratch — construct a
  minimal baseline containing only the agreed fixes, measure it, then add
  each candidate edit one at a time (and afterwards remove each retained edit
  once) with benchmarks and tests after every step. Expect some "necessary"
  edits to turn out to do nothing; revert those.
- Never describe a change as restoring previous behavior without diffing the
  actual git history.

## 8. Compiler annotations conceal as often as they cure

- Before adding `@generated`, `@assume_effects`, `@constprop`, `@inline`, or
  a `Method.recursion_relation` override, demonstrate the specific compiler
  failure it is meant to fix (e.g. show that inference gives up ONLY on
  large, deeply nested expressions); after adding one, re-test the original
  symptom — if it is still there, the annotation was hiding a structural
  problem.
- Prefer optionally-generated functions (`if @generated`) over plain
  `@generated` when the arguments may not be statically known: plain
  generated functions infer to `Any` for unknown static parameters.
- `Base.@assume_effects :foldable` does not guarantee that a call constant-
  folds in large expressions — verify with the compilation checker's `:kernel`
  stage when GPU code depends on the fold.

## 9. Downstream breakage from renames and representation changes

- When a PR renames or removes internal names that other packages use, add a
  `deprecated.jl` to the module with plain `const` aliases (no deprecation
  warnings), and **export** any alias whose old name was exported. Only alias
  names whose semantics survived; leaving a changed-meaning name undefined is
  better than silently resolving it to something subtly different.
- Find the full list by grepping the depot copies of the downstream packages
  (`~/.julia/packages/`), including any separately-registered subpackages of
  this repository, then verify by running the smallest downstream test suite
  (or at least precompilation of the heaviest downstream package) with the
  package `Pkg.develop`ed.
- Classify every downstream failure: (A) fixable here with an alias or
  method; (B) downstream code depends on a changed internal representation
  (e.g. it indexes `parent(...)` with the old array shape) — aliases cannot
  fix this; it needs a downstream release, so report it and move on; (C)
  pre-existing — confirm by checking the same job on the merge-base commit.
