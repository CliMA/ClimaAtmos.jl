---
name: fix-ci
description: Fix CI failures and performance regressions for a Julia package PR by iterating - triage the latest CI results, fix one root cause, verify locally, push. Use when a PR's GitHub Actions or Buildkite CI is failing, or when CI jobs run slower than they do on the main branch. Verifies CPU and GPU compilation locally before every push, benchmarks performance-relevant changes against main, and predicts GPU kernel times with an instruction-level model.
---

# Fix CI

The loop:

0. At session start: set up CI access (Section 1), verify the local tools
   (Section 4), and create or re-read the measurement ledger (Section 7).
1. Triage every failed job of the latest build — Buildkite `hard_failed`
   steps and failed GitHub Actions checks (Section 2). After the FIRST
   complete build of the PR, also run the job-time comparison against main
   (Section 5.2).
2. Fix ONE root cause (Sections 3, 6, 8, 9).
3. Verify the fix locally (Section 4).
4. If the change is performance-relevant — it touches hot loops, kernels,
   broadcasting or indexing machinery, type parameters of core structs, or
   compiler annotations — benchmark against main (Section 5).
5. Format, commit, push (Section 2).
6. Repeat until CI is green or you reach a decision only the repository owner
   can make (Section 3).

## 0. Autonomy

This loop performs actions that
[agent_autonomy.md](../../../workflow/agent_autonomy.md) gates behind user
approval. Invoking the skill grants exactly one of them: pushing commits to the
PR branch named in the request (never force-pushing, never to `main`). Ask
before each of the others, every time:

- editing `.buildkite/pipeline.yml` or `.github/workflows/*`, including a
  temporary `soft_fail` diagnostic step (Section 2);
- changing a stored baseline, tolerance threshold, golden file, or
  reproducibility reference (Sections 3 and 5.7);
- flipping `@test_broken`/`skip` markers or otherwise disabling a test
  (Section 3);
- editing `Manifest.toml`, `[compat]`, or adding `[deps]`.

When asking, name the file, the change, and the evidence for it, so the user
can approve in one reply.

## 1. Set up access to CI results before anything else

- **GitHub Actions**: raw job logs require authentication. Check
  `gh auth status`; if `gh` is missing or logged out, STOP and ask the user to
  install and authenticate it (`gh auth login`). Then:
  - Statuses: `gh api repos/<org>/<repo>/commits/<sha>/check-runs`
  - Logs: `gh api repos/<org>/<repo>/actions/jobs/<job id>/logs`
    (the check-run id is the job id)
- **Buildkite** (CliMA's GPU/MPI CI): public pipelines need no auth. All
  endpoints below return JSON with the header `Accept: application/json`.
  Private pipelines (e.g. downstream or performance pipelines) need a
  read-only API token (`read_builds`, `read_build_logs`) and the REST API
  instead: `curl -H "Authorization: Bearer $TOKEN"
  https://api.buildkite.com/v2/organizations/<org>/pipelines/<pipeline>/builds/<N>`
  (jobs with ids and states), `.../builds/<N>/jobs/<job id>/log` (field
  `content`; strip ANSI codes and `\x1b_bk;t=<ms>\x07` timestamp prefixes).
  Public endpoints:
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
  `gh api repos/<org>/<repo>/commits/<sha>/status` — each status has a
  `target_url`. To find a branch's latest build, resolve its head sha first
  (`gh api repos/<org>/<repo>/commits/<branch> --jq .sha`).

## 2. Run the loop

- Read logs for the *first* error and note which testsets completed; inner
  `@testset`s print no summary even when they pass — only the top-level one
  does — so a missing summary does not locate the crash. Cluster failures by
  root cause before fixing anything; one fix often clears many jobs.
- Format before every commit with the same JuliaFormatter major version the CI
  action uses — read the version from the formatting workflow under
  `.github/workflows/` or the repo's formatter environment (see
  [code_style.md §1](../../../code-quality/code_style.md)).
- If the pipeline cancels in-flight builds when the branch is pushed (check
  the pipeline settings, or whether earlier builds show as canceled after
  pushes), never push while waiting on a diagnostic job's results.
- For failures that only manifest on CI hardware (runtime GPU errors,
  segfaults), get a full answer in ONE round with a temporary diagnostic step
  in `.buildkite/pipeline.yml` marked `soft_fail: true` (a pipeline edit;
  ask first, Section 0): run each suspect case
  via `Distributed.remotecall_eval` on a worker process, catch
  `ProcessExitedException`, respawn, and print a PASS/CRASH map. Remove
  temporary steps once the cause is fixed.
- When new failures appear after your own push, suspect your latest commit
  first and bisect against the previous build's results. Confirm which commit
  a "known good" build ran (builds map to commits via Section 1)
  before treating its numbers as a baseline.

## 3. Decide what is a design question and what is a mechanical detail

- Never redesign or bypass a feature the PR exists to introduce to make a
  test pass — assume a small bug underneath and find it. Stop and ask the
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
    call site, not only the one that was measured.
  - When an operation's result is fixed by its API contract (e.g. an output
    whose size the contract determines), encode the contract as a dedicated
    method instead of computing the result generically at runtime, which
    blocks constant folding.
  - Prefer facts derived from existing machinery over hand-maintained
    parallel tables of the same facts — but verify the derivation
    constant-folds wherever the fact is used, including inside kernels.
  - An `if` on runtime argument properties (which the compiler folds away
    when the answer is fixed by the argument types) often replaces several
    methods, eliminating whole classes of method ambiguities; conversely, a
    `Vararg` method that overlaps a typed method is an ambiguity waiting to
    happen (constrain the `Vararg` method's leading arguments so that the
    signatures do not overlap, or split it by argument type).
- When a test's expectation is wrong rather than the code (stale baselines,
  `@test_broken` that now passes), propose updating the expectation with the
  evidence and wait for approval (Section 0) — flip only
  markers that pass in *every* invocation across at least two builds, and use
  `skip` instead of `broken` when the pass/fail set is unstable.

## 4. Verify locally before pushing; use CI only for what needs hardware

- **The tools.** This skill's folder ships three, each with a test file:
  - `test_compilation.jl` (module `TestCompilation`): `@test_compilation` and
    `compilation_reports` check CPU and GPU compilation without a GPU
    (this section). Needs `Adapt`, `CUDA`, `JET`.
  - `flame_diff.jl` (module `FlameDiff`): compares two flame graphs
    (Section 5.4). Needs `ProfileCanvas`.
  - `gpu_perf_model.jl` (module `GPUPerfModel`): predicts a GPU kernel's time
    from its LLVM IR or SASS (Section 5.8). Package-free except `device_ir`,
    which needs `CUDA` (no device).

  Use the repository's test or benchmark environment if it provides these
  packages, otherwise create one once with
  `julia -e 'using Pkg; Pkg.activate("<scratch>/ci_tools");
  Pkg.add(["Adapt", "CUDA", "JET", "ProfileCanvas"])'`
  (`<scratch>` is any writable directory outside the repository).
- **Verify the tools before trusting any conclusion drawn from them.** Run
  each tool's test file to completion before its first use in a session —
  `julia --project=<env> <skill folder>/<tool>_tests.jl` for
  `test_compilation`, `flame_diff`, and `gpu_perf_model` (including any
  integration tests the suites run against the package). If a tool's own
  tests fail, fix that first. Separately, the checker's *results* go stale:
  it runs through the package's `Adapt` rules, broadcasting machinery, and
  array wrappers, so
  whenever a commit changes any of those, re-run the compilation checks
  themselves on the functions you touched — earlier passes do not count.
- CPU: run the failing test files directly, matching CI's flags — read the
  exact `julia` invocation (project environment, `--check-bounds`, threads)
  from the job's entry in the CI configuration. Also run the relevant files
  with more than one thread (e.g. `--threads=4`): branches guarded by thread
  counts never execute in single-threaded runs.
- GPU without a device: almost everything except runtime errors and *novel*
  instruction-selection crashes can be checked locally (known crashers are
  caught by the checker's `:llvm_types` stage).
  `include("<skill folder>/test_compilation.jl"); using .TestCompilation`,
  then, with CPU-constructed arguments, `@test_compilation f(args...)` inside
  a `@testset` or `ok, reports = compilation_reports(f, args)` (stages
  `:cpu`, `:host`, `:kernel`, `:pointers`, `:llvm_types`; see the module
  docstring for keywords such as `type_replacements`). Check every function
  you touched over every kind of argument CI can
  pass it — plain arrays, unmaterialized broadcast expressions
  (`Base.Broadcast.Broadcasted`), and any wrapper types the package defines.
  A fix validated on only one of these is not validated.
- If a CUDA device IS available locally, iterate on the failing GPU tests
  locally.
- Julia-version-specific failures (e.g. method-ambiguity counts from Aqua)
  must be reproduced under the same Julia minor version CI uses
  (`juliaup add <version>`, then `julia +<version> ...`; see
  [ci_triage.md §1](../../../workflow/ci_triage.md)), with the same set of
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
Caveats:

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
only appear at realistic problem complexity, so keep at least one
whole-workload benchmark in the verification battery (a full model step if
the repo has one, otherwise its heaviest realistic workload). Run it
INTERLEAVED (branch, main, branch, main — via the worktree) in the same
session; never compare a fresh number against one recorded earlier. Rerun at
least twice; a single elevated run right after heavy compilation is usually
warmup.

### 5.4 Flame graphs

- Generate a profile as HTML with ProfileCanvas:
  `import Profile, ProfileCanvas; Profile.@profile <workload>;
  ProfileCanvas.html_file("flame.html")` (see
  [allocation_debugging.md §6](../../../performance/allocation_debugging.md)).
  CI flame-graph jobs usually upload the same kind of file as an artifact
  (Section 1 shows how to download it).
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

### 5.5 Hot-loop IR diagnostics

For a hot-loop throughput regression, two checks localize the cause faster
than any profile:

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
- Per-point wrapper construction (views, broadcast slices) has zero cost only
  if it is fully elided; check the typed IR for `%new` of wrapper types
  surviving in the loop body.

### 5.6 End-to-end performance pipeline

Some CliMA repositories have an end-to-end performance pipeline (e.g. an AMIP
or flagship-model run reporting SYPD — simulated years per day) that is
triggered by appending `[perf]` to the commit message. To check whether the
current repo has one, check the commit status of a recent `[perf]` commit
(Section 1) or grep the CI configuration. Where it exists, it is the
authoritative measurement of runtime performance:

- Trigger it EARLY on a performance-relevant PR; a `[perf]` tag on any
  commit is enough.
- The job log contains the SYPD measurement and a comparison to the latest
  stored reference value; other recent builds of the same pipeline show the
  acceptable range.

### 5.7 Performance-baseline hygiene

- Updating a stored perf baseline (latency thresholds, timing caps) needs
  approval (Section 0). When it is granted,
  record in the commit message which build/commit produced the new numbers
  AND how they compare to main. A baseline set mid-PR may have been measured
  on a state where the code path was not doing its full work — verify the
  path is exercised end-to-end before trusting a "best ever" number.
- Fixed absolute deltas across benchmarks of different sizes indicate
  fixed per-call overhead (e.g. launch-time work); proportional deltas
  indicate per-element work. Diagnose accordingly before retuning baselines.

### 5.8 Instruction-level GPU performance model (`gpu_perf_model.jl`)

`GPUPerfModel` predicts a CUDA kernel's device time from its LLVM IR (compiled
device-free) or SASS, and names the limit that binds: dependent latency, issue,
a pipe (FMA, FP64, MUFU), the LSU, L2/DRAM bandwidth, barriers or instruction
fetch. It simulates one SM cycle by cycle over the kernel's dynamic trace, with
per-op latencies, per-pipe throughputs, dependency-limited ILP and program
order. Occupancy or instruction counts alone are not predictions. Use it to
compare variants of a kernel and to find what binds before you spend CI runs on
them; read **Accuracy** below before you trust an absolute time. The module
docstring (`?GPUPerfModel`) lists the mechanisms, and `?simulate` the keywords.

- **Quick start** (device-free: any environment with CUDA.jl; no GPU needed):

  ```julia
  import CUDA
  using CUDA: blockIdx, blockDim, threadIdx
  include("<skill folder>/gpu_perf_model.jl"); using .GPUPerfModel

  function stencil!(out, a, b)  # out[k, c] = a[k + 1, c] - a[k, c] + b[k, c]
      i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      nk, nc = size(out)
      if i <= nk * nc
          k, c = (i - 1) % nk + 1, (i - 1) ÷ nk + 1  # consecutive threads: consecutive k
          @inbounds out[k, c] = a[k + 1, c] - a[k, c] + b[k, c]
      end
      return nothing
  end
  M = CUDA.CuDeviceMatrix{Float32, 1}
  K = parse_llvm(device_ir(stencil!, Tuple{M, M, M}))  # or parse_sass(read("k.sass", String))
  p = simulate(K; threads = 256, blocks = cld(63 * 65536, 256), regs = spill(K, 255)[2].peak)
  display(p)
  ```

  The first `device_ir`, `parse_llvm` and `simulate` calls in a process take
  about 15 s together (compilation); later calls take milliseconds. The
  report reads `45.72 µs, binding: dram_bandwidth`, `utilization: dram 72.1%,
  ...`, `ops/thread: ... LDG 3, ... STG 1` and two notes. Then:
  1. Check the trace before the time: `ops/thread` must show the loads, stores,
     barriers and divisions you expect per thread (here 3 loads and 1 store;
     none means that the walk skipped the body). `op_counts(K; dynamic = true,
     trips, conds, loops)` lists what the walk executes.
  2. Resolve every `note:` (table below). Here, `guards assumed to enter the
     kernel body: .not` is the `i <= nk * nc` bounds check (right as guessed),
     and `uniform branch probabilities ... on switch in L33` is Julia's check of
     the divisor `nk` for 0 and -1: `K.labels[K.succs[findfirst(==("L33"),
     K.labels)]]` names its targets, and `probs = Dict(("L33", "oksrem") =>
     1.0)` takes the normal one (the time stays the same here).
  3. Give what the IR cannot show: `regs`/`shmem` from ptxas or the CI kernel
     statistics (`spill(K, 255)[2].peak` is an upper bound), `hits` (default:
     every load from DRAM), `trips` for run-time loops, the four flags (below),
     and `clock_ghz` (below; `1.1395` gives 49.4 µs here).
  4. Compare variants under identical settings and explain the difference with
     the stalls, utilizations and bounds. The same kernel with consecutive
     threads on consecutive `c` (`c, k = (i - 1) % nc + 1, (i - 1) ÷ nc + 1`)
     touches 32 sectors per warp access instead of 4-5 (`K.sectors`) and takes
     362 µs, with `dram` 99% busy.
- **Inputs for package kernels.** `device_ir(f, tt)` needs the device argument
  types. Build the arguments on the CPU, rewrite `Array{T, N}` to `CuArray{T,
  N, CUDA.DeviceMemory}` in their types (recursively through type parameters),
  and take `Core.Compiler.return_type(CUDA.cudaconvert, Tuple{T})` of each
  (check `isconcretetype`). Build the objects with the GPU's memory layout: a
  package that picks layouts by device (RRTMGP stores buffers `(n, ncol)` on the
  CPU and `(ncol, n)` on the GPU) otherwise gives the CPU layout's access
  footprints. Locate loops in the source through the IR's `!dbg` lines to set
  their `trips`. Pass as `blocks` the blocks that have work.
- **IR or SASS.** With CI access, prefer SASS (`CUDA.@device_code_sass`,
  `cuobjdump -sass`): it has ptxas's instruction selection and order, spills,
  the stack frame and constant-bank operands. SASS has no addresses, so its
  accesses count as coalesced unless you set `coalescing` (Nsight sectors per
  request over the executed loads' bytes per lane). IR is close to source
  order and ptxas reorders it, so bracket IR predictions with `reorder = true`
  (an upper bound on ILP) and `reorder = n` (at most `n` live values), and
  report both.
- **Hit rates are inputs.** `hits` (the fractions of global loads served by
  L1, L2 and DRAM) defaults to all-DRAM. Sweep it, or derive it from Nsight
  Compute sector counts: `l1 = 1 - lts__t_sectors_srcunit_tex_op_read / (l1tex
  global + local load sectors)`, `dram = dram__bytes_read / 32 / (load
  sectors)`, `l2` = the rest; `store_dram = dram__bytes_write / (32 ×
  lts__t_sectors_srcunit_tex_op_write)`. Do not use `lts__t_sector_hit_rate`:
  it counts store sectors as hits. Loads of a module constant table always hit
  L1, and loads that walk an array or re-read one address across a loop's
  iterations take their own L1 hits (note `loads reuse lines`); `hits` covers
  the rest.
- **Compare at the clock of the run.** `clock_ghz` converts cycles to µs and
  the DRAM bandwidth to bytes per cycle. Nsight Compute locks the clock
  (`gpc__cycles_elapsed.avg.per_second`: 1.14 GHz on the CI A100s). Normal
  runs can run at that clock too: when a kernel's normal and Nsight times
  agree, use the Nsight clock. A kernel that runs faster than its issue limit
  allows at that clock (instructions / (SMs × 4 schedulers) cycles) ran at a
  higher clock: use the device's (1.41 GHz) or compare cycles. `time_us`
  includes `launch_us` (2 µs) per kernel.
- **Notes: what the model guessed.** Every `note:` line names a guess that
  changes the prediction. Resolve them before you read the time:

  | note | meaning | set |
  |---|---|---|
  | `loop L: unknown trip count, using n` | the loop's count is run-time data | `trips = Dict("L" => n)`, n = runs of the loop header (a loop tested at its top runs its body n - 1 times); for many nested loops, `loops` |
  | `...; outermost loop: set trips ...` | the kernel's outermost loop has a run-time count, usually the work per thread; unset, it runs once, and `loops` gives it 4-64 | `trips`: e.g. cld(items, threads × blocks) for a grid-stride loop, the inner dimension for a matmul's k loop, the point count for a per-thread loop over points |
  | `loop L: leaves early on data, using its bound n` | a constant bound with a convergence exit | `loops` or `trips` |
  | `guards assumed to enter the kernel body: c` | bounds checks and loop zero-trip guards taken as passing | `conds` where most threads exit or a loop is skipped |
  | `uniform branch probabilities on c ×n` | the 5 most visited branches left at 50/50 | `conds = Dict("c" => P(true))` by IR value name; in SASS (reused predicate registers) `probs = Dict((from, to) => p)` by labels |
  | `the walk ran x% of the kernel's ops` | under a quarter of the code ran | check the guessed branches |
  | `n global accesses with an address that is not a function of the lane` | charged as coalesced | check them; `coalescing` |
  | `call @f modeled as one branch-class op`, `approximate expansion for libdevice call` | an unknown call, a 12-op placeholder | compare with SASS |
  | `stepped into the calls of device functions: @f ×n` | non-inlined calls run in the trace | their loops' `trips` by prefixed label (`f.2.L161`) |
  | `no exit with a positive probability` | `conds`/`probs` never take the loop's exit | fix them |
  | `` `loops` changes nothing ``, `` `gather = :scattered` changes nothing `` | the flag has nothing to act on | nothing |

  The other notes (`loads reuse lines`, `local memory`, `spills`, `DRAM
  latency`, `simulated 2 and 4 trips`) report what the model did. Two branch
  kinds need `conds` even with `trips` set: a branch between two code paths by
  a run-time size (a reduction that loops sequentially below a size and splits
  pairwise above it: set it from the sizes), and a data-dependent branch that
  is constant per thread across a loop (a sunlit column): the walk alternates
  it per visit, the mean over warps, but when one block per SM runs, the
  slowest blocks (the expensive side) set the time, so set it to that side.
- **Four flags say what the kernel alone cannot show.** Decide them from the
  algorithm and the kernel's place in the program, never from measured times.
  Unset, each keeps the default behavior.

  `cache` (`:cold`, default; `:warm`): are the kernel's inputs still in L2?
  `:warm` serves the DRAM share of `hits` from L2 and keeps stores in L2
  (`store_dram = 0`). Size each kernel by its own data: the arrays it reads
  plus the arrays it writes, each array once, in MiB (2^20 bytes), without
  index (connectivity) arrays and per-grid constants (geometry, weights). Half
  the L2 is 20 MiB on an A100, 3 MiB on a V100 and 2 MiB on a P100; a kernel
  at the threshold fits. A chain of kernels gets one size per kernel.

  | situation | `cache` |
  |---|---|
  | timed under Nsight Compute (each replay flushes the caches) | `:cold` |
  | the preceding kernel(s) wrote or read its inputs (also in a sequence timed repeatedly), and its data fits in half the L2 | `:warm` |
  | a write-only kernel (fill) whose output fits in half the L2 | `:warm` |
  | one kernel timed repeatedly on its own, inputs no kernel touched recently, or data larger than half the L2 | `:cold` |

  Examples: `y .= a .* x .+ y` right after the kernel that wrote `x`, both
  8 MiB (16 MiB in all): `:warm`; the same on 16 MiB arrays: `:cold`. Two
  kernels that read the same 4 MiB inputs one after the other, the pair timed
  in a loop: `:warm` for both. A time-step kernel whose inputs come from
  several earlier kernels: `:cold`. ClimaCore's DSS chain on a 6 MiB field
  with a 4 MiB perimeter buffer (transform: field + buffer, local summation:
  buffer, untransform: buffer + field): `:warm` for each kernel.

  `gather` (`:local`, default; `:scattered`): do consecutive threads reach
  unrelated memory through a loaded index? `:local` takes a loaded index as
  its 1-based position in an identity map (lanes that read one entry share
  it; consecutive entries give consecutive indices). `:scattered` charges
  each access through a loaded index a sector and a 128-byte line per lane. An
  index computed from loaded data (a table lookup by temperature) is not a
  loaded index.

  | situation | `gather` |
  |---|---|
  | consecutive threads follow different entries of an index map to unrelated addresses: unstructured or cubed-sphere connectivity, neighbor lists, a sparse matrix's column indices | `:scattered` |
  | an identity or near-identity map (structured element lists), threads that share an entry, or no loaded indices | `:local` |

  Examples: a CSR sparse matrix-vector product, one thread per row reading
  `x[colval[k]]`: `:scattered`. ClimaCore's DSS local summation on a 2D
  space (one thread per vertex or face through `local_vertices`):
  `:scattered`; every other ClimaCore kernel, including that summation on
  3D spaces (the levels of a vertex share its entry): `:local`.

  `loops` (unset, default; `:few`, `:short`, `:long`): how many times do the
  loops that `trips` does not name iterate, when the code does not fix their
  count (`unknown trip count` notes) or fixes only a bound they can leave
  early on data (`leaves early on data` notes: Newton iterations with
  `maxiters`, iterate-until-converged loops)? They run 4, 16 or 64 source
  iterations, the geometric middle of each range (at most 2x off inside it).
  Unset, a run-time count runs once and an early exit is never taken. `trips`
  takes precedence; a loop with a constant count and no early exit keeps it.
  A warp runs a loop until its last lane leaves, so take the slowest of 32
  neighboring threads. Count one call's iterations; an unrolled loop runs the
  count over its unroll factor.

  | the loop runs over | iterations | `loops` |
  |---|---|---|
  | time substeps, Newton or secant iterations, the points of a 1D quadrature rule, a short fixed count | 2-8 | `:few` |
  | a count in the tens, the points of a 2D rule (N² for N = 3-5), bisection steps | 8-32 | `:short` |
  | the levels or layers of a column, a count in the hundreds | 32-128 | `:long` |
  | the kernel's outermost loop, when its count is the work per thread (grid-stride loops, a matmul's k loop, a per-thread solve over n points) | any | name it in `trips` |
  | nothing is known | - | unset |

  Examples: a Newton solve per point (`maxiters = 1000`) that converges in
  1-2 iterations: `:few`; Brent iterations, median 6 per point: `:short`;
  Courant-limited substeps of 5-44 per point: `:long`; `f(x, n)` with an
  `n`-iteration loop inside `f` and `n = 16` at run time: `:short` (a
  constant 16 is inferred); a non-inlined column integral over 63 levels:
  `:long`.

  `inner_extent` (unset, default; `1`): does the innermost dimension of the
  kernel's index space have size 1 in this launch? Unset, every run-time size
  that the kernel divides its thread index by (or takes its remainder by:
  `CartesianIndices`, `divrem`, `÷`/`%`) counts as at least a warp wide, so
  the lanes of a warp run along that dimension and `hits` describes their
  reuse. With `1`, those sizes are 1 in the access footprints, so the lanes
  spread over the outer dimensions, and loads take their L1 hits from the
  sectors that earlier loads of the block touched instead of `hits.l1`. In
  device IR, the run-time splits are `sdiv`/`udiv`/`srem`/`urem` by a
  kernel-argument value.

  | the launch | `inner_extent` |
  |---|---|
  | the run-time sizes that the kernel splits its thread index by are 1 | `1` |
  | the code's size-1 case uses a layout without that dimension, its threads covering the other dimensions | `1` |
  | that size is 2 or more (also fewer than 32), or the kernel splits its thread index by compile-time sizes or not at all | unset |

  Examples: a CUDA.jl kernel over an `(n, m)` matrix that takes
  `CartesianIndices(A)[i]`: `1` when `n = 1`. KernelAbstractions'
  `@index(Global, Cartesian)` with a dynamic workgroup whose leading size is
  1: `1`; with a static workgroup: unset. RRTMGP's `compute_col_gas_CUDA!`
  (`glx % nlay`): `1` for one layer, unset for 63. ClimaCore on one-level
  spaces: `1` for the DSS kernels and spectral operators, unset for pointwise
  kernels (their level count is a compile-time parameter) and for column
  reductions into a 2D field.
- **Mechanisms; switch them on separately** and report each run (each has its
  own `p.stalls` entry). Always on: dependency latency, pipe and LSU
  throughput, L2/DRAM queues, barriers. Switches: `hit_levels = true` draws
  each load's level from `hits`, so a batch of loads waits for its DRAM member
  (`false`: mean latency); `scoreboards = 6` dependency barriers per warp;
  `loaded = true` raises the DRAM latency with utilization up to the measured
  loaded value (A100); `icache = true` fetches instructions through the L0/L1
  instruction caches (`stagger` desynchronizes resident blocks); `spills =
  true` allocates `regs` registers and inserts `STL`/`LDL` (`remat`,
  `local_hits`); `expansion` charges SASS instructions beyond the IR op
  classes; `load_hits = Dict(static load index => hits)` overrides single
  loads (a prefetch's target, or `(l1 = 1, l2 = 0, dram = 0)` to bound a
  software-pipelining change before you write it). The front end merges
  adjacent loads and stores (the LoadStoreVectorizer), charges each global
  access the 32-byte sectors its warp touches (`K.sectors`), steps into
  non-inlined device functions, expands libdevice calls to ptxas's sequences,
  applies loop strength reduction, runs lane splits as SIMT divergence and
  thread-range splits as whole blocks, and adds the convergence checks that
  ptxas emits before shuffles with run-time masks. A simulation above 2
  million ops simulates 2 and 4 trips of the outermost loop and extrapolates
  linearly (noted).
- **Validate before trusting a conclusion.** Compare `time_us` with the
  measured time, and the stalls with Nsight Compute
  (`smsp__warp_issue_stalled_*`): `wait_LDG` ≈ long_scoreboard, `wait_LDL` ≈
  long_scoreboard with local-memory sectors, `pipe_*` ≈ math_pipe_throttle,
  `unit_lsu` ≈ mio/lg_throttle, `unit_bar` ≈ mio_throttle with
  `sm__inst_executed_pipe_adu` near peak, `barrier` ≈ barrier, `*_bandwidth`
  ≈ `lts__t_sectors`/`dram__bytes` against peak, `no_instruction` ≈
  no_instruction + imc_miss, `issue` ≈ `smsp__issue_active`. Check the spill
  frame (`4 × words`) against ptxas' `bytes stack frame` and dynamic op counts
  against Nsight's instructions per warp. A residual above ~20%, or a change
  that measures differently from its prediction, is a missing mechanism:
  record it in the ledger (Section 7) instead of fitting parameters.
- **Example outside ClimaCore: RRTMGP** (1.1.0, device-free sm_80 IR, no
  measured times: a trace check). Its `compute_col_gas_CUDA!` (one thread per
  layer and column, 24576 columns of 63 layers) runs 4 LDG, 1 STG and 30 FP32
  per thread: 19.5-27.5 µs. Its clear-sky two-stream solves
  (`rte_lw_2stream_solve_CUDA!`, `rte_sw_2stream_solve_CUDA!`: one thread
  per column, one 8-warp block per SM, looping over 256 or 224 g-points) run
  per g-point and layer 56 LDG, 34 LDL, 137 FP32 and 8.5 MUFU of gas optics
  (LW), matching the source's loads and stores. They are latency-bound
  (`wait_LDG`, `wait_LDL`): LW 137-427 ms, SW 93-340 ms, from `reorder =
  true` with `hits = (l1 = 0.5, l2 = 0.4, dram = 0.1)` to all-DRAM hits in IR
  order. That took GPU-layout objects, `trips` for every loop located through
  `!dbg` (the g-point loop is the outermost: unset, it would run once), and
  `conds` for the sunlit-column branch. The LDL are pointer reloads from an
  argument struct that LLVM keeps in memory (784 B per thread, above the L1
  share, so they go to L2).
- **Accuracy** (blind test, 2026-10; err = exp(|ln(model / measured)|) - 1,
  at the clock of each run). 8066 measured kernel times on A100, V100 and P100
  GPUs (2521 distinct kernels) from ClimaCore, RRTMGP, Oceananigans and
  CUDA.jl/KernelAbstractions/NNlib, none seen in development, with the flags
  set from this section alone: median err 0.17; 63% within 25%, 81% within
  1.5x, 89% within 2x (87% when each distinct kernel counts once). By
  package, within 2x: ClimaCore 96% (72% within 25%), RRTMGP 85%,
  Oceananigans 80%, plain CUDA.jl/KernelAbstractions/NNlib kernels 72%. By
  timing: Nsight Compute times 97% within 2x, CUPTI times of normal runs 80%.
  (The model was developed on about 8000 other measured CliMA kernels, median
  err 0.11-0.16.)

  Trust it for comparing variants of one kernel under identical settings (the
  direction and rough size of a change), for finding the binding unit and
  the stalls, and for checking a trace (op counts, sectors, spills). Do not
  trust a single absolute time without a measurement, or any result in the
  weak spots below. Weak spots, with workarounds:
  - Kernels whose outermost loop has a run-time count (matmul, SpMV,
    iteration counts such as Collatz or Mandelbrot, RRTMGP's g-point solves)
    come out 10-1000x too fast unless `trips` names that loop: the blind
    test gave it 1 trip, so `loops` never reached it. Set `trips` (the
    outermost-loop note says which loop).
  - CUPTI times of normal runs where the cache state depends on data flow
    that nobody recorded: decide `cache` from the program, or compare with
    Nsight times (cold).
  - Fused solvers, column reductions and accumulations: 22-39% of them
    outside 2x. Set iteration counts in `trips` and compare op counts with
    Nsight's instructions per warp.
  - GPUArrays copies and permutes come out at about 0.6x (too fast).
  - Float64 saturation-adjustment kernels from SASS came out about 1000x too
    fast: check the dynamic op count against Nsight's before trusting them.

  Not tested: tensor cores, cuBLAS/cuFFT, multi-GPU runs, Hopper and Ada
  GPUs, dynamic parallelism, textures.
- **Known error sources** that the model does not represent:
  - Data-dependent branches follow `conds` (50/50 by default). Special values
    (Inf, NaN, denormals) send divisions, square roots and libdevice
    functions to slow paths that the model does not charge.
  - Nested early exits all take the `loops` count: a Newton solve and its
    rarely-used line search both run 4 with `:few` (1.2-2.3x slow).
  - Libdevice functions without a measured SASS sequence (`sin`, `cos`,
    `fmod`, ...) are 12-op placeholders; the measured sequences are sm_80's.
  - Read-modify-write and misaligned boundary sectors are charged per warp
    (L2 merges some of them); 3D spectral operators with vector results come
    out 0.4-0.67 with Nsight hit rates (`lg_throttle` that the model lacks),
    and 2D DSS local summations with many components 0.4-0.5 (a limit on
    outstanding scattered requests).
  - Shared-memory accesses block issue while the LSU is busy (no queue);
    uneven block dispatch across SMs is not modeled (one average SM; check
    `sm__cycles_active.max/avg`); kernels launched back to back keep
    instruction and constant lines that the model fetches cold.
  - SASS control words are not in the disassembly: latency-bound kernels with
    few warps per scheduler come out ~10% fast from SASS.
- **Extend it in place.** Device rows (`OPS`, `DEVICES`) carry one citation
  each; IR mappings live in `EXPANSIONS`/`CALLS`, SASS classes in
  `SASS_CID`. Add a known-answer test to `gpu_perf_model_tests.jl` with every
  change, and replace assumption-marked (`ASM`) values when measurements
  appear.

## 6. Isolate low-level issues and fix them at their source

- Chase symptoms down to the single lowest-level function responsible, using
  JET/`@test_compilation`, then fix that function, not the symptom — e.g.
  call a branch-free internal instead of an entry point whose unreachable
  throw branch JET flags, or construct a wrapper explicitly when a library's
  constructor returns a type inference cannot determine.
- Using un-exported internals is acceptable when the fix is small, guarded
  (`@static if isdefined(...)` with a public-API fallback), and documented
  with the reason.
- Rules that recur on GPUs (see
  [gpu_performance.md §8](../../../performance/gpu_performance.md)):
  kernel-launched closures may capture only isbits values (derive types from
  argument types inside the closure, never capture a `Type`); error paths
  that build strings at runtime cannot compile in kernels (use static
  messages or move checks to the host); `Adapt` does not descend into
  `Base.Pair` or unregistered wrappers (write explicit `adapt_structure`
  rules); partial-rank views/reshapes of device arrays pull in
  `SignedMultiplicativeInverse` string-throwing constructors (index at full
  rank).

## 7. Keep hypotheses alive in a ledger; re-test them as the code evolves

- **Maintain a written ledger of measurements and hypotheses.** Keep one
  untracked Markdown file at the repository root (e.g. `ci_ledger.md` — NOT
  in a session-specific temp directory, so it survives across sessions; never
  commit it) with two tables:
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
  git history.

## 8. Compiler annotations conceal as often as they cure

- Before adding `@generated`, `@assume_effects`, `@constprop`, `@inline`, or
  a `Method.recursion_relation` override, demonstrate the compiler failure it
  is meant to fix (e.g. show that inference gives up ONLY on large, deeply
  nested expressions); after adding one, re-test the original symptom — if
  it is still there, the annotation was hiding a structural problem.
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
