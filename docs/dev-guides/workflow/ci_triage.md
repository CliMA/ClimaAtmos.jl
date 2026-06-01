# CI Failure Triage

A checklist for the most common "passes locally, fails on CI" failure modes in CliMA repos. Walk down it in order; each item is something CI exercises that a casual local run does not.

## 1. Julia version mismatch

Most CliMA repos test on a *matrix* of Julia versions (typically the current LTS plus one or two newer point releases) across Linux, macOS, and Windows. Check the `.github/workflows/ci.yml` of the repo you're working in to see which versions are exercised.

A version-specific failure is almost always one of: a method that exists only on newer Julia versions, a precompilation cache that differs between versions, or a syntactic change accepted in one version but not another. Reproduce locally with `juliaup add <version>` and `julia +<version> --project ...`.

## 2. OS mismatch (Windows in particular)

Path separators (`/` vs `\`), file-mode bits, line endings (`\r\n`), and locale-dependent string sorting all bite on Windows. Use `joinpath`, `splitpath`, and `Sys.iswindows()`; never concatenate paths with `*` or string literals.

`macOS-latest` runners occasionally differ from Linux in BLAS rounding at the last bit. If a tolerance test fails only on macOS by `eps(FT)`, the right fix is to loosen `atol` to `eps(FT) * N` for sums of length `N`, not to special-case the OS.

## 3. Downgrade and invalidations jobs

Several repos (Thermodynamics, ClimaCore, others) run two extra jobs that catch issues a plain `Pkg.test()` misses:

- **Downgrade**: re-runs tests with the *lowest* compat-allowed version of every dep. Failures here mean a `[compat]` lower bound is too permissive — usually because new code used an API added in a later version. Fix by bumping the relevant lower bound.
- **Invalidations**: compares method-table invalidations against `main`. Failures mean a change in this PR added or strengthened invalidations of upstream code, which slows TTFX for downstream users. The fix is typically to narrow a method signature or remove a type-piratical method.

## 4. Downstream tests

Library repos (Thermodynamics, CloudMicrophysics, SurfaceFluxes, ClimaTimeSteppers) run a `Downstream` workflow that builds *consumer* packages (ClimaAtmos, ClimaCoupler, ClimaLand) against the PR. A downstream failure means a change is API-compatible by the package's own tests but breaks a consumer that depends on a behavior the package's tests didn't pin. The fix is usually one of: (a) add a `NEWS.md` breaking-change entry and let the consumer adapt, or (b) restore the implicit behavior and add a test for it.

## 5. Float32 promotion

A test that uses Float64 literals (`1.2`, `Inf`, `6^x` with integer base) passes on Float64 CI runs and fails on Float32 ones with a `BroadcastInferenceError` or a `Union{Float32, Float64}`-typed result. See [type_stability.md §1](../performance/type_stability.md). The Float32 path is sometimes exercised only in a separate Buildkite job — check whether your repo's Buildkite pipeline has a `Float32` flavor before assuming GitHub Actions covers it.

## 6. GPU vs CPU

A test that runs on CPU in GitHub Actions may run on GPU on Buildkite. Common GPU-only failures:

- Scalar indexing on a `CuArray` (`field[i]`): caught by `CUDA.allowscalar(false)` in test setup. See [clima_comms.md §1](../infrastructure/clima_comms.md).
- `InvalidIRError: unsupported dynamic function invocation`: a closure captured a boxed variable or hit a `Function`-typed field. See [SDP 18](../code-quality/software_design_patterns.md).
- Object not `isbits` after `cudaconvert`: a `Vector`/`String`/abstract field reached the kernel boundary. See [gpu_performance.md §8](../performance/gpu_performance.md).
- `BroadcastInferenceError` only on GPU: a Float64 literal or an inlining-budget overflow that the CPU compiler tolerates but the GPU compiler does not.

## 7. MPI rank-count sensitivity

Tests that use global reductions or random number streams can pass on 1 rank and fail on 4. Two failure modes:

- **Reduction ordering**: `sum` over a distributed field reorders floating-point additions per rank count. Loosen tolerances to `eps(FT) * N` for sums of length `N`, or compare in integer space when possible.
- **Random state**: `rand()` is not synchronized across ranks. Seed explicitly per rank or generate on root and broadcast. See [clima_comms.md §2](../infrastructure/clima_comms.md).

## 8. Allocation regressions

Allocation tests typically use the warm-up + `@allocated == 0` pattern (see [allocation_debugging.md](../performance/allocation_debugging.md)). If they pass locally but fail on CI:

- The CI runner is colder; check that the test calls the function once before the assertion.
- A new precompilation entry in your branch may have shifted what the first call allocates — the warm-up captures it, but a sibling test running just before may not have. Add an explicit warm-up inside the test, not at module top level.

## 9. Documentation build failures

The `Documentation` workflow runs `docs/make.jl`. Two recurring failures:

- **Missing docstrings**: an exported symbol with no docstring (or with one not referenced from a docs page) triggers `Documenter`'s "missing docstring" error. Add an `@docs` block or use `@autodocs` over the module.
- **Cross-reference errors**: `[text](text)` in a docstring is parsed as a link. Use parentheses for units (`(kg/m^3)`), not brackets. See [documentation_policy.md](../code-quality/documentation_policy.md).

## 10. Formatter

A formatter failure is almost never about your changes — your local JuliaFormatter major version differs from CI's. See [code_style.md §1](../code-quality/code_style.md) for the version-matching procedure.

## 11. Aqua

`test_stale_deps` failing usually means a dev tool slipped into root `[deps]`; move it to `test/Project.toml`. `test_piracies` means you defined a method on a type you do not own; either move the method to the package that owns the type or wrap the type in your own struct. See [testing_and_validation.md](../infrastructure/testing_and_validation.md).

## 12. Buildkite shared-depot corruption

Buildkite jobs on shared CliMA clusters use a per-pipeline Julia [depot](https://pkgdocs.julialang.org/dev/depots/#Shared-depots-for-distributed-computing) to amortize precompilation across runs. The depot occasionally becomes corrupted — usually after a Julia version bump or an interrupted precompile — and the symptom is a failure in the *initialization* step, not in your code. Telltale messages:

- `Warning: The call to compilecache failed to create a usable precompiled cache file`
- `ERROR: LoadError: Failed to precompile <package>`
- `ERROR: \`Pkg=...\` depends on \`OtherPkg=...\`, but no such entry exists in the manifest`

(Exact wording varies by Julia version; the consistent signal is a precompile/manifest failure during pipeline initialization, not in the test step itself.) When you see these on a fresh PR with no manifest changes, the depot is the suspect. Clearing it is a one-line maintainer action; the next pipeline run rebuilds the cache. The procedure is documented in the [CliMA slurm-buildkite wiki](https://github.com/CliMA/slurm-buildkite/wiki/Clearing-Shared-Depots).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
