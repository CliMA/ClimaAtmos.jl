# Known issues

Open problems that are understood but not yet fixed. Each entry records what is
established, so the next person does not have to re-derive it. GitHub Issues are
disabled on this repository, so this file is where they live.

Remove an entry when it is fixed.

## 1. Tagged water closure assertions fail in the dynamics test group

**Status:** diagnosed; the two assertions are corrected in
`test/tagged_water_integration.jl`, awaiting a dynamics run that reaches them.

Two assertions in `test/tagged_water_integration.jl` failed deterministically on
`ci 1.10 - dynamics` (run
[32335353545](https://github.com/johannespletzer/ClimaAtmosResiDyn.jl/actions/runs/32335353545)):

```
Tagged water limiter rescale: Test Failed at test/tagged_water_integration.jl:267
  Expression: maximum(abs.(residual)) / scale < 0.001
   Evaluated: 0.0011732894309513337 < 0.001

Tagged water 1M sedimentation closure: Test Failed at test/tagged_water_integration.jl:441
  Expression: maximum(norm) <= 1 + 100 * eps(FT)
   Evaluated: 1.000060085395493 <= 1.0000000000000222
```

Both measure the same quantity — how far the partition tags have drifted from
`ρq_tot` — and neither is a statement the implementation makes.

  - `norm` is `Σₖ clamp(ρq_tagₖ / ρq_tot, 0, 1)` over the partition tags. Once
    `repair_water_tag_partition!` has made the tags non-negative, that is
    `Σₖ ρq_tagₖ / ρq_tot` wherever no single tag exceeds the parent, i.e. the
    *pointwise relative* closure residual. Bounding it by `1 + 100 · eps` asserts
    exact pointwise closure, which `bfd5b4a` deliberately declines to provide:
    the repair does not renormalize the tags onto `ρq_tot`, because doing so
    would drive `q_tag_res` to zero by construction and destroy the leakage
    monitor. The same file budgets that leakage at `5e-3` (column) and `1e-3`
    (sphere), and `norm` is the harsher measure of the two because it normalizes
    by the local `ρq_tot` rather than by the column maximum.

    The property the assertion's comment claims — that the denominator cannot
    amplify the shares it divides — needs no bound on `norm` at all: each clamped
    share is one of its non-negative terms, so every share is in `[0, 1]` and the
    partition's shares sum to 1 for any positive `norm`. That is now asserted
    directly on the shares, and `norm` keeps a drift monitor at `1 + 1e-2`.

  - The sphere residual tolerance of `1e-3` predates the repair. The test was
    added in `cadb2ec`, the repair in `bfd5b4a` ten hours later, and the repair
    changes exactly what the assertion measures: it zeroes the tags of a cell
    whose negatives outweigh its positives, and empties them when a constraint
    clips a non-positive `ρq_tot`, so the removed water surfaces in the residual
    by design. The repair was committed unrun ("no Julia toolchain in this
    environment"), and this repository's Actions history begins on 2026-08-19,
    after it — so no CI run has ever observed these tests green. The tolerance is
    now `1e-2`, which keeps the residual nearly two orders inside the `1e-1`
    excursion bound the individual tags get in the same testset.

Also established:

  - Deterministic, not flaky. The same two assertions failed on every run that
    reached them.

  - Resolution-dependent magnitude: `ci 1.10 - dynamics` evaluates `norm` at
    `1.000060085395493`, `Downgrade 1.10` at `1.0001545917163408`. Both are
    inside the new bound.

  - Not caused by the Levante GPU runscript work in #21. It reproduces
    identically before and after the only source changes on that branch, which
    were five blank lines inside docstrings in
    `src/diagnostics/tagged_water_diagnostics.jl` and
    `src/prognostic_equations/constrain_state.jl`.

What is not settled: whether a pointwise drift of `6e-5` in `norm`, and `1.2e-3`
in the sphere residual, is the right amount of leakage for this scheme. The
corrected assertions bound it and record it; tightening it would mean changing
the closure, not the test.

## 2. Levante 1/2/4 GPU scaling has not been measured

**Status:** open, needs a run on Levante.

`runscripts/xmodel.1gpu`, `xmodel.2gpus` and `xmodel.4gpus` are verified
correct on the machine — the binding report shows `MATCH` on every rank and the
CUDA/MPI device test passes — but the strong-scaling numbers they exist to
produce have not been collected. The measurement protocol is in
`runscripts/README.md`.

## 3. Tagged water does not close under AMD LES or under PrognosticEDMFX

**Status:** diagnosed, not fixed. Neither combination is exercised by any test,
so nothing currently fails.

Two transport paths move `ρq_tot` in ways the water tags do not follow, so
`Σᵢ ρq_tag_i = ρq_tot` stops holding. Both are properties of the tagged-water
implementation rather than of any particular run, and both predate the merge of
the passive-tracer line.

  - **AMD LES.** `parameterized_tendencies/les_sgs_models/anisotropic_minimum_dissipation.jl:135-152`
    (horizontal) and `:282-300` (vertical) recompute `ᶜD_amd` inside
    `foreach_gs_tracer` from *each tracer's own* gradient. So `ρq_tot` is
    diffused with `D(∇q_tot)` and each `ρq_tag_k` with `D(∇χ_k)`, and
    `Σₖ ∇⋅(ρ Dₖ ∇χₖ) ≠ ∇⋅(ρ D_tot ∇q_tot)` because the operator is nonlinear.
    This is not transport "the tags receive in their own right" — it is a
    genuine break of the partition that no bracket or repair corrects.
    Smagorinsky–Lilly (`smagorinsky_lilly.jl:167-179`) shares one `ᶜD_h` and
    does close, as does constant horizontal diffusion.

  - **PrognosticEDMFX.** The SGS mass-flux loops in `edmfx_sgs_flux.jl:106,121`
    are driven by `sgs_tracer_names(Y)`. Tags have no `sgsʲs` entries, so they
    are skipped — safely, but they never receive that first-order water
    transport. `check_water_tagging_supported` screens only the microphysics
    model, so the combination is accepted silently. The claim in
    `tagged_tracers/tagged_water.jl:18-20` that the implicit/explicit
    vertical-advection split is "the one irreducible source of closure leakage"
    is not true under EDMF.

Either guard the combinations in `check_water_tagging_supported`, or give the
tags the matching transport. Until then, read `q_tag_res` as a closure monitor
only for configurations that use a shared diffusivity and no prognostic EDMF.

## 4. The implicit water-microphysics attribution has no Jacobian diagonal

**Status:** diagnosed, not fixed.

`implicit/implicit_tendency.jl:55-64` puts the `:microphysics` water bracket on
the implicit path. Its increment is `min(Δ, 0) · ρq_tag / ρq_tot`, which is
proportional to `ρq_tag`, so `∂/∂ρq_tag = Δ⁻/ρq_tot` — the same O(1/dt)
quantity the file's own positivity argument names. Nothing supplies that entry:
under 0M the tags get the ordinary passive diagonal
(`manual_sparse_jacobian.jl:1286`), or a plain `-I` when diffusion is explicit;
under 1M the sedimentation diagonal carries no microphysics term.

The comment at `:300-303` justifying the *energy* bracket's `-I` ("the
attributed increment does not depend on the tags themselves") is true for
`:precipitation` and false for the water bracket added directly above it. With
a fixed Newton iteration count this is error in the answer rather than only
slower convergence. Needs a precipitating run to show up; no GitHub CI job
reaches it.

## 5. `fill_with_nans!` would destroy the tag masks if it ever descended into the cache

**Status:** latent; harmless today.

The debug helper would overwrite the static region masks and the `ᶜwater_fix`
ledger along with everything else. It does not, only because `AtmosCache` is a
plain struct and hits the `::Any` fallback — which means the feature is a no-op
in general, not that the tags are protected. Worth knowing before anyone makes
it work.

## 6. Cleanup findings that belong to upstream ClimaAtmos, not to this fork

**Status:** verified as upstream's, deliberately unchanged here.

A cleanup review of `851cafa` raised these. Each was checked against
`CliMA/ClimaAtmos.jl@main` and is still there, byte for byte. This repository
tracks upstream, so fixing them here would mean a conflict on every future
merge for no benefit to this fork. They are recorded so the next review does
not re-derive them.

  - **Circular conservation claims in the microphysics tests.**
    `test/parameterized_tendencies/microphysics/bmt_integration.jl:229-270` and
    `sgs_quadrature.jl:467-532` define the vapor tendency as the negative sum
    of the others and then only check that it is finite. That tests
    construction, not conservation. Much of `bmt_integration.jl` also exercises
    CloudMicrophysics structures directly rather than the ClimaAtmos wrappers.
  - **Gravity-wave jobs called tests.** Seven active jobs in
    `.buildkite/full_pipeline.yml:150-189` run scripts that produce plots and
    assert nothing. They pass whenever the script does not crash.
  - **Unverified downloads.** `test/artifact_funcs.jl` fetches mutable external
    files into `tempdir()` with no checksum. This fork no longer downloads them
    during unit tests (see below), but the four standalone gravity-wave scripts
    still use these functions.
  - **2M and 2MP3 microphysics advertised but rejected.**
    `src/cache/precomputed_quantities.jl:160-167` asserts against both, while
    the config parser, default help, README and microphysics documentation
    still list them as supported.
  - **Dead private functions.** `ᶠupdraft_nh_pressure_buoyancy` and
    `ᶠupdraft_nh_pressure_drag` in
    `src/prognostic_equations/mass_flux_closures.jl`, and `add_sgs_ᶜK!` in
    `src/cache/precomputed_quantities.jl` (with its commented-out call at
    `:749`), have no callers.
  - **Orphan files under `test/`.** `test/implicit/debugging_tools.jl` is
    unreferenced and says so itself;
    `test/parameterized_tendencies/gravity_wave/orographic_gravity_wave/compute_preprocessed_topography.jl`
    is a data-generation tool, not a test.
  - **Comments that record patch history.**
    `src/prognostic_equations/mass_flux_closures.jl:141` ("used to have"),
    `src/simulation/AtmosSimulations.jl:210` ("backward compatibility since"),
    `test/gpu_setups.jl:35-38`, and
    `test/prognostic_equations/tracer_mass_consistency_tests.jl:89-90`
    ("pre-fix"). The two comment sites this fork owns were rewritten.
  - **`solve_atmos!` contract.** The docstring in `src/simulation/solve.jl:99`
    says failures are caught and writers closed on every path, but the first
    `CTS.step!`, `precompile_callbacks` and `GC.gc()` all run before the
    `try` at `:128`.
  - **Restart-test duplication.** `test/restart.jl` and
    `test/restart_AtmosSimulation.jl` duplicate the checkpoint/reload/compare
    contract. Note that the review's proposed `restart_utils.jl` already
    exists; what is real is the stale signature documented at
    `test/restart_AtmosSimulation.jl:156-164`, which names
    `test_restart(simulation, model, grid; job_id, ...)` for a function that
    takes `(simulation, args; comms_ctx, more_ignore)`.
  - **Inactive pipeline history.** `.buildkite/full_pipeline.yml` carries
    several wholly commented-out jobs.
  - **`perf/flame.jl`.** The `@allocated` pass is labelled "old" and "TODO:
    remove" although it is the pass that enforces the allocation limit;
    `Profile.Allocs` only produces the report.
  - **Placeholder testsets.** `test/conservation/*.jl`,
    `test/prognostic_equations/hyperdiffusion_tests.jl` and `tendency_tests.jl`
    held twelve `@test_skip` placeholders and no assertions. They ran in this
    fork's `dynamics` group, so they were removed here; upstream still has
    them. The tests they were meant to become — global dry-air mass, total
    water mass and tracer mass conservation, hyperdiffusion tendency, and
    tendency-computation coverage — are worth writing against real reference
    values rather than restoring as scaffolds.
  - **Julia 1.9 compatibility.** Upstream still declares `julia = "1.9"` while
    testing only 1.10 and 1.11. This fork raised its own floor to 1.10.
