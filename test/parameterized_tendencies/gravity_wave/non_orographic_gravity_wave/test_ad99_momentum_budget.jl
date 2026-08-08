# ============================================================================
# AD99 non-orographic gravity wave — column momentum budget tests
#
# These are the first assertion-based tests for the AD99 machinery (every other
# `@test` under this directory covers the Beres convective source).
#
# WHAT IS BEING TESTED
#
# The scheme launches a spectrum of waves at `source_level`, each phase-speed
# bin `c[n]` carrying momentum flux `B0[n]`. Waves die by saturation, by hitting
# a critical level, or by being force-broken at the model top; the flux of the
# force-broken ones is collected and redistributed over the levels at and above
# `damp_level`. Since nothing is meant to be thrown away, the momentum actually
# deposited in the column must equal the momentum launched:
#
#     Σ_k  ρ_k · Δz_k · a_k   =   F_launch
#
# and `F_launch` has an exact closed form. Per wavenumber band `ink`, the kernel
# hands the half-layer between centres k and k+1 a flux `ρ_source · ε · fm_k`
# with `ε = (source_ampl / ρ_source / nk) / Σ_n |B0[n]|`. Every wave deposits
# exactly once (Σ_k fm_k = Σ_n B0[n]), `B0` does not depend on `ink`, and the
# `nk` cancels, so
#
#     F_launch = source_ampl · Σ_n B0[n] / Σ_n |B0[n]|
#
# independent of the density profile, the wind profile and the grid. That is the
# target both tests below compare against.
#
# The scheme has three places where that equality can fail, and each test below
# isolates exactly one of them.
#
# Test 1 — the interior. Every wave is driven into a critical level, and the
# launch amplitude is small enough that none is deleted at the source level (see
# test 3), so the budget exercises only the kernel's flux-to-acceleration
# conversion and the half-layer-to-cell remap done by `gw_average!`. Those two
# are a matched pair: the kernel divides by the mass of the layer BETWEEN two
# cell centres, `sqrt(ρ_k·ρ_k+1)·(z_k+1 - z_k)`, and `gw_average!` then splits
# each such layer's drag half into the cell below and half into the cell above.
# The residual is the ratio of the arithmetic to the geometric mean of two
# neighbouring cell masses, which for a uniform grid and ρ ∝ exp(-z/H) is
# exactly cosh(Δz / 2H) — +0.2552 % here. Second order in Δz/H, and it is
# asserted against that closed form. Test 1 PASSES.
#
# Test 2 — the escaped flux. The launch amplitude is made so small that no wave
# ever saturates and the wind is constant so none meets a critical level, so
# 100 % of the launched flux is force-broken at the model top and the entire
# deposit comes from `gw_deposit`. That routine divides by a level COUNT where
# it should divide by the receiving MASS. Test 2 FAILS on the current code.
#
# Test 3 — the source level. The kernel used to start its break test one level
# BELOW the launch level while guarding the `fm` accumulation with
# `level >= source_level`, so a wave already saturated at its own launch level
# was dropped from the spectrum with its flux discarded. Since `ε` normalises by
# the full Σ|B0|, the scheme then launched less than the nominal `Bt_0`. That
# came from MiMA, not from the port: `cg_drag.f90` loops `do k=iz0,0,-1` and
# guards with `if (k < iz0)`, the same level and the same guard once the
# opposite index directions are accounted for. It cost a few per cent of the net
# launched momentum in the extratropics and essentially nothing in the tropics,
# where `gw_flag = 0` centres the spectrum on the launch wind so the deleted set
# is symmetric and cancels. The break test now starts AT the launch level, so
# those waves deposit just above it. Test 3 runs the production launch amplitude
# where this actually bites and requires the budget to close. It PASSES; before
# the fix it read 0.94090.
#
# HOW TO READ THE OUTPUT
#
# Each case prints the measured `deposited / F_launch` next to the value
# predicted for each of the three possible states of `gw_deposit`, so the
# printout identifies which state the source tree is in.
#
# Grid: config/model_configs/single_column_nonorographic_gravity_wave.yml —
# 50 levels, uniform 1 km spacing to 50 km, AD99 only (no Beres source, so
# `postprocess_and_accumulate!` runs exactly once per band).
#
# Run with (single column, CPU):
#   CLIMACOMMS_DEVICE=CPU julia +1.11 --project=.buildkite \
#     test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/test_ad99_momentum_budget.jl
# ============================================================================

using Test
using Printf
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

# ----------------------------------------------------------------------------
# Build the single-column simulation once; both tests reuse it.
# ----------------------------------------------------------------------------
comms_ctx = ClimaComms.SingletonCommsContext()
config_file = joinpath(
    @__DIR__,
    "../../../../config/model_configs/single_column_nonorographic_gravity_wave.yml",
)
config = CA.AtmosConfig(
    config_file;
    job_id = "ad99_momentum_budget",
    comms_ctx,
)
simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u
FT = eltype(Y.c.ρ)

gwc = p.non_orographic_gravity_wave
# AD99 only. With a Beres source configured, `postprocess_and_accumulate!` runs
# a second time per band and the budget below would mix the two sources.
@assert isnothing(gwc.gw_beres_source)
(;
    gw_c,
    gw_c0,
    gw_nk,
    gw_ncval,
    uforcing,
    vforcing,
    u_waveforcing,
    v_waveforcing,
    u_waveforcing_top,
) = gwc

# Grid geometry as plain arrays. Cell thickness comes from the face
# coordinates, so the test does not assume a uniform or unstretched grid.
ᶜz = Fields.coordinate_field(Y.c).z
center_z = Array(Fields.field2array(ᶜz))[:, 1]
face_z = Array(Fields.field2array(Fields.coordinate_field(Y.f).z))[:, 1]
Δz_cell = diff(face_z)
nlev = Spaces.nlevels(axes(Y.c.ρ))
@assert length(Δz_cell) == nlev

# Spectrum half-widths and the total source flux come from the run's own
# parameters; only the amplitude `Bw` is varied between the two tests.
cw_val = FT(parent(gwc.gw_cw)[1])
cn_val = FT(parent(gwc.gw_cn)[1])
Bt_0 = FT(parent(gwc.gw_source_ampl)[1])

# Source level: same rule the column branch of `compute_tendency!` uses.
source_level = argmin(abs.(center_z .- gwc.gw_source_height))

# Isothermal-like density, ρ = ρ0 exp(-z/H). Imposed rather than taken from the
# model state so that the density scale height, and therefore the expected
# second-order interior error, is known exactly.
const H_ρ = FT(7000)
const ρ0 = FT(1.2)
ρ_profile = @. ρ0 * exp(-center_z / H_ρ)

# Constant buoyancy frequency. It has to be large enough that no wave is
# removed by total internal reflection, which is the one removal path that
# discards a wave's flux instead of depositing it, and would therefore break
# the budget for reasons that have nothing to do with the mass weighting.
# Reflection needs |c - u| ≳ ω_r/k ≈ N / sqrt(k² + 1/(4H²)) · k/k, which is
# about 13400·N m/s for the 300 km band; N = 0.03 puts the threshold near
# 400 m/s, well above the largest |c - u| in either test.
const N_const = FT(0.03)

# Scratch fields for the imposed state (Y is left untouched).
ᶜρ_t = similar(Y.c.ρ)
ᶜu_t = similar(Y.c.ρ)
ᶜv_t = similar(Y.c.ρ)
ᶜbf_t = similar(Y.c.ρ)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

"""
    launched_flux(u_source, Bw)

Exact total launched momentum flux (Pa), `source_ampl · ΣB0 / Σ|B0|`, together
with the spectrum asymmetry `|ΣB0| / Σ|B0|`. The asymmetry must be well away
from zero for the budget test to have any power: with a symmetric spectrum and
a symmetric atmosphere, each eastward wave and its westward twin break at the
same height carrying opposite flux, so a mass-weighting error cancels exactly
in the signed budget and a broken scheme would still pass.
"""
function launched_flux(u_source, Bw)
    B0 = CA.wave_source(
        gw_c,
        FT(u_source),
        FT(Bw),
        FT(0),          # Bn = 0: single Gaussian, keeps the spectrum analytic
        cw_val,
        cn_val,
        gw_c0,
        FT(1),          # flag = 1, matching the column cache
        gw_ncval,
    )
    s_signed = sum(B0)
    s_abs = sum(abs, B0)
    return (
        F_launch = Bt_0 * s_signed / s_abs,
        asymmetry = abs(s_signed) / s_abs,
    )
end

"""
    source_level_deleted_flux(u_source, Bw)

Signed momentum flux (Pa) that the kernel removes at `level == source_level - 1`
without ever depositing it, computed in closed form.

That level's saturation threshold is exactly `fac = 0.5·kwv/N`: the kernel forms
`fac = 0.5·(ρ_kp1/ρ_source)·kwv/bf_kp1`, and at `source_level - 1` the shifted
field `ρ_kp1` is the density at `source_level`, so the density ratio is exactly
one. A bin is deleted when `B0[n]/(c[n] - u_source)³ ≥ fac`. The critical-level
test cannot fire there (`c_hat` equals `c_hat0`, so their product is a square)
and reflection needs |c - u| near 400 m/s, so saturation is the only path.

Only the 300 km band exists here (`nk == 1`), so `kwv` is a single value.
"""
function source_level_deleted_flux(u_source, Bw)
    B0 = CA.wave_source(
        gw_c,
        FT(u_source),
        FT(Bw),
        FT(0),
        cw_val,
        cn_val,
        gw_c0,
        FT(1),
        gw_ncval,
    )
    kwv = 2 * FT(π) / (FT(30) * 10 * FT(1000))
    fac = FT(0.5) * kwv / N_const
    deleted = FT(0)
    for n in eachindex(gw_c)
        c_hat = gw_c[n] - FT(u_source)
        c_hat == 0 && continue
        if B0[n] / c_hat^3 - fac >= 0
            deleted += B0[n]
        end
    end
    return Bt_0 * deleted / sum(abs, B0)
end

"""
    run_case(; u_profile, Bw, damp_level)

Impose the state, run the AD99 forcing once, and return the column momentum
budget. `deposited` integrates ρ·Δz·a over cells, which is the momentum the
model actually receives when the forcing is added to the `uₕ` tendency.

`escaped` reconstructs the flux that was force-broken at the model top from
`u_waveforcing_top`, undoing the kernel's conversion with the same ghost-level
mass the kernel used: `rbh_top = sqrt(ρ_end · ρ_end²/ρ_end-1)` over a
centre-to-centre spacing of `z_end - z_end-1`. Valid here because `nk == 1`, so
the single band's value is the one left in the cache field.
"""
function run_case(; u_profile, Bw, damp_level)
    @assert gw_nk == 1 "the `escaped` diagnostic assumes a single wave band"

    fill!(gwc.gw_Bw, FT(Bw))
    fill!(gwc.gw_Bn, FT(0))
    fill!(gwc.gw_cw, cw_val)
    fill!(gwc.gw_cn, cn_val)
    fill!(gwc.gw_flag, FT(1))
    fill!(gwc.gw_source_ampl, Bt_0)

    parent(ᶜρ_t) .= ρ_profile
    parent(ᶜu_t) .= u_profile
    parent(ᶜv_t) .= FT(0)
    parent(ᶜbf_t) .= N_const

    uforcing .= 0
    vforcing .= 0

    CA.non_orographic_gravity_wave_forcing(
        ᶜu_t,
        ᶜv_t,
        ᶜbf_t,
        ᶜρ_t,
        ᶜz,
        gwc.ᶜlevel,
        source_level,
        damp_level,
        Fields.level(ᶜρ_t, source_level),
        Fields.level(ᶜu_t, source_level),
        Fields.level(ᶜv_t, source_level),
        uforcing,
        vforcing,
        gw_ncval,
        u_waveforcing,
        v_waveforcing,
        p,
    )

    a_u = Array(Fields.field2array(uforcing))[:, 1]
    a_v = Array(Fields.field2array(vforcing))[:, 1]

    ρ_ghost = ρ_profile[end]^2 / ρ_profile[end - 1]
    Δ_top = center_z[end] - center_z[end - 1]
    m_top_kernel = sqrt(ρ_profile[end] * ρ_ghost) * Δ_top

    return (
        deposited_u = sum(ρ_profile .* Δz_cell .* a_u),
        deposited_v = sum(ρ_profile .* Δz_cell .* a_v),
        escaped = parent(u_waveforcing_top)[1] * m_top_kernel,
    )
end

"""
    deposit_predictions(damp_level)

`deposited / escaped` for the redistributed flux, for each of the three states
`gw_deposit` can be in. The redistribution writes a uniform acceleration
`a_top_stored / D` onto cells `damp_level:nlev`, so the momentum it delivers is
`M_R · a_top_stored / D` while the flux it was given is
`m_top_kernel · a_top_stored`.
"""
function deposit_predictions(damp_level)
    M_R = sum(ρ_profile[damp_level:end] .* Δz_cell[damp_level:end])
    ρ_ghost = ρ_profile[end]^2 / ρ_profile[end - 1]
    m_top_kernel =
        sqrt(ρ_profile[end] * ρ_ghost) * (center_z[end] - center_z[end - 1])
    n_recv = nlev - damp_level + 1
    return (
        shipped = M_R / (m_top_kernel * (n_recv + 1)),   # divisor N+1
        divisor_only = M_R / (m_top_kernel * n_recv),    # divisor N (matches MiMA)
        mass_weighted = FT(1),                           # the proposed fix
    )
end

report(label, ratio, pred) = @info """
$label
  measured deposited / F_launch = $(@sprintf("%.5f", ratio))
  predicted, shipped divisor (N+1)      = $(@sprintf("%.5f", pred.shipped))
  predicted, divisor-only fix (N)       = $(@sprintf("%.5f", pred.divisor_only))
  predicted, mass-weighted fix (target) = $(@sprintf("%.5f", pred.mass_weighted))"""

# ----------------------------------------------------------------------------
# Wind profile shared by tests 1 and 3.
#
# The wind rises from the source value to +120 m/s at 30.5 km, then falls to
# -130.3 m/s at the model top, so every phase-speed bin carrying non-negligible
# flux meets a critical level (u = c) inside the domain: eastward bins on the
# rising branch, westward bins on the falling branch. Bins outside
# [-130.3, +120] never do, but at four or more Gaussian half-widths from the
# centre they carry ~3e-5 of the total flux.
#
# The source wind is placed exactly on a phase-speed bin. A bin whose phase
# speed sits a hair away from the local wind has a huge B0/(c-u)³ and saturates
# instantly; landing exactly on a bin instead gives that bin B0 = 0, because the
# spectrum carries a sign(c - u_source) factor. The endpoint -130.3 is likewise
# chosen NOT to coincide with a bin: an exact coincidence anywhere above the
# source sends that bin down the `c_hat == 0` branch, which discards its flux
# instead of depositing it. Both conditions are asserted below.
# ----------------------------------------------------------------------------
u_source_shared = gw_c[argmin(abs.(collect(gw_c) .- 25))]

reversing_u_profile = let
    z_src = center_z[source_level]
    z_mid = FT(30500)
    z_top = center_z[end]
    u_hi = FT(120)
    u_lo = FT(-130.3)
    map(center_z) do z
        if z <= z_src
            u_source_shared
        elseif z <= z_mid
            u_source_shared +
            (u_hi - u_source_shared) * (z - z_src) / (z_mid - z_src)
        else
            u_hi + (u_lo - u_hi) * (z - z_mid) / (z_top - z_mid)
        end
    end
end

# ============================================================================
# Test 1 — interior budget: nothing escapes, nothing is deleted at the source
#
# `Bw = 1e-4` puts every bin's B0/(c - u_source)³ below the source-level
# saturation threshold, so the deletion measured by test 3 is exactly zero here,
# while the critical levels still remove every wave before the model top. What
# is left is the kernel/`gw_average!` pair, whose residual is cosh(Δz / 2H).
#
# `Bw` only sets WHERE waves break: `ε` renormalises by Σ|B0|, so the launched
# total is `Bt_0` regardless. That is what makes this isolation possible.
#
# Expected to PASS, at 1.00255.
# ============================================================================
@testset "AD99 column momentum budget — interior only" begin
    u_profile = reversing_u_profile
    u_source = u_source_shared
    Bw = FT(1e-4)

    lf = launched_flux(u_source, Bw)
    @test lf.asymmetry > 0.1        # setup guard: spectrum is not degenerate

    # Setup guard: no wind above the source coincides exactly with a bin.
    @test !any(u -> any(==(u), gw_c), u_profile[(source_level + 1):end])

    # Setup guard: no bin is deleted at the source level, so test 3's effect is
    # absent from this budget.
    @test source_level_deleted_flux(u_source, Bw) == 0

    res = run_case(; u_profile, Bw, damp_level = nlev)
    ratio = res.deposited_u / lf.F_launch
    report("Test 1: interior only, damp_level = $nlev", ratio, deposit_predictions(nlev))
    @info "Test 1 escaped fraction = $(@sprintf("%.2e", abs(res.escaped / lf.F_launch)))"

    # Setup guard: the redistribution must not be what this test is measuring.
    # The expected value is ~4e-5 (the far tails of the Gaussian).
    @test abs(res.escaped / lf.F_launch) < 5e-3

    # The interior pairing is exactly the ratio of the arithmetic to the
    # geometric mean of two neighbouring cell masses, which on a uniform grid
    # with ρ ∝ exp(-z/H) is cosh(Δz/2H). Asserting against the closed form is a
    # far sharper check than asserting closure to 1 %.
    interior_expected = cosh(Δz_cell[end] / (2 * H_ρ))
    @info "Test 1 closed form cosh(Δz/2H) = $(@sprintf("%.6f", interior_expected))"
    @test isapprox(ratio, interior_expected; rtol = 1e-4)

    # The physics requirement: second order in Δz/H is acceptable, first order
    # is not.
    @test isapprox(ratio, 1; atol = 0.01)

    # The meridional spectrum is exactly antisymmetric (v_source = 0, c bins
    # symmetric about zero), so it must deposit no net momentum. Catches sign
    # and index asymmetries in the kernel.
    @test abs(res.deposited_v) < 1e-6 * abs(lf.F_launch)
end

# ============================================================================
# Test 3 — the production launch amplitude, where waves saturate at launch
#
# Identical to test 1 but with the default `nogw_Bw = 0.4`, which puts the bins
# nearest the source wind above their own saturation limit at the launch level.
# Those are the waves the kernel used to delete: it dropped them from the
# spectrum at `source_level - 1` while skipping the `fm` accumulation, so their
# flux went nowhere. They now break in the layer just above the launch level and
# deposit there, so the budget closes at cosh(Δz/2H) exactly as in test 1.
#
# `source_level_deleted_flux` is kept as a regression guard rather than a
# prediction: it computes what the OLD code would have thrown away for this
# setup, and the test asserts it is large. That proves this case genuinely
# exercises the launch-level path, so if the guard in
# `waveforcing_column_accumulate!` ever slips back to `source_level - 1` this
# test fails rather than quietly passing as a duplicate of test 1.
# ============================================================================
@testset "AD99 column momentum budget — saturation at the launch level" begin
    u_profile = reversing_u_profile
    u_source = u_source_shared
    Bw = FT(0.4)          # the ClimaParams default, i.e. what production runs use

    lf = launched_flux(u_source, Bw)
    would_have_been_deleted = source_level_deleted_flux(u_source, Bw) / lf.F_launch

    res = run_case(; u_profile, Bw, damp_level = nlev)
    ratio = res.deposited_u / lf.F_launch
    interior_expected = cosh(Δz_cell[end] / (2 * H_ρ))

    @info """
    Test 3: saturation at the launch level, damp_level = $nlev
      measured deposited / F_launch      = $(@sprintf("%.5f", ratio))
      closed form cosh(Δz/2H)            = $(@sprintf("%.5f", interior_expected))
      flux the old code would have lost  = $(@sprintf("%.5f", would_have_been_deleted)) of F_launch"""

    # Regression guard: this setup must actually have waves that saturate at the
    # launch level, or the test proves nothing.
    @test would_have_been_deleted > 0.01

    # Setup guard: still nothing escaping the top.
    @test abs(res.escaped / lf.F_launch) < 5e-3

    # Conservation. Before the launch-level fix this read 0.94090.
    @test isapprox(ratio, interior_expected; rtol = 1e-4)
    @test isapprox(ratio, 1; atol = 0.01)

    @test abs(res.deposited_v) < 1e-6 * abs(lf.F_launch)
end

# ============================================================================
# Test 2 — escaped-flux budget: everything reaches the model top
#
# `Bw = 1e-10` makes B0/(c-u)³ four orders of magnitude below the saturation
# threshold `fac` everywhere, so no wave breaks. The wind is constant, so no
# wave meets a critical level either. All of the launched flux is therefore
# force-broken at the top level and the whole deposit comes from `gw_deposit`.
#
# Note the launched total is unaffected: `ε` renormalises by Σ|B0|, so `Bw`
# sets only WHERE waves break, while `source_ampl` sets HOW MUCH flux is
# launched. The target is the same `F_launch` as in test 1.
#
# Two sub-cases, because they constrain the divisor differently:
#   (a) damp_level = nlev, the production column setting: one receiving level,
#       shipped divisor 2. Predicted 0.537, i.e. 46 % of the escaped momentum
#       is silently dropped.
#   (b) damp_level = nlev - 3: four receiving levels, shipped divisor 5. This
#       is the sphere-like arrangement. Predicted 1.078, an over-deposit — the
#       error changes SIGN between the two arrangements.
#
# Both sub-cases are expected to FAIL on the current code.
# ============================================================================
@testset "AD99 column momentum budget — escaped flux redistribution" begin
    n0 = argmin(abs.(collect(gw_c) .- 25))
    u_source = gw_c[n0]
    u_profile = fill(u_source, nlev)
    Bw_tiny = FT(1e-10)

    lf = launched_flux(u_source, Bw_tiny)
    @test lf.asymmetry > 0.1        # setup guard

    for damp_level in (nlev, nlev - 3)
        res = run_case(; u_profile, Bw = Bw_tiny, damp_level)
        ratio = res.deposited_u / lf.F_launch
        pred = deposit_predictions(damp_level)

        report("Test 2: damp_level = $damp_level", ratio, pred)
        @info "Test 2 escaped fraction = $(@sprintf("%.4f", abs(res.escaped / lf.F_launch)))"

        # Setup guard: this test is only meaningful if essentially all of the
        # launched flux really did reach the model top.
        @test abs(res.escaped / lf.F_launch) > 0.99

        # Diagnostic, not the requirement: confirms the measured number matches
        # the algebra for whichever divisor the source tree currently has, so a
        # failure of the next assertion is unambiguous.
        @test isapprox(ratio, pred.shipped; rtol = 1e-3) ||
              isapprox(ratio, pred.divisor_only; rtol = 1e-3) ||
              isapprox(ratio, pred.mass_weighted; rtol = 1e-3)

        # The requirement: the redistribution deposits exactly the flux it was
        # handed. Fails today.
        @test isapprox(ratio, 1; atol = 0.01)

        @test abs(res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end
