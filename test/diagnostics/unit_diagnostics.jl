#=
Unit tests for diagnostic compute functions defined in:
  - conservation_diagnostics.jl
  - core_diagnostics.jl
  - edmfx_diagnostics.jl
  - gravitywave_diagnostics.jl
  - radiation_diagnostics.jl

Design
======
Tests are driven by two tables:

  VALID_CASES  — each entry is case(name, key) or case(name, (key1, key2, ...))
                 where each key identifies a fixture in the `states` dict.
                 The test loop runs the diagnostic against every listed fixture
                 and asserts the result is all-finite. Multi-key entries exercise
                 different model-dispatch paths for the same diagnostic.

  SKIP_CASES   — Set of names from out-of-scope files (tracer_diagnostics.jl,
                 negative_scalars_diagnostics.jl). These are excluded from the
                 exhaustiveness check; their own test files should cover them.

An exhaustiveness @testset verifies every registered diagnostic name appears in
exactly one of the two tables. The check fails if a new diagnostic is added to a
source file without a corresponding entry here.

Additional spot-checks for model-dispatch error paths (variables that must throw
on unsupported models) are in the "Model dispatch error paths" @testset below.
These are not subject to exhaustiveness.

Adding a new diagnostic
=======================
1. Identify which existing fixture key(s) cover the compute path(s). Prefer
   reusing an existing key from `states` over adding a new fixture — most
   model combinations are already present.

2. Add case(name, key) or case(name, (key1, key2, ...)) to VALID_CASES, grouped
   with related variables and annotated with the dispatch rationale.

3. If no existing fixture covers the required model combination, add a new
   fixture at the end of the fixture section and register it in `states` with a
   descriptive key. Keep fixtures minimal: use the smallest grid (column unless
   a sphere is required) and only enable the model components the diagnostic
   actually needs.

4. If the diagnostic must error on unsupported models, add a @test_throws case
   to the "Model dispatch error paths" @testset.

Fixtures
========
All fixtures are built once at module load time and shared across all cases.
The `states` dict maps symbol keys to (Y, p) tuples. Available keys:

  :dry            — DryModel + SlabOceanSST, column
  :sphere         — DryModel + SlabOceanSST, sphere (also used for rv, orog, energyo)
  :ssv            — DryModel + steady-state velocity, plane grid
  :m0             — EquilibriumMicrophysics0M, column
  :m1             — NonEquilibriumMicrophysics1M, column
  :m2             — NonEquilibriumMicrophysics2M, column
  :nogw           — 0M + NonOrographicGravityWave, column
  :m0_slab_sphere — 0M + SlabOceanSST, sphere (watero)
  :smag           — SmagorinskyLilly, sphere
  :m0_pedmfx      — 0M + PrognosticEDMFX, column
  :m1_pedmfx      — 1M + PrognosticEDMFX, column
  :m1_dedmfx      — 1M + DiagnosticEDMFX, column
  :m2_pedmfx      — 2M + PrognosticEDMFX, column
  :vd             — VerticalDiffusion, column
  :dwh            — DecayWithHeightDiffusion, column
  :allsky         — AllSkyRadiationWithClearSkyDiagnostics + aerosol_radiation=true, column

Pattern: Arrange – Act – Assert.
=#

using Test
using Dates

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Spaces

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
    build_state_cache(FT, model; grid, kwargs...) -> (Y, p)

Construct a minimal state vector `Y` and cache `p` for `model` on `grid`.
All keyword arguments have sensible defaults; override only what the diagnostic
under test actually requires (e.g. `aerosol_names` for aerosol diagnostics,
`set_steady_state_velocity = true` for steady-state error diagnostics).
"""
function build_state_cache(FT, model; grid,
    params = CA.ClimaAtmosParameters(FT),
    ic = CA.Setups.DecayingProfile(; params),
    surface_setup = CA.SurfaceConditions.DefaultExchangeCoefficients(),
    dt = FT(1.0), start_date = DateTime(2010, 1, 1),
    aerosol_names = [], time_varying_trace_gas_names = (),
    set_steady_state_velocity = false,
    vwb_species = nothing,
)
    spaces = CA.get_spaces(grid)
    Y = CA.Setups.initial_state(ic, params, model, spaces.center_space, spaces.face_space)
    # steady state velocity is only needed for some diagnostics
    steady_state_velocity =
        set_steady_state_velocity ?
        CA.get_steady_state_velocity(
            params, Y, CA.NoTopography(), "ConstantBuoyancyFrequencyProfile", "Linear",
        ) : nothing
    p = CA.build_cache(
        Y, model, params, surface_setup, dt, start_date,
        aerosol_names, time_varying_trace_gas_names, steady_state_velocity, vwb_species,
    )
    return Y, p
end

"""
    compute_diag(diag, Y, p, t = 0)

Invoke the compute function of a `DiagnosticVariable`, dispatching to either
`diag.compute(Y, p, t)` or `diag.compute!(nothing, Y, p, t)` as appropriate.
Also tests that exactly one of `compute` / `compute!` is defined.
"""
function compute_diag(diag, Y, p, t = 0)
    has_compute = !isnothing(diag.compute)
    has_compute! = !isnothing(diag.compute!)
    # Exactly one of compute / compute! must be defined
    @test has_compute != has_compute!
    return has_compute ? diag.compute(Y, p, t) : diag.compute!(nothing, Y, p, t)
end

"""
    field_data(result) -> AbstractArray

Materialise a (possibly lazy) diagnostic result and return a flat numeric array
suitable for `all(isfinite, ...)` checks.
"""
field_data(r) = parent(Base.materialize(r))

# Look up a registered DiagnosticVariable by short name.
getdiag(name) = CA.Diagnostics.get_diagnostic_variable(name)

# ---------------------------------------------------------------------------
# Fixtures — built once, shared across all test cases
# ---------------------------------------------------------------------------

FT = Float32
params = CA.ClimaAtmosParameters(FT)

sphere = CA.SphereGrid(FT; h_elem = 2)
column = CA.ColumnGrid(FT)

# Model configurations, state, cache

## Dry model, also tests slab ocean
model_dry =
    CA.AtmosModel(; microphysics_model = CA.DryModel(), surface_model = CA.SlabOceanSST())
(Y_dry, p_dry) = build_state_cache(FT, model_dry; grid = column);

## Sphere with dry model
(Y_sphere, p_sphere) = build_state_cache(FT, model_dry; grid = sphere);

## Model with steady-state velocity
# Mirrors the plane_no_topography_float64_test.yml config:
#   PlaneGrid + LinearWarp (the default for PlaneGrid) + NoTopography
#   + ConstantBuoyancyFrequencyProfile initial condition.
plane = CA.PlaneGrid(FT; x_elem = 4, z_elem = 5, z_stretch = false)
(Y_ssv, p_ssv) = build_state_cache(FT, model_dry; grid = plane,
    ic = CA.Setups.ConstantBuoyancyFrequencyProfile(),
    set_steady_state_velocity = true,
);

## Microphysics-specific models
model_0m = CA.AtmosModel(; microphysics_model = CA.EquilibriumMicrophysics0M())
model_1m = CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics1M())
model_2m = CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics2M())
(Y_0m, p_0m) = build_state_cache(FT, model_0m; grid = column);
(Y_1m, p_1m) = build_state_cache(FT, model_1m; grid = column);
(Y_2m, p_2m) = build_state_cache(FT, model_2m; grid = column);

## Non-orographic gravity wave
nogw_params = CA.NonOrographicGravityWaveParameters(FT)
non_orographic_gravity_wave = CA.NonOrographicGravityWave(;
    (f => getfield(nogw_params, f) for f in fieldnames(typeof(nogw_params)))...,
)
microphysics_model = CA.EquilibriumMicrophysics0M()
model_nogw = CA.AtmosModel(; microphysics_model, non_orographic_gravity_wave)
(Y_nogw, p_nogw) = build_state_cache(FT, model_nogw; grid = column);

## Sphere with moist model + slab ocean (watero needs MoistMicrophysics + SpectralElementSpace2D)
model_0m_slab = CA.AtmosModel(;
    microphysics_model = CA.EquilibriumMicrophysics0M(),
    surface_model = CA.SlabOceanSST(),
)
(Y_0m_slab_sphere, p_0m_slab_sphere) = build_state_cache(FT, model_0m_slab; grid = sphere);

## Smagorinsky-Lilly LES model
model_smag = CA.AtmosModel(smagorinsky_lilly = CA.SmagorinskyLilly(; axes = :UV_W))
(Y_smag, p_smag) = build_state_cache(FT, model_smag; grid = sphere);

## Radiation models
radiation_mode = CA.RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics(;
    aerosol_radiation = true,
)
model_allsky = CA.AtmosModel(; radiation_mode)
(Y_allsky, p_allsky) = build_state_cache(FT, model_allsky; grid = column,
    aerosol_names = ("DST01",),
);
# Radiation flux arrays are initialized to NaN by set_and_save! (filled only after solver runs).
# Zero them so diagnostics return finite values when tested outside a time-stepping loop.
let rrtm = p_allsky.radiation.rrtmgp_model
    for f in propertynames(rrtm)
        a = getproperty(rrtm, f)
        if a isa AbstractArray && any(isnan, a)
            @. a = ifelse(isnan(a), 0, a)
        end
    end
end

## EDMFX models
# Shared EDMF configuration — minimal (no entr/detr, no SGS fluxes)
tcp = CAP.turbconv_params(params)
edmfx_model = CA.EDMFXModel(;
    entr_model = CA.NoEntrainment(), detr_model = CA.NoDetrainment(),
    scale_blending_method = CA.SmoothMinimumBlending(),
)
pedmfx = CA.PrognosticEDMFX(; area_fraction = tcp.min_area)
dedmfx = CA.DiagnosticEDMFX(; area_fraction = tcp.min_area)

model_0m_pedmfx = CA.AtmosModel(; microphysics_model = CA.EquilibriumMicrophysics0M(),
    turbconv_model = pedmfx, edmfx_model)
model_1m_pedmfx = CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics1M(),
    turbconv_model = pedmfx, edmfx_model)
model_1m_dedmfx = CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics1M(),
    turbconv_model = dedmfx, edmfx_model)
model_2m_pedmfx = CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics2M(),
    turbconv_model = pedmfx, edmfx_model)
(Y_0m_pedmfx, p_0m_pedmfx) = build_state_cache(FT, model_0m_pedmfx; grid = column);
(Y_1m_pedmfx, p_1m_pedmfx) = build_state_cache(FT, model_1m_pedmfx; grid = column);
(Y_1m_dedmfx, p_1m_dedmfx) = build_state_cache(FT, model_1m_dedmfx; grid = column);
(Y_2m_pedmfx, p_2m_pedmfx) = build_state_cache(FT, model_2m_pedmfx; grid = column);

## VerticalDiffusion and DecayWithHeightDiffusion (no EDMF)
vdp = CAP.vert_diff_params(params)
model_vd = CA.AtmosModel(;
    vertical_diffusion = CA.VerticalDiffusion{FT}(;
        disable_momentum_vertical_diffusion = false, C_E = vdp.C_E))
model_dwh = CA.AtmosModel(;
    vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
        disable_momentum_vertical_diffusion = false, H = vdp.H, D₀ = vdp.D₀))
(Y_vd, p_vd) = build_state_cache(FT, model_vd; grid = column)
(Y_dwh, p_dwh) = build_state_cache(FT, model_dwh; grid = column)

# ---------------------------------------------------------------------------
# Case tables
# ---------------------------------------------------------------------------

# All fixtures indexed by symbol — used by the `case` constructor below.
#! format: off
states = Dict(
    :dry            => (Y_dry,            p_dry),
    :sphere         => (Y_sphere,         p_sphere),
    :ssv            => (Y_ssv,            p_ssv),
    :m0             => (Y_0m,             p_0m),
    :m1             => (Y_1m,             p_1m),
    :m2             => (Y_2m,             p_2m),
    :nogw           => (Y_nogw,           p_nogw),
    :m0_slab_sphere => (Y_0m_slab_sphere, p_0m_slab_sphere),
    :smag           => (Y_smag,           p_smag),
    :m0_pedmfx      => (Y_0m_pedmfx,      p_0m_pedmfx),
    :m1_pedmfx      => (Y_1m_pedmfx,      p_1m_pedmfx),
    :m1_dedmfx      => (Y_1m_dedmfx,      p_1m_dedmfx),
    :m2_pedmfx      => (Y_2m_pedmfx,      p_2m_pedmfx),
    :vd             => (Y_vd,             p_vd),
    :dwh            => (Y_dwh,            p_dwh),
    :allsky         => (Y_allsky,         p_allsky),
)

# case(name, key)  or  case(name, (key1, key2, ...))
# Each key refers to an entry in `states`; the test loop runs the diagnostic
# against every listed fixture, producing one test per (name, key) pair.
case(name, key::Symbol) = (; name, keys = (key,))
case(name, keys::Tuple) = (; name, keys)
# Helper function to create multiple cases with the same name but different keys
cases(names, keys) = map(n -> case(n, keys), names)

#
# VALID_CASES — diagnostic must compute without error; result must be all-finite
# Each entry lists all fixture keys for which the diagnostic should be tested.
# Multi-key entries exercise different model-dispatch paths.
#
VALID_CASES = [
    # ---------------------------------------------------------------------------
    # conservation_diagnostics.jl
    # ---------------------------------------------------------------------------
    case("massa",   :dry),
    case("energya", :dry),
    case("watera",  :m0),             # MoistMicrophysics only
    case("energyo", :sphere),         # SlabOceanSST + SpectralElementSpace2D
    case("watero",  :m0_slab_sphere), # MoistMicrophysics + SlabOceanSST + sphere

    # ---------------------------------------------------------------------------
    # core_diagnostics.jl
    # ---------------------------------------------------------------------------
    # no dispatch (universal)
    cases((
        "rhoa", "ua", "va", "wa", "ta", "thetaa", "ha", "pfull", "zg",
        "bgrad", "strain", "cl", "ke", "ts", "tas", "uas", "vas", "tauu", "tauv",
        "hfes", "dsevi", "env_q_tot_variance", "env_temperature_variance",
        "env_q_tot_temperature_covariance", "env_q_tot_temperature_correlation",
    ), :dry)...,
    # sphere-only (DSS / hypsography)
    cases(("rv", "orog"), :sphere)...,
    # MoistMicrophysics, single path
    cases((
        "hus", "hur", "husv", "hussfc", "evspsbl", "hfls",
        "clivi", "clvi", "prw", "hurvi", "cape", "mslp"
    ), :m0)...,
    # Union{DryModel, MoistMicrophysics}: single method
    cases(("pr", "prra", "prsn"), :dry)...,
    # EquilibriumMicrophysics0M (precomputed cache), NonEquilibriumMicrophysics (state)
    cases(("clw", "cli", "clwvi", "lwp"), (:m0, :m1))...,
    # DryModel, MoistMicrophysics (different flux computation)
    case("hfss",  (:dry, :m0)),
    # Non-EDMF (Smagorinsky formula), EDMF (mixing-length closure)
    case("lmix",  (:dry, :m0_pedmfx)),
    # 1M / 2M microphysics
    cases(("husra", "hussn", "rwp"), :m1)...,  # Union{1M, 2M}, single method
    cases(("cdnc", "ncra"), :m2)...,  # 2M only
    # Smagorinsky-Lilly
    cases(("Dh_smag", "Dv_smag", "strainh_smag", "strainv_smag"), :smag)...,
    # steady-state velocity
    cases(("uapredicted", "vapredicted", "wapredicted", "uaerror", "vaerror", "waerror"), :ssv)...,

    # ---------------------------------------------------------------------------
    # gravitywave_diagnostics.jl
    # ---------------------------------------------------------------------------
    cases(("utendnogw", "vtendnogw"), :nogw)...,

    # ---------------------------------------------------------------------------
    # radiation_diagnostics.jl
    # All paths covered by :allsky (AllSkyRadiationWithClearSkyDiagnostics +
    # aerosol_radiation = true), the most featureful radiation mode.
    # ---------------------------------------------------------------------------
    cases(("rsd", "rsdt", "rsds", "rsu", "rsut", "rsus"), :allsky)...,
    cases(("rld", "rlds", "rlu", "rlut", "rlus"), :allsky)...,
    cases(("rsdcs", "rsdscs", "rsucs", "rsutcs"), :allsky)...,
    cases(("rsuscs", "rldcs", "rldscs", "rlucs", "rlutcs"), :allsky)...,
    cases(("reffclw", "reffcli", "od550aer", "odsc550aer"), :allsky)...,

    # ---------------------
    # edmfx_diagnostics.jl
    # ---------------------
    # Union{PrognosticEDMFX, DiagnosticEDMFX}: single method
    cases(("rhoaup", "waup", "taup", "thetaaup", "haup", "husup"), :m0_pedmfx)...,
    cases(("hurup", "entr", "turbentr", "detr", "waen", "tke", "lmixw", "lmixtke", "lmixb"), :m0_pedmfx)...,
    # PrognosticEDMFX only
    cases(("rhoaen", "taen", "thetaaen", "haen", "husen", "huren"), :m0_pedmfx)...,
    # PrognosticEDMFX, DiagnosticEDMFX
    cases(("arup", "aren"), (:m0_pedmfx, :m1_dedmfx))...,
    # 0M+Union{P,D}, NonEq+PrognosticEDMFX, NonEq+DiagnosticEDMFX
    cases(("clwup", "cliup"), (:m0_pedmfx, :m1_dedmfx, :m1_pedmfx))...,
    # Union{1M,2M}+PrognosticEDMFX, 1M+DiagnosticEDMFX
    cases(("husraup", "hussnup"), (:m1_pedmfx, :m1_dedmfx))...,
    # 0M+PrognosticEDMFX, NonEq+PrognosticEDMFX
    cases(("clwen", "clien"), (:m0_pedmfx, :m1_pedmfx))...,
    # 1M+PrognosticEDMFX
    cases(("husraen", "hussnen"), :m1_pedmfx)...,
    # 2M + PrognosticEDMFX
    cases(("cdncup", "cdncen", "ncraup", "ncraen"), :m2_pedmfx)...,
    # VerticalDiffusion, DecayWithHeightDiffusion, EDMF
    cases(("edt", "evu"), (:vd, :dwh, :m0_pedmfx))...,
]
#! format: on
#
# SKIP_CASES — out-of-scope diagnostic files:
#   - tracer_diagnostics.jl  : require tracer model (o3, mmr*, loadss)
#   - negative_scalars_diagnostics.jl : budget monitoring (hus_neg_sum, etc.)
# These are skipped here; their own test files should cover them.
#
SKIP_CASES = Set([
    # tracer_diagnostics.jl
    "loadss", "mmrbcpi", "mmrbcpo", "mmrdust", "mmrocpi", "mmrocpo",
    "mmrso4", "mmrss", "o3",
    # negative_scalars_diagnostics.jl
    "cli_max", "cli_min", "cli_neg_frac", "cli_neg_mean", "cli_neg_sum",
    "cli_pos_frac", "cli_pos_mean", "cli_pos_sum",
    "clw_max", "clw_min", "clw_neg_frac", "clw_neg_mean", "clw_neg_sum",
    "clw_pos_frac", "clw_pos_mean", "clw_pos_sum",
    "hus_max", "hus_min", "hus_neg_frac", "hus_neg_mean", "hus_neg_sum",
    "hus_pos_frac", "hus_pos_mean", "hus_pos_sum",
    "husra_max", "husra_min", "husra_neg_frac", "husra_neg_mean", "husra_neg_sum",
    "husra_pos_frac", "husra_pos_mean", "husra_pos_sum",
    "hussn_max", "hussn_min", "hussn_neg_frac", "hussn_neg_mean", "hussn_neg_sum",
    "hussn_pos_frac", "hussn_pos_mean", "hussn_pos_sum",
])

# ---------------------------------------------------------------------------
# Exhaustiveness check
# — every name in ALL_DIAGNOSTICS must appear in exactly one table
# ---------------------------------------------------------------------------

@testset "All diagnostic variables are accounted for" begin
    all_names = Set(keys(CA.Diagnostics.ALL_DIAGNOSTICS))
    covered = Set(c.name for c in VALID_CASES) ∪ SKIP_CASES
    missing_ = setdiff(all_names, covered)
    unexpected_ = setdiff(covered, all_names)
    @test isempty(missing_)    # new diagnostic added without a test entry
    @test isempty(unexpected_) # stale entry (diagnostic was removed)
end

# ---------------------------------------------------------------------------
# Main test loop
# Each (name, key) pair becomes one @testset — key identifies the fixture.
# ---------------------------------------------------------------------------

@testset "$(c.name) [$(key)]" for c in VALID_CASES, key in c.keys
    (Y, p) = states[key]
    diag = getdiag(c.name)
    result = compute_diag(diag, Y, p)
    @test all(isfinite, field_data(result))
end

# ---------------------------------------------------------------------------
# Spot-checks for model-dispatch error paths
# (not subject to exhaustiveness — these are additional regression guards)
# ---------------------------------------------------------------------------

@testset "Model dispatch error paths" begin
    for name in ("hus", "hur", "husv", "clw", "cli", "hussfc",
        "evspsbl", "hfls", "clwvi", "lwp", "clivi", "clvi",
        "prw", "hurvi", "husra", "hussn", "rwp",
        "cdnc", "ncra", "utendnogw", "vtendnogw")
        @testset "$name errors on dry model" begin
            @test_throws Exception compute_diag(getdiag(name), Y_dry, p_dry)
        end
    end

    @testset "orog errors on column grid (no hypsography field)" begin
        @test_throws Exception compute_diag(getdiag("orog"), Y_dry, p_dry)
    end
end

# ---------------------------------------------------------------------------
# Pure-Julia helper: geometric_scaling (radiation_diagnostics.jl)
# Not a DiagnosticVariable, so not in ALL_DIAGNOSTICS.
# ---------------------------------------------------------------------------

@testset "geometric_scaling" begin
    gs = CA.Diagnostics.geometric_scaling
    R = FT(CAP.planet_radius(params))
    @test gs(FT(0), R) ≈ FT(1)   # scaling = 1 at surface
    @test gs(R, R) ≈ FT(4)   # (2R/R)² = 4
    @test gs(FT(10_000), R) > FT(1)
    @test isfinite(gs(FT(30_000), R))
end
