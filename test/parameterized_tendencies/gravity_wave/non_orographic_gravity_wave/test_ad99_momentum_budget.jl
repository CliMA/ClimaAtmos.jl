# ============================================================================
# AD99 non-orographic gravity wave — column momentum budget tests
#
# The scheme launches a spectrum at `source_level`, phase-speed bin `c[n]`
# carrying momentum flux `B0[n]`. Waves die by saturation, at a critical level,
# or force-broken at the model top (that flux is redistributed over the levels
# at and above `damp_level`). Nothing is meant to be thrown away, so
#
#     Σ_k ρ_k·Δz_k·a_k  =  F_launch  =  source_ampl · Σ_n B0[n] / Σ_n |B0[n]|
#
# The kernel hands the half-layer between centres k and k+1 a flux
# `ρ_source·ε·fm_k` with `ε = (source_ampl/ρ_source/nk) / Σ_n|B0[n]|`; every
# wave deposits exactly once (Σ_k fm_k = Σ_n B0[n]) and `nk` cancels, so
# `F_launch` is independent of the density profile, the wind profile, the grid,
# and the number of phase-speed bins. Three ways it can fail, one per test:
#
# 1. Interior. The kernel divides by the mass of the layer BETWEEN two cell
#    centres, sqrt(ρ_k·ρ_k+1)·(z_k+1 - z_k), and `gw_average!` splits each such
#    layer's drag half into the cell below and half into the cell above. The
#    residual is the arithmetic/geometric mean ratio of neighbouring cell
#    masses, exactly cosh(Δz/2H) for a uniform grid with ρ ∝ exp(-z/H).
#
# 2. Escaped flux. Nothing saturates and nothing meets a critical level, so all
#    of the launched flux is force-broken at the top and the whole deposit comes
#    from `gw_deposit`, which rescales by `m_top / M_R`: the mass of the ghost
#    layer the kernel divided by, over the mass of the receiving levels.
#
# 3. Source level. The kernel used to start its break test one level BELOW the
#    launch level while guarding the `fm` accumulation with
#    `level >= source_level`, so a wave already saturated at its own launch
#    level was dropped with its flux discarded; since `ε` normalises by the full
#    Σ|B0|, the scheme then launched less than the nominal `Bt_0`. Inherited
#    from MiMA `cg_drag.f90`, and worth a few per cent of the net launched
#    momentum in the extratropics. The break test now starts AT the launch level.
#
# Each test runs on a single column and on a cubed sphere whose `damp_level`
# varies by column — the arrangement production sphere runs use, and the only
# one that exercises `damp_level` and `M_R` as per-column fields.
#
# NOT tested: `non_orographic_gravity_wave_apply_tendency!` clips each component
# to ±3e-3 m/s² before adding it to the tendency. These tests read
# `uforcing`/`vforcing` directly, so they check the scheme's own budget, not the
# momentum the model ultimately receives.
#
# No `get_simulation`: the forcing only reads `p.non_orographic_gravity_wave`
# and a few `p.scratch` fields, so the test builds the grid with the solver's
# own helpers, calls the real cache constructor, and assembles a minimal `p`.
# The scratch names below must stay in step with
# `src/cache/temporary_quantities.jl`.
#
# The phase-speed grid is deliberately coarse (21 bins, against 251 at the
# ClimaParams defaults): the kernel unrolls a reduction over the bins, so bin
# count dominates compile cost, and the identity above is exactly independent of
# it. Every other parameter is the ClimaParams default.
#
# Run with (CPU):
#   CLIMACOMMS_DEVICE=CPU julia +1.11 --project=.buildkite \
#     test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/test_ad99_momentum_budget.jl
# ============================================================================

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

const FT = Float64

comms_ctx = ClimaComms.SingletonCommsContext()

# 50 uniform 1 km levels, shared by the column and the sphere so that the closed
# form cosh(Δz/2H) is the same number in both.
const Z_MAX = FT(50000)
const Z_ELEM = 50

# Imposed density profile, rather than one taken from a model state, so that the
# expected second-order interior error is known exactly.
const H_ρ = FT(7000)
const ρ0 = FT(1.2)

# Constant buoyancy frequency, large enough that no wave is removed by total
# internal reflection. Reflection (like an exactly vanishing intrinsic phase
# speed) discards a wave's flux instead of depositing it, and would break the
# budget for reasons unrelated to the mass weighting. It needs
# |c - u| ≳ ω_r/k ≈ 13400·N m/s for the 300 km band, so N = 0.03 puts the
# threshold near 400 m/s, well above the largest |c - u| below.
const N_const = FT(0.03)

# Launch level index: z = 14.5 km, nearest the 15 km ClimaParams
# `nogw_source_height`.
const SOURCE_LEVEL = 15

# Production `NonOrographicGravityWave` at ClimaParams defaults, except for the
# coarse phase-speed grid. `beres_source = nothing`, so the budget below is the
# AD99 budget alone.
build_gw() = CA.NonOrographicGravityWave{FT, Nothing}(;
    source_pressure = FT(31500),
    damp_pressure = FT(85),
    source_height = FT(15000),
    Bw = FT(0.4),
    Bn = FT(0),
    dc = FT(4),
    cmax = FT(40),
    c0 = FT(0),
    nk = FT(1),
    cw = FT(35),
    cw_tropics = FT(35),
    cn = FT(2),
    Bt_0 = FT(0.0043),
    Bt_n = FT(0),
    Bt_s = FT(0),
    Bt_eq = FT(0.0043),
    ϕ0_n = FT(15),
    ϕ0_s = FT(-15),
    dϕ_n = FT(10),
    dϕ_s = FT(-10),
    beres_source = nothing,
)

# The vertical profile of `field` in its first column, as a plain `Vector`.
# Handles both the `VF` layout of a column and the `VIJFH` layout of an
# extruded space.
function first_column(field)
    A = Array(parent(field))
    return A[:, ntuple(_ -> 1, ndims(A) - 1)...]
end

# The per-column values of a level field, flattened. Two level fields on the
# same space flatten in the same order, so the vectors line up elementwise. The
# `copy` matters: on CPU `Array(parent(f))` returns the field's own storage.
per_column(level_field) = copy(vec(Array(parent(level_field))))

# Everything `non_orographic_gravity_wave_forcing` needs for `kind` (`:column`
# or `:sphere`), with no simulation: the cache, a minimal `p`, the imposed-state
# scratch fields, and the grid geometry as plain arrays. The sphere is
# deliberately tiny (2 elements per panel edge, 96 columns) and
# shallow-atmosphere, so the measure `column_integral_definite!` uses is the
# plain mesh spacing.
function build_state(kind::Symbol)
    grid = if kind == :column
        CA.ColumnGrid(
            FT;
            context = comms_ctx,
            z_elem = Z_ELEM,
            z_max = Z_MAX,
            z_stretch = false,
        )
    elseif kind == :sphere
        CA.SphereGrid(
            FT;
            context = comms_ctx,
            z_elem = Z_ELEM,
            z_max = Z_MAX,
            z_stretch = false,
            h_elem = 2,
            deep_atmosphere = false,
        )
    else
        error("unknown grid kind $kind")
    end
    (; center_space, face_space) = CA.get_spaces(grid)

    # A minimal `Y`: the cache constructor only touches `Y.c.ρ`, `axes(Y.c)`,
    # and (on the sphere) the level-1 local geometry.
    ᶜY = Fields.Field(NamedTuple{(:ρ,), Tuple{FT}}, center_space)
    fill!(parent(ᶜY), FT(1))
    Y = (; c = ᶜY)

    # `issphere` reaches for a horizontal space, which a column does not have.
    @assert (kind == :column) == CA.iscolumn(center_space)
    kind == :sphere && @assert CA.issphere(center_space)

    gwc = CA.non_orographic_gravity_wave_cache(Y, build_gw())

    ᶜρ_t = Fields.Field(FT, center_space)
    # These names must match `temporary_quantities`; the forcing routine reads
    # exactly these, so a rename there fails here on field access.
    scratch = (;
        ᶜtemp_scalar = similar(ᶜρ_t),      # ᶜρ_p1
        ᶜtemp_scalar_2 = similar(ᶜρ_t),    # ᶜz_p1
        ᶜtemp_scalar_3 = similar(ᶜρ_t),    # ᶜu_p1
        ᶜtemp_scalar_4 = similar(ᶜρ_t),    # ᶜv_p1
        ᶜtemp_scalar_5 = similar(ᶜρ_t),    # ᶜbf_p1
        ᶜtemp_scalar_6 = similar(ᶜρ_t),    # gw_average! scratch
        temp_field_level_2 = similar(gwc.damp_level), # gw_deposit_scale
    )
    p = (; non_orographic_gravity_wave = gwc, scratch)

    # Materialized rather than used as the `.z` property view, so that `parent`
    # sees a plain scalar array instead of the full coordinate struct.
    ᶜz = Fields.Field(FT, center_space)
    ᶠz = Fields.Field(FT, face_space)
    ᶜz .= Fields.coordinate_field(center_space).z
    ᶠz .= Fields.coordinate_field(face_space).z

    center_z = first_column(ᶜz)
    face_z = first_column(ᶠz)
    nlev = Spaces.nlevels(center_space)
    @assert length(center_z) == nlev
    @assert length(face_z) == nlev + 1

    return (;
        kind,
        center_space,
        gwc,
        p,
        ᶜz,
        ᶜρ_t,
        ᶜu_t = similar(ᶜρ_t),
        ᶜv_t = similar(ᶜρ_t),
        ᶜbf_t = similar(ᶜρ_t),
        ᶜmom = similar(ᶜρ_t),
        deposited_field = similar(gwc.damp_level),
        source_level = similar(gwc.damp_level),
        ρ_source = similar(gwc.damp_level),
        u_source = similar(gwc.damp_level),
        v_source = similar(gwc.damp_level),
        nlev,
        center_z,
        Δz_cell = diff(face_z),
        ρ_profile = ρ0 .* exp.(-center_z ./ H_ρ),
    )
end

# ----------------------------------------------------------------------------
# Spectrum helpers. These reimplement the launch algebra in plain Julia, so the
# targets below are independent of the kernel being tested.
# ----------------------------------------------------------------------------

const GW = build_gw()
const GW_C = ntuple(
    n -> FT((n - 1) * GW.dc - GW.cmax),
    Int(floor(FT(2 * GW.cmax / GW.dc + 1))),
)
const GW_NCVAL = Val(length(GW_C))

# Bn = 0 (a single Gaussian) and flag = 1 (ground-relative phase speeds), the
# ClimaParams defaults, matching what `run_case!` fills into the cache.
spectrum(u_source, Bw) = CA.wave_source(
    GW_C,
    FT(u_source),
    FT(Bw),
    FT(0),
    GW.cw,
    GW.cn,
    GW.c0,
    FT(1),
    GW_NCVAL,
)

# Exact launched momentum flux (Pa), `source_ampl · ΣB0 / Σ|B0|`, and the
# spectrum asymmetry `|ΣB0| / Σ|B0|`. The asymmetry must be well away from zero
# for the budget test to have power: with a symmetric spectrum and a symmetric
# atmosphere, each eastward wave and its westward twin break at the same height
# carrying opposite flux, so a mass-weighting error cancels exactly in the
# signed budget and a broken scheme would still pass.
function launched_flux(u_source, Bw)
    B0 = spectrum(u_source, Bw)
    s_signed = sum(B0)
    s_abs = sum(abs, B0)
    return (
        F_launch = GW.Bt_0 * s_signed / s_abs,
        asymmetry = abs(s_signed) / s_abs,
    )
end

# Signed momentum flux (Pa) the OLD kernel removed at `source_level - 1` without
# depositing it, in closed form. That level's saturation threshold is exactly
# `fac = 0.5·kwv/N`: the kernel forms `fac = 0.5·(ρ_kp1/ρ_source)·kwv/bf_kp1`,
# and there the shifted `ρ_kp1` is the density at `source_level`, so the ratio
# is one. A bin was deleted when `B0[n]/(c[n] - u_source)³ ≥ fac`; the
# critical-level test cannot fire there (`c_hat` equals `c_hat0`, so their
# product is a square) and reflection needs |c - u| near 400 m/s, so saturation
# was the only path. `nk == 1`, so `kwv` is a single value.
function source_level_deleted_flux(u_source, Bw)
    B0 = spectrum(u_source, Bw)
    kwv = 2 * FT(π) / (FT(30) * 10 * FT(1000))
    fac = FT(0.5) * kwv / N_const
    deleted = FT(0)
    for n in eachindex(GW_C)
        c_hat = GW_C[n] - FT(u_source)
        c_hat == 0 && continue
        if B0[n] / c_hat^3 - fac >= 0
            deleted += B0[n]
        end
    end
    return GW.Bt_0 * deleted / sum(abs, B0)
end

# ----------------------------------------------------------------------------
# Wind profiles
# ----------------------------------------------------------------------------

# The source wind sits exactly on a phase-speed bin. A bin a hair away from the
# local wind has a huge B0/(c-u)³ and saturates instantly; landing exactly on a
# bin gives that bin B0 = 0, because the spectrum carries a sign(c - u_source).
const U_SOURCE = FT(24)
@assert any(==(U_SOURCE), GW_C) "the launch wind must sit exactly on a phase-speed bin"

# Rises from the source value to +48.3 m/s at 30.5 km, then falls to -49.7 m/s
# at the model top. Both endpoints lie outside [-cmax, cmax] and neither
# coincides with a bin, so every phase-speed bin meets a critical level (u = c)
# strictly inside the domain: eastward bins on the rising branch, westward bins
# on the falling branch. Nothing reaches the model top.
const Z_MID = FT(30500)
const U_HI = FT(48.3)
const U_LO = FT(-49.7)

reversing_wind(z, z_src, z_top) = ifelse(
    z <= z_src,
    U_SOURCE,
    ifelse(
        z <= Z_MID,
        U_SOURCE + (U_HI - U_SOURCE) * (z - z_src) / (Z_MID - z_src),
        U_HI + (U_LO - U_HI) * (z - Z_MID) / (z_top - Z_MID),
    ),
)

# The winds the kernel sees above the launch level, including the ghost level it
# extrapolates above the model top. An exact coincidence with a phase-speed bin
# sends that bin down the `c_hat == 0` branch, which discards its flux.
function winds_above_source(st)
    z_src = st.center_z[SOURCE_LEVEL]
    z_top = st.center_z[end]
    u = map(z -> reversing_wind(z, z_src, z_top), st.center_z)
    return vcat(u[(SOURCE_LEVEL + 1):end], 2 * u[end] - u[end - 1])
end

# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

# `spec` is either an integer, giving every column the same `damp_level`, or
# `Val(:varying)`, which spreads `damp_level` over `nlev-4:nlev` across the
# sphere's columns so that the receiving mass `M_R` differs by column.
function set_damp_level!(st, spec::Int)
    fill!(st.gwc.damp_level, FT(spec))
    return nothing
end

function set_damp_level!(st, ::Val{:varying})
    lat = Fields.local_geometry_field(st.gwc.damp_level).coordinates.lat
    @. st.gwc.damp_level = FT(st.nlev - 4) + floor(mod(abs(lat), FT(5)))
    return nothing
end

# Mass of the ghost layer the kernel divided the escaped flux by: it
# extrapolates ρ_ghost = ρ_end²/ρ_end-1 above the top centre.
function ghost_layer_mass(st)
    ρ_ghost = st.ρ_profile[end]^2 / st.ρ_profile[end - 1]
    return sqrt(st.ρ_profile[end] * ρ_ghost) *
           (st.center_z[end] - st.center_z[end - 1])
end

# Impose the state, run the AD99 forcing once, and return the momentum budget
# per column. `deposited` integrates ρ·a with the same operator the scheme uses
# for `M_R`, i.e. the momentum the model receives (before the ±3e-3 clip).
# `escaped` reconstructs the flux force-broken at the top from
# `u_waveforcing_top`, undoing the kernel's conversion with the same ghost-layer
# mass; valid because `nk == 1`, so the single band's value is the one left in
# the cache field.
function run_case!(st; Bw, wind::Symbol, damp_spec)
    (; gwc, p, ᶜz, ᶜρ_t, ᶜu_t, ᶜv_t, ᶜbf_t, ᶜmom, deposited_field) = st
    @assert gwc.gw_nk == 1 "the `escaped` diagnostic assumes a single wave band"

    # Uniform spectrum parameters in every column: the only per-column variation
    # this test wants is `damp_level`. On the sphere `gw_flag` is 0 inside ±10°
    # latitude, which centres the spectrum on the launch wind and makes ΣB0 (and
    # so F_launch) vanish by symmetry, leaving the budget ratio 0/0 there.
    # Forcing flag = 1 everywhere keeps every column non-degenerate.
    fill!(gwc.gw_Bw, FT(Bw))
    fill!(gwc.gw_Bn, FT(0))
    fill!(gwc.gw_cw, GW.cw)
    fill!(gwc.gw_cn, GW.cn)
    fill!(gwc.gw_flag, FT(1))
    fill!(gwc.gw_source_ampl, GW.Bt_0)

    z_src = st.center_z[SOURCE_LEVEL]
    z_top = st.center_z[end]
    @. ᶜρ_t = ρ0 * exp(-(ᶜz / H_ρ))
    if wind == :reversing
        @. ᶜu_t = reversing_wind(ᶜz, z_src, z_top)
    elseif wind == :constant
        fill!(ᶜu_t, U_SOURCE)
    else
        error("unknown wind $wind")
    end
    fill!(ᶜv_t, FT(0))
    fill!(ᶜbf_t, N_const)

    # Launch-level state as level fields, matching what `compute_tendency!`
    # hands the forcing on the sphere.
    fill!(st.source_level, FT(SOURCE_LEVEL))
    for (dst, src) in
        ((st.ρ_source, ᶜρ_t), (st.u_source, ᶜu_t), (st.v_source, ᶜv_t))
        Fields.field_values(dst) .=
            Fields.field_values(Fields.level(src, SOURCE_LEVEL))
    end
    set_damp_level!(st, damp_spec)

    fill!(gwc.uforcing, FT(0))
    fill!(gwc.vforcing, FT(0))

    CA.non_orographic_gravity_wave_forcing(
        ᶜu_t,
        ᶜv_t,
        ᶜbf_t,
        ᶜρ_t,
        ᶜz,
        gwc.ᶜlevel,
        st.source_level,
        gwc.damp_level,
        st.ρ_source,
        st.u_source,
        st.v_source,
        gwc.uforcing,
        gwc.vforcing,
        gwc.gw_ncval,
        gwc.u_waveforcing,
        gwc.v_waveforcing,
        p,
    )

    @. ᶜmom = ᶜρ_t * gwc.uforcing
    Operators.column_integral_definite!(deposited_field, ᶜmom)
    deposited_u = per_column(deposited_field)

    @. ᶜmom = ᶜρ_t * gwc.vforcing
    Operators.column_integral_definite!(deposited_field, ᶜmom)
    deposited_v = per_column(deposited_field)

    return (;
        deposited_u,
        deposited_v,
        escaped = per_column(gwc.u_waveforcing_top) .* ghost_layer_mass(st),
        damp_level = round.(Int, per_column(gwc.damp_level)),
    )
end

# ----------------------------------------------------------------------------
# Expected values
# ----------------------------------------------------------------------------

const COLUMN_STATE = build_state(:column)
const SPHERE_STATE = build_state(:sphere)
const NLEV = COLUMN_STATE.nlev

# Arithmetic/geometric mean ratio of two neighbouring cell masses, which on a
# uniform grid with ρ ∝ exp(-z/H) is cosh(Δz/2H).
const INTERIOR_EXPECTED = cosh(COLUMN_STATE.Δz_cell[end] / (2 * H_ρ))

# Closure tolerance for the interior tests, and the largest escaped fraction
# consistent with it. The escaped flux deposits at ratio 1 while the interior
# deposits at INTERIOR_EXPECTED, so an escaped fraction f moves the measured
# ratio by f·(INTERIOR_EXPECTED - 1)/INTERIOR_EXPECTED. Deriving the setup guard
# from the assertion tolerance keeps the two from disagreeing.
const RTOL_CLOSURE = 1e-4
const MAX_ESCAPED_FRAC =
    FT(0.1) * RTOL_CLOSURE * INTERIOR_EXPECTED / (INTERIOR_EXPECTED - 1)

# ============================================================================
# Test 1 — interior budget: nothing escapes, nothing is deleted at the source
#
# `Bw = 1e-4` puts every bin's B0/(c - u_source)³ below the source-level
# saturation threshold, so test 3's deletion is exactly zero here, while the
# critical levels still remove every wave before the model top. What is left is
# the kernel/`gw_average!` pair, whose residual is cosh(Δz/2H).
#
# `Bw` only sets WHERE waves break: `ε` renormalises by Σ|B0|, so the launched
# total is `Bt_0` regardless. That is what makes this isolation possible.
# ============================================================================
@testset "AD99 column momentum budget — interior only" begin
    Bw = FT(1e-4)
    lf = launched_flux(U_SOURCE, Bw)

    @test lf.asymmetry > 0.1        # setup guard: spectrum is not degenerate
    @test source_level_deleted_flux(U_SOURCE, Bw) == 0  # test 3's effect absent

    for (name, st, damp_spec) in (
        ("column", COLUMN_STATE, NLEV),
        ("sphere, per-column damp_level", SPHERE_STATE, Val(:varying)),
    )
        # Setup guard: no wind above the source coincides with a bin.
        @test !any(u -> any(==(u), GW_C), winds_above_source(st))

        res = run_case!(st; Bw, wind = :reversing, damp_spec)
        ratios = res.deposited_u ./ lf.F_launch
        escaped_frac = maximum(abs, res.escaped ./ lf.F_launch)

        @info "Test 1 ($name): deposited / F_launch = $(round(ratios[1]; digits = 6)), cosh(Δz/2H) = $(round(INTERIOR_EXPECTED; digits = 6)), escaped fraction = $(round(escaped_frac; sigdigits = 3))"

        # Setup guard: the redistribution must not be what this test measures.
        @test escaped_frac < MAX_ESCAPED_FRAC

        # Asserting against the closed form is far sharper than closure to 1 %.
        @test all(isapprox.(ratios, INTERIOR_EXPECTED; rtol = RTOL_CLOSURE))

        # The physics requirement: second order in Δz/H is acceptable, first
        # order is not.
        @test all(isapprox.(ratios, 1; atol = 0.01))

        # The meridional spectrum is exactly antisymmetric (v_source = 0, bins
        # symmetric about zero), so it must deposit no net momentum. Catches
        # sign and index asymmetries in the kernel.
        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end

# ============================================================================
# Test 2 — escaped-flux budget: everything reaches the model top
#
# `Bw = 1e-10` puts B0/(c-u)³ orders of magnitude below the saturation threshold
# everywhere, so no wave breaks, and the wind is constant, so none meets a
# critical level. All of the launched flux is force-broken at the top and the
# whole deposit comes from `gw_deposit`. The launched total is unchanged (`ε`
# renormalises by Σ|B0|), so the target is the same `F_launch` as in test 1.
#
# The column runs two arrangements, which constrain the divisor differently and
# in which the old error changed SIGN: `damp_level = nlev` (the production
# column setting, one receiving level) and `damp_level = nlev - 3` (four). The
# sphere runs both at once, one per column.
# ============================================================================
@testset "AD99 column momentum budget — escaped flux redistribution" begin
    Bw = FT(1e-10)
    lf = launched_flux(U_SOURCE, Bw)

    @test lf.asymmetry > 0.1        # setup guard

    for (name, st, damp_spec) in (
        ("column, damp_level = $NLEV", COLUMN_STATE, NLEV),
        ("column, damp_level = $(NLEV - 3)", COLUMN_STATE, NLEV - 3),
        ("sphere, per-column damp_level", SPHERE_STATE, Val(:varying)),
    )
        res = run_case!(st; Bw, wind = :constant, damp_spec)
        ratios = res.deposited_u ./ lf.F_launch
        escaped_frac = minimum(abs, res.escaped ./ lf.F_launch)

        @info "Test 2 ($name): deposited / F_launch = $(round(ratios[1]; digits = 6)), escaped fraction = $(round(escaped_frac; digits = 4)), receiving levels = $(sort(unique(NLEV .- res.damp_level .+ 1)))"

        # Setup guard: only meaningful if essentially all of the launched flux
        # really did reach the model top.
        @test escaped_frac > 0.99

        # The requirement: the redistribution deposits exactly the flux it was
        # handed, in every column.
        @test all(isapprox.(ratios, 1; atol = 1e-3))

        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end

# ============================================================================
# Test 3 — the production launch amplitude, where waves saturate at launch
#
# Identical to test 1 but with `nogw_Bw = 0.4`, the ClimaParams default, which
# puts the bins nearest the source wind above their own saturation limit at the
# launch level. Those are the waves the kernel used to delete at
# `source_level - 1` while skipping the `fm` accumulation, so their flux went
# nowhere. They now break in the layer just above the launch level and deposit
# there, so the budget closes at cosh(Δz/2H) exactly as in test 1.
#
# `source_level_deleted_flux` is a regression guard, not a prediction: it
# computes what the OLD code would have thrown away here, and the test requires
# it to be large. That proves this case genuinely exercises the launch-level
# path, so if the guard in `waveforcing_column_accumulate!` ever slips back to
# `source_level - 1` this test fails rather than quietly duplicating test 1.
# ============================================================================
@testset "AD99 column momentum budget — saturation at the launch level" begin
    Bw = GW.Bw           # the ClimaParams default, i.e. what production runs use
    lf = launched_flux(U_SOURCE, Bw)
    would_have_been_deleted =
        source_level_deleted_flux(U_SOURCE, Bw) / lf.F_launch

    # Regression guard: this setup must actually have waves that saturate at the
    # launch level, or the test proves nothing.
    @test would_have_been_deleted > 0.01

    for (name, st, damp_spec) in (
        ("column", COLUMN_STATE, NLEV),
        ("sphere, per-column damp_level", SPHERE_STATE, Val(:varying)),
    )
        res = run_case!(st; Bw, wind = :reversing, damp_spec)
        ratios = res.deposited_u ./ lf.F_launch

        @info "Test 3 ($name): deposited / F_launch = $(round(ratios[1]; digits = 6)), cosh(Δz/2H) = $(round(INTERIOR_EXPECTED; digits = 6)), flux the old code would have lost = $(round(would_have_been_deleted; digits = 5)) of F_launch"

        # Setup guard: still nothing escaping the top.
        @test maximum(abs, res.escaped ./ lf.F_launch) < MAX_ESCAPED_FRAC

        @test all(isapprox.(ratios, INTERIOR_EXPECTED; rtol = RTOL_CLOSURE))
        @test all(isapprox.(ratios, 1; atol = 0.01))

        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end
