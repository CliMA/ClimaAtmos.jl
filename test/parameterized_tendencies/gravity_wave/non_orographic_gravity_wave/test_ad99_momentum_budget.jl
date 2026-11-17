# ============================================================================
# AD99 non-orographic gravity wave — column momentum budget tests
#
# What these tests check, in one sentence: every bit of momentum the scheme
# launches at the source level must show up somewhere in the column — none of
# it may be silently lost.
#
# How the scheme works: at `source_level` it launches a set of waves, one per
# phase-speed bin `c[n]`, each carrying momentum flux `B0[n]`. Each wave gives
# up its momentum in exactly one place: where it breaks (saturation), where it
# hits a critical level, or — if it reaches the model top without breaking —
# it is broken there by force and its momentum is spread over the levels at
# and above `damp_level`. The one exception is a wave removed by total internal
# reflection, whose momentum is discarded; the setup below keeps every wave far
# from reflection. Because every other wave deposits exactly once, the
# column-integrated drag equals the launched flux,
#
#     Σ_k ρ_k·Δz_k·a_k  =  F_launch  =  source_ampl · Σ_n B0[n] / Σ_n |B0[n]|,
#
# to round-off, where the column integral is the model's own (`ρ·J/ΔA` per
# cell, so on a deep-atmosphere grid it includes the growth of each cell's area
# with height). Tests 1-3 run on a uniform grid, on a stretched grid, and on a
# deep-atmosphere sphere. There are four independent ways the code could break
# this budget, so there are four tests:
#
# 1. Interior deposition. When a wave breaks between two cell centres, the
#    kernel converts its flux into a drag, and `postprocess_and_accumulate!`
#    divides it by the mean mass of the two cells and gives half to each, so
#    the two cells together receive exactly the layer's flux. Dividing by any
#    other mass (the kernel's own `√(ρ_k ρ_{k+1})·Δz`, say) leaves a residual
#    of second order on uniform and stretched grids and, on a deep-atmosphere
#    grid, equal to the area growth from the surface (about 1.6% at 50 km),
#    which the round-off tolerance catches.
#
# 2. Escaped flux. The wind and stability are set so that no wave breaks on
#    the way up: everything reaches the top and goes through the forced-break
#    path in `gw_deposit`. That path must rescale by `m_top / M_R` (the mass
#    of the fictitious top layer the kernel divided by, over the total mass of
#    the levels that receive the momentum) for the budget to close exactly.
#
# 3. Source level. A wave can be unstable already at its own launch level. The
#    old code (as in MiMA's `cg_drag.f90`) ran the AD99 launch-level test
#    (Alexander & Dunkerton 1999, Eqs. 20-22): its loop step
#    `source_level - 1` tests the values at `source_level` and removes the
#    waves that fail without depositing them, a few per cent of the launched
#    momentum in the extratropics. The scheme now deliberately skips that test
#    and deposits these waves in the first layer above the launch level (see
#    the "Waves that are unstable at the launch level" section of the docs for
#    why); the test would catch a return to the old behaviour.
#
# 4. Exact critical level. When the wind at a level equals a wave's phase speed
#    exactly, `c - u = 0` and the kernel takes a separate branch (the
#    reflection test cannot be evaluated there). That branch used to remove the
#    wave without depositing it; it now deposits the flux like any other
#    critical level.
#
# 5. The Beres convective source. It launches a per-column spectrum from a
#    per-column level and deposits it with its coverage instead of the AD99
#    intermittency; its budget is a_cover times its launched flux, and the
#    full drag with both sources must be the sum of the two alone.
#
# The budget closes wherever the waves deposit, so tests 6-8 check where and
# whether they do: 6, the height where each wave breaks against the analytic
# height; 7, total internal reflection; 8, the AD99 source amplitude, frame and
# spectrum by latitude. Test 2 also passes `damp_level` as one level index, the
# form single-column callers use, and test 4 checks that every wave reaches its
# exact critical level before anything else removes it.
#
# Each of tests 1-4 runs twice: on a single column, and on a cubed sphere where
# `damp_level` differs from column to column. Only test 2 sends momentum through
# `damp_level` and `M_R`, so only there does the sphere check the per-column
# hand-back, the production setup; in tests 1, 3 and 4 every column is the same
# and the sphere run checks that the kernel works on the sphere's data layout.
#
# What is deliberately NOT tested: the model later clips each drag component
# to ±3e-3 m/s² in `non_orographic_gravity_wave_apply_tendency!`. These tests
# read `uforcing`/`vforcing` before that clipping, so they check the scheme's
# own consistency, not the momentum the model finally receives.
#
# Setup notes:
# - No `get_simulation`: the forcing only reads `p.non_orographic_gravity_wave`
#   and a few `p.scratch` fields, so the test builds the grid directly, calls
#   the real cache constructor, and assembles a minimal `p`.
# - The phase-speed grid is coarse on purpose (21 bins instead of the default
#   251): compile time grows with the bin count because the kernel unrolls a
#   loop over the bins, and the budget identity above does not depend on the
#   bin count anyway. Every other parameter is the ClimaParams default.
#
# Run with (CPU):
#   CLIMACOMMS_DEVICE=CPU julia --project=.buildkite \
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

# 50 levels over 50 km: uniform 1 km levels, or, when `z_stretch` is set,
# stretched by the two-scale rule from 100 m at the surface to 2.5 km at the top
# (a combination the rule admits for 50 levels in 50 km).
const Z_MAX = FT(50000)
const Z_ELEM = 50
const STRETCH = (; dz_bottom = FT(100), dz_top = FT(2500), z_pbl = FT(2200))

# Imposed density profile, rather than one taken from a model state, so that the
# test does not depend on a model run.
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

# Production `NonOrographicGravityWave` at ClimaParams defaults, read from the
# ClimaParams TOML (so the test cannot drift from production values) with the
# coarse phase-speed grid as the only override. `beres_source = nothing`, so
# the budget below is the AD99 budget alone.
function build_gw()
    gw_params = CA.NonOrographicGravityWaveParameters(
        FT,
        (; dc = FT(4), cmax = FT(40)),
    )
    P = typeof(gw_params)
    kwargs = NamedTuple{fieldnames(P)}(
        ntuple(i -> getfield(gw_params, i), fieldcount(P)),
    )
    return CA.NonOrographicGravityWave{FT, Nothing}(;
        kwargs...,
        beres_source = nothing,
    )
end

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
# deliberately tiny (2 elements per panel edge). `z_stretch` selects the stretched
# vertical grid and `deep_atmosphere` the deep-atmosphere sphere, whose cell areas
# grow with height.
function build_state(
    kind::Symbol;
    z_stretch = false,
    deep_atmosphere = false,
    gw = build_gw(),
)
    grid = if kind == :column
        CA.ColumnGrid(
            FT;
            context = comms_ctx,
            z_elem = Z_ELEM,
            z_max = Z_MAX,
            z_stretch,
            STRETCH...,
        )
    elseif kind == :sphere
        CA.SphereGrid(
            FT;
            context = comms_ctx,
            z_elem = Z_ELEM,
            z_max = Z_MAX,
            z_stretch,
            STRETCH...,
            h_elem = 2,
            deep_atmosphere,
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

    gwc = CA.non_orographic_gravity_wave_cache(Y, gw)

    ᶜρ_t = Fields.Field(FT, center_space)

    scratch = (;
        ᶜtemp_scalar = similar(ᶜρ_t),      # ᶜρ_p1
        ᶜtemp_scalar_2 = similar(ᶜρ_t),    # ᶜz_p1
        ᶜtemp_scalar_3 = similar(ᶜρ_t),    # ᶜu_p1
        ᶜtemp_scalar_4 = similar(ᶜρ_t),    # ᶜv_p1
        ᶜtemp_scalar_5 = similar(ᶜρ_t),    # ᶜbf_p1
        ᶜtemp_scalar_6 = similar(ᶜρ_t),    # ᶜlayer_ratio
        temp_field_level = similar(gwc.damp_level), # gw_deposit_scale
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
# Spectrum helpers. These redo the launch algebra in plain Julia, reusing only
# the kernel's spectrum function `CA.wave_source`, so the launched-flux target
# does not depend on how the kernel moves momentum.
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

# Signed momentum flux (Pa) of the waves that are too strong to be stable at
# the launch level itself, i.e. the waves the old code removed there without
# depositing them. Test 1 requires this flux to be exactly zero, and test 3
# requires it to be large. A wave with phase speed `c[n]` and amplitude `B0[n]`
# breaks (saturates) when `B0[n]/(c[n] - u_source)³ ≥ fac`. The kernel's
# threshold is `fac = 0.5·(ρ_kp1/ρ_source)·kwv/bf_kp1`; the old code's first
# step took `kp1` at the launch level, where this reduces to `0.5·kwv/N` (the
# test's buoyancy frequency is constant), which is what this helper uses. At
# that step the kernel's other two ways of removing a wave cannot trigger: a
# critical level requires `c - u` to have changed sign since launch, and
# reflection requires |c - u| near 400 m/s. So the helper gives the old code's
# loss exactly. It is a setup check, not a prediction of the new code: the new
# kernel's first step takes `kp1` one level higher, with a lower density and a
# different wind, so the waves it removes in the first layer can differ.
# `nk == 1`, so `kwv` is a single wavenumber.
function launch_level_saturated_flux(u_source, Bw)
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

# The source wind sits exactly on a phase-speed bin. A bin a bit away from the
# local wind has a huge B0/(c-u)³ and saturates instantly; landing exactly on a
# bin gives that bin B0 = 0, because the spectrum carries a sign(c - u_source).
const U_SOURCE = FT(24)
@assert any(==(U_SOURCE), GW_C) "the launch wind must sit exactly on a phase-speed bin"

# Rises from the source value to +48.3 m/s at 30.5 km, then falls to -49.7 m/s
# at the model top. A wave is absorbed at the first height where the wind
# catches up with its phase speed (its critical level, u = c). The endpoints
# bracket [-cmax, cmax], so the wind sweeps through every bin's phase speed
# inside the domain — eastward bins are absorbed on the rising branch,
# westward bins on the falling branch — and nothing reaches the model top.
# Neither endpoint sits exactly on a bin, which would send that bin down the
# separate `c_hat == 0` branch (see `winds_above_source` below); test 4 covers
# that branch on its own.
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

# Steps through the phase-speed bins exactly, one bin per level: up by `dc` from
# the source wind to +48 m/s, then down by `dc` to -48 m/s, then constant. Every
# bin with nonzero flux therefore meets a level where `c - u` is exactly zero,
# which sends it down the `c_hat == 0` branch of the kernel (test 4). Indexed by
# level so the hits are exact in floating point; the grid is uniform with
# centres at (k - 1/2)·Δz.
function on_bin_wind(z)
    j = round(Int, z / (Z_MAX / Z_ELEM) + FT(0.5)) - SOURCE_LEVEL
    step = FT(4)
    return j <= 0 ? U_SOURCE :
           j <= 6 ? U_SOURCE + step * j :
           j <= 30 ? U_SOURCE + step * 6 - step * (j - 6) : FT(-48)
end

# The winds the kernel sees above the launch level, including the ghost level it
# extrapolates above the model top. An exact coincidence with a phase-speed bin
# sends that bin down the `c_hat == 0` branch.
function winds_above_source(st; wind = :reversing)
    z_src = st.center_z[SOURCE_LEVEL]
    z_top = st.center_z[end]
    u = map(
        z -> wind == :reversing ? reversing_wind(z, z_src, z_top) : on_bin_wind(z),
        st.center_z,
    )
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

# Set up the idealized atmosphere, run the AD99 scheme once, and count where
# the launched momentum went. For each column:
# - `deposited` (Pa): the total momentum per second the air received. The
#   scheme outputs accelerations, so multiply by density and integrate over
#   the column to get back to flux units. If the scheme loses nothing, this
#   equals `F_launch`; that is every test's budget check. (Same integral
#   operator as the scheme's `M_R`; measured before the ±3e-3 m/s² clip
#   applied later in the tendency.)
# - `escaped` (Pa): how much flux reached the model top. Already included in
#   `deposited` (`gw_deposit` puts it back into the column); returned only to
#   check each test took the intended path — ≈0 in tests 1/3, ≈all in test 2.
#   Recovered from the ghost layer's stored acceleration by multiplying with
#   the ghost-layer mass (needs `nk == 1`, asserted below).
function run_case!(st; Bw, wind::Symbol, damp_spec, damp_index = false)
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
    elseif wind == :on_bins
        @. ᶜu_t = on_bin_wind(ᶜz)
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

    # `damp_index` passes `damp_level` as one level index instead of a level field.
    damp_level = damp_index ? damp_spec::Int : gwc.damp_level
    CA.non_orographic_gravity_wave_forcing(
        ᶜu_t,
        ᶜv_t,
        ᶜbf_t,
        ᶜρ_t,
        ᶜz,
        gwc.ᶜlevel,
        st.source_level,
        damp_level,
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

    # Drag profiles as a level-by-column matrix, for checks of where the drag goes.
    nlev = Spaces.nlevels(axes(gwc.uforcing))
    profile(f) = permutedims(reduce(hcat, (per_column(Fields.level(f, k)) for k in 1:nlev)))

    return (;
        deposited_u,
        deposited_v,
        forcing_u = profile(gwc.uforcing),
        forcing_v = profile(gwc.vforcing),
        # `postprocess_and_accumulate!` leaves the escaped flux, as an acceleration,
        # in the top level of `u_waveforcing`.
        escaped = per_column(Fields.level(gwc.u_waveforcing, st.nlev)) .*
                  ghost_layer_mass(st),
        damp_level = round.(Int, per_column(gwc.damp_level)),
    )
end

# ----------------------------------------------------------------------------
# Expected values
# ----------------------------------------------------------------------------

const COLUMN_STATE = build_state(:column)
const SPHERE_STATE = build_state(:sphere)
const STRETCHED_COLUMN_STATE = build_state(:column; z_stretch = true)
const DEEP_SPHERE_STATE = build_state(:sphere; deep_atmosphere = true)
const NLEV = COLUMN_STATE.nlev

# The budget closes to round-off on every grid.
const RTOL_CLOSURE = 1e-10

# Setup guard for the interior tests: the escaped-flux route must carry almost
# none of the launched flux, so that the test measures the interior route.
const MAX_ESCAPED_FRAC = 1e-3

# The grids tests 1-3 run on. The stretched grid has uneven cell masses even
# where the density changes slowly, and the deep sphere has cell areas that grow
# with height. Dividing by any mass other than the mean of the two cells' masses
# misses the budget on both.
const INTERIOR_CASES = (
    ("column", COLUMN_STATE, NLEV),
    ("stretched column", STRETCHED_COLUMN_STATE, NLEV),
    ("sphere, per-column damp_level", SPHERE_STATE, Val(:varying)),
    ("deep sphere, per-column damp_level", DEEP_SPHERE_STATE, Val(:varying)),
)

# ============================================================================
# Test 1 — interior budget: nothing escapes, nothing saturates at the launch
#
# The small amplitude (`Bw = 1e-4`) keeps every wave stable at the launch
# level (no test-3 breaking there), and the reversing wind absorbs every wave
# at a critical level before the model top (no test-2 escape). The only
# machinery left running is the interior path — the kernel putting drag on
# half-levels, then `postprocess_and_accumulate!` moving it to full levels — whose
# deposited/launched ratio must be 1 on every grid.
#
# Shrinking `Bw` does not shrink the launched total: the intermittency `ε`
# renormalises by Σ|B0|, so the launch is always `Bt_0`. `Bw` only moves
# WHERE waves break, which is what makes this isolation possible.
# ============================================================================
@testset "AD99 column momentum budget — interior only" begin
    Bw = FT(1e-4)
    lf = launched_flux(U_SOURCE, Bw)

    @test lf.asymmetry > 0.1        # setup guard: spectrum is not degenerate
    @test launch_level_saturated_flux(U_SOURCE, Bw) == 0  # test 3's effect absent

    for (name, st, damp_spec) in INTERIOR_CASES
        # Setup guard: no wind above the source coincides with a bin.
        @test !any(u -> any(==(u), GW_C), winds_above_source(st))

        res = run_case!(st; Bw, wind = :reversing, damp_spec)
        ratios = res.deposited_u ./ lf.F_launch
        escaped_frac = maximum(abs, res.escaped ./ lf.F_launch)

        @info "Test 1 ($name): deposited / F_launch = $(round(ratios[1]; digits = 12)), escaped fraction = $(round(escaped_frac; sigdigits = 3))"

        # Setup guard: the redistribution must not be what this test measures.
        @test escaped_frac < MAX_ESCAPED_FRAC

        @test all(isapprox.(ratios, 1; rtol = RTOL_CLOSURE))

        # The meridional spectrum is exactly antisymmetric (v_source = 0, bins
        # symmetric about zero), so it must deposit no net momentum. Catches
        # sign and index asymmetries in the kernel.
        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end

# ============================================================================
# Test 2 — escaped-flux budget: everything reaches the model top
#
# The waves are launched so weak (`Bw = 1e-10`) that none break on the way up,
# and the wind is constant with height, so none are absorbed at a critical
# level. Every wave therefore rides to the model top, where momentum is not
# allowed to leave the column: `gw_deposit` must hand all of it back to the top
# few layers. This test checks it hands back exactly what it was given. Making
# the waves weak does not change how much momentum is launched (the spectrum is
# renormalised by Σ|B0|), so the target is the same `F_launch` as in test 1.
#
# The check runs with two different numbers of receiving layers at the top
# (`damp_level = nlev` = one layer, the production column setting;
# `nlev - 3` = four), because a wrong divisor in the hand-back scales with that
# count and can even flip the sign of the error. The sphere runs both at once,
# one per column.
# ============================================================================
@testset "AD99 column momentum budget — escaped flux redistribution" begin
    Bw = FT(1e-10)
    lf = launched_flux(U_SOURCE, Bw)

    @test lf.asymmetry > 0.1        # setup guard

    for (name, st, damp_spec) in (
        ("column, damp_level = $NLEV", COLUMN_STATE, NLEV),
        ("column, damp_level = $(NLEV - 3)", COLUMN_STATE, NLEV - 3),
        ("stretched column, damp_level = $(NLEV - 3)", STRETCHED_COLUMN_STATE, NLEV - 3),
        ("sphere, per-column damp_level", SPHERE_STATE, Val(:varying)),
        ("deep sphere, per-column damp_level", DEEP_SPHERE_STATE, Val(:varying)),
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
        @test all(isapprox.(ratios, 1; rtol = RTOL_CLOSURE))

        # The hand-back is one uniform acceleration over the receiving levels,
        # as `gw_deposit` documents.
        uniform = all(eachindex(res.damp_level)) do col
            received = res.forcing_u[res.damp_level[col]:end, col]
            all(isapprox.(received, received[1]; rtol = 1e-6))
        end
        @test uniform

        # The sphere case checks the per-column hand-back only if the receiving
        # layer really differs between columns.
        startswith(name, "sphere") && @test length(unique(res.damp_level)) > 1

        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end

    # Single-column callers pass `damp_level` as one level index, not a level
    # field; that path must deposit exactly as the field does.
    for damp_spec in (NLEV, NLEV - 3)
        res_field = run_case!(COLUMN_STATE; Bw, wind = :constant, damp_spec)
        res_index =
            run_case!(COLUMN_STATE; Bw, wind = :constant, damp_spec, damp_index = true)
        @test res_index.forcing_u == res_field.forcing_u
        @test res_index.forcing_v == res_field.forcing_v
        @test all(isapprox.(res_index.deposited_u ./ lf.F_launch, 1; rtol = RTOL_CLOSURE))
    end
end

# ============================================================================
# Test 3 — the production launch amplitude, where waves saturate at launch
#
# Same as test 1 but at the real production amplitude (`nogw_Bw = 0.4`, the
# ClimaParams default). At this strength the bins nearest the launch wind are
# already over their breaking limit the moment they are launched. The scheme
# must treat them like any other breaking wave and deposit their momentum in
# the layer just above the launch; the budget then closes exactly as in
# test 1.
#
# Why this deserves its own test: the kernel used to start its loop at
# `source_level - 1`, which applies the AD99 launch-level test and throws these
# waves' momentum away entirely. `launch_level_saturated_flux` is a setup check, not
# a prediction: it computes how much momentum rides on these launch-saturated
# waves, and the test requires that to be large, proving this case really does
# exercise the launch level. If `waveforcing_column_accumulate!` ever slips
# back to `source_level - 1`, this test fails instead of quietly
# becoming a copy of test 1: the budget drops by that flux if the waves are
# dropped again, and drag appears below the launch level if they are deposited
# there instead.
# ============================================================================
@testset "AD99 column momentum budget — saturation at the launch level" begin
    Bw = GW.Bw           # the ClimaParams default, i.e. what production runs use
    lf = launched_flux(U_SOURCE, Bw)
    would_have_been_deleted =
        launch_level_saturated_flux(U_SOURCE, Bw) / lf.F_launch

    # Regression guard: this setup must actually have waves that saturate at the
    # launch level, or the test proves nothing.
    @test would_have_been_deleted > 0.01

    for (name, st, damp_spec) in INTERIOR_CASES
        res = run_case!(st; Bw, wind = :reversing, damp_spec)
        ratios = res.deposited_u ./ lf.F_launch

        @info "Test 3 ($name): deposited / F_launch = $(round(ratios[1]; digits = 12)), flux the old code would have lost = $(round(would_have_been_deleted; digits = 5)) of F_launch"

        # Setup guard: still nothing escaping the top.
        @test maximum(abs, res.escaped ./ lf.F_launch) < MAX_ESCAPED_FRAC

        @test all(isapprox.(ratios, 1; rtol = RTOL_CLOSURE))

        # The launch-saturated waves go into the first layer above the launch
        # level, so no drag lands below it. Moving the loop start back to
        # `source_level - 1` without the old code's guards would deposit them
        # below the launch level and still close the budget; this catches that.
        @test all(iszero, res.forcing_u[1:(SOURCE_LEVEL - 1), :])
        @test all(iszero, res.forcing_v[1:(SOURCE_LEVEL - 1), :])

        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end

# ============================================================================
# Test 4 — waves that meet their critical level exactly (c = u)
#
# Same as test 1 (small amplitude, nothing escapes, nothing saturates at the
# launch level), but the wind steps through the phase-speed bins exactly, so
# every wave is absorbed through the `c_hat == 0` branch rather than the
# sign-change test. A version of that branch that dropped the flux would give a
# ratio far below 1; the deposit must close exactly, as in test 1. Uniform grids
# only, because the on-bin wind is indexed on the uniform levels.
# ============================================================================
# Whether bin `n` of the AD99 spectrum (launched in the on-bin wind with amplitude
# `Bw`) passes every level between the launch level and its exact critical level
# without breaking or meeting a sign change, as the kernel tests them: at level `k`
# it tests the values at `k + 1`, breaking when B0/ĉ³ ≥ ½·(ρ_{k+1}/ρ_source)·k/N.
# Bins without flux deposit nothing either way.
function reaches_exact_critical_level(st, n, Bw)
    B0 = spectrum(U_SOURCE, Bw)[n]
    B0 == 0 && return true
    kwv = 2 * FT(π) / (FT(30) * 10 * FT(1000))
    c = GW_C[n]
    c_hat0 = c - U_SOURCE
    for kp1 in (SOURCE_LEVEL + 1):(st.nlev)
        c_hat = c - on_bin_wind(st.center_z[kp1])
        c_hat == 0 && return true
        c_hat0 * c_hat < 0 && return false
        fac = FT(0.5) * (st.ρ_profile[kp1] / st.ρ_profile[SOURCE_LEVEL]) * kwv / N_const
        B0 / c_hat^3 >= fac && return false
    end
    return false
end

@testset "AD99 column momentum budget — exact critical levels (c = u)" begin
    Bw = FT(1e-4)
    lf = launched_flux(U_SOURCE, Bw)
    @test launch_level_saturated_flux(U_SOURCE, Bw) == 0

    for (name, st, damp_spec) in (
        ("column", COLUMN_STATE, NLEV),
        ("sphere, per-column damp_level", SPHERE_STATE, Val(:varying)),
    )
        # Setup guard: every bin is hit exactly by some wind above the source.
        u_above = winds_above_source(st; wind = :on_bins)
        @test all(c -> any(==(c), u_above), GW_C)
        # Setup guard: every bin with flux reaches the level where the wind equals
        # its phase speed, so it is absorbed there through the `c_hat == 0` branch,
        # not earlier by breaking (with the real density ratio ρ(z)/ρ_source of each
        # level) or by a sign change of c - u.
        @test all(n -> reaches_exact_critical_level(st, n, Bw), eachindex(GW_C))

        res = run_case!(st; Bw, wind = :on_bins, damp_spec)
        ratios = res.deposited_u ./ lf.F_launch

        @info "Test 4 ($name): deposited / F_launch = $(round(ratios[1]; digits = 12))"

        @test maximum(abs, res.escaped ./ lf.F_launch) < MAX_ESCAPED_FRAC
        @test all(isapprox.(ratios, 1; rtol = RTOL_CLOSURE))
        @test maximum(abs, res.deposited_v) < 1e-6 * abs(lf.F_launch)
    end
end

# ============================================================================
# Test 5 — the Beres convective source, alone and on top of AD99
#
# The Beres path differs from AD99 in what it launches and from where: a
# precomputed per-column spectrum B0 (a stress, Pa), launched from the first
# level at or above the top of the convective heating, which differs between
# columns, and deposited with the coverage a_cover instead of the AD99
# intermittency, a_cover/(ρ_source·nk) per wavenumber band. Its budget is
#
#     Σ_k ρ_k·Δz_k·a_k  =  a_cover · Σ_n B0[n]
#
# summed over the nk bands (a_cover·ΣB0/nk each). The two sources accumulate
# into the same forcing, so the full drag must be the AD99-only drag plus the
# Beres-only drag. This would catch an `=` in place of a `+=` in the
# accumulation, which the coverage-linearity test of the single-column file
# differences away. Here nk = 2, a_cover = 0.3, and the launch height varies by
# column on the sphere; B0 is set by hand, asymmetric in c so its net flux is
# not zero, with a meridional spectrum that rides to the model top (v = 0, so
# no critical level), which sends it through the escaped-flux route.
# ============================================================================
const A_COVER = FT(0.3)

function build_gw_beres()
    gw_params = CA.NonOrographicGravityWaveParameters(
        FT,
        (; dc = FT(4), cmax = FT(40), nk = FT(2)),
    )
    P = typeof(gw_params)
    kwargs = NamedTuple{fieldnames(P)}(
        ntuple(i -> getfield(gw_params, i), fieldcount(P)),
    )
    beres_source = CA.BeresSourceParams{FT}(;
        Q0_threshold = FT(0),
        beres_scale_factor = FT(1),
        σ_x = FT(4000),
        ν_min = FT(8.727e-4),
        ν_max = FT(1.047e-2),
        n_ν = 9,
    )
    return CA.NonOrographicGravityWave{FT, typeof(beres_source)}(;
        kwargs...,
        beres_source,
    )
end

# Hand-set launch spectra, each signed like the intrinsic phase speed with a larger
# eastward lobe so the net flux is not zero: zonal, a Gaussian in c - U_SOURCE;
# meridional, a Gaussian in c.
const BERES_B0_U = ntuple(
    n ->
        GW_C[n] == U_SOURCE ? FT(0) :
        sign(GW_C[n] - U_SOURCE) * FT(1e-3) *
        (GW_C[n] > U_SOURCE ? FT(1) : FT(0.4)) *
        exp(-((GW_C[n] - U_SOURCE) / 15)^2),
    length(GW_C),
)
const BERES_B0_V = ntuple(
    n ->
        sign(GW_C[n]) * FT(5e-4) * (GW_C[n] > 0 ? FT(1) : FT(0.5)) *
        exp(-(GW_C[n] / 15)^2),
    length(GW_C),
)

# Heating top per column: 8 km on a column; on the sphere 6-11 km by latitude,
# so the Beres launch level differs from column to column.
function set_beres_ztop!(st)
    (; gw_ztop) = st.gwc
    if st.kind == :column
        fill!(gw_ztop, FT(8000))
    else
        lat = Fields.local_geometry_field(gw_ztop).coordinates.lat
        @. gw_ztop = FT(6000) + FT(5000) * abs(sind(lat))
    end
    return nothing
end

# Run AD99 and Beres together with the given AD99 source amplitude and Beres
# coverage, and return the column-integrated drag and the drag profiles.
function run_beres_case!(
    st;
    source_ampl,
    a_cover,
    damp_spec,
    wind = :reversing,
    B0_u = BERES_B0_U,
    B0_v = BERES_B0_V,
)
    (; gwc, p, ᶜz, ᶜρ_t, ᶜu_t, ᶜv_t, ᶜbf_t, ᶜmom, deposited_field) = st
    @assert gwc.gw_nk == 2

    fill!(gwc.gw_Bw, GW.Bw)
    fill!(gwc.gw_Bn, FT(0))
    fill!(gwc.gw_cw, GW.cw)
    fill!(gwc.gw_cn, GW.cn)
    fill!(gwc.gw_flag, FT(1))
    fill!(gwc.gw_source_ampl, source_ampl)

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

    fill!(st.source_level, FT(SOURCE_LEVEL))
    for (dst, src) in
        ((st.ρ_source, ᶜρ_t), (st.u_source, ᶜu_t), (st.v_source, ᶜv_t))
        Fields.field_values(dst) .=
            Fields.field_values(Fields.level(src, SOURCE_LEVEL))
    end
    set_damp_level!(st, damp_spec)

    # The Beres source: hand-set spectra and coverage, the launch state as
    # production sets it. The scalar source fields are carried by the kernel's
    # input tuple but not read; they are set so they hold finite values.
    fill!(gwc.gw_beres_B0_u, CA.BeresB0(B0_u))
    fill!(gwc.gw_beres_B0_v, CA.BeresB0(B0_v))
    fill!(gwc.gw_a_cover, a_cover)
    for f in (gwc.gw_Q0, gwc.gw_h_heat, gwc.gw_u_heat, gwc.gw_v_heat)
        fill!(f, FT(0))
    end
    fill!(gwc.gw_N_source, N_const)
    set_beres_ztop!(st)
    CA.set_beres_launch_state!(
        gwc.beres_source_ρ_z_u_v_level, ᶜρ_t, ᶜz, ᶜu_t, ᶜv_t, gwc.ᶜlevel,
        gwc.gw_ztop,
    )

    fill!(gwc.uforcing, FT(0))
    fill!(gwc.vforcing, FT(0))
    CA.non_orographic_gravity_wave_forcing(
        ᶜu_t, ᶜv_t, ᶜbf_t, ᶜρ_t, ᶜz, gwc.ᶜlevel, st.source_level,
        gwc.damp_level, st.ρ_source, st.u_source, st.v_source, gwc.uforcing,
        gwc.vforcing, gwc.gw_ncval, gwc.u_waveforcing, gwc.v_waveforcing, p,
    )

    @. ᶜmom = ᶜρ_t * gwc.uforcing
    Operators.column_integral_definite!(deposited_field, ᶜmom)
    deposited_u = per_column(deposited_field)
    @. ᶜmom = ᶜρ_t * gwc.vforcing
    Operators.column_integral_definite!(deposited_field, ᶜmom)
    deposited_v = per_column(deposited_field)
    nlev = Spaces.nlevels(axes(gwc.uforcing))
    profile(f) = permutedims(reduce(hcat, (per_column(Fields.level(f, k)) for k in 1:nlev)))
    return (;
        deposited_u,
        deposited_v,
        forcing_u = profile(gwc.uforcing),
        forcing_v = profile(gwc.vforcing),
        launch_level = per_column(gwc.beres_source_ρ_z_u_v_level.:5),
    )
end

const BERES_CASES = (
    ("column", build_state(:column; gw = build_gw_beres()), NLEV),
    (
        "stretched column",
        build_state(:column; z_stretch = true, gw = build_gw_beres()),
        NLEV,
    ),
    (
        "deep sphere, per-column damp_level",
        build_state(:sphere; deep_atmosphere = true, gw = build_gw_beres()),
        Val(:varying),
    ),
)

@testset "Beres column momentum budget, alone and on top of AD99" begin
    lf = launched_flux(U_SOURCE, GW.Bw)
    F_beres_u = A_COVER * sum(BERES_B0_U)
    F_beres_v = A_COVER * sum(BERES_B0_V)
    @test abs(F_beres_u) > 0.1 * A_COVER * sum(abs, BERES_B0_U)  # not degenerate
    @test abs(F_beres_v) > 0.1 * A_COVER * sum(abs, BERES_B0_V)

    for (name, st, damp_spec) in BERES_CASES
        beres_only =
            run_beres_case!(st; source_ampl = FT(0), a_cover = A_COVER, damp_spec)
        ad99_only =
            run_beres_case!(st; source_ampl = GW.Bt_0, a_cover = FT(0), damp_spec)
        both = run_beres_case!(st; source_ampl = GW.Bt_0, a_cover = A_COVER, damp_spec)

        @info "Test 5 ($name): Beres deposited / launched = $(round(beres_only.deposited_u[1] / F_beres_u; digits = 12)) (u), $(round(beres_only.deposited_v[1] / F_beres_v; digits = 12)) (v); launch levels $(sort(unique(round.(Int, beres_only.launch_level))))"

        # Setup: the Beres launch level lies above the ground and below the
        # damping layer and, on the sphere, differs between columns.
        @test all(>(1), beres_only.launch_level)
        @test all(<(NLEV - 3), beres_only.launch_level)
        st.kind == :sphere && @test length(unique(beres_only.launch_level)) > 1

        @test all(isapprox.(beres_only.deposited_u ./ F_beres_u, 1; rtol = RTOL_CLOSURE))
        @test all(isapprox.(beres_only.deposited_v ./ F_beres_v, 1; rtol = RTOL_CLOSURE))
        @test all(isapprox.(ad99_only.deposited_u ./ lf.F_launch, 1; rtol = RTOL_CLOSURE))
        @test all(
            isapprox.(
                both.deposited_u ./ (lf.F_launch + F_beres_u),
                1;
                rtol = RTOL_CLOSURE,
            ),
        )
        # Nothing lands below the Beres launch level.
        for (col, level) in enumerate(round.(Int, beres_only.launch_level))
            @test all(iszero, beres_only.forcing_u[1:(level - 1), col])
        end
        # The two sources add.
        scale = maximum(abs, both.forcing_u)
        @test maximum(
            abs,
            both.forcing_u .- (ad99_only.forcing_u .+ beres_only.forcing_u),
        ) <
              1e-12 * scale
        @test maximum(
            abs,
            both.forcing_v .- (ad99_only.forcing_v .+ beres_only.forcing_v),
        ) < 1e-12 * maximum(abs, both.forcing_v)
    end
end

# ============================================================================
# Test 6 — where each wave breaks, against the analytic breaking height
#
# In a constant wind with constant N and ρ = ρ0·exp(-z/H), a wave of flux B0
# and intrinsic phase speed ĉ saturates where B0/ĉ³ first reaches the breaking
# limit ½·(ρ(z)/ρ_ref)·k/N, with ρ_ref = ρ_source for the AD99 flux per unit
# mass and ρ_ref = 1 for the Beres stress. So it breaks at
#
#     z* = H·ln(ρ0·k/(2·N·ρ_ref·B0/ĉ³)).
#
# The kernel tests the level above each level, so the wave deposits in the
# layer between the first centre at or above z* (and above the launch level)
# and the centre below it, half in each of those two cells; a wave still
# unbroken at the top goes to the damping layer. The cells that receive drag
# must be exactly the union of these cells over all waves and wavenumber bands.
# A factor of 2 in the limit moves each z* by H·ln 2 ≈ 4.9 km, five layers.
# ============================================================================

# The set of cells that receive drag from the waves `B0` (intrinsic phase speeds
# `c_hat`) launched at `launch_level`, for `nk` wavenumber bands, with the
# reference density `ρ_ref` of the breaking limit. Every z* must lie more than
# 1 m from a cell centre, so rounding cannot move a wave to another layer.
function predicted_drag_cells(st, B0, c_hat, launch_level, ρ_ref; nk, damp_level)
    cells = Set{Int}()
    for ink in 1:nk, n in eachindex(B0)
        B0[n] == 0 && continue
        kwv = 2 * FT(π) / (30 * FT(10)^ink * 1000)
        ratio = B0[n] / c_hat[n]^3
        @assert ratio > 0 "B0 must carry the sign of the intrinsic phase speed"
        z_star = H_ρ * log(ρ0 * kwv / (2 * N_const * ρ_ref * ratio))
        @assert minimum(abs, st.center_z .- z_star) > 1 "z* = $z_star m is on a centre"
        kp1 = findfirst(
            k -> k > launch_level && st.center_z[k] >= z_star,
            1:(st.nlev),
        )
        isnothing(kp1) ? union!(cells, damp_level:(st.nlev)) : union!(cells, (kp1 - 1, kp1))
    end
    return cells
end

drag_cells(forcing_column) = Set(findall(!iszero, forcing_column))

@testset "Breaking heights match the analytic saturation height" begin
    # AD99: the ClimaParams spectrum at 50 times the production amplitude, so the
    # waves break from the first layer above the launch level up to about 31 km.
    Bw = FT(20)
    B0 = spectrum(U_SOURCE, Bw)
    c_hat = GW_C .- U_SOURCE
    ρ_source = COLUMN_STATE.ρ_profile[SOURCE_LEVEL]
    expected = predicted_drag_cells(
        COLUMN_STATE, B0, c_hat, SOURCE_LEVEL, ρ_source; nk = 1, damp_level = NLEV,
    )
    @test length(expected) >= 8                    # the waves break at many heights
    @test maximum(expected) < NLEV - 5             # and none reaches the top
    res = run_case!(COLUMN_STATE; Bw, wind = :constant, damp_spec = NLEV)
    @test drag_cells(res.forcing_u[:, 1]) == expected

    # Beres: two bins, launched at the first level above 8 km, each breaking at two
    # heights, one per wavenumber band: the second band's k is 10 times smaller, so
    # its breaking limit is too, and it breaks H·ln 10 ≈ 16 km lower (here at 9.7
    # and 15.1 km, the first band at 25.8 and 31.2 km).
    (_, st, _) = BERES_CASES[1]
    n_east = findfirst(==(FT(40)), GW_C)           # ĉ = 16 m/s
    n_west = findfirst(==(FT(0)), GW_C)            # ĉ = -24 m/s
    B0_u = ntuple(
        n -> n == n_east ? FT(0.043) : n == n_west ? FT(-0.067) : FT(0),
        length(GW_C),
    )
    zero_B0 = ntuple(_ -> FT(0), length(GW_C))
    res_beres = run_beres_case!(
        st;
        source_ampl = FT(0),
        a_cover = FT(1),
        damp_spec = NLEV,
        wind = :constant,
        B0_u,
        B0_v = zero_B0,
    )
    launch_level = round(Int, res_beres.launch_level[1])
    expected_beres = predicted_drag_cells(
        st, B0_u, GW_C .- U_SOURCE, launch_level, FT(1); nk = 2, damp_level = NLEV,
    )
    @test length(expected_beres) == 8              # four separate layers
    @test drag_cells(res_beres.forcing_u[:, 1]) == expected_beres
    @test all(iszero, res_beres.forcing_v)
end

# ============================================================================
# Test 7 — total internal reflection
#
# A wave whose intrinsic horizontal phase speed exceeds ω_r/k, with
# ω_r = √(N²k²/(k² + 1/(4H_b²))) and H_b the density scale height of the layer, is
# reflected: the kernel removes it from the spectrum and deposits nothing. Just
# below that speed it keeps propagating, and at the model top it is deposited.
# One step of the kernel, `waveforcing_accumulate_step`, is called directly with
# one phase-speed bin, a weak wave (far from breaking) and the AD99 layout of
# its inputs.
# ============================================================================
@testset "AD99 kernel: waves faster than the reflection speed are removed" begin
    N = FT(0.01)
    kwv = 2 * FT(π) / (FT(30) * 10 * FT(1000))
    (z_k, z_kp1) = (FT(20000), FT(21000))
    (ρ_k, ρ_kp1) = (ρ0 * exp(-z_k / H_ρ), ρ0 * exp(-z_kp1 / H_ρ))
    H_b = (z_kp1 - z_k) / log(ρ_k / ρ_kp1)
    ω_r = sqrt(N^2 * kwv^2 / (kwv^2 + 1 / (4 * H_b^2)))
    c_reflect = ω_r / kwv
    level_end = 50
    BitVector1 = CA.UnrolledUtilities.StaticBitVector{1}
    # One kernel step at `level` for a wave of intrinsic phase speed `c_hat`
    # (u = 0, so c = c_hat) launched at level 2 with a weak eastward flux.
    function step(c_hat, level)
        c = (c_hat,)
        B0 = (FT(1e-12),)
        state = (FT(0), BitVector1(_ -> true), sum(abs, B0), B0)
        inp = (
            FT(0), N, ρ_k, ρ_kp1, z_kp1, z_k, level,          # shared slots 1-7
            FT(0), ρ_k, 2, FT(0.4), FT(0), FT(35), FT(2), FT(1), FT(0.0043), # AD99 8-16
        )
        (wave_forcing, mask, _, _) = CA.waveforcing_accumulate_step(
            state, inp, c, FT(0), 1, kwv, level_end, Val(1), Val(:ad99),
        )
        return (; wave_forcing, kept = mask[1])
    end
    above = step(FT(1.01) * c_reflect, 10)
    below = step(FT(0.99) * c_reflect, 10)
    @test !above.kept
    @test above.wave_forcing == 0
    @test below.kept
    @test below.wave_forcing == 0
    # At the model top an unreflected wave is deposited, a reflected one is not.
    @test step(FT(0.99) * c_reflect, level_end).wave_forcing > 0
    @test step(FT(1.01) * c_reflect, level_end).wave_forcing == 0
end

# ============================================================================
# Test 8 — the AD99 source by latitude, and the frame of the tropical spectrum
#
# On the sphere the AD99 source amplitude follows the MiMA rule: `Bt_eq` within
# |lat| ≤ |dϕ| (10°), a linear ramp to `Bt_0` at |lat| = |ϕ0| (15°), and beyond
# that `Bt_0` plus hemispheric tanh enhancements. Inside the tropical band the
# spectrum is centred on the launch wind (flag 0, no narrow component, the
# tropical width); outside it is centred on c0 in the ground frame (flag 1).
# The ClimaParams defaults make the amplitude the same everywhere, so this test
# sets distinct values.
# ============================================================================
@testset "AD99 source amplitude, frame and width by latitude" begin
    (Bt_0, Bt_eq, Bt_n, Bt_s) = (FT(0.004), FT(0.006), FT(0.002), FT(0.001))
    (Bn, cw, cw_tropics) = (FT(0.1), FT(35), FT(25))
    gw_params = CA.NonOrographicGravityWaveParameters(
        FT,
        (; dc = FT(4), cmax = FT(40), Bt_0, Bt_eq, Bt_n, Bt_s, Bn, cw, cw_tropics),
    )
    P = typeof(gw_params)
    gw = CA.NonOrographicGravityWave{FT, Nothing}(;
        NamedTuple{fieldnames(P)}(ntuple(i -> getfield(gw_params, i), fieldcount(P)))...,
        beres_source = nothing,
    )
    (; ϕ0_n, ϕ0_s, dϕ_n, dϕ_s) = gw
    @test (ϕ0_n, ϕ0_s, dϕ_n, dϕ_s) == (15, -15, 10, -10)
    gwc = build_state(:sphere; gw).gwc
    lat = per_column(Fields.local_geometry_field(gwc.gw_source_ampl).coordinates.lat)
    ampl = per_column(gwc.gw_source_ampl)
    flag = per_column(gwc.gw_flag)
    gw_Bn = per_column(gwc.gw_Bn)
    gw_cw = per_column(gwc.gw_cw)
    tropics = abs.(lat) .<= 10
    ramp = 10 .< abs.(lat) .<= 15
    extratropics = abs.(lat) .> 15
    # Setup: the grid has columns in all three regions.
    @test any(tropics) && any(ramp) && any(extratropics)

    expected_ampl = map(lat) do ϕ
        if abs(ϕ) <= 10
            Bt_eq
        elseif abs(ϕ) <= 15
            Bt_eq + (Bt_0 - Bt_eq) * (abs(ϕ) - 10) / 5
        else
            Bt_0 + Bt_n * (1 + tanh((ϕ - 15) / 10)) / 2 +
            Bt_s * (1 + tanh((ϕ + 15) / -10)) / 2
        end
    end
    @test ampl ≈ expected_ampl rtol = 1e-12
    @test all(flag[tropics] .== 0) && all(flag[.!tropics] .== 1)
    @test all(gw_Bn[tropics] .== 0) && all(gw_Bn[.!tropics] .== Bn)
    @test all(gw_cw[tropics] .== cw_tropics) && all(gw_cw[.!tropics] .== cw)

    # The tropical spectrum (flag 0) is the ground-frame one (flag 1, u = 0)
    # shifted by the launch wind: for u = 8 m/s, two bins.
    u = FT(8)
    ground = CA.wave_source(GW_C, FT(0), FT(0.4), FT(0), cw, GW.cn, FT(0), FT(1), GW_NCVAL)
    moving = CA.wave_source(GW_C, u, FT(0.4), FT(0), cw, GW.cn, FT(0), FT(0), GW_NCVAL)
    @test collect(moving[3:end]) == collect(ground[1:(end - 2)])
    @test moving !=
          CA.wave_source(GW_C, u, FT(0.4), FT(0), cw, GW.cn, FT(0), FT(1), GW_NCVAL)
end
