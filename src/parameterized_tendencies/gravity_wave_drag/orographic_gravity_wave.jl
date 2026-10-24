#####
##### Orographic drag parameterization
#####

# This orographic gravity wave drag follows the paper by Garner 2005:
# https://journals.ametsoc.org/view/journals/atsc/62/7/jas3496.1.xml?tab_body=pdf
# and the GFDL implementation:
# https://github.com/NOAA-GFDL/atmos_phys/blob/main/atmos_param/topo_drag/topo_drag.F90

using ClimaUtilities.ClimaArtifacts
using ClimaCore: InputOutput
import .AtmosArtifacts as AA

"""
    orographic_gravity_wave_cache(Y, atmos::AtmosModel)
    orographic_gravity_wave_cache(Y, ::Nothing)
    orographic_gravity_wave_cache(Y, ogw::OrographicGravityWave, topo_info = nothing)

Allocate the orographic gravity wave (OGW) cache for the state `Y`.

Returns an empty `NamedTuple` when the model is disabled. Otherwise the cache holds the
shape parameters in `ogw_params`, the orographic drag input `topo_info`, the per-column
base-flux and Froude-number fields, the saturation profile `topo_ᶜτ_sat`/`topo_ᶠτ_sat`, and
the accumulated forcings `ᶜuforcing`/`ᶜvforcing` [m/s²].

`topo_info` is loaded by `get_topo_info` unless the caller supplies it, which lets a
restart or a test reuse an already-regridded drag tensor. Only spherical domains are
supported; anything else trips an assertion.

See also `orographic_gravity_wave_compute_tendency!`. The scheme is documented on the
*Orographic Gravity Waves* page.
"""
orographic_gravity_wave_cache(Y, atmos::AtmosModel) =
    orographic_gravity_wave_cache(Y, atmos.orographic_gravity_wave)

orographic_gravity_wave_cache(Y, ::Nothing) = (;)

"""
    get_topo_info(Y, ogw::OrographicGravityWave)

Obtain the orographic drag input `(; hmax, hmin, t11, t12, t21, t22)` on the surface space.

The source is selected by `ogw.topo_info`:

  - `Val(:gfdl_restart)`: regrid the GFDL `topo_drag.res.nc` restart from the `topo_drag`
    ClimaArtifact onto the model grid with `regrid_OGW_info`.
  - `Val(:raw_topo)`: build the drag from the configured topography with
    `compute_ogw_drag`, which loads a preprocessed HDF5 artifact for Earth topography and
    computes the tensor on the fly for the analytical test topographies.
  - `Val(:raw_topo_online)`: build the drag from the raw ETOPO elevation at initialization
    with `compute_ogw_drag_online`, which runs the `compute_OGW_info` preprocessing
    pipeline for Earth topography and falls back to the analytical path for the test
    topographies.
  - `Val(:linear)`: user-defined analytical drag input, for idealized tests.

Any other value is an error. Called from `orographic_gravity_wave_cache`.

# Returns

A `NamedTuple` of surface-level `Field`s: the effective maximum and minimum subgrid
obstacle heights `hmax`, `hmin` [m], and the four components `t11`, `t12`, `t21`, `t22` of
the orographic tensor `T = −∇χ (∇h)ᵀ`, stored with `tᵢⱼ = −∂χ/∂xⱼ · ∂h/∂xᵢ`.
"""
# Shape parameters the shipped `raw_topo` (`ogw_computed_drag_h*`) artifacts were
# generated with, i.e. the ClimaParams defaults for `ogw_mountain_height_width_exponent`,
# `ogw_critical_height_threshold`, and `ogw_smoothing_scale_fraction`.
const RAW_TOPO_ARTIFACT_γ = 0.4
const RAW_TOPO_ARTIFACT_H_FRAC = 0.1
const RAW_TOPO_ARTIFACT_α_SMOOTHING = 0.15

"""
    warn_if_stale_raw_topo_artifact(ogw)

Warn when `raw_topo` is used for Earth topography with `γ`, `h_frac`, or `α_smoothing` that
differ from the values the shipped artifact was built with.

The `raw_topo` artifact fixes `hmax`, `hmin`, and the drag tensor at generation time, so
overriding these parameters leaves the drag input inconsistent with the runtime physics
(which does use the current parameters). Switch to `raw_topo_online` to rebuild the drag
input from the current parameters. Only Earth topography loads an artifact, so no warning is
issued for the analytical test topographies.
"""
function warn_if_stale_raw_topo_artifact(ogw)
    (ogw.topography == Val(:Earth) || ogw.topography == Val(:NoWarp)) ||
        return nothing
    FT = typeof(ogw.h_frac)
    stale = String[]
    isapprox(ogw.γ, FT(RAW_TOPO_ARTIFACT_γ)) ||
        push!(stale, "γ = $(ogw.γ) (artifact $(RAW_TOPO_ARTIFACT_γ))")
    isapprox(ogw.h_frac, FT(RAW_TOPO_ARTIFACT_H_FRAC)) ||
        push!(stale, "h_frac = $(ogw.h_frac) (artifact $(RAW_TOPO_ARTIFACT_H_FRAC))")
    isapprox(ogw.α_smoothing, FT(RAW_TOPO_ARTIFACT_α_SMOOTHING)) || push!(
        stale,
        "α_smoothing = $(ogw.α_smoothing) (artifact $(RAW_TOPO_ARTIFACT_α_SMOOTHING))",
    )
    isempty(stale) && return nothing
    @warn "orographic_gravity_wave = \"raw_topo\" loads a preprocessed artifact built at \
           the default shape parameters, so the drag input will not reflect the \
           overridden $(join(stale, ", ")). Use \"raw_topo_online\" to recompute the drag \
           input from the current parameters."
    return nothing
end

function get_topo_info(Y, ogw::OrographicGravityWave)
    if ogw.topo_info == Val(:gfdl_restart)
        topo_path = @clima_artifact("topo_drag", ClimaComms.context(Y.c))
        orographic_info_rll = joinpath(topo_path, "topo_drag.res.nc")
        topo_info = regrid_OGW_info(Y, orographic_info_rll)
    elseif ogw.topo_info == Val(:raw_topo)
        warn_if_stale_raw_topo_artifact(ogw)
        earth_radius =
            Spaces.topology(
                Spaces.horizontal_space(axes(Y.c)),
            ).mesh.domain.radius
        topo_info = compute_ogw_drag(
            Y,
            earth_radius,
            ogw.topography,
            ogw.h_frac,
        )
    elseif ogw.topo_info == Val(:raw_topo_online)
        earth_radius =
            Spaces.topology(
                Spaces.horizontal_space(axes(Y.c)),
            ).mesh.domain.radius
        topo_info = compute_ogw_drag_online(
            Y,
            earth_radius,
            ogw.topography,
            ogw.γ,
            ogw.h_frac,
            ogw.α_smoothing,
        )
    elseif ogw.topo_info == Val(:linear)
        # For user-defined analytical tests
        topo_info = initialize_drag_input_as_fields(Y, ogw.drag_input)
    else
        error(
            "topo_info must be Val(:gfdl_restart), Val(:raw_topo), Val(:raw_topo_online), or Val(:linear)",
        )
    end
    return topo_info
end

function orographic_gravity_wave_cache(Y, ogw::OrographicGravityWave, topo_info = nothing)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations
    @assert Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain isa
            Domains.SphereDomain

    FT = Spaces.undertype(axes(Y.c))
    (; γ, ϵ, β, ρscale, L0, a0, a1, Fr_crit, a_tofd) = ogw

    if topo_info === nothing
        topo_info = get_topo_info(Y, ogw)
    end

    center_space, face_space = axes(Y.c), axes(Y.f)

    # The TOFD drag coefficient depends only on the grid and the orography, so it is
    # built once here and combined with the current wind every step. Its input, the
    # filtered-orography variance, is computed from the Earth elevation data in the same
    # cases in which the gravity-wave drag uses Earth orography.
    ᶜtofd_Cd = zero(Y.c.ρ)
    if !iszero(a_tofd)
        uses_earth_orography =
            ogw.topography in (Val(:Earth), Val(:NoWarp)) &&
            ogw.topo_info in (Val(:gfdl_restart), Val(:raw_topo), Val(:raw_topo_online))
        if uses_earth_orography
            earth_radius =
                Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain.radius
            σ²_flt = compute_filtered_orography_variance(
                Y,
                AA.earth_orography_file_path(context = ClimaComms.context(Y.c)),
                earth_radius,
            )
            tofd_drag_coefficient!(
                ᶜtofd_Cd,
                Fields.coordinate_field(Y.c).z,
                Fields.level(Fields.coordinate_field(Y.f).z, half),
                σ²_flt,
                a_tofd,
            )
        else
            @warn "Turbulent orographic form drag needs Earth orography; it is off for \
                   topography = $(ogw.topography) and orographic_gravity_wave = \
                   $(ogw.topo_info)."
        end
    end

    # Prepare cache
    return (;
        ogw_params = (;
            Fr_crit = Fr_crit,
            topo_ρscale = ρscale,
            topo_L0 = L0,
            topo_a0 = a0,
            topo_a1 = a1,
            topo_γ = γ,
            topo_β = β,
            topo_ϵ = ϵ,
            topo_a_tofd = a_tofd,
        ),
        topo_ᶜτ_sat = Fields.Field(FT, axes(Y.c)),
        topo_ᶠτ_sat = Fields.Field(FT, axes(Y.f.u₃)),
        topo_ᶠVτ = Fields.Field(FT, axes(Y.f.u₃)),
        topo_τ_x = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_y = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_l = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_p = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_np = similar(Fields.level(Y.c.ρ, 1)),
        topo_U_sat = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_sat = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_max = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_min = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_clp = similar(Fields.level(Y.c.ρ, 1)),
        topo_ᶜz_pbl = similar(Fields.level(Y.c.ρ, 1)),
        topo_ᶠz_blocked_start = similar(Fields.level(Y.f.u₃, half)),
        values_at_z_pbl = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT}),
        topo_info = topo_info,
        ᶜbuoyancy_frequency = Fields.Field(FT, center_space),
        ᶠbuoyancy_frequency = Fields.Field(FT, face_space),
        ᶜuforcing = zero(Y.c.ρ),
        ᶜvforcing = zero(Y.c.ρ),
        ᶜtofd_Cd,
        ᶜdTdz = similar(Y.c.ρ),
        ᶠp_ref = similar(Fields.level(Y.f.u₃, half), FT),
        ᶜmask = Fields.Field(Bool, center_space),
    )

end

"""
    orographic_gravity_wave_compute_tendency!(Y, p, ::Nothing)
    orographic_gravity_wave_compute_tendency!(Y, p, ::FullOrographicGravityWave)

Compute the orographic gravity wave drag and store it in the OGW cache.

Mutates `p.orographic_gravity_wave` (notably `ᶜuforcing` and `ᶜvforcing` [m/s²], which are
zeroed and then recomputed) and reads `Y`. This is the expensive half of the
parameterization and runs on the `dt_ogw` callback `ogw_model_callback!`; the cheap half,
`orographic_gravity_wave_apply_tendency!`, applies the cached forcing (plus the
per-step implicit TOFD) every integrator step. The `::Nothing` method is a no-op.

Prepares the inputs the forcing routine needs and then calls
`orographic_gravity_wave_forcing!`:

  - The buoyancy frequency `N = sqrt(g/T · (dT/dz + g/c_pm))`, using the moist isobaric
    specific heat, at cell centers and interpolated to faces, floored at `sqrt(eps(FT))`.
  - Face pressure `ᶠp` from `ogw_face_pressure!`: interpolated from cell centers at
    interior faces and hydrostatic at the bottom and top faces. The non-propagating drag
    takes each cell's pressure thickness from it.
  - The physical horizontal wind components from `Y.c.uₕ`.

Reads `p.precomputed.ᶜp`, `ᶜT`, `ᶜq_tot_nonneg`, `ᶜq_liq`, `ᶜq_ice`, and clobbers
`p.scratch.ᶠtemp_scalar` along with the scratch fields listed for
`orographic_gravity_wave_forcing!`. Described in [garner2005](@cite); see the
*Orographic Gravity Waves* page for the derivations.
"""
orographic_gravity_wave_compute_tendency!(Y, p, ::Nothing) = nothing

function orographic_gravity_wave_compute_tendency!(Y, p, ::FullOrographicGravityWave)
    @debug begin
        if !hasfield(typeof(p), :ogwd_call_counter)
            @info "OGWD tendency function called for the first time"
        end

        # DEBUG: Check if Y has NaNs at entry
        if any(isnan, Y.c.ρ)
            @error "OGWD: Input Y.c.ρ already has NaNs at function entry!"
            error("Cannot compute OGWD tendency with NaN inputs")
        end
    end

    # unpack cache
    (; ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; params) = p
    (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave
    (; ᶜdTdz) = p.orographic_gravity_wave
    (; ᶜbuoyancy_frequency, ᶠbuoyancy_frequency) = p.orographic_gravity_wave

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶠdz = Fields.Δz_field(axes(Y.f))
    FT = Spaces.undertype(axes(Y.c))
    ᶜρ = Y.c.ρ

    # parameters
    cp_d = CAP.cp_d(params)
    grav = CAP.grav(params)
    thermo_params = CAP.thermodynamics_params(params)

    # compute buoyancy frequency
    ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1
    @. ᶜbuoyancy_frequency =
        (grav / ᶜT) *
        (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    @. ᶜbuoyancy_frequency =
        ifelse(ᶜbuoyancy_frequency < eps(FT), sqrt(eps(FT)), sqrt(abs(ᶜbuoyancy_frequency))) # to avoid small numbers
    @. ᶠbuoyancy_frequency = ᶠinterp(ᶜbuoyancy_frequency)

    # face pressure, from which the non-propagating drag takes each cell's
    # pressure thickness
    ᶠp = p.scratch.ᶠtemp_scalar
    ogw_face_pressure!(ᶠp, ᶜp, ᶜρ, ᶜz, ᶠz, grav)

    # prepare physical uv input variables for gravity_wave_forcing()
    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    @. ᶜuforcing = 0
    @. ᶜvforcing = 0

    orographic_gravity_wave_forcing!(
        ᶜu,
        ᶜv,
        ᶜbuoyancy_frequency,
        ᶠbuoyancy_frequency,
        ᶜz,
        ᶠz,
        ᶠdz,
        ᶜuforcing,
        ᶜvforcing,
        ᶜρ,
        ᶜp,
        ᶠp,
        ᶜT,
        grav,
        cp_d,
        p,
    )

    @debug begin
        # Checkpoint 2b: Check computed forcing magnitude
        @info "  Computed forcing: u_max=$(maximum(abs, ᶜuforcing)) m/s², v_max=$(maximum(abs, ᶜvforcing)) m/s²"

        # Warn if forcing is very large (>0.01 m/s² = 1 cm/s² is already strong)
        max_forcing = max(maximum(abs, ᶜuforcing), maximum(abs, ᶜvforcing))
        if max_forcing > 0.01
            @warn "OGWD forcing very large! This may cause instability."
            @warn "  max_accel=$(max_forcing) m/s² (threshold: 0.01 m/s²)"
        end
    end
end

"""
    orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, ::Nothing)
    orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, ::OrographicGravityWave)

Add the orographic drag to the horizontal momentum tendency.

Mutates `Yₜ.c.uₕ`. The `::Nothing` method is a no-op. Two contributions are added every
integrator step:

  - The cached propagating and blocked gravity-wave forcing `ᶜuforcing`/`ᶜvforcing`
    [m/s²]. It is refreshed only on the `dt_ogw` callback by
    `orographic_gravity_wave_compute_tendency!`, which also limits its magnitude to
    3e-3·√2 m/s² (preserving direction); between callbacks the same forcing is reapplied.
  - The Beljaars (2004) turbulent orographic form drag, evaluated from the current wind
    `Y.c.uₕ` with the static drag coefficient `ᶜtofd_Cd` and the backward-Euler rate
    `tofd_implicit_rate`. Near the surface over steep orography its damping time is far
    shorter than `dt_ogw` and can be shorter than `p.dt`, so it is neither cached nor
    passed through the magnitude limiter; the implicit rate bounds the per-step decrement
    by the wind itself instead. Skipped when `a_tofd == 0`.
"""
orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, ::Nothing) = nothing

function orographic_gravity_wave_apply_tendency!(
    Yₜ,
    Y,
    p,
    ::OrographicGravityWave,
)
    (; ᶜuforcing, ᶜvforcing, ᶜtofd_Cd, ogw_params) = p.orographic_gravity_wave

    @. Yₜ.c.uₕ +=
        C12.(Geometry.UVVector.(ᶜuforcing, ᶜvforcing))

    iszero(ogw_params.topo_a_tofd) && return nothing
    dt = eltype(ᶜtofd_Cd)(p.dt)
    @. Yₜ.c.uₕ -=
        tofd_implicit_rate(ᶜtofd_Cd, sqrt(norm_sqr(Y.c.uₕ)), dt) * Y.c.uₕ
    return nothing
end


"""
    orographic_gravity_wave_forcing!(u_phy, v_phy, ᶜbuoyancy_frequency,
                                     ᶠbuoyancy_frequency, ᶜz, ᶠz, ᶠdz, ᶜuforcing,
                                     ᶜvforcing, ᶜρ, ᶜp, ᶠp, ᶜT, grav, cp_d, p)

Assemble the propagating and non-propagating orographic drag into the forcing fields.

Mutates `ᶜuforcing` and `ᶜvforcing` [m/s²], which the caller has zeroed, along with the
intermediate OGW cache fields. Called from
`orographic_gravity_wave_compute_tendency!`.

Steps, in order:

 1. `get_pbl_z!` finds the PBL top `z_pbl`, the center height of the PBL-top cell, and
    `blocked_layer_start_height!` stores, in a face-level field, the height of the top
    face of the cell just above it. That cell is the lowest cell of the blocked layer
    (GFDL `kpbl`), and `calc_nonpropagating_forcing!` counts the phase from its top
    face.
 2. `calc_base_flux!` evaluates the low-level flow at `z_pbl` and splits the base
    momentum flux into the linear, propagating, and non-propagating parts `τ_l`, `τ_p`,
    `τ_np`.
 3. `calc_saturation_profile!` marches the propagating flux `τ_sat(z)` upward.
 4. `calc_propagate_forcing!` converts `dτ_sat/dz` into a tendency, and
    `calc_nonpropagating_forcing!` adds the blocked-flow tendency from the first cell
    above the PBL-top cell up to the reference level.
 5. The forcing vector magnitude is limited to 3e-3·√2 m/s² (scaling `(u, v)` together so
    the drag direction is preserved) so a large tendency cannot destabilize the integrator.
    The cap is the diagonal of the ±3e-3 m/s² per-component box of GFDL `topo_drag`
    (`max_udt`), so it is isotropic and no tighter than that box in any direction.

Clobbers `p.scratch.ᶜtemp_scalar`, `ᶜtemp_scalar_2`, `temp_field_level`, and
`ᶠtemp_field_level`.
"""
function orographic_gravity_wave_forcing!(
    u_phy,
    v_phy,
    ᶜbuoyancy_frequency,
    ᶠbuoyancy_frequency,
    ᶜz,
    ᶠz,
    ᶠdz,
    ᶜuforcing,
    ᶜvforcing,
    ᶜρ,
    ᶜp,
    ᶠp,
    ᶜT,
    grav,
    cp_d,
    p,
)

    FT = eltype(ᶠbuoyancy_frequency)

    (;
        topo_ᶜz_pbl,
        topo_ᶠz_blocked_start,
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_p,
        topo_τ_np,
    ) =
        p.orographic_gravity_wave
    (; topo_ᶜτ_sat, topo_ᶠτ_sat) = p.orographic_gravity_wave
    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; topo_ᶠVτ, values_at_z_pbl, topo_info) = p.orographic_gravity_wave
    (; ᶜmask, ᶠp_ref) = p.orographic_gravity_wave

    # Extract parameters
    ogw_params = p.orographic_gravity_wave.ogw_params

    # z_pbl is the center height of the PBL-top cell. The blocked layer starts in the
    # cell above it and counts its phase from that cell's top face (GFDL topo_drag:
    # kpbl, zhalf(kpbl)), which calc_nonpropagating_forcing! needs as a face-level
    # field.
    get_pbl_z!(topo_ᶜz_pbl, ᶜp, ᶜT, ᶜz, grav, cp_d)
    topo_ᶠz_blocked_start = topo_ᶠz_blocked_start.components.data.:1
    blocked_layer_start_height!(topo_ᶠz_blocked_start, topo_ᶜz_pbl, ᶠz)

    # compute base flux at the planetary boundary layer height
    calc_base_flux!(
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_p,
        topo_τ_np,
        #
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜz_pbl,
        #
        values_at_z_pbl,
        #
        ogw_params,
        topo_info,
        #
        ᶜρ,
        u_phy,
        v_phy,
        ᶜz,
        ᶜbuoyancy_frequency,
    )

    calc_saturation_profile!(
        topo_ᶠτ_sat,
        topo_ᶠVτ,
        #
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜτ_sat,
        topo_τ_x,
        topo_τ_y,
        topo_τ_p,
        topo_ᶜz_pbl,
        #
        ogw_params,
        #
        ᶜρ,
        u_phy,
        v_phy,
        ᶜp,
        ᶜbuoyancy_frequency,
        ᶜz,
    )

    # compute drag tendencies due to propagating part
    ᶜdτ_sat_dz = p.scratch.ᶜtemp_scalar
    calc_propagate_forcing!(
        ᶜuforcing,
        ᶜvforcing,
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_ᶠτ_sat,
        ᶜdτ_sat_dz,
        ᶜρ,
    )

    ᶜweights = p.scratch.ᶜtemp_scalar
    ᶜdiff = p.scratch.ᶜtemp_scalar_2
    ᶜwtsum = p.scratch.temp_field_level
    ᶠz_ref = p.scratch.ᶠtemp_field_level
    calc_nonpropagating_forcing!(
        ᶜuforcing,
        ᶜvforcing,
        #
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_np,
        topo_ᶠVτ,
        topo_ᶠz_blocked_start,
        #
        ᶠz_ref,
        ᶠp_ref,
        ᶜmask,
        ᶜweights,
        ᶜdiff,
        ᶜwtsum,
        #
        ᶠp,
        ᶠbuoyancy_frequency,
        ᶠz,
        ᶠdz,
        grav,
    )

    # Check before the limiter: it maps an infinite component to NaN (F_max/Inf = 0,
    # then 0·Inf), which would hide the original problem.
    @debug begin
        if !all(isfinite, ᶜuforcing) || !all(isfinite, ᶜvforcing)
            @error "Non-finite OGWD forcing before the magnitude limiter!"
            @error "  ᶜuforcing: has_nan=$(any(isnan, ᶜuforcing)), has_inf=$(any(isinf, ᶜuforcing)), min=$(minimum(ᶜuforcing)), max=$(maximum(ᶜuforcing))"
            @error "  ᶜvforcing: has_nan=$(any(isnan, ᶜvforcing)), has_inf=$(any(isinf, ᶜvforcing)), min=$(minimum(ᶜvforcing)), max=$(maximum(ᶜvforcing))"
            error("OGWD produced non-finite forcing - aborting")
        end
    end

    # Limit the forcing magnitude as a numerical safety rail for the explicit
    # application, preserving its direction: the (u, v) vector is scaled by
    # min(1, F_max/|F|), so the drag stays antiparallel to the flow where the limiter
    # binds (clipping each component separately would rotate it). The scale is
    # materialized first so scaling ᶜuforcing does not change the |F| seen by
    # ᶜvforcing. F_max = 3e-3·√2 m/s² is the diagonal of the ±3e-3 m/s² per-component
    # box of GFDL topo_drag, so the limit is isotropic and no tighter than that box in
    # any direction.
    F_max = FT(3e-3) * sqrt(FT(2))
    ᶜdrag_scale = p.scratch.ᶜtemp_scalar
    @. ᶜdrag_scale = ifelse(
        sqrt(ᶜuforcing^2 + ᶜvforcing^2) > F_max,
        F_max / sqrt(ᶜuforcing^2 + ᶜvforcing^2),
        FT(1),
    )
    @. ᶜuforcing *= ᶜdrag_scale
    @. ᶜvforcing *= ᶜdrag_scale
end

"""
    ogw_face_pressure!(ᶠp, ᶜp, ᶜρ, ᶜz, ᶠz, grav)

Fill `ᶠp` [Pa] with the face pressure the blocked drag takes its pressure thicknesses from.

Interior faces get the mean of the two adjacent cell-center pressures (`ᶠinterp`). The
boundary faces are integrated hydrostatically over the half cell from the adjacent center,

    p_bottom = p_1 + ρ_1·g·(z_1 − z_bottom),
    p_top    = max(p_N − ρ_N·g·(z_top − z_N), eps),

because `ᶠinterp` copies the nearest center value onto a boundary face, which would leave
the lowest and highest cells with about half their true pressure thickness. The model keeps
no surface or top-of-model pressure, so the half-cell integral is the closest consistent
value. Called from `orographic_gravity_wave_compute_tendency!`; it only broadcasts over
levels, so it runs on the GPU.

# Arguments

  - `ᶠp`: Face pressure [Pa]; overwritten.
  - `ᶜp`, `ᶜρ`: Pressure [Pa] and density [kg/m³] at cell centers.
  - `ᶜz`, `ᶠz`: Heights of cell centers and faces [m].
  - `grav`: Gravitational acceleration [m/s²].
"""
function ogw_face_pressure!(ᶠp, ᶜp, ᶜρ, ᶜz, ᶠz, grav)
    FT = eltype(ᶠp)
    @. ᶠp = ᶠinterp(ᶜp)

    # The boundary face levels and their adjacent center levels are different spaces
    # with the same layout, so combine their raw data (views into the fields).
    n = Spaces.nlevels(axes(ᶜp))
    bot(x) = Fields.field_values(Fields.level(x, 1))
    top(x) = Fields.field_values(Fields.level(x, n))
    ᶠp_bot = Fields.field_values(Fields.level(ᶠp, half))
    ᶠp_top = Fields.field_values(Fields.level(ᶠp, n + half))
    z_bot = Fields.field_values(Fields.level(ᶠz, half))
    z_top = Fields.field_values(Fields.level(ᶠz, n + half))
    p_floor = eps(FT)
    ᶠp_bot .= bot(ᶜp) .+ bot(ᶜρ) .* grav .* (bot(ᶜz) .- z_bot)
    ᶠp_top .= max.(top(ᶜp) .- top(ᶜρ) .* grav .* (z_top .- top(ᶜz)), p_floor)
    return nothing
end

"""
    blocked_layer_start_height!(ᶠz_start, ᶜz_pbl, ᶠz)

Set `ᶠz_start` [m] to the height of the top face of the lowest blocked cell, the cell just
above the PBL-top cell.

`ᶜz_pbl` from `get_pbl_z!` is the center height of the PBL-top cell. The first face above
that center is the top face of the PBL-top cell, which is the bottom face of the lowest
blocked cell; the second face above it is the top face of the lowest blocked cell, and is
the result. This follows GFDL `topo_drag`, whose `get_pbl` sets `kpbl` to the first level
above the PBL and whose phase sum starts at `zhalf(kpbl)`, the top face of that level. Both
faces are found per column with `Operators.column_reduce!`, so the result is exact on
stretched and terrain-following grids.

If the PBL-top cell is the second-highest cell, the result is the model top face, and the
blocked layer is the top cell alone. If it is the top cell, there is no cell above it, the
result is `Inf`, and `calc_nonpropagating_forcing!` adds no blocked drag in that column.
Neither happens in practice, since `get_pbl_z!` only accepts levels with `p ≥ 0.5·p_sfc`.

`ᶠz_start` lives on the bottom face level and `ᶜz_pbl` on the first center level, which are
different spaces with the same layout, so `ᶜz_pbl` is rewrapped on the face level space.
Called from `orographic_gravity_wave_forcing!`.
"""
function blocked_layer_start_height!(ᶠz_start, ᶜz_pbl, ᶠz)
    FT = eltype(ᶠz_start)
    z_pbl_on_faces = Fields.Field(Fields.field_values(ᶜz_pbl), axes(ᶠz_start))
    input = @. lazy(tuple(ᶠz, z_pbl_on_faces))
    # Accumulator: (result, whether the first face above z_pbl has been passed).
    Operators.column_reduce!(
        ᶠz_start,
        input;
        init = (FT(Inf), false),
        transform = first,
    ) do (z_acc, passed_first), (z_face, z_center)
        z_face > z_center || return (z_acc, passed_first)
        passed_first || return (z_acc, true)
        return (ifelse(isinf(z_acc), z_face, z_acc), true)
    end
    return nothing
end

"""
    calc_nonpropagating_forcing!(ᶜuforcing, ᶜvforcing, τ_x, τ_y, τ_l, τ_np, ᶠVτ, ᶠz_start,
                                 ᶠz_ref, ᶠp_ref, ᶜmask, ᶜweights, ᶜdiff, ᶜwtsum,
                                 ᶠp, ᶠN, ᶠz, ᶠdz, grav)

Add the blocked-flow (non-propagating) orographic drag to the forcing fields.

Increments `ᶜuforcing` and `ᶜvforcing` [m/s²] and fills the working fields `ᶠz_ref`,
`ᶠp_ref`, `ᶜmask`, `ᶜweights`, `ᶜdiff`, and `ᶜwtsum`. Called from
`orographic_gravity_wave_forcing!` after `calc_propagate_forcing!`.

The drag is confined to a layer that starts in the first cell above the PBL-top cell and
ends at a reference level `z_ref`, and is distributed within it by pressure weighting,
which concentrates it near the surface. No cell of the PBL gets blocked drag, as in GFDL
`topo_drag`, whose blocked layer runs from `kref` down to `kpbl`, the first level above
the PBL. `z_start` is `ᶠz_start`, the top face of that lowest blocked cell (GFDL
`zhalf(kpbl)`). `z_ref` is the first face above `z_start` at which the accumulated
vertical phase of a stationary hydrostatic wave, `Σ Δz·N/V_τ` over the layers from
`z_start` upward, exceeds `π`, i.e. half a vertical wavelength. Each layer uses `N` and
`V_τ` at its upper face, with `N` clamped to `[0.7e-2, 1.7e-2]` 1/s and `V_τ` floored at
1 m/s so the layer depth stays physical. If the phase never reaches `π`, or `z_start` is
the model top face, `z_ref` is the model top face. If `z_start` is `Inf` (no cell above
the PBL-top cell), `z_ref` is `Inf` and the column gets no blocked drag.

The mask keeps the cells whose top face is at or above `z_start` and whose bottom face is
below `z_ref`, so the lowest blocked cell is always in it, and drops cells of zero weight,
which contribute nothing to the pressure-weighted sum. The tendency is then
`g·τ_x·τ_np/(τ_l·wtsum)·weight` and likewise for `τ_y`, and is set to zero in columns where
the mask is empty. `Δp` is each cell's own pressure thickness, the difference of `ᶠp`
across its two faces, and `wtsum = Σ Δp·weight`, both as in GFDL `topo_drag`, so the
column-integrated blocked stress `Σ (du/dt)·Δp/g` equals `τ_x·τ_np/τ_l` exactly.

# Arguments

  - `τ_x`, `τ_y`: Zonal and meridional components of the base momentum flux, from
    `calc_base_flux!`.
  - `τ_l`, `τ_np`: Corrected linear drag and non-propagating drag, from `calc_base_flux!`.
  - `ᶠVτ`: Wind projected onto the drag direction, at faces [m/s].
  - `ᶠz_start`: Height of the top face of the lowest blocked cell, the cell just above the
    PBL-top cell, on the face level space, from `blocked_layer_start_height!` [m].
  - `ᶠp`: Pressure at faces, from `ogw_face_pressure!` [Pa].
  - `ᶠN`: Buoyancy frequency at faces [1/s].
  - `grav`: Gravitational acceleration [m/s²].

# Notes

A `NaN` in the weight sum is warned about but not treated as fatal.

Described in [garner2005](@cite); see the *Orographic Gravity Waves* page.
"""
function calc_nonpropagating_forcing!(
    ᶜuforcing,
    ᶜvforcing,
    #
    τ_x,
    τ_y,
    τ_l,
    τ_np,
    ᶠVτ,
    ᶠz_start,
    #
    ᶠz_ref,
    ᶠp_ref,
    ᶜmask,
    ᶜweights,
    ᶜdiff,
    ᶜwtsum,
    #
    ᶠp,
    ᶠN,
    ᶠz,
    ᶠdz,
    grav,
)
    FT = eltype(ᶠN)

    # Convert type parameters to values before using in closure
    pi_val = FT(π)
    min_n_val = FT(0.7e-2)
    max_n_val = FT(1.7e-2)
    min_Vτ_val = FT(1.0)

    # z_ref is the first face above z_start where the WKB phase, accumulated over
    # each layer's own thickness, Σ Δz·N/V_τ, as GFDL topo_drag does, exceeds π.
    # Accumulator: (z_ref, phase, started, done). The face at z_start only starts
    # the sum; z_ref then tracks the latest face, so it ends at the model top if the
    # phase never exceeds π, and stays Inf if z_start is Inf.
    input = @. lazy(
        tuple(ᶠz_start, ᶠz, ᶠN, ᶠVτ, pi_val, min_n_val, max_n_val, min_Vτ_val),
    )

    Operators.column_reduce!(
        ᶠz_ref,
        input;
        init = (FT(Inf), FT(0), false, false),
        transform = first,
    ) do (z_ref_acc, phase_acc, started, done),
    (
        z_start,
        z_face,
        N_face,
        Vτ_face,
        pi_val,
        min_n_val,
        max_n_val,
        min_Vτ_val,
    )
        if done || z_face < z_start
            return (z_ref_acc, phase_acc, started, done)
        end
        if !started
            return (z_face, phase_acc, true, false)
        end
        phase_acc +=
            (z_face - z_ref_acc) * max(min_n_val, min(max_n_val, N_face)) /
            max(min_Vτ_val, Vτ_face)
        return (z_face, phase_acc, true, phase_acc > pi_val)
    end

    eps_val = eps(FT)
    half_val = FT(0.5)
    nan_val = FT(NaN)

    input = @. lazy(tuple(ᶠz_ref, ᶠp, ᶠz, ᶠdz, eps_val, half_val))

    Operators.column_reduce!(
        ᶠp_ref,
        input;
        init = nan_val,
    ) do ᶠp_ref, (z_ref, ᶠp, ᶠz, ᶠdz, eps_val, half_val)
        if abs(ᶠz - z_ref) < (half_val * ᶠdz + eps_val)
            if isnan(ᶠp_ref)
                ᶠp_ref = ᶠp
            end
        end
        return ᶠp_ref
    end

    # Blocked cells, from bottom to top:
    #   PBL-top cell:        top face one cell below z_start -> no drag
    #   first cell above it: top face = z_start              -> drag (lowest blocked cell)
    #   ...
    #   cell below z_ref:    top face = z_ref                -> drag (highest blocked cell)
    # So a cell gets drag if its top face is at or above z_start (ᶜtop_bias) and its
    # bottom face is below z_ref (ᶜbottom_bias). z_start is copied from a face height,
    # so the ≥ comparison is exact.
    @. ᶜmask =
        isone(ᶜtop_bias.((ᶠz .>= ᶠz_start))) .&& isone(ᶜbottom_bias.((ᶠz .< ᶠz_ref)))
    @. ᶜweights = ᶜinterp.(ᶠp .- ᶠp_ref)
    # Each cell's own pressure thickness (GFDL topo_drag: dp = phalf(k+1) - phalf(k))
    @. ᶜdiff = ᶜbottom_bias(ᶠp) - ᶜtop_bias(ᶠp)

    # Exclude cells with zero weight: p == p_ref there, so they contribute
    # nothing to the pressure-weighted sum or the forcing anyway.
    @. ᶜmask = ᶜmask && (!iszero(ᶜweights))

    ᶜweights .= ᶜweights .* ᶜmask

    # Column stress sum wtsum = Σ Δp·weight (GFDL topo_drag: `wtsum += dp*weight`).
    # The forcing below divides by wtsum, so the column-integrated blocked stress
    # Σ (du/dt)·Δp/g equals τ_x·τ_np/τ_l.
    input = @. lazy(ifelse(ᶜmask == true, ᶜdiff * ᶜweights, FT(0)))

    Operators.column_reduce!(ᶜwtsum, input; init = FT(0)) do acc, wtsum_field
        return acc + wtsum_field
    end

    if any(isnan.(ᶜwtsum))
        @warn "NaN encountered in weight sum calculation of orographic gravity wave drag"
    end

    # Compute drag, handling empty mask case (wtsum=0) gracefully
    # When wtsum=0, the mask is empty (no cell above the PBL-top cell),
    # so we set forcing to 0 for those columns
    @. ᶜuforcing += ifelse(
        iszero(ᶜwtsum),
        FT(0),
        grav * τ_x * τ_np / τ_l / ᶜwtsum * ᶜweights,
    )
    @. ᶜvforcing += ifelse(
        iszero(ᶜwtsum),
        FT(0),
        grav * τ_y * τ_np / τ_l / ᶜwtsum * ᶜweights,
    )

end

"""
    tofd_drag_coefficient!(ᶜCd, ᶜz, z_sfc, σ²_flt, a_tofd)

Fill `ᶜCd` [m⁻¹] with the turbulent orographic form drag (TOFD) coefficient of
[beljaars2004](@cite).

TOFD represents the form drag of orography with horizontal scales below about 5 km, too small
to launch resolved gravity waves. The deceleration is `∂u/∂t = −C_d(z)·|U|·u` with the IFS
form of the scheme (IFS Documentation CY47R1, Part IV, Eqs. 3.58 to 3.60)

```
C_d(z) = a_tofd · α · β · C_md · C_corr · 2.109 · exp[−(z/1500)^{1.5}] · a₂ · z^{−1.2},
a₂ = σ_flt² · k₁^{n₁−n₂} / (I_H · k_flt^{n₁}),
```

where `z` is the height above the local surface (floored at 1 m so `z^{−1.2}` stays finite)
and `σ_flt²` is the variance of the band-pass-filtered orography from
`compute_filtered_orography_variance`. The constants are those of the IFS:
`α = 35` (the large-eddy-simulation value 12 of Beljaars et al., multiplied by 1.7² in the IFS
tuning), `β = 1`, `C_md = 0.005`, `C_corr = 0.6`, the orography spectrum slopes `n₁ = −1.9`
and `n₂ = −2.8` with the break at `k₁ = 0.003 m⁻¹`, and the filter integral
`I_H = 0.00102 m⁻¹` and representative wavenumber `k_flt = 0.00035 m⁻¹` of the band pass that
defines `σ_flt`. The profile concentrates the drag in the lowest ~1 km, below the PBL-top
layer where the propagating and blocked components act.

`C_d` depends only on the grid and the orography, so it is computed once when the cache is
built; `orographic_gravity_wave_apply_tendency!` combines it with the current wind through
`tofd_implicit_rate`. `a_tofd` scales the IFS amplitude: `1` reproduces the IFS and `0` gives
`C_d ≡ 0`.

# Arguments

  - `ᶜz`: Cell-center height (altitude) [m].
  - `z_sfc`: Terrain surface height, e.g. the bottom face level `Fields.level(ᶠz, half)`
    [m]; it broadcasts up the column, so `z − z_sfc` is height above ground.
  - `σ²_flt`: Variance of the filtered orography on the surface space [m²].
  - `a_tofd`: TOFD amplitude relative to the IFS [-].
"""
function tofd_drag_coefficient!(ᶜCd, ᶜz, z_sfc, σ²_flt, a_tofd)
    FT = eltype(ᶜCd)

    # IFS constants (IFS Documentation CY47R1, Part IV, Eq. 3.58).
    α_tofd = FT(35)        # 12·1.7²
    β_tofd = FT(1)
    C_md = FT(0.005)
    C_corr = FT(0.6)
    I_H = FT(0.00102)      # m⁻¹
    k_flt = FT(0.00035)    # m⁻¹
    k₁ = FT(0.003)         # m⁻¹
    n₁ = FT(-1.9)
    n₂ = FT(-2.8)
    # a₂ / σ_flt², in m^(1+n₂) = m^-1.8
    spectrum_const = k₁^(n₁ - n₂) / (I_H * k_flt^n₁)
    amplitude = α_tofd * β_tofd * C_md * C_corr * FT(2.109) * spectrum_const

    z_decay = FT(1500)     # [m]
    decay_exp = FT(1.5)
    power_exp = FT(-1.2)
    z_floor = FT(1)        # keeps z^-1.2 finite [m]

    # σ²_flt is a surface field and broadcasts up the column.
    z_agl = @. lazy(max(z_floor, ᶜz - z_sfc))
    @. ᶜCd =
        FT(a_tofd) * amplitude * max(FT(0), σ²_flt) *
        exp(-(z_agl / z_decay)^decay_exp) * z_agl^power_exp
    return nothing
end

"""
    tofd_implicit_rate(Cd, speed, dt)

Return the backward-Euler TOFD damping rate `r / (1 + r·dt)` [s⁻¹], with `r = Cd·speed`.

The tendency `−tofd_implicit_rate(Cd, |U|, dt)·u` is the increment of the linearized
implicit update `u⁺ = u / (1 + r·dt)` divided by `dt`. It equals the explicit drag
`−Cd·|U|·u` when `r·dt ≪ 1` and saturates at `−u/dt` when `r·dt ≫ 1`, so one step can
bring the wind to rest but never reverse it, however large `Cd` is.

# Arguments

  - `Cd`: TOFD drag coefficient from `tofd_drag_coefficient!` [m⁻¹].
  - `speed`: Horizontal wind speed `|U|` [m/s].
  - `dt`: Model time step [s].
"""
function tofd_implicit_rate(Cd, speed, dt)
    r = Cd * speed
    return r / (1 + r * dt)
end

"""
    calc_propagate_forcing!(ᶜuforcing, ᶜvforcing, τ_x, τ_y, τ_l, τ_sat, dτ_sat_dz, ᶜρ)

Add the propagating (vertically radiating) orographic drag to the forcing fields.

Decrements `ᶜuforcing` and `ᶜvforcing` [m/s²] by `(τ_x/τ_l)·(1/ρ)·dτ_sat/dz` and the `τ_y`
counterpart, writes the vertical derivative into `dτ_sat_dz`, and returns `nothing`. The
loss of saturated flux with height is exactly the momentum the breaking wave returns to the
flow, hence the minus sign.

The tendency vanishes below the PBL top, where `calc_saturation_profile!` holds `τ_sat`
constant at `τ_p`. Called from `orographic_gravity_wave_forcing!`; `dτ_sat_dz` is a scratch
field.
"""
function calc_propagate_forcing!(
    ᶜuforcing,
    ᶜvforcing,
    τ_x,
    τ_y,
    τ_l,
    τ_sat,
    dτ_sat_dz,
    ᶜρ,
)
    dτ_sat_dz .=
        Geometry.WVector.(ᶜgradᵥ.(τ_sat)).components.data.:1

    @. ᶜuforcing -= τ_x / τ_l / ᶜρ * dτ_sat_dz
    @. ᶜvforcing -= τ_y / τ_l / ᶜρ * dτ_sat_dz
    return nothing
end

"""
    get_pbl_z!(result, ᶜp, ᶜT, ᶜz, grav, cp_d)

Compute the planetary boundary layer (PBL) top height of each column.

Mutates `result`, setting it to the height of the highest cell center satisfying both

 1. `p ≥ 0.5·p_sfc`, which restricts the search to the lower atmosphere, and
 2. `T_sfc + T_boost − T > (g/c_pd)·(z − z_sfc)` with `T_boost = 1.5` K,

where the surface values are those of the lowest cell center. Within the well-mixed
boundary layer, turbulent mixing keeps the profile at or steeper than the dry adiabat, so
the second inequality holds; it fails in the stably stratified free atmosphere above, which
marks the transition. The temperature boost improves the estimate near the surface. The
lowest cell center always qualifies (its left side is `T_boost > 0` and its right side is
zero), so if no higher level qualifies, `result` is the lowest cell center height, and the
blocked drag starts in the second cell, as GFDL `topo_drag` does with its fallback
`kpbl = kdim - 1`.

Called from `orographic_gravity_wave_forcing!`. Implemented with
`Operators.column_reduce!` so it runs on the GPU.

# Arguments

  - `result`: Surface-level output field for the PBL top height [m]; overwritten.
  - `ᶜp`: Pressure at cell centers [Pa].
  - `ᶜT`: Temperature at cell centers [K].
  - `ᶜz`: Geometric height of cell centers [m].
  - `grav`: Gravitational acceleration [m/s²].
  - `cp_d`: Isobaric specific heat of dry air [J/(kg·K)].

See the *Orographic Gravity Waves* page.
"""
function get_pbl_z!(result, ᶜp, ᶜT, ᶜz, grav, cp_d)
    FT = eltype(ᶜp)

    # Get surface values (first level values)
    p_sfc = Fields.level(ᶜp, 1)
    T_sfc = Fields.level(ᶜT, 1)
    z_sfc = Fields.level(ᶜz, 1)

    half_val = FT(0.5)
    temp_offset = FT(1.5)
    grav_val = FT(grav)
    cp_d_val = FT(cp_d)
    zero_val = FT(0)

    # Create a lazy tuple of inputs for column_reduce
    input = @. lazy(
        tuple(
            ᶜp,
            ᶜT,
            ᶜz,
            p_sfc,
            T_sfc,
            z_sfc,
            grav_val,
            cp_d_val,
            half_val,
            temp_offset,
            zero_val,
        ),
    )

    # Perform the column reduction
    Operators.column_reduce!(
        result,
        input;
        init = FT(0),
        transform = first, # Extract just the z_pbl value
    ) do z_pbl,
    (
        p_col,
        T_col,
        z_col,
        p_sfc,
        T_sfc,
        z_sfc,
        grav_val,
        cp_d_val,
        half_val,
        temp_offset,
        zero_val,
    )

        if z_pbl == zero_val
            z_pbl = z_sfc
        end
        # Check conditions
        p_threshold = p_col >= (half_val * p_sfc)
        T_threshold =
            (T_sfc + temp_offset - T_col) >
            (grav_val / cp_d_val * (z_col - z_sfc))

        # If both conditions are met, update z_pbl to current height
        if p_threshold && T_threshold
            z_pbl = z_col
        end

        # Move to next level
        return z_pbl
    end
end



"""
    calc_base_flux!(τ_x, τ_y, τ_l, τ_p, τ_np, U_sat, FrU_sat, FrU_clp, FrU_max, FrU_min,
                    z_pbl, values_at_z_pbl, ogw_params, topo_info,
                    ᶜρ, u_phy, v_phy, ᶜz, ᶜN)

Compute the base momentum flux at the PBL top and split it into drag components.

Mutates all of `τ_x`, `τ_y`, `τ_l`, `τ_p`, `τ_np`, `U_sat`, `FrU_sat`, `FrU_clp`,
`FrU_max`, `FrU_min`, and the working tuple field `values_at_z_pbl`, and returns `nothing`.
Called from `orographic_gravity_wave_forcing!`.

The low-level flow is sampled at the source level, the highest cell center with
`z ≤ z_pbl`, giving `(ρ_pbl, u_pbl, v_pbl, N_pbl)`. The linear base flux is
`τ = ρ_pbl·N_pbl·⟨T⟩ᵀ·V_pbl`, evaluated componentwise as
`τ_x = ρ_pbl·N_pbl·(t11·u_pbl + t21·v_pbl)` and
`τ_y = ρ_pbl·N_pbl·(t12·u_pbl + t22·v_pbl)`.

`V_τ` is the low-level wind projected onto the drag direction, and the saturation velocity
is `U_sat = sqrt(ρ_pbl/ρ_scale · V_τ³/(N_pbl·L₀))`, the largest wave amplitude the flow can
carry before breaking. Together with the obstacle Froude numbers
`Fr_{max,min} = max(0, h_{max,min})·N_pbl/V_τ` it defines the `FrU` limits, `FrU_clp`
marking the saturation point.

The drag is then obtained by integrating an assumed power-law distribution of subgrid
obstacle heights: `τ_l` over the whole range, `τ_p` over the unsaturated part (linear waves
that propagate), and `τ_np` over the saturated part (blocked flow forced around the
obstacles). Finally `τ_np` is divided by `max(Fr_crit, Fr_max)`, converting it to drag per
unit blocked depth, with `Fr_crit` as a floor so short mountains, which block nothing, do
not blow the division up.

# Arguments

  - `z_pbl`: PBL top height per column [m].
  - `values_at_z_pbl`: Four-slot tuple field holding `(ρ, u, v, N)` at the source level.
  - `ogw_params`: Shape and scale parameters `Fr_crit`, `ρscale`, `L0`, `a0`, `a1`, `γ`, `β`,
    `ϵ`.
  - `topo_info`: Orographic input `hmax`, `hmin`, and the tensor components.
  - `ᶜN`: Buoyancy frequency at cell centers [1/s].

Described in [garner2005](@cite); see the *Orographic Gravity Waves* page for the
integrals.
"""
function calc_base_flux!(
    τ_x,
    τ_y,
    τ_l,
    τ_p,
    τ_np,
    #
    U_sat,
    FrU_sat,
    FrU_clp,
    FrU_max,
    FrU_min,
    z_pbl,
    #
    values_at_z_pbl,
    #
    ogw_params,
    topo_info,
    #
    ᶜρ,
    u_phy,
    v_phy,
    ᶜz,
    ᶜN,
)
    (;
        Fr_crit,
        topo_ρscale,
        topo_L0,
        topo_a0,
        topo_a1,
        topo_γ,
        topo_β,
        topo_ϵ,
    ) = ogw_params
    (; hmax, hmin, t11, t12, t21, t22) = topo_info

    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ

    input = @. lazy(tuple(ᶜρ, u_phy, v_phy, ᶜN, ᶜz, z_pbl))

    Operators.column_reduce!(
        values_at_z_pbl,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), FT(0.0)),
    ) do (ρ_acc, u_acc, v_acc, N_acc), (ρ, u, v, N, z_col, z_target)

        # Check if current level height is at or above z_pbl
        # Use the last valid level that satisfies z_col <= z_target
        if z_col <= z_target
            return (ρ, u, v, N)
        else
            return (ρ_acc, u_acc, v_acc, N_acc)
        end
    end

    # These are views
    ρ_pbl = values_at_z_pbl.:1
    u_pbl = values_at_z_pbl.:2
    v_pbl = values_at_z_pbl.:3
    N_pbl = values_at_z_pbl.:4

    # Calculate τ components
    @. τ_x = ρ_pbl * N_pbl * (t11 * u_pbl + t21 * v_pbl)
    @. τ_y = ρ_pbl * N_pbl * (t12 * u_pbl + t22 * v_pbl)

    # Calculate Vτ using field operations
    Vτ = @. lazy(
        max(
            eps(FT),
            -(u_pbl * τ_x + v_pbl * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2)),
        ),
    )

    # Calculate Froude numbers
    Fr_max = @. lazy(max(FT(0), hmax) * N_pbl / Vτ)
    Fr_min = @. lazy(max(FT(0), hmin) * N_pbl / Vτ)

    # Calculate U_sat
    @. U_sat = sqrt(ρ_pbl / topo_ρscale * Vτ^3 / N_pbl / topo_L0)

    # Calculate FrU values
    @. FrU_sat = Fr_crit * U_sat
    @. FrU_min = Fr_min * U_sat
    @. FrU_max = max(Fr_max * U_sat, FrU_min + eps(FT))
    @. FrU_clp = min(FrU_max, max(FrU_min, FrU_sat))

    # Calculate drag components
    @. τ_l = ((FrU_max)^(2 + γ - ϵ) - (FrU_min)^(2 + γ - ϵ)) / (2 + γ - ϵ)

    # Calculate propagating drag
    @. τ_p =
        topo_a0 * (
            (FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) / (2 + γ - ϵ) +
            FrU_sat^(β + 2) * (FrU_max^(γ - ϵ - β) - FrU_clp^(γ - ϵ - β)) /
            (γ - ϵ - β)
        )

    # Calculate non-propagating drag
    @. τ_np =
        topo_a1 * U_sat / (1 + β) * (
            (FrU_max^(1 + γ - ϵ) - FrU_clp^(1 + γ - ϵ)) / (1 + γ - ϵ) -
            FrU_sat^(β + 1) * (FrU_max^(γ - ϵ - β) - FrU_clp^(γ - ϵ - β)) /
            (γ - ϵ - β)
        )

    # Apply scaling
    @. τ_np = τ_np / max(Fr_crit, Fr_max)

    return nothing
end

"""
    calc_saturation_profile!(ᶠτ_sat, ᶠVτ, U_sat, FrU_sat, FrU_clp, FrU_max, FrU_min,
                             ᶜτ_sat, τ_x, τ_y, τ_p, z_pbl, ogw_params,
                             ᶜρ, u_phy, v_phy, ᶜp, ᶜN, ᶜz)

Compute the vertical profile of saturated momentum flux for the propagating component.

Mutates `ᶜτ_sat`, `ᶠτ_sat`, and `ᶠVτ`, and returns `nothing`. Only the propagating part
needs a profile; the non-propagating part stays the scalar `τ_np` and is distributed by
pressure weighting in `calc_nonpropagating_forcing!`. Called from
`orographic_gravity_wave_forcing!`.

At each cell center `V_τ` is the wind projected onto the drag direction and `L₁` is an
effective obstacle width, `L₀` rescaled by the flow curvature
`1 − 2·V_τ·d²V_τ/N²` and clamped to `[0.5, 2]·L₀`. A column accumulator then carries the
saturation velocity upward as
`U_sat[k] = min(U_sat[k-1], sqrt(ρ/ρ_scale · V_τ³/(N·L₁)))`, seeded at the lowest level
with the base-flux `U_sat`. The `min` makes `U_sat` monotonically non-increasing: a wave
can lose flux to breaking but never recover it.

Below the PBL top (`z ≤ z_pbl`) the profile is held at the base-flux value `τ_p`, so its
derivative vanishes there and the propagating tendency is confined to the free atmosphere.
Above it, the same obstacle-distribution integral as `τ_p` is re-evaluated with the lowered
breaking line `FrU_sat = Fr_crit·U_sat[k]`, the launch-level values `FrU_sat0`, `FrU_clp0`
being retained from `calc_base_flux!`.

If the wave reaches the model top without breaking (`τ_sat[end] > 0`), the residual is
removed from the whole column with a pressure weighting `(p_sfc − p)/(p_sfc − p_top)`, so
momentum is conserved; the correction is skipped when `τ_sat[end] ≤ 0`. Finally `τ_sat` and
`V_τ` are interpolated to faces for the routines that need them there.

Described in [garner2005](@cite); see the *Orographic Gravity Waves* page for the
integrals.
"""
function calc_saturation_profile!(
    ᶠτ_sat,
    ᶠVτ,
    #
    U_sat,
    FrU_sat,
    FrU_clp,
    FrU_max,
    FrU_min,
    ᶜτ_sat,
    τ_x,
    τ_y,
    τ_p,
    z_pbl,
    #
    ogw_params,
    #
    ᶜρ,
    u_phy,
    v_phy,
    ᶜp,
    ᶜN,
    ᶜz,
)
    # Extract parameters from tuple
    (; Fr_crit, topo_ρscale, topo_L0, topo_a0, topo_γ, topo_β, topo_ϵ) =
        ogw_params

    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ

    # Calculate Vτ at cell centers using field operations
    ᶜVτ = @. lazy(
        max(
            eps(FT),
            (-(u_phy * τ_x + v_phy * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2))),
        ),
    )

    # Second vertical derivatives of the wind, at cell centers
    d2udz = lazy.(ᶜd2dz2(u_phy))
    d2vdz = lazy.(ᶜd2dz2(v_phy))
    # Project them onto the drag direction, as for Vτ above; this feeds L1
    d2Vτdz = @. lazy(
        max(
            eps(FT),
            -(d2udz * τ_x + d2vdz * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2)),
        ),
    )

    # Effective obstacle width L1: L0 rescaled by the flow curvature, clamped
    L1 = @. lazy(
        topo_L0 *
        max(FT(0.5), min(FT(2.0), FT(1.0) - FT(2.0) * ᶜVτ * d2Vτdz / ᶜN^2)),
    )

    # Local saturation-velocity ceiling at each cell center
    U_k_field = @. lazy(sqrt(ᶜρ / topo_ρscale * ᶜVτ^3 / ᶜN / L1))

    z_surf = Fields.level(ᶜz, 1)
    # Create combined input for column_accumulate
    input = @. lazy(
        tuple(
            FrU_clp,
            FrU_sat,
            U_k_field,
            FrU_max,
            FrU_min,
            Fr_crit,
            z_surf,
            ᶜz,
            z_pbl,
            topo_a0,
            τ_p,
            U_sat,
        ),
    )

    # Zero the result field before the accumulator overwrites it level by level
    fill!(ᶜτ_sat, 0.0)

    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), FT(0.0)),
        transform = first,
    ) do (tau_sat_val, U_sat_val, local_FrU_sat, local_FrU_clp),
    (
        FrU_clp0,
        FrU_sat0,
        U,
        FrU_max,
        FrU_min,
        Fr_crit_val,
        z_surf,
        z_col,
        z_target,
        topo_a0,
        τ_p,
        U_sat,
    )

        if z_col == z_surf
            U_sat_val = U_sat
        end

        U_sat_val = min(U_sat_val, U)
        local_FrU_sat = Fr_crit_val * U_sat_val  # Use local variable instead
        local_FrU_clp = min(FrU_max, max(FrU_min, local_FrU_sat))  # Use local variable instead

        if z_col <= z_target
            tau_sat_val = τ_p
        else
            tau_sat_val =
                topo_a0 * (
                    (local_FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) /
                    (2 + γ - ϵ) +
                    local_FrU_sat^2 *
                    FrU_sat0^β *
                    (FrU_max^(γ - ϵ - β) - FrU_clp0^(γ - ϵ - β)) / (γ - ϵ - β) +
                    local_FrU_sat^2 *
                    (FrU_clp0^(γ - ϵ) - local_FrU_clp^(γ - ϵ)) / (γ - ϵ)
                )
        end

        return (tau_sat_val, U_sat_val, local_FrU_sat, local_FrU_clp)
    end

    top_values = Fields.level(ᶜτ_sat, Spaces.nlevels(axes(ᶜτ_sat)))
    p_surf = Fields.level(ᶜp, 1)
    p_top = Fields.level(ᶜp, Spaces.nlevels(axes(ᶜp)))

    zero_val = FT(0.0)

    input = @. lazy(tuple(top_values, ᶜτ_sat, p_surf, p_top, ᶜp, zero_val))

    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = FT(0.0),
        transform = identity,
    ) do τ_sat_val, (top_values, ᶜτ_sat, p_surf, p_top, ᶜp, zero_val)

        τ_sat_val = ᶜτ_sat

        if top_values > zero_val
            τ_sat_val -= (top_values * (p_surf - ᶜp) / (p_surf - p_top))
        end

        return τ_sat_val
    end

    @. ᶠτ_sat = ᶠinterp(ᶜτ_sat)
    @. ᶠVτ = ᶠinterp(ᶜVτ)

    return nothing
end


"""
    compute_ogw_drag(Y, earth_radius, topography, h_frac)

Build the orographic drag input for the configured topography.

Returns `(; hmax, hmin, t11, t21, t12, t22)` on the surface cell-center space. Called from
`get_topo_info` for `topo_info = Val(:raw_topo)`.

For Earth topography (`Val(:Earth)` or `Val(:NoWarp)`) the drag was computed offline by the
preprocessing pipeline, so this loads it: first a local
`computed_drag_Earth_false_1_<h_elem>.hdf5` if present, otherwise the matching
`ogw_computed_drag_h*` ClimaArtifact for the resolution.

For the analytical test topographies (`DCMIP200`, `Hughes2023`, `Agnesi`, `Schar`,
`Cosine2d`, `Cosine3d`) the tensor is instead computed at startup with ClimaCore horizontal
gradient operators: `hmax` is the elevation above the surface, `hmin = h_frac·hmax`, the
velocity potential is approximated as `χ = hmax·A_cell·R/(2π)` with `A_cell` the bottom cell
area, and `tᵢⱼ = −∂χ/∂xⱼ · ∂h/∂xᵢ`. The gradient of `χ` is negated so the drag opposes the
low-level flow, matching the offline pipeline and [garner2005](@cite) Eq. 6; without it the
tensor would carry the wrong sign and accelerate the flow. The drag vector is zeroed south
of 88°S, where the grid convergence makes the horizontal gradients unreliable.

# Arguments

  - `Y`: Prognostic state, used only for its spaces and communications context.
  - `earth_radius`: Sphere radius [m].
  - `topography`: `Val` of the configured topography name.
  - `h_frac`: Ratio of the minimum to the maximum obstacle height [-].

# Notes

The analytical-topography path is not yet covered by tests.
"""
function compute_ogw_drag(
    Y,
    earth_radius,
    topography,
    h_frac,
)
    FT = eltype(Y)
    center_space = axes(Y.c)
    h_elem = Spaces.n_elements_per_panel_direction(center_space)
    face_space = axes(Y.f)
    ᶜsurface_space = Fields.level(center_space, 1)
    J_bot = Fields.level(Fields.local_geometry_field(face_space).J, half)
    Δz_bot = Fields.level(Fields.Δz_field(face_space), half)
    cell_area_bot = @. J_bot / Δz_bot

    z_surface = Fields.level(Fields.coordinate_field(Y.f).z, half)

    cg_lat = Fields.level(Fields.coordinate_field(Y.f).lat, half)

    if topography == Val(:Earth) || topography == Val(:NoWarp)
        # Try local file first (for development when preprocessing has been run)
        local_filename = "computed_drag_Earth_false_1_$(h_elem)"
        local_path = joinpath(pkgdir(@__MODULE__), "$(local_filename).hdf5")

        if isfile(local_path)
            @info "Loading computed drag from local file: $(local_path)"
            topo_info = load_preprocessed_topography(
                local_filename,
                ClimaComms.context(Y.c),
            )

            @debug begin
                # Checkpoint 1: Validate loaded drag tensor
                @info "OGWD drag tensor loaded from LOCAL FILE (h_elem=$(h_elem)):"
                @info "  hmax: min=$(minimum(parent(topo_info.hmax))), max=$(maximum(parent(topo_info.hmax))), mean=$(sum(parent(topo_info.hmax))/length(parent(topo_info.hmax)))"
                @info "  t11: min=$(minimum(parent(topo_info.t11))), max=$(maximum(parent(topo_info.t11)))"
                @info "  NaN/Inf: hmax_nan=$(any(isnan, parent(topo_info.hmax))), t11_inf=$(any(isinf, parent(topo_info.t11)))"
            end
        else
            # Fall back to ClimaArtifacts
            @info "Local file not found, loading from ClimaArtifacts (h_elem=$(h_elem))..."
            artifact_path =
                AA.ogw_computed_drag_file_path(; h_elem, context = ClimaComms.context(Y.c))
            @info "Loading from: $(artifact_path)"
            reader = InputOutput.HDF5Reader(artifact_path, ClimaComms.context(Y.c))
            topo_info = InputOutput.read_field(reader, "computed_drag")
            Base.close(reader)

            @debug begin
                # Checkpoint 1: Validate loaded drag tensor
                @info "OGWD drag tensor loaded from ARTIFACT (h_elem=$(h_elem)):"
                @info "  hmax: min=$(minimum(topo_info.hmax)), max=$(maximum(topo_info.hmax)), mean=$(sum(parent(topo_info.hmax))/length(parent(topo_info.hmax)))"
                @info "  t11: min=$(minimum(topo_info.t11)), max=$(maximum(topo_info.t11))"
                @info "  NaN/Inf: hmax_nan=$(any(isnan, topo_info.hmax)), t11_inf=$(any(isinf, topo_info.t11))"
            end
        end

        return set_topo_info_target_space(topo_info, ᶜsurface_space)

        ### Handle analytical test cases
        # NOTE: OGW for analytical topography cases is not yet tested.
    elseif topography == Val(:DCMIP200)
        topography_function = topography_dcmip200
    elseif topography == Val(:Hughes2023)
        topography_function = topography_hughes2023
    elseif topography == Val(:Agnesi)
        topography_function = topography_agnesi
    elseif topography == Val(:Schar)
        topography_function = topography_schar
    elseif topography == Val(:Cosine2d)
        topography_function = topography_cosine_2d
    elseif topography == Val(:Cosine3d)
        topography_function = topography_cosine_3d
    else
        error("Topography required for orographic gravity wave drag: $topography")
    end

    real_elev = SpaceVaryingInput(topography_function, face_space)
    real_elev = Fields.level(real_elev, half)
    @. real_elev = max(0, real_elev)

    hmax = @. real_elev - z_surface
    hmin = @. h_frac * hmax

    χ = @. hmax * cell_area_bot * earth_radius / (FT(2) * FT(pi))

    ∇ₕχ = @. Geometry.UVVector(gradₕ(χ))
    ∇ₕhmax = @. Geometry.UVVector(gradₕ(hmax))

    # Negate the velocity-potential gradient so the drag opposes the low-level
    # flow, matching the offline pipeline (calc_orographic_tensor uses
    # `.-calc_∇A(χ, …)`) and Garner (2005) Eq. 6/8. Without this the analytical
    # topography tensor would carry the wrong sign and accelerate the flow.
    dχdx = @. -∇ₕχ.components.data.:1
    dχdy = @. -∇ₕχ.components.data.:2

    dhdx = ∇ₕhmax.components.data.:1
    dhdy = ∇ₕhmax.components.data.:2

    # Handle drag vector elements at the antarctic region
    @. dχdx = ifelse(cg_lat < FT(-88), 0, dχdx)
    @. dχdy = ifelse(cg_lat < FT(-88), 0, dχdy)

    # We convert the face-centered drag vector elements to cell-centered
    # quantities as these are used to compute the physics associated with the
    # orographic gravity wave drag in the cell.
    hmax = Fields.Field(Fields.field_values(hmax), ᶜsurface_space)
    hmin = Fields.Field(Fields.field_values(hmin), ᶜsurface_space)
    t11 = Fields.Field(Fields.field_values(dχdx .* dhdx), ᶜsurface_space)
    t21 = Fields.Field(Fields.field_values(dχdx .* dhdy), ᶜsurface_space)
    t12 = Fields.Field(Fields.field_values(dχdy .* dhdx), ᶜsurface_space)
    t22 = Fields.Field(Fields.field_values(dχdy .* dhdy), ᶜsurface_space)

    return (; hmax, hmin, t11, t21, t12, t22)

end

"""
    compute_ogw_drag_online(Y, earth_radius, topography, γ, h_frac, α_smoothing)

Build the orographic drag input `(; hmax, hmin, t11, t12, t21, t22)` at initialization by
running the preprocessing pipeline directly, rather than reading a stored artifact.

For Earth topography (`Val(:Earth)` or `Val(:NoWarp)`) this calls `compute_OGW_info` on the
raw ETOPO elevation (`AA.earth_orography_file_path`), so the result reflects the run's
resolution together with `γ`, `h_frac`, and `α_smoothing`. The neighborhood statistics are
computed on the host and regridded onto the model surface space with the GPU-capable
`SpaceVaryingInput`, so both CPU and GPU runs are supported at a one-time initialization
cost.

For the analytical test topographies the computation already lives in `compute_ogw_drag`,
so this delegates to it.

# Arguments

  - `Y`: Prognostic state, used only for its spaces.
  - `earth_radius`: Sphere radius [m].
  - `topography`: `Val` of the configured topography name.
  - `γ`: Mountain height-width exponent [-].
  - `h_frac`: Ratio of the minimum to the maximum obstacle height [-].
  - `α_smoothing`: Smoothing scale as a fraction of the model grid spacing [-].
"""
function compute_ogw_drag_online(
    Y,
    earth_radius,
    topography,
    γ,
    h_frac,
    α_smoothing,
)
    if topography == Val(:Earth) || topography == Val(:NoWarp)
        @info "Computing orographic gravity wave drag input at initialization (raw_topo_online)" α_smoothing
        elevation_data =
            AA.earth_orography_file_path(context = ClimaComms.context(Y.c))
        return compute_OGW_info(
            Y,
            elevation_data,
            earth_radius,
            γ,
            h_frac;
            α_smoothing,
        )
    end
    return compute_ogw_drag(Y, earth_radius, topography, h_frac)
end


"""
    ᶜd2dz2(ᶜscalar)

Return the lazy second vertical derivative of a center-valued scalar, at cell centers.

Composed as `ᶜddz ∘ ᶠddz`, so the intermediate first derivative lives at faces and the
stencil is compact. Used by `calc_saturation_profile!` for the flow curvature.
"""
ᶜd2dz2(ᶜscalar) =
    lazy.(Geometry.WVector.(ᶜgradᵥ.(ᶠddz(ᶜscalar))).components.data.:1)

"""
    ᶜddz(ᶠscalar)

Return the lazy vertical derivative of a face-valued scalar, at cell centers [·/m].
"""
ᶜddz(ᶠscalar) = lazy.(Geometry.WVector.(ᶜgradᵥ.(ᶠscalar)).components.data.:1)

"""
    ᶠddz(ᶜscalar)

Return the lazy vertical derivative of a center-valued scalar, at cell faces [·/m].
"""
ᶠddz(ᶜscalar) = lazy.(Geometry.WVector.(ᶠgradᵥ.(ᶜscalar)).components.data.:1)
