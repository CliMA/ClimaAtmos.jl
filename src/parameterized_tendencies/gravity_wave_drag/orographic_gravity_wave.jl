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
  - `Val(:linear)`: user-defined analytical drag input, for idealized tests.

Any other value is an error. Called from `orographic_gravity_wave_cache`.

# Returns

A `NamedTuple` of surface-level `Field`s: the effective maximum and minimum subgrid
obstacle heights `hmax`, `hmin` [m], and the four components `t11`, `t12`, `t21`, `t22` of
the orographic tensor `T = −∇χ (∇h)ᵀ`, stored with `tᵢⱼ = −∂χ/∂xⱼ · ∂h/∂xᵢ`.
"""
function get_topo_info(Y, ogw::OrographicGravityWave)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations

    if ogw.topo_info == Val(:gfdl_restart)
        topo_path = @clima_artifact("topo_drag", ClimaComms.context(Y.c))
        orographic_info_rll = joinpath(topo_path, "topo_drag.res.nc")
        topo_info = regrid_OGW_info(Y, orographic_info_rll)
    elseif ogw.topo_info == Val(:raw_topo)
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
    elseif ogw.topo_info == Val(:linear)
        # For user-defined analytical tests
        topo_info = initialize_drag_input_as_fields(Y, ogw.drag_input)
    else
        error("topo_info must be a symbol of type gfdl_restart, raw_topo, or linear")
    end

    return topo_info

end

function orographic_gravity_wave_cache(Y, ogw::OrographicGravityWave, topo_info = nothing)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations
    @assert Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain isa
            Domains.SphereDomain

    FT = Spaces.undertype(axes(Y.c))
    (; γ, ϵ, β, ρscale, L0, a0, a1, Fr_crit) = ogw

    if topo_info === nothing
        topo_info = get_topo_info(Y, ogw)
    end

    center_space, face_space = axes(Y.c), axes(Y.f)

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
        topo_ᶠz_pbl = similar(Fields.level(Y.f.u₃, half)),
        values_at_z_pbl = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT}),
        topo_info = topo_info,
        ᶜbuoyancy_frequency = Fields.Field(FT, center_space),
        ᶠbuoyancy_frequency = Fields.Field(FT, face_space),
        ᶜuforcing = zero(Y.c.ρ),
        ᶜvforcing = zero(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
        ᶠp_m1 = Fields.Field(FT, face_space),
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
`orographic_gravity_wave_apply_tendency!`, applies the cached forcing every integrator
step. The `::Nothing` method is a no-op.

Prepares the inputs the forcing routine needs and then calls
`orographic_gravity_wave_forcing!`:

  - The buoyancy frequency `N = sqrt(g/T · (dT/dz + g/c_pm))`, using the moist isobaric
    specific heat, at cell centers and interpolated to faces, floored at `sqrt(eps(FT))`.
  - Face pressure `ᶠp` and its one-level-down shift `ᶠp_m1`, needed for the layer
    thicknesses in the non-propagating drag. The value below the bottom face is
    extrapolated with the barometric formula, using a scale height diagnosed from the two
    lowest faces.
  - The physical horizontal wind components from `Y.c.uₕ`.

Reads `p.precomputed.ᶜp`, `ᶜT`, `ᶜq_tot_nonneg`, `ᶜq_liq`, `ᶜq_ice`, and clobbers
`p.scratch.ᶠtemp_scalar`, `ᶠtemp_field_level`, and `temp_data_face_level`. Described in
[garner2005](@cite); see the *Orographic Gravity Waves* page for the derivations.
"""
orographic_gravity_wave_compute_tendency!(Y, p, ::Nothing) = nothing

function orographic_gravity_wave_compute_tendency!(Y, p, ::FullOrographicGravityWave)
    @debug begin
        if !hasfield(typeof(p), :ogwd_call_counter)
            @info "OGWD tendency function called for the first time"
        end

        # DEBUG: Check if Y has NaNs at entry
        if any(isnan, parent(Y.c.ρ))
            @error "OGWD: Input Y.c.ρ already has NaNs at function entry!"
            error("Cannot compute OGWD tendency with NaN inputs")
        end
    end

    # unpack cache
    (; ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; params) = p
    (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave
    (; ᶜdTdz) = p.orographic_gravity_wave
    (; ᶠp_m1) = p.orographic_gravity_wave
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

    # compute ᶠp and ᶠp_m1
    # load array from scratch
    ᶠp = p.scratch.ᶠtemp_scalar
    @. ᶠp = ᶠinterp(ᶜp)
    scale_height_values = p.scratch.ᶠtemp_field_level
    z_extrapolated_values = p.scratch.temp_data_face_level

    # explicit scale height approach for pressure extrapolation
    # Fields.level returns by reference
    ᶠz_bottom = Fields.level(ᶠz, half)
    ᶠz_second = Fields.level(ᶠz, 1 + half)
    ᶠp_bottom = Fields.level(ᶠp, half)
    ᶠp_second = Fields.level(ᶠp, 1 + half)

    # Calculate scale height from the two levels
    Fields.field_values(scale_height_values) .=
        (Fields.field_values(ᶠz_second) .- Fields.field_values(ᶠz_bottom)) ./
        log.(Fields.field_values(ᶠp_bottom) ./ Fields.field_values(ᶠp_second))

    # Calculate the extrapolated height (one level below bottom)
    z_extrapolated_values .=
        Fields.field_values(ᶠz_bottom) .-
        (Fields.field_values(ᶠz_second) .- Fields.field_values(ᶠz_bottom))

    # Extrapolate pressure using barometric formula: p = p₀ * exp(-z/H)
    Boundary_value = Fields.Field(
        Fields.field_values(ᶠp_bottom) .*
        exp.(
            (z_extrapolated_values .- Fields.field_values(ᶠz_bottom)) ./
            Fields.field_values(scale_height_values),
        ),
        axes(ᶠp_bottom),
    )

    field_shiftface_down!(ᶠp, ᶠp_m1, Boundary_value)

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
        ᶠp_m1,
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
    orographic_gravity_wave_apply_tendency!(Yₜ, p, ::Nothing)
    orographic_gravity_wave_apply_tendency!(Yₜ, p, ::OrographicGravityWave)

Add the cached orographic gravity wave drag to the horizontal momentum tendency.

Mutates `Yₜ.c.uₕ` by adding the covariant form of the cached `ᶜuforcing`/`ᶜvforcing`
[m/s²]. The `::Nothing` method is a no-op.

This runs every integrator step, while the forcing itself is refreshed only on the `dt_ogw`
callback by `orographic_gravity_wave_compute_tendency!`, which also clamps it to
±3e-3 m/s²; between callbacks the same forcing is reapplied.
"""
orographic_gravity_wave_apply_tendency!(Yₜ, p, ::Nothing) = nothing

function orographic_gravity_wave_apply_tendency!(
    Yₜ,
    p,
    ::OrographicGravityWave,
)
    (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

    @. Yₜ.c.uₕ +=
        C12.(Geometry.UVVector.(ᶜuforcing, ᶜvforcing))

end


"""
    orographic_gravity_wave_forcing!(u_phy, v_phy, ᶜbuoyancy_frequency,
                                     ᶠbuoyancy_frequency, ᶜz, ᶠz, ᶠdz, ᶜuforcing,
                                     ᶜvforcing, ᶜρ, ᶜp, ᶠp, ᶠp_m1, ᶜT, grav, cp_d, p)

Assemble the propagating and non-propagating orographic drag into the forcing fields.

Mutates `ᶜuforcing` and `ᶜvforcing` [m/s²], which the caller has zeroed, along with the
intermediate OGW cache fields. Called from
`orographic_gravity_wave_compute_tendency!`.

Steps, in order:

 1. `get_pbl_z!` finds the PBL top `z_pbl`, which is copied to a face field (shifted down
    half a cell) because `calc_nonpropagating_forcing!` needs it there on the GPU.
 2. `calc_base_flux!` evaluates the low-level flow at `z_pbl` and splits the base
    momentum flux into the linear, propagating, and non-propagating parts `τ_l`, `τ_p`,
    `τ_np`.
 3. `calc_saturation_profile!` marches the propagating flux `τ_sat(z)` upward.
 4. `calc_propagate_forcing!` converts `dτ_sat/dz` into a tendency, and
    `calc_nonpropagating_forcing!` adds the blocked-flow tendency between `z_pbl` and the
    reference level.
 5. Both components are clamped to ±3e-3 m/s² so a large tendency cannot destabilize the
    integrator.

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
    ᶠp_m1,
    ᶜT,
    grav,
    cp_d,
    p,
)

    FT = eltype(ᶠbuoyancy_frequency)
    Δz_bot = Fields.level(ᶠdz, half)

    (; topo_ᶜz_pbl, topo_ᶠz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
        p.orographic_gravity_wave
    (; topo_ᶜτ_sat, topo_ᶠτ_sat) = p.orographic_gravity_wave
    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; topo_ᶠVτ, values_at_z_pbl, topo_info) = p.orographic_gravity_wave
    (; ᶜmask, ᶠp_ref) = p.orographic_gravity_wave

    # Extract parameters
    ogw_params = p.orographic_gravity_wave.ogw_params

    # we copy the z_pbl from a cell-centered to face array.
    # the z-values don't change, but this is necessary for
    # calc_nonpropagating_forcing! to work on the GPU
    get_pbl_z!(topo_ᶜz_pbl, ᶜp, ᶜT, ᶜz, grav, cp_d)
    parent(topo_ᶠz_pbl) .= parent(topo_ᶜz_pbl) .- FT(1 / 2) .* parent(Δz_bot)
    topo_ᶠz_pbl = topo_ᶠz_pbl.components.data.:1

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
        topo_ᶠz_pbl,
        #
        ᶠz_ref,
        ᶠp_ref,
        ᶜmask,
        ᶜweights,
        ᶜdiff,
        ᶜwtsum,
        #
        ᶠp,
        ᶠp_m1,
        ᶠbuoyancy_frequency,
        ᶠz,
        ᶠdz,
        grav,
    )

    # constrain forcing
    @. ᶜuforcing = max(FT(-3e-3), min(FT(3e-3), ᶜuforcing))
    @. ᶜvforcing = max(FT(-3e-3), min(FT(3e-3), ᶜvforcing))

    @debug begin
        # DEBUG: Check for NaNs in OGWD forcing
        if any(isnan, parent(ᶜuforcing)) || any(isnan, parent(ᶜvforcing))
            @error "NaN detected in OGWD forcing!"
            @error "  ᶜuforcing: has_nan=$(any(isnan, parent(ᶜuforcing))), min=$(minimum(parent(ᶜuforcing))), max=$(maximum(parent(ᶜuforcing)))"
            @error "  ᶜvforcing: has_nan=$(any(isnan, parent(ᶜvforcing))), min=$(minimum(parent(ᶜvforcing))), max=$(maximum(parent(ᶜvforcing)))"
            error("OGWD produced NaN forcing - aborting")
        end
    end
end

"""
    calc_nonpropagating_forcing!(ᶜuforcing, ᶜvforcing, τ_x, τ_y, τ_l, τ_np, ᶠVτ, ᶠz_pbl,
                                 ᶠz_ref, ᶠp_ref, ᶜmask, ᶜweights, ᶜdiff, ᶜwtsum,
                                 ᶠp, ᶠp_m1, ᶠN, ᶠz, ᶠdz, grav)

Add the blocked-flow (non-propagating) orographic drag to the forcing fields.

Increments `ᶜuforcing` and `ᶜvforcing` [m/s²] and fills the working fields `ᶠz_ref`,
`ᶠp_ref`, `ᶜmask`, `ᶜweights`, `ᶜdiff`, and `ᶜwtsum`. Called from
`orographic_gravity_wave_forcing!` after `calc_propagate_forcing!`.

The drag is confined to the layer between the PBL top and a reference level `z_ref`, and is
distributed within it by pressure weighting, which concentrates it near the surface. `z_ref`
is the first face above `z_pbl` at which the accumulated vertical phase of a stationary
hydrostatic wave, `Σ (z − z_pbl)·N/V_τ`, exceeds `π`, i.e. half a vertical wavelength; `N`
is clamped to `[0.7e-2, 1.7e-2]` 1/s and `V_τ` floored at 1 m/s so the layer depth stays
physical. If the phase never reaches `π`, `z_ref` ends up at the model top.

The mask keeps cells overlapping `[z_pbl, z_ref)` (upper face above `z_pbl` and lower face
below `z_ref`, so at least one cell survives whenever `z_ref > z_pbl`) and drops cells of
zero weight, which contribute nothing and would divide by zero. The tendency is then
`g·τ_x·τ_np/(τ_l·wtsum)·weight` and likewise for `τ_y`, and is set to zero in columns where
the mask is empty.

# Arguments

  - `τ_x`, `τ_y`: Zonal and meridional components of the base momentum flux, from
    `calc_base_flux!`.
  - `τ_l`, `τ_np`: Corrected linear drag and non-propagating drag, from `calc_base_flux!`.
  - `ᶠVτ`: Wind projected onto the drag direction, at faces [m/s].
  - `ᶠz_pbl`: PBL top height on the face space [m].
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
    ᶠz_pbl,
    #
    ᶠz_ref,
    ᶠp_ref,
    ᶜmask,
    ᶜweights,
    ᶜdiff,
    ᶜwtsum,
    #
    ᶠp,
    ᶠp_m1,
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

    # Compute z_ref using column_reduce
    input = @. lazy(
        tuple(ᶠz_pbl, ᶠz, ᶠN, ᶠVτ, pi_val, min_n_val, max_n_val, min_Vτ_val),
    )

    Operators.column_reduce!(
        ᶠz_ref,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), false),
        transform = first,
    ) do (z_ref_acc, ᶠz_pbl_acc, phase_acc, done),
    (
        ᶠz_pbl_itr,
        z_face,
        N_face,
        Vτ_face,
        pi_val,
        min_n_val,
        max_n_val,
        min_Vτ_val,
    )
        if done
            # If already done, return the accumulated values
            return (z_ref_acc, ᶠz_pbl_acc, phase_acc, true)
        end
        if (z_face > ᶠz_pbl_itr)
            # Only accumulate phase above z_pbl
            phase_acc +=
                (z_face - ᶠz_pbl_itr) * max(min_n_val, min(max_n_val, N_face)) /
                max(min_Vτ_val, Vτ_face)

            # If phase exceeds π, stop and return current z_col as z_ref
            if phase_acc > pi_val
                return (z_face, ᶠz_pbl_itr, phase_acc, true)
            end
            # Update z_ref only when above z_pbl
            # If phase never exceeds π, z_ref will end up at model top
            return (z_face, ᶠz_pbl_itr, phase_acc, false)
        end
        # Below z_pbl, keep previous z_ref_acc unchanged
        return (z_ref_acc, ᶠz_pbl_acc, phase_acc, false)
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

    # Include cells that overlap with [z_pbl, z_ref):
    # - ᶜright_bias checks upper face > z_pbl (cell extends above z_pbl)
    # - ᶜleft_bias checks lower face < z_ref (cell starts below z_ref)
    # This ensures at least one cell is included when z_ref > z_pbl
    @. ᶜmask = ᶜright_bias.((ᶠz .> ᶠz_pbl)) .&& ᶜleft_bias.((ᶠz .< ᶠz_ref))
    @. ᶜweights = ᶜinterp.(ᶠp .- ᶠp_ref)
    @. ᶜdiff = ᶜinterp.(ᶠp_m1 .- ᶠp)

    # Exclude cells with zero weights from the mask to avoid division by zero.
    # Zero weight means p == p_ref at that cell, so it contributes nothing
    # to the pressure-weighted average.
    @. ᶜmask = ᶜmask && (!iszero(ᶜweights))

    parent(ᶜweights) .= parent(ᶜweights .* ᶜmask)

    input = @. lazy(ifelse(ᶜmask == true, ᶜdiff / ᶜweights, FT(0)))

    Operators.column_reduce!(ᶜwtsum, input; init = FT(0)) do acc, wtsum_field
        return acc + wtsum_field
    end

    if any(isnan.(parent(ᶜwtsum)))
        @warn "NaN encountered in weight sum calculation of orographic gravity wave drag"
    end

    # Compute drag, handling empty mask case (wtsum=0) gracefully
    # When wtsum=0, the mask is empty (no cells between z_pbl and z_ref),
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
    parent(dτ_sat_dz) .=
        parent(Geometry.WVector.(ᶜgradᵥ.(τ_sat)).components.data.:1)

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
marks the transition. The temperature boost improves the estimate near the surface. If no
level qualifies, `result` falls back to the surface height.

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
    field_shiftface_down!(source_field, shifted_field, boundary_value)

Shift a face-centered field downward by one level, storing the result in `shifted_field`.

This is needed to access face values at level `k-1` from within a level-`k` computation
(e.g., computing `ᶠp[k-1]` for pressure differences across cell layers). ClimaCore `column_reduce` and `column_accumulate` do not support direct `field[k-1]` indexing in broadcast expressions, so we
construct the shifted view via a round-trip through the cell-center grid:

 1. `LeftBiasedF2C` interpolates faces → cell centers using the value from below.
 2. `LeftBiasedC2F` interpolates cell centers → faces using the value from below,
    with `boundary_value` prescribed at the bottom face.

The net effect is `shifted_field[k] = source_field[k-1]` for interior faces,
and `shifted_field[bottom] = boundary_value` at the lowest face.

Called from `orographic_gravity_wave_compute_tendency!` to build `ᶠp_m1`.
"""
function field_shiftface_down!(source_field, shifted_field, boundary_value)
    L1 = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(boundary_value))
    shifted_field .= L1.(ᶜleft_bias.(source_field))
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
            topo_info = load_preprocessed_topography(local_filename)

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
