#####
##### External forcing for single column experiments, drawing on
##### Shen et al. (2022), "A Library of Large-Eddy Simulations Forced by Global
##### Climate Models", JAMES 14, e2021MS002631. https://doi.org/10.1029/2021MS002631
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import NCDatasets as NC
import Interpolations as Intp
import Dates
using Statistics: mean
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs: LinearInterpolation, TimeVaryingInput, evaluate!
import UnrolledUtilities: unrolled_map, unrolled_foreach

"""
    interp_vertical_prof(x, xp, fp)

Interpolate the vertical profile `fp`, defined at the points `xp`, to the query
points `x`.

Linear between the points of `xp`, flat (nearest-boundary value) outside their
range.

# Arguments

  - `x`: Query points, an array or a `ClimaCore` `Field` (e.g. the model heights
    [m]); the result keeps its shape.
  - `xp`: Vector of points at which `fp` is defined, in the same units as `x`.
  - `fp`: Vector of profile values at `xp`.

# Returns

An array of interpolated values shaped like `x` (like `parent(x)` when `x` is a
`Field`).
"""
function interp_vertical_prof(x, xp, fp)
    spl = Intp.extrapolate(
        Intp.interpolate((xp,), fp, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    # Interpolate on a flattened view and reshape back to the original shape.
    x_data = x isa Fields.Field ? parent(x) : x
    return reshape(spl(vec(x_data)), size(x_data))
end


"""
    gcm_vert_advection!(ᶜχₜ, ᶜχ, ᶜls_subsidence)

Add the mean vertical advection `⟨w̃⟩ ∂⟨χ̃⟩/∂z` to `ᶜχₜ`, converting a total
vertical advection tendency into its eddy part.

Used when building the GCM-driven forcing cache: `ᶜχₜ` is initialized with the
GCM's total vertical advection tendency `-⟨w̃ ∂χ̃/∂z⟩`, and adding the mean
advection back leaves the eddy fluctuation term, following the decomposition of
Shen et al. (2022), Eqs. 9–10.

# Arguments

  - `ᶜχₜ`: Tendency field of `χ`, modified in place.
  - `ᶜχ`: The GCM's time-mean profile of the specific scalar `χ`.
  - `ᶜls_subsidence`: The GCM's large-scale mean subsidence velocity `⟨w̃⟩` [m/s].
"""
function gcm_vert_advection!(ᶜχₜ, ᶜχ, ᶜls_subsidence)
    @. ᶜχₜ +=
        Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜχ))).components.data.:1 *
        ᶜls_subsidence
end

"""
    compute_gcm_driven_scalar_inv_τ(z::FT, params) where {FT}

Return the height-dependent inverse relaxation timescale `Γᵣ(z)` [1/s] for nudging
scalars (temperature, humidity) toward target profiles.

Follows Shen et al. (2022), Eq. 11: no relaxation below `zᵢ`, a raised-cosine ramp
between `zᵢ` and `zᵣ`, and the full rate `1/τᵣ` above `zᵣ`,

```math
Γ_r(z) = \\begin{cases}
0, & z < z_i \\\\
\\frac{1}{2τ_r}\\left[1 - \\cos\\!\\left(π \\frac{z - z_i}{z_r - z_i}\\right)\\right],
  & z_i \\le z \\le z_r \\\\
1/τ_r, & z > z_r
\\end{cases}
```

# Arguments

  - `z`: Height [m].
  - `params`: Parameter set supplying `gcmdriven_scalar_relaxation_timescale` (`τᵣ`)
    [s], `gcmdriven_relaxation_minimum_height` (`zᵢ`) [m], and
    `gcmdriven_relaxation_maximum_height` (`zᵣ`) [m].
"""
function compute_gcm_driven_scalar_inv_τ(z::FT, params) where {FT}
    τᵣ = CAP.gcmdriven_scalar_relaxation_timescale(params)
    zᵢ = CAP.gcmdriven_relaxation_minimum_height(params)
    zᵣ = CAP.gcmdriven_relaxation_maximum_height(params)

    if z < zᵢ
        return FT(0)
    elseif zᵢ <= z <= zᵣ
        cos_arg = pi * ((z - zᵢ) / (zᵣ - zᵢ))
        return (FT(0.5) / τᵣ) * (1 - cos(cos_arg))
    else
        return (1 / τᵣ)
    end
end

"""
    compute_gcm_driven_momentum_inv_τ(z::FT, params) where {FT}

Return the inverse relaxation timescale [1/s] for nudging horizontal momentum
toward target profiles.

Following Shen et al. (2022), this is height-independent: `1/τᵣ` with `τᵣ` the
parameter `gcmdriven_momentum_relaxation_timescale`.

# Arguments

  - `z`: Height [m]; unused, present so this can be broadcast over a height field
    like `compute_gcm_driven_scalar_inv_τ`.
  - `params`: Parameter set supplying `gcmdriven_momentum_relaxation_timescale` [s].
"""
function compute_gcm_driven_momentum_inv_τ(z::FT, params) where {FT}
    τᵣ = CAP.gcmdriven_momentum_relaxation_timescale(params)
    return FT(1) / τᵣ
end

"""
    external_forcing_cache(Y, atmos::AtmosModel, params, start_date)
    external_forcing_cache(Y, external_forcing, params, start_date)

Build the cache that `external_forcing_tendency!` reads as `p.external_forcing`,
dispatching on the external forcing type.

Together with `external_forcing_tendency!`, this is the complete extension
interface for a custom single-column forcing: define a forcing type and add one
method of each, as shown on the "Single Column Models" page of the docs
(`docs/src/single_column.md`). The cache typically holds interpolated profiles,
`TimeVaryingInput`s, working fields, and nudging timescales.

The `AtmosModel` method resolves which surface variables the rest of the model
needs (`ts` for an `ExternalTemperature` surface, `coszen` and `rsdt` for
`ExternalTVInsolation` under RRTMGP) and forwards them to the
`ExternalDrivenTVForcing` method; for any other forcing it simply forwards
`atmos.external_forcing`.

# Arguments

  - `Y`: Initial state vector, used for the field structure and coordinates.
  - `atmos` or `external_forcing`: The model, or the external forcing object to
    dispatch on.
  - `params`: Parameter set.
  - `start_date`: Simulation start date, used to anchor time-varying inputs.

# Returns

A `NamedTuple` of cached data, empty when `external_forcing` is `nothing`.
"""
function external_forcing_cache(Y, atmos::AtmosModel, params, start_date)
    external_forcing = atmos.external_forcing
    if external_forcing isa ExternalDrivenTVForcing
        # Surface variables are required by the resolved model components that
        # consume them, not by the forcing terms: `ts` by an `ExternalTemperature`
        # surface, `coszen`/`rsdt` by `ExternalTVInsolation` under RRTMGP.
        insolation_vars =
            atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode ?
            required_surface_variables(atmos.insolation) : ()
        surface_vars = (
            required_surface_variables(atmos.surface.temperature)...,
            insolation_vars...,
        )
        return external_forcing_cache(
            Y,
            external_forcing,
            params,
            start_date;
            surface_vars,
        )
    end
    return external_forcing_cache(Y, external_forcing, params, start_date)
end

external_forcing_cache(Y, external_forcing::Nothing, params, _) = (;)

"""
    external_forcing_cache(Y, external_forcing::GCMForcing, params, _)

Build the cache for a GCM-driven single-column experiment, from the NetCDF file
`external_forcing.external_forcing_file` and the group `external_forcing.cfsite_number`.

Reads the time-mean vertical profiles of the GCM horizontal-advection tendencies
(`tntha`, `tnhusha`), the vertical-advection tendencies (`tntva`, `tnhusva`), the
GCM state used as the nudging target (`ta`, `hus`, `ua`, `va`), and the pressure
velocity `wap`, from which the subsidence velocity follows by the hydrostatic
approximation `w ≈ -ω α / g`. All profiles are interpolated to the model grid with
`interp_vertical_prof`. The vertical-advection tendencies are converted into eddy
fluctuation terms with `gcm_vert_advection!`, and the nudging inverse timescales
are evaluated with `compute_gcm_driven_scalar_inv_τ` and
`compute_gcm_driven_momentum_inv_τ`. The insolation entries store the TOA flux,
recovered as `rsdt / coszen`, and the cosine of the solar zenith angle. The
methodology is that of Shen et al. (2022).

# Returns

A `NamedTuple` of `ClimaCore` `Field`s: `ᶜdTdt_fluc`, `ᶜdqtdt_fluc`, `ᶜdTdt_hadv`,
`ᶜdqtdt_hadv`, `ᶜT_nudge`, `ᶜqt_nudge`, `ᶜu_nudge`, `ᶜv_nudge`, `ᶜinv_τ_wind`,
`ᶜinv_τ_scalar`, `ᶜls_subsidence`, `toa_flux`, and `cos_zenith`.
"""
function external_forcing_cache(Y, external_forcing::GCMForcing, params, _)
    FT = Spaces.undertype(axes(Y.c))
    ᶜdTdt_fluc = similar(Y.c, FT)
    ᶜdqtdt_fluc = similar(Y.c, FT)
    ᶜdTdt_hadv = similar(Y.c, FT)
    ᶜdqtdt_hadv = similar(Y.c, FT)
    ᶜT_nudge = similar(Y.c, FT)
    ᶜqt_nudge = similar(Y.c, FT)
    ᶜu_nudge = similar(Y.c, FT)
    ᶜv_nudge = similar(Y.c, FT)
    ᶜinv_τ_wind = similar(Y.c, FT)
    ᶜinv_τ_scalar = similar(Y.c, FT)
    ᶜls_subsidence = similar(Y.c, FT)
    toa_flux = similar(Fields.level(Y.c.ρ, 1), FT)
    cos_zenith = similar(Fields.level(Y.c.ρ, 1), FT)

    (; external_forcing_file, cfsite_number) = external_forcing

    NC.Dataset(external_forcing_file, "r") do ds

        function setvar!(cc_field, varname, zc_gcm, zc_forcing)
            parent(cc_field) .= interp_vertical_prof(
                zc_gcm,
                zc_forcing,
                gcm_driven_profile_tmean(ds.group[cfsite_number], varname),
            )
        end

        function setvar_subsidence!(cc_field, varname, zc_gcm, zc_forcing, params)
            # Computes subsidence velocity from the hydrostatic approximation
            # w \approx - ω α / g, where ω is pressure velocity and α = 1/ρ is
            # the specific volume
            parent(cc_field) .= interp_vertical_prof(
                zc_gcm,
                zc_forcing,
                gcm_driven_profile_tmean(ds.group[cfsite_number], varname) .* .-(
                    gcm_driven_profile_tmean(ds.group[cfsite_number], "alpha"),
                ) ./ CAP.grav(params),
            )
        end

        function set_toa_flux!(cc_field)
            # rsdt is TOA insolation. We need
            # TOA flux and the solar zenith angle separately. So compute
            #`toa_flux = rsdt/cos(SZA)`.
            parent(cc_field) .= mean(
                ds.group[cfsite_number]["rsdt"][:] ./
                ds.group[cfsite_number]["coszen"][:],
            )
        end

        function set_cos_zenith!(cc_field)
            parent(cc_field) .= ds.group[cfsite_number]["coszen"][1]
        end

        zc_forcing = gcm_height(ds.group[cfsite_number])
        zc_gcm = Fields.coordinate_field(Y.c).z

        setvar!(ᶜdTdt_hadv, "tntha", zc_gcm, zc_forcing)
        setvar!(ᶜdqtdt_hadv, "tnhusha", zc_gcm, zc_forcing)
        setvar_subsidence!(ᶜls_subsidence, "wap", zc_gcm, zc_forcing, params)
        # GCM states, used for nudging + vertical eddy advection
        setvar!(ᶜT_nudge, "ta", zc_gcm, zc_forcing)
        setvar!(ᶜqt_nudge, "hus", zc_gcm, zc_forcing)
        setvar!(ᶜu_nudge, "ua", zc_gcm, zc_forcing)
        setvar!(ᶜv_nudge, "va", zc_gcm, zc_forcing)

        # Vertical eddy advection (Shen et al., 2022; eqn. 9,10)
        # sum of two terms to give total tendency. First term:
        setvar!(ᶜdTdt_fluc, "tntva", zc_gcm, zc_forcing)
        setvar!(ᶜdqtdt_fluc, "tnhusva", zc_gcm, zc_forcing)

        # subtract mean vertical advection to obtain eddy part:
        gcm_vert_advection!(ᶜdTdt_fluc, ᶜT_nudge, ᶜls_subsidence)
        gcm_vert_advection!(ᶜdqtdt_fluc, ᶜqt_nudge, ᶜls_subsidence)

        set_toa_flux!(toa_flux)
        set_cos_zenith!(cos_zenith)

        @. ᶜinv_τ_wind = compute_gcm_driven_momentum_inv_τ(zc_gcm, params)
        @. ᶜinv_τ_scalar = compute_gcm_driven_scalar_inv_τ(zc_gcm, params)
    end

    return (;
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜinv_τ_wind,
        ᶜinv_τ_scalar,
        ᶜls_subsidence,
        toa_flux,
        cos_zenith,
    )
end

"""
    external_forcing_tendency!(Yₜ, Y, p, t, external_forcing)

Add the tendencies of an external forcing (GCM data, reanalysis, or an idealized
case such as ISDAC), dispatching on the forcing type.

Together with `external_forcing_cache`, this is the complete extension interface
for a custom single-column forcing: define a forcing type and add one method of
each, as shown on the "Single Column Models" page of the docs
(`docs/src/single_column.md`). The method for `nothing` is a no-op.

Methods typically increment `Yₜ.c.uₕ` (momentum nudging), `Yₜ.c.ρe_tot`, and
`Yₜ.c.ρq_tot`, the last two obtained from temperature and specific-humidity
tendencies by `apply_Tq_forcing!`.

# Arguments

  - `Yₜ`: Tendency state vector, modified in place.
  - `Y`: Current state vector.
  - `p`: Cache; the forcing data is in `p.external_forcing`, and the thermodynamic
    state in `p.precomputed`.
  - `t`: Current simulation time, at which time-varying inputs are evaluated.
  - `external_forcing`: The external forcing object to dispatch on.

Called from `additional_tendency!`, i.e., treated explicitly. Returns `nothing`.
"""
external_forcing_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

# ============================================================================
# Shared forcing-tendency kernels
#
# The wind/scalar nudging, the temperature-and-humidity → energy conversion,
# and the large-scale subsidence math are common to the file-driven
# (`ExternalDrivenTVForcing`), GCM, and ISDAC forcings. These kernels are the
# single implementation; each forcing's tendency composes them.
# ============================================================================

"""
    nudge_uv!(Yₜ, Y, p, ᶜu_nudge, ᶜv_nudge, ᶜinv_τ_wind)

Relax the horizontal momentum `Y.c.uₕ` toward the target `(ᶜu_nudge, ᶜv_nudge)`
[m/s] with inverse timescale `ᶜinv_τ_wind` [1/s], adding the tendency
`-(uₕ - uₕ_nudge) / τ` to `Yₜ.c.uₕ`.

Uses the scratch field `p.scratch.ᶜtemp_C12` to hold the target vector. Shared by
the GCM, file-driven, and per-term `Nudging` forcings. Returns `nothing`.
"""
function nudge_uv!(Yₜ, Y, p, ᶜu_nudge, ᶜv_nudge, ᶜinv_τ_wind)
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge = C12(Geometry.UVVector(ᶜu_nudge, ᶜv_nudge), ᶜlg)
    @. Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind
    return nothing
end

"""
    nudge_Tq!(ᶜdTdt, ᶜdqtdt, Y, p, ᶜT_nudge, ᶜqt_nudge, ᶜinv_τ_scalar)

Write the temperature and total-specific-humidity nudging tendencies
`-(ψ - ψ_nudge) * ᶜinv_τ_scalar` into `ᶜdTdt` [K/s] and `ᶜdqtdt` [1/s].

Both outputs are overwritten, not accumulated. Reads the temperature from
`p.precomputed.ᶜT` and the specific humidity from `Y`. Returns `nothing`.
"""
function nudge_Tq!(ᶜdTdt, ᶜdqtdt, Y, p, ᶜT_nudge, ᶜqt_nudge, ᶜinv_τ_scalar)
    (; ᶜT) = p.precomputed
    @. ᶜdTdt = -(ᶜT - ᶜT_nudge) * ᶜinv_τ_scalar
    @. ᶜdqtdt = -(specific(Y.c.ρq_tot, Y.c.ρ) - ᶜqt_nudge) * ᶜinv_τ_scalar
    return nothing
end

"""
    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt, ᶜdqtdt)

Convert temperature (`ᶜdTdt` [K/s]) and total-specific-humidity (`ᶜdqtdt` [1/s])
tendencies into total-energy and total-water tendencies, adding them to
`Yₜ.c.ρe_tot` and `Yₜ.c.ρq_tot`.

The energy conversion is `ρ [c_{v,m} dT/dt + (c_{v,v}(T - T₀) + L_{v,0} - R_v T₀) dq_tot/dt]`, i.e., the moist-mixture heat capacity times the temperature tendency
plus the internal energy of the added vapor; no potential-energy term, since the
water is added at constant height. Reads the thermodynamic state from
`p.precomputed`. Shared by the GCM, file-driven, and ISDAC forcings. Returns
`nothing`.
"""
function apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt, ᶜdqtdt)
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) * ᶜdTdt +
            (cv_v * (ᶜT - T_0) + Lv_0 - R_v * T_0) * ᶜdqtdt
        )
    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt
    return nothing
end

"""
    apply_subsidence_forcing!(Yₜ, Y, p, ᶜls_subsidence)

Apply the large-scale subsidence velocity `ᶜls_subsidence` [m/s] to total energy
and total water, adding the tendencies to `Yₜ.c.ρe_tot` and `Yₜ.c.ρq_tot`.

Interpolates the velocity to faces and calls `subsidence!` with the first-order
upwind scheme, advecting the total enthalpy `p.precomputed.ᶜh_tot` and the total
specific humidity. Uses the scratch field `p.scratch.ᶠtemp_CT3`. Returns `nothing`.
"""
function apply_subsidence_forcing!(Yₜ, Y, p, ᶜls_subsidence)
    (; ᶜh_tot) = p.precomputed
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠls_subsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠls_subsidence³ =
        ᶠinterp(ᶜls_subsidence * CT3(unit_basis_vector_data(CT3, ᶜlg)))
    subsidence!(
        Yₜ.c.ρe_tot,
        Y.c.ρ,
        ᶠls_subsidence³,
        ᶜh_tot,
        Val{:first_order}(),
    )
    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    subsidence!(
        Yₜ.c.ρq_tot,
        Y.c.ρ,
        ᶠls_subsidence³,
        ᶜq_tot,
        Val{:first_order}(),
    )
    return nothing
end

"""
    external_forcing_tendency!(Yₜ, Y, p, t, ::GCMForcing)

Apply GCM-driven forcing from the always-populated cache: horizontal advection,
vertical eddy fluctuation, nudging of winds, temperature, and humidity toward the
GCM profiles, and subsidence.

The three temperature and humidity contributions are summed into two scratch
fields and converted once to `ρe_tot` and `ρq_tot` tendencies, composing the same
shared kernels (`nudge_uv!`, `nudge_Tq!`, `apply_Tq_forcing!`,
`apply_subsidence_forcing!`) that the file-driven `ExternalDrivenTVForcing` uses.
Returns `nothing`.
"""
function external_forcing_tendency!(Yₜ, Y, p, t, ::GCMForcing)
    (;
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜls_subsidence,
        ᶜinv_τ_wind,
        ᶜinv_τ_scalar,
    ) = p.external_forcing

    nudge_uv!(Yₜ, Y, p, ᶜu_nudge, ᶜv_nudge, ᶜinv_τ_wind)

    # Sum horizontal-advection, nudging, and vertical-fluctuation tendencies.
    # `ᶜdTdt_sum`/`ᶜdqtdt_sum` alias the scratch fields the nudging tendency is
    # written into, so the `@.` sum reads the nudging value pointwise.
    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    nudge_Tq!(ᶜdTdt_sum, ᶜdqtdt_sum, Y, p, ᶜT_nudge, ᶜqt_nudge, ᶜinv_τ_scalar)
    @. ᶜdTdt_sum = ᶜdTdt_hadv + ᶜdTdt_sum + ᶜdTdt_fluc
    @. ᶜdqtdt_sum = ᶜdqtdt_hadv + ᶜdqtdt_sum + ᶜdqtdt_fluc

    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt_sum, ᶜdqtdt_sum)
    apply_subsidence_forcing!(Yₜ, Y, p, ᶜls_subsidence)
    return nothing
end

# ============================================================================
# Per-term protocol for the composed file-driven forcing
#
# Each `AbstractForcingTerm` implements:
#   - `required_surface_variables` (dispatched on the model components, not the
#     terms): the surface variables the resolved model needs
#   - `forcing_term_cache`: the build-time cache
#   - `update_forcing_term!`: the per-step refresh from the `TimeVaryingInput`s
#   - `accumulate_Tq_tendency!`: adds into the shared (dT, dq) buffers
#   - `apply_direct_forcing!`: applies to the state, for momentum nudging and
#     subsidence
# ============================================================================

"""
    required_surface_variables(model_component)

Return the tuple of surface file variables a resolved model component needs from a
file-driven forcing dataset.

Dispatches on the model component rather than on the forcing terms: an
`ExternalTemperature` surface needs `(:ts,)` and an `ExternalTVInsolation` needs
`(:coszen, :rsdt)`; anything else needs nothing. `external_forcing_cache` takes
the union and requires the dataset to provide it.
"""
required_surface_variables(_) = ()
required_surface_variables(::SurfaceConditions.ExternalTemperature) = (:ts,)
required_surface_variables(::ExternalTVInsolation) = (:coszen, :rsdt)

"""
    materialize_inv_τ(term::Nudging, ᶜz, params)

Return the inverse relaxation timescale field [1/s] of a [`Nudging`](@ref) term,
evaluated once at cache build.

The rate follows `term.timescale`: `DefaultTimescale()` uses the Shen et al.
(2022) profiles (`compute_gcm_driven_scalar_inv_τ` for `(:ta, :hus)`,
`compute_gcm_driven_momentum_inv_τ` for `(:ua, :va)`), a `Number` is a constant
relaxation timescale `τ` [s], and anything else is called as `z -> τ(z)`. The
result is then multiplied by `term.mask`, which may be `nothing`, a `Number`, a
`Field`, or a function of height.
"""
function materialize_inv_τ(term::Nudging, ᶜz, params)
    FT = Spaces.undertype(axes(ᶜz))
    ᶜinv_τ = similar(ᶜz, FT)
    _set_inv_τ_rate!(ᶜinv_τ, term.timescale, term.variables, ᶜz, params)
    _apply_inv_τ_mask!(ᶜinv_τ, term.mask, ᶜz)
    return ᶜinv_τ
end

function _set_inv_τ_rate!(ᶜinv_τ, ::DefaultTimescale, variables, ᶜz, params)
    if all(in(NUDGING_SCALAR_VARS), variables)
        @. ᶜinv_τ = compute_gcm_driven_scalar_inv_τ(ᶜz, params)
    else # momentum (mixed sets are rejected at `Nudging` construction)
        @. ᶜinv_τ = compute_gcm_driven_momentum_inv_τ(ᶜz, params)
    end
    return nothing
end
_set_inv_τ_rate!(ᶜinv_τ, τ::Number, variables, ᶜz, params) =
    (ᶜinv_τ .= 1 / τ; nothing)
_set_inv_τ_rate!(ᶜinv_τ, f, variables, ᶜz, params) =
    (@. ᶜinv_τ = 1 / f(ᶜz); nothing)

_apply_inv_τ_mask!(ᶜinv_τ, ::Nothing, ᶜz) = nothing
_apply_inv_τ_mask!(ᶜinv_τ, w::Number, ᶜz) = (ᶜinv_τ .*= w; nothing)
_apply_inv_τ_mask!(ᶜinv_τ, m::Fields.Field, ᶜz) = (ᶜinv_τ .*= m; nothing)
_apply_inv_τ_mask!(ᶜinv_τ, f, ᶜz) = (@. ᶜinv_τ *= f(ᶜz); nothing)

# --- HorizontalAdvection / VerticalFluctuation: a (dT, dq) tendency pair ---
function _tendency_pair_cache(Y, cd, start_date, method, dT_var, dq_var)
    FT = Spaces.undertype(axes(Y.c))
    inputs = ColumnDatasets.column_timevaryinginputs(
        cd,
        (dT_var, dq_var),
        axes(Y.c),
        start_date;
        method,
    )
    return (;
        input_dT = inputs[dT_var],
        input_dq = inputs[dq_var],
        ᶜdT = similar(Y.c, FT),
        ᶜdq = similar(Y.c, FT),
    )
end
"""
    forcing_term_cache(term, Y, cd, start_date, method, params, ᶜz)

Build the per-term cache of a file-driven forcing term: its `TimeVaryingInput`s,
the fields they are evaluated into, and any quantity materialized once at build
time.

One method per concrete [`AbstractForcingTerm`](@ref):
[`HorizontalAdvection`](@ref) and [`VerticalFluctuation`](@ref) each hold a
`(dT, dq)` input pair, [`Subsidence`](@ref) holds the vertical velocity `wa`, and
[`Nudging`](@ref) holds one target field per nudged variable (`nothing` for the
others) plus the inverse timescale from `materialize_inv_τ`.

# Arguments

  - `term`: The forcing term to dispatch on.
  - `Y`: Initial state vector, used for the field structure.
  - `cd`: The `ColumnDataset` holding the forcing file.
  - `start_date`: Simulation start date, anchoring the time axis.
  - `method`: Time interpolation method for the `TimeVaryingInput`s.
  - `params`: Parameter set.
  - `ᶜz`: Cell-center height field [m].

# Returns

A `NamedTuple`, stored in `p.external_forcing.term_caches` alongside its term.
"""
forcing_term_cache(::HorizontalAdvection, Y, cd, start_date, method, params, ᶜz) =
    _tendency_pair_cache(Y, cd, start_date, method, :tntha, :tnhusha)
forcing_term_cache(::VerticalFluctuation, Y, cd, start_date, method, params, ᶜz) =
    _tendency_pair_cache(Y, cd, start_date, method, :tntva, :tnhusva)

"""
    update_forcing_term!(cache, term, t)

Refresh a forcing term's working fields from its `TimeVaryingInput`s at time `t`,
in place.

One method per concrete [`AbstractForcingTerm`](@ref). Called once per cache
update, before the tendency hooks read the fields. Returns `nothing`.
"""
function update_forcing_term!(
    cache,
    ::Union{HorizontalAdvection, VerticalFluctuation},
    t,
)
    evaluate!(cache.ᶜdT, cache.input_dT, t)
    evaluate!(cache.ᶜdq, cache.input_dq, t)
    return nothing
end
"""
    accumulate_Tq_tendency!(ᶜdTdt, ᶜdqtdt, term, cache, Y, p)

Add a forcing term's temperature [K/s] and total-specific-humidity [1/s]
tendencies into the shared buffers `ᶜdTdt` and `ᶜdqtdt`.

The buffers are accumulated across all terms of a composed forcing and converted
once to `ρe_tot` and `ρq_tot` tendencies by `apply_Tq_forcing!`, so a term must
never overwrite them. [`HorizontalAdvection`](@ref) and
[`VerticalFluctuation`](@ref) add their prescribed tendencies directly, and
[`Nudging`](@ref) adds `-(ψ - ψ_nudge) / τ` for whichever of `ta` and `hus` it
nudges. Terms with no scalar contribution fall through to the no-op method for
`::AbstractForcingTerm`. Returns `nothing`.
"""
function accumulate_Tq_tendency!(
    ᶜdTdt,
    ᶜdqtdt,
    ::Union{HorizontalAdvection, VerticalFluctuation},
    cache,
    Y,
    p,
)
    @. ᶜdTdt += cache.ᶜdT
    @. ᶜdqtdt += cache.ᶜdq
    return nothing
end

# --- Subsidence ---
function forcing_term_cache(::Subsidence, Y, cd, start_date, method, params, ᶜz)
    FT = Spaces.undertype(axes(Y.c))
    inputs =
        ColumnDatasets.column_timevaryinginputs(cd, (:wa,), axes(Y.c), start_date; method)
    return (; input_wa = inputs.wa, ᶜls_subsidence = similar(Y.c, FT))
end
update_forcing_term!(cache, ::Subsidence, t) =
    (evaluate!(cache.ᶜls_subsidence, cache.input_wa, t); nothing)
"""
    apply_direct_forcing!(Yₜ, Y, p, term, cache)

Add a forcing term's contributions that do not go through the shared temperature
and humidity buffers directly to `Yₜ`.

Used for [`Subsidence`](@ref), which advects total energy and total water with
`apply_subsidence_forcing!`, and for [`Nudging`](@ref) of `(:ua, :va)`, which
relaxes `Yₜ.c.uₕ` with `nudge_uv!`. Terms with no direct contribution fall through
to the no-op method for `::AbstractForcingTerm`. Returns `nothing`.
"""
apply_direct_forcing!(Yₜ, Y, p, ::Subsidence, cache) =
    apply_subsidence_forcing!(Yₜ, Y, p, cache.ᶜls_subsidence)

# --- Nudging: per-role target fields (field-or-nothing) + inverse timescale ---
function forcing_term_cache(term::Nudging, Y, cd, start_date, method, params, ᶜz)
    FT = Spaces.undertype(axes(Y.c))
    vars = term.variables
    inputs =
        ColumnDatasets.column_timevaryinginputs(cd, vars, axes(Y.c), start_date; method)
    slot(v) = v in term.variables ? similar(Y.c, FT) : nothing
    input(v) = v in term.variables ? inputs[v] : nothing
    return (;
        ᶜinv_τ = materialize_inv_τ(term, ᶜz, params),
        ᶜT_nudge = slot(:ta),
        ᶜqt_nudge = slot(:hus),
        ᶜu_nudge = slot(:ua),
        ᶜv_nudge = slot(:va),
        input_ta = input(:ta),
        input_hus = input(:hus),
        input_ua = input(:ua),
        input_va = input(:va),
    )
end
function update_forcing_term!(cache, ::Nudging, t)
    isnothing(cache.ᶜT_nudge) || evaluate!(cache.ᶜT_nudge, cache.input_ta, t)
    isnothing(cache.ᶜqt_nudge) || evaluate!(cache.ᶜqt_nudge, cache.input_hus, t)
    isnothing(cache.ᶜu_nudge) || evaluate!(cache.ᶜu_nudge, cache.input_ua, t)
    isnothing(cache.ᶜv_nudge) || evaluate!(cache.ᶜv_nudge, cache.input_va, t)
    return nothing
end
function accumulate_Tq_tendency!(ᶜdTdt, ᶜdqtdt, ::Nudging, cache, Y, p)
    (; ᶜT) = p.precomputed
    ᶜinv_τ = cache.ᶜinv_τ
    isnothing(cache.ᶜT_nudge) ||
        @. ᶜdTdt += -(ᶜT - cache.ᶜT_nudge) * ᶜinv_τ
    isnothing(cache.ᶜqt_nudge) || @. ᶜdqtdt +=
        -(specific(Y.c.ρq_tot, Y.c.ρ) - cache.ᶜqt_nudge) * ᶜinv_τ
    return nothing
end
function apply_direct_forcing!(Yₜ, Y, p, ::Nudging, cache)
    isnothing(cache.ᶜu_nudge) ||
        nudge_uv!(Yₜ, Y, p, cache.ᶜu_nudge, cache.ᶜv_nudge, cache.ᶜinv_τ)
    return nothing
end

# Terms that implement only one hook fall through to these no-ops.
accumulate_Tq_tendency!(ᶜdTdt, ᶜdqtdt, ::AbstractForcingTerm, cache, Y, p) =
    nothing
apply_direct_forcing!(Yₜ, Y, p, ::AbstractForcingTerm, cache) = nothing

"""
    external_forcing_cache(Y, external_forcing::ExternalDrivenTVForcing, params, start_date; surface_vars)

Build the cache for the file-driven forcing.

Consumes only the column variables its composed `forcing` terms require (the union
over `required_column_variables`) plus the `surface_vars` the resolved model
components need; missing data for any of them is a loud error.

# Returns

A `NamedTuple` with `forcing_terms` (the terms themselves), `term_caches` (one
`forcing_term_cache` per term, holding `TimeVaryingInput`s, working
fields, and materialized nudging timescales), and `surface_fields` and
`surface_timevaryinginputs` for the `ExternalTemperature` and
`ExternalTVInsolation` paths.
"""
function external_forcing_cache(
    Y,
    external_forcing::ExternalDrivenTVForcing,
    params,
    start_date;
    surface_vars = (),
)
    cd = external_forcing.dataset
    forcing = external_forcing.forcing
    surface_vars = Tuple(surface_vars)

    column_vars =
        isempty(forcing) ? () :
        Tuple(union(unrolled_map(required_column_variables, forcing)...))
    ColumnDatasets.require_forcing_variables(cd, column_vars, surface_vars)

    method = external_forcing.time_interpolation_method
    ᶜz = Fields.coordinate_field(Y.c).z
    term_caches = unrolled_map(
        term -> forcing_term_cache(term, Y, cd, start_date, method, params, ᶜz),
        forcing,
    )

    surface_target_space = axes(Fields.level(Y.f.u₃, ClimaCore.Utilities.half))
    surface_timevaryinginputs =
        isempty(surface_vars) ?
        (;) :
        ColumnDatasets.surface_timevaryinginputs(
            cd,
            surface_vars,
            surface_target_space,
            start_date;
            method,
        )
    FT = Spaces.undertype(axes(Y.c))
    surface_fields =
        isempty(surface_vars) ?
        (;) :
        similar(
            Fields.level(Y.f.u₃, ClimaCore.Utilities.half),
            NamedTuple{surface_vars, NTuple{length(surface_vars), FT}},
        )

    return (;
        forcing_terms = forcing,
        term_caches,
        surface_fields,
        surface_timevaryinginputs,
    )
end

"""
    external_forcing_tendency!(Yₜ, Y, p, t, ::ExternalDrivenTVForcing)

Apply the composed file-driven forcing.

Each term's `(dT, dq)` contribution is accumulated into two zeroed scratch buffers
by `accumulate_Tq_tendency!` and converted once to `ρe_tot` and `ρq_tot`
tendencies by `apply_Tq_forcing!`. Each term's direct contributions (momentum
nudging, subsidence) are then added by `apply_direct_forcing!`. Returns `nothing`.
"""
function external_forcing_tendency!(Yₜ, Y, p, t, ::ExternalDrivenTVForcing)
    (; forcing_terms, term_caches) = p.external_forcing

    ᶜdTdt = p.scratch.ᶜtemp_scalar
    ᶜdqtdt = p.scratch.ᶜtemp_scalar_2
    ᶜdTdt .= 0
    ᶜdqtdt .= 0
    unrolled_foreach(forcing_terms, term_caches) do term, cache
        accumulate_Tq_tendency!(ᶜdTdt, ᶜdqtdt, term, cache, Y, p)
    end
    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt, ᶜdqtdt)

    unrolled_foreach(forcing_terms, term_caches) do term, cache
        apply_direct_forcing!(Yₜ, Y, p, term, cache)
    end
    return nothing
end

"""
    external_forcing_cache(Y, external_forcing::ISDACForcing, params, _)

Return an empty cache for ISDAC (Indirect and Semi-Direct Aerosol Campaign)
forcing.

The ISDAC profiles are analytic functions of height, evaluated directly in
`external_forcing_tendency!`, so nothing needs to be pre-loaded.
"""
external_forcing_cache(Y, external_forcing::ISDACForcing, params, _) = (;)  # Don't need to cache anything

"""
    external_forcing_tendency!(Yₜ, Y, p, t, ::ISDACForcing)

Apply the ISDAC (Indirect and Semi-Direct Aerosol Campaign) forcing: relaxation of
the horizontal winds, temperature, and total specific humidity toward the analytic
`APL.ISDAC_*` profiles.

Increments `Yₜ.c.uₕ` directly and, through `apply_Tq_forcing!`, `Yₜ.c.ρe_tot` and
`Yₜ.c.ρq_tot`. The target temperature is obtained by saturation adjustment of the
prescribed liquid-ice potential temperature and total specific humidity at the
current model pressure, and the two relaxation rates are the height-dependent
`ISDAC_inv_τ_scalar` and `ISDAC_inv_τ_wind` [1/s]. `t` is unused, the profiles
being steady.
"""
function external_forcing_tendency!(Yₜ, Y, p, t, ::ISDACForcing)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, ᶜT) = p.precomputed

    ᶜinv_τ_scalar = APL.ISDAC_inv_τ_scalar(FT)  # s⁻¹
    ᶜinv_τ_wind = APL.ISDAC_inv_τ_wind(FT)  # s⁻¹
    θ = APL.ISDAC_θ_liq_ice(FT)
    u = APL.ISDAC_u(FT)
    v = APL.ISDAC_v(FT)
    q_tot = APL.ISDAC_q_tot(FT)

    # Convert ISDAC potential temperature to air temperature
    FT = Spaces.undertype(axes(Y.c))
    ta_ISDAC =
        (pres, z) ->
            TD.saturation_adjustment(
                thermo_params,
                TD.pθ_li(),
                pres,
                θ(z),
                q_tot(z);
                maxiter = 4,
            ).T

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge = C12(Geometry.UVVector(u(ᶜz), v(ᶜz)), ᶜlg)
    @. Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind(ᶜz)

    # TODO: May make more sense to use initial ISDAC (hydrostatic) pressure, but would need to add it to cache,
    # so for now just use current pressure.
    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging =
        -(ᶜT - ta_ISDAC(ᶜp, ᶜz)) *
        ᶜinv_τ_scalar(ᶜz)
    @. ᶜdqtdt_nudging =
        -(specific(Y.c.ρq_tot, Y.c.ρ) - q_tot(ᶜz)) * ᶜinv_τ_scalar(ᶜz)

    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt_nudging, ᶜdqtdt_nudging)
end
