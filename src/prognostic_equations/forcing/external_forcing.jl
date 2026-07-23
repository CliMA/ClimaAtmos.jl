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

Interpolates a 1D vertical profile `fp` defined at points `xp` to new query points `x`.

Uses linear interpolation between points in `xp` and flat extrapolation (using the
value at the nearest boundary) for points `x` outside the range of `xp`.

Arguments:

  - `x`: Vector of query points (e.g., heights) at which to interpolate.
  - `xp`: Vector of points at which the profile `fp` is defined.
  - `fp`: Vector of profile values corresponding to `xp`.

Returns:

  - A vector of interpolated values at points `x`.
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

Computes the vertical advection tendency term for a scalar quantity `χ` given the
large-scale subsidence velocity This term arises from the decomposition of vertical
eddy advection in GCM forcings, as described in Shen et al. (2022, e.g., Equations 9-10).

The term calculated and added to `ᶜχₜ` is of the form:
`tendency += <w̃> * ∂<χ̃>/∂z`
where `<w̃>` is the large-scale mean subsidence velocity (`ᶜls_subsidence`) and
`∂<χ̃>/∂z` is the vertical gradient of the GCM's time-mean profile of the scalar `χ`.

This function assumes that `ᶜχₜ` already contains the total vertical eddy advection
term (`-<w̃ ∂χ̃/∂z>`), and it subtracts the mean advection to obtain the eddy part.

Arguments:

  - `ᶜχₜ`: Field of tendencies for `χ`, modified in place.
  - `ᶜχ`: Field representing the GCM's time-mean profile of the specific scalar `χ`.
  - `ᶜls_subsidence`: Field of the GCM's large-scale mean subsidence velocity `<w̃>`.
"""
function gcm_vert_advection!(ᶜχₜ, ᶜχ, ᶜls_subsidence)
    @. ᶜχₜ +=
        Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜχ))).components.data.:1 *
        ᶜls_subsidence
end

"""
    compute_gcm_driven_scalar_inv_τ(z::FT, params) where {FT}

Calculates the height-dependent inverse relaxation timescale (Γᵣ(z)) for nudging
scalar quantities (such as temperature and humidity) towards GCM profiles.

The formulation follows Shen et al. (2022, Equation 11):

  - Γᵣ(z) = 0 for z < zᵢ (no relaxation below minimum height zᵢ)
  - Γᵣ(z) = (0.5 / τᵣ) * (1 - cos(π * (z - zᵢ) / (zᵣ - zᵢ))) for zᵢ ≤ z ≤ zᵣ (smooth transition)
  - Γᵣ(z) = 1 / τᵣ for z > zᵣ (full relaxation timescale τᵣ)

Arguments:

  - `z`: Height [m].
  - `params`: Parameter set containing `CAP.gcmdriven_scalar_relaxation_timescale` (τᵣ),
    `CAP.gcmdriven_relaxation_minimum_height` (zᵢ), and
    `CAP.gcmdriven_relaxation_maximum_height` (zᵣ).

Returns:

  - The inverse relaxation timescale [s⁻¹] at height `z`.
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

Calculates the inverse relaxation timescale for nudging horizontal momentum
toward GCM profiles.

Following Shen et al. (2022), this is a constant timescale.

Arguments:

  - `z`: Height [m] (often unused if timescale is constant).
  - `params`: Parameter set containing `CAP.gcmdriven_momentum_relaxation_timescale`.

Returns:

  - The constant inverse relaxation timescale [s⁻¹].
"""
function compute_gcm_driven_momentum_inv_τ(z::FT, params) where {FT}
    τᵣ = CAP.gcmdriven_momentum_relaxation_timescale(params)
    return FT(1) / τᵣ
end

"""
    external_forcing_cache(Y, atmos::AtmosModel, params, start_date)
    external_forcing_cache(Y, external_forcing_type, params, start_date)

Sets up and returns a cache for external forcing data based on the specified
`external_forcing_type`. This cache typically holds pre-interpolated GCM profiles,
tendencies, and nudging parameters.

Dispatches to specific methods based on `atmos.external_forcing` or the explicit
`external_forcing_type`.

Arguments:

  - `Y`: The initial state vector (used for defining field structures and coordinates).
  - `atmos::AtmosModel` or `external_forcing_type`: The atmospheric model or specific
    external forcing configuration object.
  - `params`: Parameter set.
  - `start_date`: Simulation start date, used for time-varying inputs.

Returns:

  - A `NamedTuple` containing cached fields for external forcing, or an empty
    `NamedTuple` if `external_forcing_type` is `Nothing`.
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

Prepares cached fields for GCM-driven single-column model experiments by reading
forcing data from a NetCDF file specified in `external_forcing.external_forcing_file`.

This involves:

  - Reading time-mean vertical profiles of GCM tendencies (horizontal advection of
    temperature and moisture, radiative heating, vertical eddy advection components)
    and GCM state variables (temperature, moisture, winds) for a specified `cfsite_number`.
  - Reading large-scale subsidence (`wap`).
  - Reading TOA flux and cosine of solar zenith angle.
  - Interpolating these profiles to the model's vertical grid using `interp_vertical_prof`.
  - Computing inverse relaxation timescales for nudging.
  - Calculating the full vertical eddy fluctuation term for temperature and moisture by
    combining GCM-diagnosed terms with `gcm_vert_advection!`.

The methodology is that described by Shen et al. (2022) for forcing LES or SCMs with
GCM output.

Returns:

  - A `NamedTuple` of `ClimaCore.Fields.Field`s containing the interpolated and
    processed GCM forcing data.
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
    external_forcing_tendency!(Yₜ, Y, p, t, external_forcing_type)

Applies pre-processed external forcings (e.g., from GCM data, reanalysis, or
idealized case specifications like ISDAC) to the model tendencies.

Dispatches to specific methods based on `external_forcing_type`.

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector.
  - `p`: Cache containing parameters, precomputed fields, and the external forcing cache
    (`p.external_forcing`).
  - `t`: Current simulation time (used by time-varying forcings).
  - `external_forcing_type`: The specific external forcing configuration object.
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
with inverse timescale `ᶜinv_τ_wind`, adding the tendency to `Yₜ.c.uₕ`.
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
`-(ψ - ψ_nudge) * ᶜinv_τ_scalar` into `ᶜdTdt` and `ᶜdqtdt`.
"""
function nudge_Tq!(ᶜdTdt, ᶜdqtdt, Y, p, ᶜT_nudge, ᶜqt_nudge, ᶜinv_τ_scalar)
    (; ᶜT) = p.precomputed
    @. ᶜdTdt = -(ᶜT - ᶜT_nudge) * ᶜinv_τ_scalar
    @. ᶜdqtdt = -(specific(Y.c.ρq_tot, Y.c.ρ) - ᶜqt_nudge) * ᶜinv_τ_scalar
    return nothing
end

"""
    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt, ᶜdqtdt)

Convert temperature (`ᶜdTdt`) and total-specific-humidity (`ᶜdqtdt`)
tendencies into total-energy (`ρe_tot`) and total-specific-humidity (`ρq_tot`)
tendencies, adding them to `Yₜ`.
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

Apply first-order large-scale subsidence `ᶜls_subsidence` to total energy and
total specific humidity.
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
vertical eddy fluctuation, nudging of winds/temperature/humidity toward the GCM
profiles, and subsidence. Temperature and moisture tendencies are converted to
`ρe_tot`/`ρq_tot`, composing the same shared kernels `ExternalDrivenTVForcing`
uses.
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

required_surface_variables(_) = ()
required_surface_variables(::SurfaceConditions.ExternalTemperature) = (:ts,)
required_surface_variables(::ExternalTVInsolation) = (:coszen, :rsdt)

# The nudging inverse timescale (rate × mask), materialized once at cache build
# into a Field.
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
forcing_term_cache(::HorizontalAdvection, Y, cd, start_date, method, params, ᶜz) =
    _tendency_pair_cache(Y, cd, start_date, method, :tntha, :tnhusha)
forcing_term_cache(::VerticalFluctuation, Y, cd, start_date, method, params, ᶜz) =
    _tendency_pair_cache(Y, cd, start_date, method, :tntva, :tnhusva)

function update_forcing_term!(
    cache,
    ::Union{HorizontalAdvection, VerticalFluctuation},
    t,
)
    evaluate!(cache.ᶜdT, cache.input_dT, t)
    evaluate!(cache.ᶜdq, cache.input_dq, t)
    return nothing
end
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

Build the cache for the file-driven forcing. It consumes only the column
variables its composed `forcing` terms require, plus `surface_vars`. Missing
data for any is a loud error. The cache holds the `forcing` terms, a per-term
cache (`TimeVaryingInput`s, working fields, and materialized nudging
timescales), and the surface inputs used by the
`ExternalTemperature`/`ExternalTVInsolation` paths.
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

Apply the composed file-driven forcing. Each term's `(dT, dq)` contribution is
accumulated into shared buffers and converted once to `ρe_tot`/`ρq_tot`
tendencies. Each term's direct contributions (momentum nudging, subsidence) are
then applied to the state.
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

Returns an empty cache for ISDAC (Indirect and Semi-Direct Aerosol Campaign)
forcing. ISDAC forcing profiles are analytical functions of height, not requiring
pre-loading from files into cached fields.
"""
external_forcing_cache(Y, external_forcing::ISDACForcing, params, _) = (;)  # Don't need to cache anything

"""
    external_forcing_tendency!(Yₜ, Y, p, t, ::ISDACForcing)

Applies tendencies based on the ISDAC (Indirect and Semi-Direct Aerosol Campaign)
case specifications. This involves nudging (relaxation) of horizontal winds,
temperature, and total specific humidity towards idealized profiles defined by the
`APL.ISDAC_...` functions.

The nudging target temperature profile is derived from prescribed potential
temperature (`θ`) and total specific humidity (`q_tot`) profiles, using the
current model pressure. Tendencies for temperature and `q_tot` from nudging are
then converted into tendencies for total energy (`ρe_tot`) and total specific
humidity (`ρq_tot`).
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
