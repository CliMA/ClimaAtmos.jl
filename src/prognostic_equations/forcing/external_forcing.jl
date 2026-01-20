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
external_forcing_cache(Y, atmos::AtmosModel, params, start_date) =
    external_forcing_cache(Y, atmos.external_forcing, params, start_date)

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
- Reading TOA insolation and cosine of solar zenith angle.
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
    insolation = similar(Fields.level(Y.c.ρ, 1), FT)
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

        function set_insolation!(cc_field)
            # rsdt is TOA insolation on a horizontal plane. We need
            # total solar irradiance and the solar zenith angle separately. So compute 
            #`TSI = rsdt/cos(SZA)`.
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

        set_insolation!(insolation)
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
        insolation,
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

"""
    external_forcing_tendency!(Yₜ, Y, p, t, ::Union{GCMForcing, ExternalDrivenTVForcing})

Applies tendencies from GCM or reanalysis-driven external forcings. This includes:
- Horizontal advection tendencies for temperature (`ᶜdTdt_hadv`) and total specific
  humidity (`ᶜdqtdt_hadv`).
- Vertical eddy fluctuation tendencies (`ᶜdTdt_fluc`, `ᶜdqtdt_fluc`).
- Nudging (relaxation) of horizontal winds (`uₕ`), temperature, and total specific
  humidity (`q_tot`) towards prescribed GCM/reanalysis profiles (`ᶜu_nudge`,
  `ᶜv_nudge`, `ᶜT_nudge`, `ᶜqt_nudge`) using precalculated inverse relaxation
  timescales (`ᶜinv_τ_wind`, `ᶜinv_τ_scalar`).
- Subsidence effects on total energy and total specific humidity, using the
  large-scale subsidence rate `ᶜls_subsidence`.

The sum of horizontal advection, nudging, and vertical fluctuation tendencies for
temperature and moisture are converted into tendencies for total energy (`ρe_tot`)
and total specific humidity (`ρq_tot`).

A top boundary condition is applied by zeroing out the `ρe_tot` and `ρq_tot`
tendencies at the highest model level.
"""
function external_forcing_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{GCMForcing, ExternalDrivenTVForcing},
)
    # horizontal advection, vertical fluctuation, nudging, subsidence (need to add),
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜT) = p.precomputed
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

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge = C12(Geometry.UVVector(ᶜu_nudge, ᶜv_nudge), ᶜlg)
    @. Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind

    (; ᶜh_tot, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    # nudging tendency
    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging =
        -(ᶜT - ᶜT_nudge) * ᶜinv_τ_scalar
    @. ᶜdqtdt_nudging =
        -(specific(Y.c.ρq_tot, Y.c.ρ) - ᶜqt_nudge) * ᶜinv_τ_scalar

    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_sum = ᶜdTdt_hadv + ᶜdTdt_nudging + ᶜdTdt_fluc
    @. ᶜdqtdt_sum = ᶜdqtdt_hadv + ᶜdqtdt_nudging + ᶜdqtdt_fluc

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # total energy
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) * ᶜdTdt_sum +
            (
                cv_v * (ᶜT - T_0) + Lv_0 -
                R_v * T_0
            ) * ᶜdqtdt_sum
        )
    # total specific humidity
    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt_sum

    ## subsidence -->
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

    # Hard set tendencies of ρe_tot and ρq_tot at the top to 0. Otherwise upper 
    # portion of domain is anomalously cold
    ρe_tot_top = Fields.level(Yₜ.c.ρe_tot, Spaces.nlevels(axes(Y.c)))
    @. ρe_tot_top = 0.0

    ρq_tot_top = Fields.level(Yₜ.c.ρq_tot, Spaces.nlevels(axes(Y.c)))
    @. ρq_tot_top = 0.0
    # <-- subsidence

    return nothing
end

"""
    external_forcing_cache(Y, external_forcing::ExternalDrivenTVForcing, params, start_date)

Sets up cache structures for time-varying external forcing, typically from reanalysis
data such as ERA5, as specified in `external_forcing.external_forcing_file`.

This involves:
- Creating `TimeVaryingInput` objects for various atmospheric column variables
  (e.g., "ta", "hus", "wap") and surface variables (e.g., "coszen", "rsdt", "ts").
  These objects handle on-the-fly reading and interpolation of time-dependent data.
- Allocating `ClimaCore.Fields.Field`s to store the instantaneous values of these
  forcing fields at each timestep.
- Pre-calculating nudging timescale profiles.

The cached `TimeVaryingInput` objects are updated during the simulation via callbacks.
This cache does not load all data at once but prepares for its retrieval.
"""
function external_forcing_cache(
    Y,
    external_forcing::ExternalDrivenTVForcing,
    params,
    start_date,
)
    # current support is for time varying era5 data which does not require vertical advective tendencies
    # or surface latent and sensible heat fluxes, i.e., the surface state is set with surface temperature
    # only. This could be modified to include these terms if needed.
    (; external_forcing_file) = external_forcing

    # generate forcing files

    column_tendencies = [
        "ta",
        "hus",
        "tntva",
        "wa",
        "tntha",
        "tnhusha",
        "ua",
        "va",
        "tnhusva",
        "rho",
        "wap",
    ]
    surface_tendencies = ["coszen", "rsdt", "hfls", "hfss", "ts"]
    column_target_space = axes(Y.c)
    surface_target_space = axes(Fields.level(Y.f.u₃, ClimaCore.Utilities.half))

    extrapolation_bc = (Intp.Flat(), Intp.Flat(), Intp.Linear())

    column_timevaryinginputs = [
        TimeVaryingInput(
            external_forcing_file,
            name,
            column_target_space;
            reference_date = start_date,
            regridder_kwargs = (; extrapolation_bc),
            # useful for monthly averaged diurnal data - does not affect hourly era5 case because of time bounds flag
            method = TimeVaryingInputs.LinearInterpolation(
                TimeVaryingInputs.PeriodicCalendar(),
            ),
        ) for name in column_tendencies
    ]

    surface_timevaryinginputs = [
        TimeVaryingInput(
            external_forcing_file,
            name,
            surface_target_space;
            reference_date = start_date,
            regridder_kwargs = (; extrapolation_bc),
            method = TimeVaryingInputs.LinearInterpolation(
                TimeVaryingInputs.PeriodicCalendar(),
            ),
        ) for name in surface_tendencies
    ]

    column_variable_names_as_symbols = Symbol.(column_tendencies)
    surface_variable_names_as_symbols = Symbol.(surface_tendencies)

    column_inputs = similar(
        Y.c,
        NamedTuple{
            Tuple(column_variable_names_as_symbols),
            NTuple{length(column_variable_names_as_symbols), eltype(Y.c.ρ)},
        },
    )

    surface_inputs = similar(
        Fields.level(Y.f.u₃, ClimaCore.Utilities.half),
        NamedTuple{
            Tuple(surface_variable_names_as_symbols),
            NTuple{length(surface_variable_names_as_symbols), eltype(params)},
        },
    )

    column_timevaryinginputs =
        (; zip(column_variable_names_as_symbols, column_timevaryinginputs)...)
    surface_timevaryinginputs =
        (; zip(surface_variable_names_as_symbols, surface_timevaryinginputs)...)

    era5_tv_column_cache = (; column_inputs, column_timevaryinginputs)
    era5_tv_surface_cache = (; surface_inputs, surface_timevaryinginputs)

    # create cache for external forcing data that will be populated in callbacks
    FT = Spaces.undertype(axes(Y.c))
    era5_cache = (;
        ᶜdTdt_fluc = similar(Y.c, FT),
        ᶜdqtdt_fluc = similar(Y.c, FT),
        ᶜdTdt_hadv = similar(Y.c, FT),
        ᶜdqtdt_hadv = similar(Y.c, FT),
        ᶜT_nudge = similar(Y.c, FT),
        ᶜqt_nudge = similar(Y.c, FT),
        ᶜu_nudge = similar(Y.c, FT),
        ᶜv_nudge = similar(Y.c, FT),
        ᶜinv_τ_wind = FT(1 / (6 * 3600)),  # TODO: consider making timescale configurable in params
        # set relaxation profile toward reference state
        ᶜinv_τ_scalar = compute_gcm_driven_scalar_inv_τ.(
            Fields.coordinate_field(Y.c).z,
            params,
        ),
        ᶜls_subsidence = similar(Y.c, FT),
        insolation = similar(
            Fields.level(Y.f.u₃, ClimaCore.Utilities.half),
            FT,
        ),
        cos_zenith = similar(
            Fields.level(Y.f.u₃, ClimaCore.Utilities.half),
            FT,
        ),
    )
    return (; era5_tv_column_cache..., era5_tv_surface_cache..., era5_cache...)
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
    ta_ISDAC =
        (pres, z) -> TD.air_temperature(
            thermo_params,
            TD.PhaseEquil_pθq(thermo_params, pres, θ(z), q_tot(z)),
        )

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

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # total energy
    (; ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) * ᶜdTdt_nudging +
            (
                cv_v * (ᶜT - T_0) + Lv_0 -
                R_v * T_0
            ) * ᶜdqtdt_nudging
        )

    # total specific humidity
    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt_nudging
end
