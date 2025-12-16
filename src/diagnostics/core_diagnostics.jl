"""
This file is included in Diagnostics.jl

README: Adding a new core diagnostic:

In addition to the metadata (names, comments, ...), the most important step in adding a
new DiagnosticVariable is defining its compute function. `compute` has to take three
arguments: (state, cache, time), and has to return the diagnostic.

Often, it is possible to compute certain diagnostics only for specific models (e.g.,
humidity for moist models). For that, it is convenient to adopt the following pattern:

1. Define a catch base function that does the computation we want to do for the case we know
how to handle, for example

```julia
function compute_hur(state, cache, time, ::Union{EquilMoistModel, NonEquilMoistModel})
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.relative_humidity(thermo_params, cache.ᶜts))
end
```

2. Define a function that has the correct signature and calls this function

```julia
compute_hur(state, cache, time) = compute_hur(state, cache, time, cache.atmos.moisture_model)
```

3. Define a function that returns an error when the model is incorrect

```julia
compute_hur(_, _, _, model) = error_diagnostic_variable("relative_humidity", model)
```

We can also output a specific error message

```julia
compute_hur(_, _, _, model) =
    error_diagnostic_variable("relative humidity makes sense only for moist models")
```

# General helper functions for undefined diagnostics for a particular model

```julia
error_diagnostic_variable(variable, model::T) where {T} =
    error_diagnostic_variable("Cannot compute $variable with model = $T")
```
"""

###
# Density (3d)
###
add_diagnostic_variable!(short_name = "rhoa", units = "kg m^-3",
    long_name = "Air Density",
    standard_name = "air_density",
    compute = (state, cache, time) -> state.c.ρ,
)

###
# U velocity (3d)
###
add_diagnostic_variable!(short_name = "ua", units = "m s^-1",
    long_name = "Eastward Wind",
    standard_name = "eastward_wind",
    comments = "Eastward (zonal) wind component",
    compute = (state, cache, time) ->
        (@. lazy(u_component(Geometry.UVector(cache.precomputed.ᶜu)))),
)

###
# V velocity (3d)
###
add_diagnostic_variable!(short_name = "va", units = "m s^-1",
    long_name = "Northward Wind",
    standard_name = "northward_wind",
    comments = "Northward (meridional) wind component",
    compute = (state, cache, time) ->
        (@. lazy(v_component(Geometry.VVector(cache.precomputed.ᶜu)))),
)

###
# W velocity (3d)
###
# TODO: may want to convert to omega (Lagrangian pressure tendency) as standard output,
# but this is probably more useful for now
#
add_diagnostic_variable!(short_name = "wa", units = "m s^-1",
    long_name = "Upward Air Velocity",
    standard_name = "upward_air_velocity",
    comments = "Vertical wind component",
    compute = (state, cache, time) ->
        (@. lazy(w_component(Geometry.WVector(cache.precomputed.ᶠu)))),
)

###
# Temperature (3d)
###
add_diagnostic_variable!(short_name = "ta", units = "K",
    long_name = "Air Temperature",
    standard_name = "air_temperature",
    compute = (state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        return @. lazy(TD.air_temperature(thermo_params, cache.precomputed.ᶜts))
    end,
)

###
# Potential temperature (3d)
###
add_diagnostic_variable!(short_name = "thetaa", units = "K",
    long_name = "Air Potential Temperature",
    standard_name = "air_potential_temperature",
    compute = (state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        return @. lazy(TD.dry_pottemp(thermo_params, cache.precomputed.ᶜts))
    end,
)

###
# Enthalpy (3d)
###
add_diagnostic_variable!(short_name = "ha", units = "m^2 s^-2",
    long_name = "Air Specific Enthalpy",
    compute = (state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        return @. lazy(TD.specific_enthalpy(thermo_params, cache.precomputed.ᶜts))
    end,
)

###
# Air pressure (3d)
###
add_diagnostic_variable!(short_name = "pfull", units = "Pa",
    long_name = "Pressure at Model Full-Levels",
    compute = (state, cache, time) -> cache.precomputed.ᶜp,
)

###
# Vorticity (3d)
###
add_diagnostic_variable!(short_name = "rv", units = "s^-1",
    long_name = "Relative Vorticity",
    standard_name = "relative_vorticity",
    comments = "Vertical component of relative vorticity",
    compute! = (out, state, cache, time) -> begin
        vort = @. lazy(w_component(Geometry.WVector(wcurlₕ(cache.precomputed.ᶜu))))
        vort = isnothing(out) ? copy(vort) : (out .= vort)
        # We need to ensure smoothness, so we call DSS
        Spaces.weighted_dss!(vort)
        return vort
    end,
)

###
# Geopotential height (3d)
###
add_diagnostic_variable!(short_name = "zg", units = "m",
    long_name = "Geopotential Height",
    standard_name = "geopotential_height",
    compute = (state, cache, time) -> (@. lazy(cache.core.ᶜΦ / CAP.grav(cache.params))),
)

###
# Cloud fraction (3d)
###
add_diagnostic_variable!(short_name = "cl", units = "%",
    long_name = "Cloud fraction",
    compute = (state, cache, time) ->
        (@. lazy(cache.precomputed.cloud_diagnostics_tuple.cf * 100)),
)

###
# Total kinetic energy
###
add_diagnostic_variable!(short_name = "ke", units = "m^2 s^-2",
    long_name = "Total Kinetic Energy",
    standard_name = "total_kinetic_energy",
    comments = "The kinetic energy on cell centers",
    compute = (state, cache, time) -> cache.precomputed.ᶜK,
)

###
# Mixing length (3d)
###
add_diagnostic_variable!(short_name = "lmix", units = "m",
    long_name = "Environment Mixing Length",
    comments = """
    Calculated as smagorinsky length scale without EDMF SGS model,
    or from mixing length closure with EDMF SGS model.
    """,
    compute = compute_lmix,
)
compute_lmix(state, cache, time) =
    compute_lmix(state, cache, time, cache.atmos.turbconv_model)
compute_lmix(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX}) =
    ᶜmixing_length(state, cache)
function compute_lmix(state, cache, _, _)
    ᶜ∂b∂z = cache.precomputed.ᶜ∂b∂z
    ᶜS = cache.precomputed.ᶜS
    ᶜdz = Fields.Δz_field(axes(state.c))
    ᶜprandtl_nvec = @. lazy(turbulent_prandtl_number(cache.params, ᶜ∂b∂z, ᶜS))
    N_eff = @. lazy(sqrt(max(ᶜ∂b∂z, 0)))
    c_smag = CAP.c_smag(cache.params)
    return @. lazy(smagorinsky_lilly_length(c_smag, N_eff, ᶜdz, ᶜprandtl_nvec, ᶜS))
end

###
# Buoyancy gradient (3d)
###
add_diagnostic_variable!(short_name = "bgrad", units = "s^-2",
    long_name = "Linearized Buoyancy Gradient",
    compute = (state, cache, time) -> cache.precomputed.ᶜlinear_buoygrad,
)

###
# Strain rate magnitude (3d)
###
add_diagnostic_variable!(short_name = "strain", units = "s^-2",
    long_name = "String Rate Magnitude",
    compute = (state, cache, time) -> cache.precomputed.ᶜstrain_rate_norm,
)

###
# Smagorinsky Lilly diffusivity
###
add_diagnostic_variable!(short_name = "Dh_smag", units = "m^2 s^-1",
    long_name = "Horizontal smagorinsky diffusivity",
    compute = (state, cache, time) -> cache.precomputed.ᶜD_h,
)
add_diagnostic_variable!(short_name = "Dv_smag", units = "m^2 s^-1",
    long_name = "Vertical smagorinsky diffusivity",
    compute = (state, cache, time) -> cache.precomputed.ᶜD_v,
)
add_diagnostic_variable!(short_name = "strainh_smag", units = "s",
    long_name = "Horizontal strain rate magnitude (for Smagorinsky)",
    compute = (state, cache, time) -> cache.precomputed.ᶜS_norm_h,
)
add_diagnostic_variable!(short_name = "strainv_smag", units = "s",
    long_name = "Vertical strain rate magnitude (for Smagorinsky)",
    compute = (state, cache, time) -> cache.precomputed.ᶜS_norm_v,
)

###
# Relative humidity (3d)
###
compute_hur(state, cache, time) =
    compute_hur(state, cache, time, cache.atmos.moisture_model)
compute_hur(_, _, _, model) = error_diagnostic_variable("hur", model)

function compute_hur(_, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.relative_humidity(thermo_params, cache.precomputed.ᶜts))
end

add_diagnostic_variable!(short_name = "hur", units = "",
    long_name = "Relative Humidity",
    standard_name = "relative_humidity",
    comments = "Total amount of water vapor in the air relative to the amount \
                achievable by saturation at the current temperature",
    compute = compute_hur,
)

###
# Total specific humidity (3d)
###
compute_hus(state, cache, time) =
    compute_hus(state, cache, time, cache.atmos.moisture_model)
compute_hus(_, _, _, model) = error_diagnostic_variable("hus", model)

compute_hus(state, _, _, ::Union{EquilMoistModel, NonEquilMoistModel}) =
    @. lazy(state.c.ρq_tot / state.c.ρ)

add_diagnostic_variable!(short_name = "hus", units = "kg kg^-1",
    long_name = "Specific Humidity",
    standard_name = "specific_humidity",
    comments = "Mass of all water phases per mass of air",
    compute = compute_hus,
)

###
# Liquid water specific humidity (3d)
###
compute_clw(state, cache, time) =
    compute_clw(state, cache, time, cache.atmos.moisture_model)
compute_clw(_, _, _, model) = error_diagnostic_variable("clw", model)

compute_clw(_, cache, _, _::Union{EquilMoistModel, NonEquilMoistModel})
    cache.precomputed.cloud_diagnostics_tuple.q_liq

add_diagnostic_variable!(short_name = "clw", units = "kg kg^-1",
    long_name = "Mass Fraction of Cloud Liquid Water",
    standard_name = "mass_fraction_of_cloud_liquid_water_in_air",
    comments = """
    Includes both large-scale and convective cloud.
    This is calculated as the mass of cloud liquid water in the grid cell divided by
    the mass of air (including the water in all phases) in the grid cells.
    """,
    compute = compute_clw,
)

###
# Ice water specific humidity (3d)
###
compute_cli(state, cache, time) =
    compute_cli(state, cache, time, cache.atmos.moisture_model)
compute_cli(_, _, _, model) = error_diagnostic_variable("cli", model)

compute_cli(_, cache, _, _::Union{EquilMoistModel, NonEquilMoistModel}) =
    cache.precomputed.cloud_diagnostics_tuple.q_ice

add_diagnostic_variable!(short_name = "cli", units = "kg kg^-1",
    long_name = "Mass Fraction of Cloud Ice",
    standard_name = "mass_fraction_of_cloud_ice_in_air",
    comments = """
    Includes both large-scale and convective cloud.
    This is calculated as the mass of cloud ice in the grid cell divided by
    the mass of air (including the water in all phases) in the grid cell.
    """,
    compute = compute_cli,
)

###
# Surface specific humidity (2d)
###
compute_hussfc(state, cache, time) =
    compute_hussfc(state, cache, time, cache.atmos.moisture_model)
compute_hussfc(_, _, _, model) = error_diagnostic_variable("hussfc", model)

function compute_hussfc(_, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    thermo_params = CAP.thermodynamics_params(cache.params)
    ts_sfc = cache.precomputed.sfc_conditions.ts
    return @. lazy(TD.total_specific_humidity(thermo_params, ts_sfc))
end

add_diagnostic_variable!(short_name = "hussfc", units = "kg kg^-1",
    long_name = "Surface Specific Humidity",
    standard_name = "specific_humidity",
    comments = "Mass of all water phases per mass of air in the layer \
                infinitely close to the surface",
    compute = compute_hussfc,
)

###
# Surface temperature (2d)
###
add_diagnostic_variable!(short_name = "ts", units = "K",
    long_name = "Surface Temperature",
    standard_name = "surface_temperature",
    comments = "Temperature of the lower boundary of the atmosphere",
    compute = (state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        ts_sfc = cache.precomputed.sfc_conditions.ts
        return @. lazy(TD.air_temperature(thermo_params, ts_sfc))
    end,
)

###
# Near-surface air temperature (2d)
###
add_diagnostic_variable!(short_name = "tas", units = "K",
    long_name = "Near-Surface Air Temperature",
    standard_name = "air_temperature",
    comments = "Temperature at the bottom cell center of the atmosphere",
    compute = (state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        ts_int = Fields.level(cache.precomputed.ᶜts, 1)
        return @. lazy(TD.air_temperature(thermo_params, ts_int))
    end,
)

###
# Near-surface U velocity (2d)
###
add_diagnostic_variable!(short_name = "uas", units = "m s^-1",
    long_name = "Eastward Near-Surface Wind",
    standard_name = "eastward_wind",
    comments = "Eastward component of the wind at the bottom cell center of the atmosphere",
    compute = (state, cache, time) -> begin
        u_int = Fields.level(cache.precomputed.ᶜu, 1)
        return @. lazy(u_component(Geometry.UVector(u_int)))
    end,
)

###
# Near-surface V velocity (2d)
###
add_diagnostic_variable!(short_name = "vas", units = "m s^-1",
    long_name = "Northward Near-Surface Wind",
    standard_name = "northward_wind",
    comments = "Northward (meridional) wind component at the bottom cell center of the atmosphere",
    compute = (state, cache, time) -> begin
        v_int = Fields.level(cache.precomputed.ᶜu, 1)
        return @. lazy(v_component(Geometry.VVector(v_int)))
    end,
)

###
# Eastward and northward surface drag component (2d)
###
function compute_tau(cache, component)
    (; surface_ct3_unit) = cache.core
    (; ρ_flux_uₕ) = cache.precomputed.sfc_conditions
    τ_sfc = Geometry.UVVector(adjoint(ρ_flux_uₕ) * surface_ct3_unit)
    return @. lazy(getproperty(τ_sfc.components.data, component))
end

add_diagnostic_variable!(short_name = "tauu", units = "Pa",
    long_name = "Surface Downward Eastward Wind Stress",
    standard_name = "downward_eastward_stress",
    comments = "Eastward component of the surface drag",
    compute = (state, cache, time) -> compute_tau(cache, :1),
)

add_diagnostic_variable!(short_name = "tauv", units = "Pa",
    long_name = "Surface Downward Northward Wind Stress",
    standard_name = "downward_northward_stress",
    comments = "Northward component of the surface drag",
    compute = (state, cache, time) -> compute_tau(cache, :2),
)

###
# Surface energy flux (2d) - TODO: this may need to be split into sensible and latent heat fluxes
###
add_diagnostic_variable!(short_name = "hfes", units = "W m^-2",
    long_name = "Surface Upward Energy Flux",
    comments = "Energy flux at the surface",
    compute = (state, cache, time) -> begin
        (; ρ_flux_h_tot) = cache.precomputed.sfc_conditions
        (; surface_ct3_unit) = cache.core
        return @. lazy(dot(ρ_flux_h_tot, surface_ct3_unit))
    end,
)

###
# Surface evaporation (2d)
###
compute_evspsbl(state, cache, time) =
    compute_evspsbl(state, cache, time, cache.atmos.moisture_model)
compute_evspsbl(_, _, _, model) = error_diagnostic_variable("evspsbl", model)

function compute_evspsbl(_, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    return @. lazy(dot(ρ_flux_q_tot, surface_ct3_unit))
end

add_diagnostic_variable!(short_name = "evspsbl", units = "kg m^-2 s^-1",
    long_name = "Evaporation Including Sublimation and Transpiration",
    comments = "evaporation at the surface",
    compute = compute_evspsbl,
)

###
# Latent heat flux (2d)
###
compute_hfls(state, cache, time) =
    compute_hfls(state, cache, time, cache.atmos.moisture_model)
compute_hfls(_, _, _, model) = error_diagnostic_variable("hfls", model)

function compute_hfls(_, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    thermo_params = CAP.thermodynamics_params(cache.params)
    LH_v0 = TD.Parameters.LH_v0(thermo_params)

    return @. lazy(dot(ρ_flux_q_tot, surface_ct3_unit) * LH_v0)
end

add_diagnostic_variable!(short_name = "hfls", units = "W m^-2",
    long_name = "Surface Upward Latent Heat Flux",
    standard_name = "surface_upward_latent_heat_flux",
    compute = compute_hfls,
)

###
# Sensible heat flux (2d)
###
compute_hfss(state, cache, time) =
    compute_hfss(state, cache, time, cache.atmos.moisture_model)
compute_hfss(_, _, _, model) = error_diagnostic_variable("hfss", model)

function compute_hfss(_, cache, _, ::DryModel)
    (; ρ_flux_h_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    return @. lazy(dot(ρ_flux_h_tot, surface_ct3_unit))
end

function compute_hfss(_, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; ρ_flux_h_tot, ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    thermo_params = CAP.thermodynamics_params(cache.params)
    LH_v0 = TD.Parameters.LH_v0(thermo_params)

    return @. lazy(
        dot(ρ_flux_h_tot, surface_ct3_unit) -
        dot(ρ_flux_q_tot, surface_ct3_unit) * LH_v0,
    )
end

add_diagnostic_variable!(short_name = "hfss", units = "W m^-2",
    long_name = "Surface Upward Sensible Heat Flux",
    standard_name = "surface_upward_sensible_heat_flux",
    compute = compute_hfss,
)

###
# Precipitation (2d)
###
compute_pr(state, cache, time) =
    compute_pr(state, cache, time, cache.atmos.microphysics_model)
compute_pr(_, _, _, model) = error_diagnostic_variable("pr", model)

compute_pr(_, cache, _,
    ::Union{
        NoPrecipitation,
        Microphysics0Moment, Microphysics1Moment,
        Microphysics2Moment, Microphysics2MomentP3,
    },
) = @. lazy(cache.precomputed.surface_rain_flux + cache.precomputed.surface_snow_flux)

add_diagnostic_variable!(short_name = "pr", units = "kg m^-2 s^-1",
    long_name = "Precipitation",
    standard_name = "precipitation",
    comments = "Total precipitation including rain and snow",
    compute = compute_pr,
)

compute_prra(state, cache, time) =
    compute_prra(state, cache, time, cache.atmos.microphysics_model)
compute_prra(_, _, _, microphysics_model::T) where {T} =
    error_diagnostic_variable("prra", microphysics_model)

compute_prra(_, cache, _,
    ::Union{
        NoPrecipitation,
        Microphysics0Moment, Microphysics1Moment,
        Microphysics2Moment, Microphysics2MomentP3,
    },
) = cache.precomputed.surface_rain_flux

add_diagnostic_variable!(short_name = "prra", units = "kg m^-2 s^-1",
    long_name = "Rainfall Flux",
    standard_name = "rainfall_flux",
    comments = "Precipitation including all forms of water in the liquid phase",
    compute = compute_prra,
)

compute_prsn(state, cache, time) =
    compute_prsn(state, cache, time, cache.atmos.microphysics_model)
compute_prsn(_, _, _, microphysics_model::T) where {T} =
    error_diagnostic_variable("prsn", microphysics_model)

compute_prsn(_, cache, _,
    ::Union{
        NoPrecipitation,
        Microphysics0Moment, Microphysics1Moment,
        Microphysics2Moment, Microphysics2MomentP3,
    },
) = cache.precomputed.surface_snow_flux

add_diagnostic_variable!(short_name = "prsn", units = "kg m^-2 s^-1",
    long_name = "Snowfall Flux",
    standard_name = "snowfall_flux",
    comments = "Precipitation including all forms of water in the solid phase",
    compute = compute_prsn,
)

###
# Precipitation (3d)
###
compute_husra(state, cache, time) =
    compute_husra(state, cache, time, cache.atmos.microphysics_model)
compute_husra(_, _, _, model) = error_diagnostic_variable("husra", model)

compute_husra(state, _, _,
    ::Union{Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3},
) = @. lazy(state.c.ρq_rai / state.c.ρ)

add_diagnostic_variable!(short_name = "husra", units = "kg kg^-1",
    long_name = "Mass Fraction of Rain",
    standard_name = "mass_fraction_of_rain_in_air",
    comments = """
    This is calculated as the mass of rain water in the grid cell divided by
    the mass of air (dry air + water vapor + cloud condensate) in the grid cells.
    """,
    compute = compute_husra,
)

compute_hussn(state, cache, time) =
    compute_hussn(state, cache, time, cache.atmos.microphysics_model)
compute_hussn(_, _, _, model::T) where {T} =
    error_diagnostic_variable("hussn", model)

compute_hussn(state, _, _,
    ::Union{Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3},
) = @. lazy(state.c.ρq_sno / state.c.ρ)

add_diagnostic_variable!(short_name = "hussn", units = "kg kg^-1",
    long_name = "Mass Fraction of Snow",
    standard_name = "mass_fraction_of_snow_in_air",
    comments = """
    This is calculated as the mass of snow in the grid cell divided by
    the mass of air (dry air + water vapor + cloud condensate) in the grid cells.
    """,
    compute = compute_hussn,
)

compute_cdnc(state, cache, time) =
    compute_cdnc(state, cache, time, cache.atmos.microphysics_model)
compute_cdnc(_, _, _, model) = error_diagnostic_variable("cdnc", model)

compute_cdnc(state, _, _, ::Union{Microphysics2Moment, Microphysics2MomentP3}) = 
    state.c.ρn_liq

add_diagnostic_variable!(short_name = "cdnc", units = "m^-3",
    long_name = "Cloud Liquid Droplet Number Concentration",
    standard_name = "number_concentration_of_cloud_liquid_water_particles_in_air",
    comments = """
    This is calculated as the number of cloud liquid water droplets in the grid 
    cell divided by the cell volume.
    """,
    compute = compute_cdnc,
)

compute_ncra(state, cache, time) =
    compute_ncra(state, cache, time, cache.atmos.microphysics_model)
compute_ncra(_, _, _, model) = error_diagnostic_variable("ncra", model)

compute_ncra(state, _, _, ::Union{Microphysics2Moment, Microphysics2MomentP3}) = 
    state.c.ρn_rai

add_diagnostic_variable!(short_name = "ncra", units = "m^-3",
    long_name = "Raindrop Number Concentration",
    standard_name = "number_concentration_of_raindrops_in_air",
    comments = """
    This is calculated as the number of raindrops in the grid cell divided
    by the cell volume.
    """,
    compute = compute_ncra,
)

###
# Topography
###
compute_orog(state, cache, time) =
    compute_orog(state, cache, time, axes(state.c).grid.hypsography)

# When we have a Flat topography, we just have to return a field of zeros
compute_orog(state, _, _, ::Grids.Flat) = 
    @. lazy(zeros(Spaces.horizontal_space(axes(state.c.ρ))))

compute_orog(_, _, _, hypsography) = @. lazy(Geometry.tofloat.(hypsography.surface))

add_diagnostic_variable!(short_name = "orog", units = "m",
    long_name = "Surface Altitude",
    standard_name = "surface_altitude",
    comments = "Elevation of the horizontal coordinates",
    compute = compute_orog,
)

###
# Condensed water path (2d)
###
compute_clwvi(state, cache, time) =
    compute_clwvi(state, cache, time, cache.atmos.moisture_model)
compute_clwvi(_, _, _, model) = error_diagnostic_variable("clwvi", model)

function compute_clwvi!(out, state, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; q_liq, q_ice) = cache.precomputed.cloud_diagnostics_tuple
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    clw = @. cache.scratch.ᶜtemp_scalar = state.c.ρ * (q_liq + q_ice)
    Operators.column_integral_definite!(out′, clw)
    return out′
end

add_diagnostic_variable!(short_name = "clwvi", units = "kg m^-2",
    long_name = "Condensed Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_condensed_water",
    comments = """
    Mass of condensed (liquid + ice) water in the column divided by the area of the column
    (not just the area of the cloudy portion of the column). It doesn't include precipitating hydrometeors.
    """,
    compute! = compute_clwvi!,
)

###
# Liquid water path (2d)
###
compute_lwp(state, cache, time) =
    compute_lwp(state, cache, time, cache.atmos.moisture_model)
compute_lwp(_, _, _, model) = error_diagnostic_variable("lwp", model)

function compute_lwp!(out, state, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; q_liq) = cache.precomputed.cloud_diagnostics_tuple
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    lw = @. cache.scratch.ᶜtemp_scalar = state.c.ρ * q_liq
    Operators.column_integral_definite!(out′, lw)
    return out′
end

add_diagnostic_variable!(short_name = "lwp", units = "kg m^-2",
    long_name = "Liquid Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_liquid_water",
    comments = """
    The total mass of liquid water in cloud per unit area.
    (not just the area of the cloudy portion of the column). It doesn't include precipitating hydrometeors.
    """,
    compute! = compute_lwp!,
)

###
# Ice water path (2d)
###
compute_clivi(state, cache, time) =
    compute_clivi(state, cache, time, cache.atmos.moisture_model)
compute_clivi(_, _, _, model) = error_diagnostic_variable("clivi", model)

function compute_clivi!(out, state, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; q_ice) = cache.precomputed.cloud_diagnostics_tuple
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    cli = @. cache.scratch.ᶜtemp_scalar = state.c.ρ * q_ice
    Operators.column_integral_definite!(out′, cli)
    return out′
end

add_diagnostic_variable!(short_name = "clivi", units = "kg m^-2",
    long_name = "Ice Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_ice",
    comments = """
    The total mass of ice in cloud per unit area.
    (not just the area of the cloudy portion of the column). It doesn't include precipitating hydrometeors.
    """,
    compute! = compute_clivi!,
)


###
# Vertical integrated dry static energy (2d)
###
function compute_dsevi!(out, state, cache, _)
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    thermo_params = CAP.thermodynamics_params(cache.params)
    cp = CAP.cp_d(cache.params)
    ᶜta = @. lazy(TD.air_temperature(thermo_params, cache.precomputed.ᶜts))
    dse = @. cache.scratch.ᶜtemp_scalar = state.c.ρ * (cp * ᶜta + cache.core.ᶜΦ)
    Operators.column_integral_definite!(out′, dse)
    return out′
end

add_diagnostic_variable!(short_name = "dsevi", units = "",
    long_name = "Dry Static Energy Vertical Integral",
    compute! = compute_dsevi!,
)

###
# column integrated cloud fraction (2d)
###
compute_clvi(state, cache, time) =
    compute_clvi(state, cache, time, cache.atmos.moisture_model)
compute_clvi(_, _, _, model) = error_diagnostic_variable("clvi", model)

function compute_clvi!(out, state, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    FT = Spaces.undertype(axes(cloud_cover))
    (; cf) = cache.precomputed.cloud_diagnostics_tuple
    cloud_cover = @. cache.scratch.ᶜtemp_scalar = FT(cf > zero(FT))
    Operators.column_integral_definite!(out′, cloud_cover)
    return out′
end

add_diagnostic_variable!(short_name = "clvi", units = "m",
    long_name = "Vertical Cloud Fraction Integral",
    comments = "The total height of the column occupied at least partially by cloud.",
    compute! = compute_clvi!,
)


###
# Column integrated total specific humidity (2d)
###
compute_prw(state, cache, time) =
    compute_prw(state, cache, time, cache.atmos.moisture_model)
compute_prw(_, _, _, model) = error_diagnostic_variable("prw", model)

function compute_prw!(out, state, _, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    Operators.column_integral_definite!(out′, state.c.ρq_tot)
    return out′
end

add_diagnostic_variable!(short_name = "prw", units = "kg m^-2",
    long_name = "Water Vapor Path",
    standard_name = "atmospheric_mass_content_of_water_vapor",
    comments = "Vertically integrated specific humidity",
    compute! = compute_prw!,
)

###
# Column integrated relative humidity (2d)
###
compute_hurvi(state, cache, time) =
    compute_hurvi(state, cache, time, cache.atmos.moisture_model)
compute_hurvi(_, _, _, model) = error_diagnostic_variable("hurvi", model)

function compute_hurvi!(out, state, _, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜts) = cache.precomputed
    # compute vertical integral of saturation specific humidity
    # note next line currently allocates; currently no correct scratch space
    sat_vi = zeros(axes(Fields.level(state.f, half)))
    sat = cache.scratch.ᶜtemp_scalar
    @. sat = state.c.ρ * TD.q_vap_saturation(thermo_params, ᶜts)
    Operators.column_integral_definite!(sat_vi, sat)
    # compute saturation-weighted vertical integral of specific humidity
    hur = cache.scratch.ᶜtemp_scalar
    qv = @. lazy(TD.vapor_specific_humidity(thermo_params, ᶜts))
    @. hur = state.c.ρ * qv * sat
    Operators.column_integral_definite!(out′, hur)
    @. out′ = out′ / sat_vi
    return out′
end

add_diagnostic_variable!(short_name = "hurvi", units = "kg m^-2",
    long_name = "Relative Humidity Saturation-Weighted Vertical Integral",
    standard_name = "relative_humidity_vi",
    comments = "Integrated relative humidity over the vertical column",
    compute! = compute_hurvi!,
)


###
# Vapor specific humidity (3d)
###
compute_husv(state, cache, time) =
    compute_husv(state, cache, time, cache.atmos.moisture_model)
compute_husv(_, _, _, model) = error_diagnostic_variable("husv", model)

function compute_husv(_, cache, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.vapor_specific_humidity(thermo_params, cache.precomputed.ᶜts))
end

add_diagnostic_variable!(short_name = "husv", units = "kg kg^-1",
    long_name = "Vapor Specific Humidity",
    comments = "Mass of water vapor per mass of air",
    compute = compute_husv,
)

###
# Analytic Steady-State Approximations
###

# These are only available when `check_steady_state` is `true`.

add_diagnostic_variable!(
    short_name = "uapredicted",
    long_name = "Predicted Eastward Wind",
    standard_name = "predicted_eastward_wind",
    units = "m s^-1",
    comments = "Predicted steady-state eastward (zonal) wind component",
    compute = (state, cache, time) ->
        (@. lazy(u_component(cache.steady_state_velocity.ᶜu))),
)

add_diagnostic_variable!(
    short_name = "vapredicted",
    long_name = "Predicted Northward Wind",
    standard_name = "predicted_northward_wind",
    units = "m s^-1",
    comments = "Predicted steady-state northward (meridional) wind component",
    compute = (state, cache, time) ->
        (@. lazy(v_component(cache.steady_state_velocity.ᶜu))),
)

add_diagnostic_variable!(
    short_name = "wapredicted",
    long_name = "Predicted Upward Air Velocity",
    standard_name = "predicted_upward_air_velocity",
    units = "m s^-1",
    comments = "Predicted steady-state vertical wind component",
    compute = (state, cache, time) ->
        (@. lazy(w_component(cache.steady_state_velocity.ᶠu))),
)

add_diagnostic_variable!(
    short_name = "uaerror",
    long_name = "Error of Eastward Wind",
    standard_name = "error_eastward_wind",
    units = "m s^-1",
    comments = "Error of steady-state eastward (zonal) wind component",
    compute = (state, cache, time) -> (@. lazy(
        u_component(Geometry.UVWVector(cache.precomputed.ᶜu)) -
        u_component(cache.steady_state_velocity.ᶜu),
    )),
)

add_diagnostic_variable!(
    short_name = "vaerror",
    long_name = "Error of Northward Wind",
    standard_name = "error_northward_wind",
    units = "m s^-1",
    comments = "Error of steady-state northward (meridional) wind component",
    compute = (state, cache, time) -> (@. lazy(
        v_component(Geometry.UVWVector(cache.precomputed.ᶜu)) -
        v_component(cache.steady_state_velocity.ᶜu),
    )),
)

add_diagnostic_variable!(
    short_name = "waerror",
    long_name = "Error of Upward Air Velocity",
    standard_name = "error_upward_air_velocity",
    units = "m s^-1",
    comments = "Error of steady-state vertical wind component",
    compute = (state, cache, time) -> (@. lazy(
        w_component(Geometry.UVWVector(cache.precomputed.ᶠu)) -
        w_component(cache.steady_state_velocity.ᶠu),
    )),
)

###
# Convective Available Potential Energy (2d)
###
function compute_cape!(out, state, cache, _)
    thermo_params = CAP.thermodynamics_params(cache.params)
    g = TD.Parameters.grav(thermo_params)
    FT = Spaces.undertype(axes(state.c))
    (; ᶜp, sfc_conditions, ᶜts) = cache.precomputed

    # Get surface parcel properties
    q_sfc = @. lazy(TD.total_specific_humidity(thermo_params, sfc_conditions.ts))
    θ_liq_ice_sfc = @. lazy(TD.liquid_ice_pottemp(thermo_params, sfc_conditions.ts))

    # Create parcel thermodynamic states at each level based on energy & moisture at surface
    parcel_ts_moist = @. lazy(TD.PhaseEquil_pθq(thermo_params, ᶜp, θ_liq_ice_sfc, q_sfc))

    # Calculate virtual temperatures for parcel & environment
    parcel_Tv = @. lazy(TD.virtual_temperature(thermo_params, parcel_ts_moist))
    env_Tv = @. lazy(TD.virtual_temperature(thermo_params, ᶜts))

    # Calculate buoyancy from the difference in virtual temperatures
    # restrict to tropospheric buoyancy (generously below 20km) TODO: integrate from LFC to LNB 
    ᶜz = Fields.coordinate_field(state.c.ρ).z
    ᶜbuoyancy = @. lazy(g * (parcel_Tv - env_Tv) / env_Tv * FT(ᶜz < 20_000))

    # Integrate positive buoyancy to get CAPE
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    Operators.column_integral_definite!(out′, @. lazy(max(ᶜbuoyancy, 0)))
    return out′
end

add_diagnostic_variable!(
    short_name = "cape",
    long_name = "Convective Available Potential Energy",
    standard_name = "convective_available_potential_energy",
    units = "J kg^-1",
    comments = "Energy available to a parcel lifted moist adiabatically from the surface. \
                We assume fully reversible phase changes and no precipitation.",
    compute! = compute_cape!,
)

###
# Mean sea level pressure (2d)
###
function compute_mslp(state, cache, _)
    thermo_params = CAP.thermodynamics_params(cache.params)
    g = TD.Parameters.grav(thermo_params)
    ts_level = Fields.level(cache.precomputed.ᶜts, 1)
    R_m_surf = @. lazy(TD.gas_constant_air(thermo_params, ts_level))

    p_level = Fields.level(cache.precomputed.ᶜp, 1)
    t_level = @. lazy(TD.air_temperature(thermo_params, ts_level))
    z_level = Fields.level(Fields.coordinate_field(state.c.ρ).z, 1)

    # Reduce to mean sea level using hypsometric formulation with lapse rate adjustment
    # Using constant lapse rate Γ = 6.5 K/km, with virtual temperature
    # represented via R_m_surf. This reduces biases over
    # very cold or very warm high-topography regions.
    FT = Spaces.undertype(Fields.axes(state.c.ρ))
    Γ = FT(6.5e-3) # K m^-1

    #   p_msl = p_z0 * [1 + Γ * z / T_z0]^( g / (R_m Γ))
    # where:
    #   - p_z0 pressure at the lowest model level
    #   - T_z0 air temperature at the lowest model level
    #   - R_m moist-air gas constant at the surface (R_m_surf), which
    #     accounts for virtual-temperature effects in the exponent
    #   - Γ constant lapse rate (6.5 K/km here)

    return @. lazy(p_level * (1 + Γ * z_level / t_level)^(g / Γ / R_m_surf))
end

add_diagnostic_variable!(
    short_name = "mslp",
    long_name = "Mean Sea Level Pressure",
    standard_name = "mean_sea_level_pressure",
    units = "Pa",
    comments = "Mean sea level pressure computed using a lapse-rate-dependent \
                hypsometric reduction (ERA-style; Γ=6.5 K/km with \
                virtual temperature via moist gas constant).",
    compute = compute_mslp,
)

###
# Rainwater path (2d)
###
compute_rwp!(out, state, cache, time) =
    compute_rwp!(out, state, cache, time, cache.atmos.microphysics_model)
compute_rwp!(_, _, _, _, model) = error_diagnostic_variable("rwp", model)

function compute_rwp!(out, state, _, _,
    ::Union{Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3}
)
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    Operators.column_integral_definite!(out′, state.c.ρq_rai)
    return out′
end

add_diagnostic_variable!(short_name = "rwp", units = "kg m^-2",
    long_name = "Rainwater Path",
    standard_name = "atmosphere_mass_content_of_rainwater",
    comments = """
    The total mass of rainwater per unit area.
    (not just the area of the cloudy portion of the column).
    """,
    compute! = compute_rwp!,
)

###
# Covariances (3d)
###
function compute_covariance_diagnostics!(out, state, cache, time, type)
    turbconv_model = cache.atmos.turbconv_model
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isa(turbconv_model, PrognosticEDMFX)
        ᶜts = cache.precomputed.ᶜts⁰
    else
        ᶜts = cache.precomputed.ᶜts
    end

    # Reuse central compute_covariance function
    (ᶜq′q′, ᶜθ′θ′, ᶜθ′q′) = compute_covariance(
        state, cache, thermo_params, ᶜts,
    )

    result = if type == :qt_qt
        ᶜq′q′
    elseif type == :tht_tht
        ᶜθ′θ′
    elseif type == :qt_tht
        ᶜθ′q′
    else
        error("Unknown variance type")
    end

    if isnothing(out)
        return Base.materialize(result)
    else
        out .= result
    end
end

compute_env_q_tot_variance!(out, state, cache, time) =
    compute_covariance_diagnostics!(out, state, cache, time, :qt_qt)
compute_env_theta_liq_ice_variance!(out, state, cache, time) =
    compute_covariance_diagnostics!(out, state, cache, time, :tht_tht)
compute_env_q_tot_theta_liq_ice_covariance!(out, state, cache, time) =
    compute_covariance_diagnostics!(out, state, cache, time, :qt_tht)

add_diagnostic_variable!(
    short_name = "env_q_tot_variance",
    long_name = "Environment Variance of Total Specific Humidity",
    units = "kg^2 kg^-2",
    compute! = compute_env_q_tot_variance!,
)

add_diagnostic_variable!(
    short_name = "env_theta_liq_ice_variance",
    long_name = "Environment Variance of Liquid Ice Potential Temperature",
    units = "K^2",
    compute! = compute_env_theta_liq_ice_variance!,
)

add_diagnostic_variable!(
    short_name = "env_q_tot_theta_liq_ice_covariance",
    long_name = "Environment Covariance of Total Specific Humidity and Liquid Ice Potential Temperature",
    units = "kg kg^-1 K",
    compute! = compute_env_q_tot_theta_liq_ice_covariance!,
)
