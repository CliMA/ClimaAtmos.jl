# This file is included in Diagnostics.jl
#
# README: Adding a new core diagnostic:
#
# In addition to the metadata (names, comments, ...), the most important step in adding a
# new DiagnosticVariable is defining its compute! function. `compute!` has to take four
# arguments: (out, state, cache, time), and has to write the diagnostic in place into the
# `out` variable.
#
# Often, it is possible to compute certain diagnostics only for specific models (e.g.,
# humidity for moist models). For that, it is convenient to adopt the following pattern:
#
# 1. Define a catch base function that does the computation we want to do for the case we know
# how to handle, for example
#
# function compute_hur!(
#     out,
#     state,
#     cache,
#     time,
#     ::T,
# ) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
#     thermo_params = CAP.thermodynamics_params(cache.params)
#     @lazy @. out = TD.relative_humidity.(thermo_params, cache.ᶜts)
# end
#
# 2. Define a function that has the correct signature and calls this function
#
# compute_hur!(out, state, cache, time) =
#     compute_hur!(out, state, cache, time, cache.atmos.moisture_model)
#
#
# compute_hur!(_, _, _, _, model::T) where {T} =
#     error_diagnostic_variable("relative_humidity", model)
#
# We can also output a specific error message
#
# compute_hur!(_, _, _, _, model::T) where {T} =
#     error_diagnostic_variable("relative humidity makes sense only for moist models")

# General helper functions for undefined diagnostics for a particular model
error_diagnostic_variable(
    message = "Cannot compute $variable with model = $T",
) = error(message)
error_diagnostic_variable(variable, model::T) where {T} =
    error_diagnostic_variable("Cannot compute $variable with model = $T")

###
# Density (3d)
###
add_diagnostic_variable!(
    short_name = "rhoa",
    long_name = "Air Density",
    standard_name = "air_density",
    units = "kg m^-3",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = state.c.ρ
    end,
)

###
# U velocity (3d)
###
add_diagnostic_variable!(
    short_name = "ua",
    long_name = "Eastward Wind",
    standard_name = "eastward_wind",
    units = "m s^-1",
    comments = "Eastward (zonal) wind component",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = u_component(Geometry.UVector.(cache.precomputed.ᶜu))
    end,
)

###
# V velocity (3d)
###
add_diagnostic_variable!(
    short_name = "va",
    long_name = "Northward Wind",
    standard_name = "northward_wind",
    units = "m s^-1",
    comments = "Northward (meridional) wind component",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = v_component(Geometry.VVector.(cache.precomputed.ᶜu))
    end,
)

###
# W velocity (3d)
###
# TODO: may want to convert to omega (Lagrangian pressure tendency) as standard output,
# but this is probably more useful for now
#
add_diagnostic_variable!(
    short_name = "wa",
    long_name = "Upward Air Velocity",
    standard_name = "upward_air_velocity",
    units = "m s^-1",
    comments = "Vertical wind component",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = w_component(Geometry.WVector.(cache.precomputed.ᶜu))
    end,
)

###
# Temperature (3d)
###
add_diagnostic_variable!(
    short_name = "ta",
    long_name = "Air Temperature",
    standard_name = "air_temperature",
    units = "K",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        @lazy @. out =
            TD.air_temperature(thermo_params, cache.precomputed.ᶜts)
    end,
)

###
# Potential temperature (3d)
###
add_diagnostic_variable!(
    short_name = "thetaa",
    long_name = "Air Potential Temperature",
    standard_name = "air_potential_temperature",
    units = "K",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        @lazy @. out = TD.dry_pottemp(thermo_params, cache.precomputed.ᶜts)
    end,
)

###
# Enthalpy (3d)
###
add_diagnostic_variable!(
    short_name = "ha",
    long_name = "Air Specific Enthalpy",
    units = "m^2 s^-2",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        @lazy @. out =
            TD.specific_enthalpy(thermo_params, cache.precomputed.ᶜts)
    end,
)

###
# Air pressure (3d)
###
add_diagnostic_variable!(
    short_name = "pfull",
    long_name = "Pressure at Model Full-Levels",
    units = "Pa",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = cache.precomputed.ᶜp
    end,
)

###
# Vorticity (3d)
###
add_diagnostic_variable!(
    short_name = "rv",
    long_name = "Relative Vorticity",
    standard_name = "relative_vorticity",
    units = "s^-1",
    comments = "Vertical component of relative vorticity",
    compute! = (out, state, cache, time) -> begin
        vort = @. w_component.(Geometry.WVector.(cache.precomputed.ᶜu))
        # We need to ensure smoothness, so we call DSS
        Spaces.weighted_dss!(vort)
        @lazy @. out = vort
    end,
)

###
# Cloud fraction (3d)
###
add_diagnostic_variable!(
    short_name = "cl",
    long_name = "Cloud fraction",
    units = "%",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = cache.precomputed.cloud_diagnostics_tuple.cf .* 100
    end,
)

###
# Total kinetic energy
###
add_diagnostic_variable!(
    short_name = "ke",
    long_name = "Total Kinetic Energy",
    standard_name = "total_kinetic_energy",
    units = "m^2 s^-2",
    comments = "The kinetic energy on cell centers",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = cache.precomputed.ᶜK
    end,
)

###
# Mixing length (3d)
###
add_diagnostic_variable!(
    short_name = "lmix",
    long_name = "Environment Mixing Length",
    units = "m",
    comments = """
    Calculated as smagorinsky length scale without EDMF SGS model,
    or from mixing length closure with EDMF SGS model.
    """,
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = cache.precomputed.ᶜmixing_length
    end,
)

###
# Buoyancy gradient (3d)
###
add_diagnostic_variable!(
    short_name = "bgrad",
    long_name = "Linearized Buoyancy Gradient",
    units = "s^-2",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = cache.precomputed.ᶜlinear_buoygrad
    end,
)

###
# Strain rate magnitude (3d)
###
add_diagnostic_variable!(
    short_name = "strain",
    long_name = "String Rate Magnitude",
    units = "s^-2",
    compute! = (out, state, cache, time) -> begin
        @lazy @. out = cache.precomputed.ᶜstrain_rate_norm
    end,
)

###
# Relative humidity (3d)
###
compute_hur!(out, state, cache, time) =
    compute_hur!(out, state, cache, time, cache.atmos.moisture_model)
compute_hur!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hur", model)

function compute_hur!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    @lazy @. out = TD.relative_humidity(thermo_params, cache.precomputed.ᶜts)
end

add_diagnostic_variable!(
    short_name = "hur",
    long_name = "Relative Humidity",
    standard_name = "relative_humidity",
    units = "",
    comments = "Total amount of water vapor in the air relative to the amount achievable by saturation at the current temperature",
    compute! = compute_hur!,
)

###
# Total specific humidity (3d)
###
compute_hus!(out, state, cache, time) =
    compute_hus!(out, state, cache, time, cache.atmos.moisture_model)
compute_hus!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hus", model)

function compute_hus!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    @lazy @. out = state.c.ρq_tot ./ state.c.ρ
end

add_diagnostic_variable!(
    short_name = "hus",
    long_name = "Specific Humidity",
    standard_name = "specific_humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air",
    compute! = compute_hus!,
)

###
# Liquid water specific humidity (3d)
###
compute_clw!(out, state, cache, time) =
    compute_clw!(out, state, cache, time, cache.atmos.moisture_model)
compute_clw!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clw", model)

function compute_clw!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    @lazy @. out = cache.precomputed.cloud_diagnostics_tuple.q_liq
end

add_diagnostic_variable!(
    short_name = "clw",
    long_name = "Mass Fraction of Cloud Liquid Water",
    standard_name = "mass_fraction_of_cloud_liquid_water_in_air",
    units = "kg kg^-1",
    comments = """
    Includes both large-scale and convective cloud.
    This is calculated as the mass of cloud liquid water in the grid cell divided by
    the mass of air (including the water in all phases) in the grid cells.
    """,
    compute! = compute_clw!,
)

###
# Ice water specific humidity (3d)
###
compute_cli!(out, state, cache, time) =
    compute_cli!(out, state, cache, time, cache.atmos.moisture_model)
compute_cli!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("cli", model)

function compute_cli!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    @lazy @. out = cache.precomputed.cloud_diagnostics_tuple.q_ice
end

add_diagnostic_variable!(
    short_name = "cli",
    long_name = "Mass Fraction of Cloud Ice",
    standard_name = "mass_fraction_of_cloud_ice_in_air",
    units = "kg kg^-1",
    comments = """
    Includes both large-scale and convective cloud.
    This is calculated as the mass of cloud ice in the grid cell divided by
    the mass of air (including the water in all phases) in the grid cell.
    """,
    compute! = compute_cli!,
)

###
# Surface specific humidity (2d)
###
compute_hussfc!(out, state, cache, time) =
    compute_hussfc!(out, state, cache, time, cache.atmos.moisture_model)
compute_hussfc!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussfc", model)

function compute_hussfc!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    @lazy @. out = TD.total_specific_humidity(
        thermo_params,
        cache.precomputed.sfc_conditions.ts,
    )
end

add_diagnostic_variable!(
    short_name = "hussfc",
    long_name = "Surface Specific Humidity",
    standard_name = "specific_humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air in the layer infinitely close to the surface",
    compute! = compute_hussfc!,
)

###
# Surface temperature (2d)
###
add_diagnostic_variable!(
    short_name = "ts",
    long_name = "Surface Temperature",
    standard_name = "surface_temperature",
    units = "K",
    comments = "Temperature of the lower boundary of the atmosphere",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        @lazy @. out = TD.air_temperature(
            thermo_params,
            cache.precomputed.sfc_conditions.ts,
        )
    end,
)

###
# Eastward and northward surface drag component (2d)
###
function compute_tau!(out, state, cache, component)
    (; surface_ct3_unit) = cache.core
    (; ρ_flux_uₕ) = cache.precomputed.sfc_conditions

    if isnothing(out)
        return getproperty(
            Geometry.UVVector.(
                adjoint.(ρ_flux_uₕ) .* surface_ct3_unit
            ).components.data,
            component,
        )
    else
        out .= getproperty(
            Geometry.UVVector.(
                adjoint.(ρ_flux_uₕ) .* surface_ct3_unit
            ).components.data,
            component,
        )
    end

    return
end

compute_tauu!(out, state, cache, time) = compute_tau!(out, state, cache, :1)
compute_tauv!(out, state, cache, time) = compute_tau!(out, state, cache, :2)

add_diagnostic_variable!(
    short_name = "tauu",
    long_name = "Surface Downward Eastward Wind Stress",
    standard_name = "downward_eastward_stress",
    units = "Pa",
    comments = "Eastward component of the surface drag",
    compute! = compute_tauu!,
)

add_diagnostic_variable!(
    short_name = "tauv",
    long_name = "Surface Downward Northward Wind Stress",
    standard_name = "downward_northward_stress",
    units = "Pa",
    comments = "Northward component of the surface drag",
    compute! = compute_tauv!,
)

###
# Surface energy flux (2d) - TODO: this may need to be split into sensible and latent heat fluxes
###
function compute_hfes!(out, state, cache, time)
    (; ρ_flux_h_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    @lazy @. out = dot(ρ_flux_h_tot, surface_ct3_unit)
end

add_diagnostic_variable!(
    short_name = "hfes",
    long_name = "Surface Upward Energy Flux",
    units = "W m^-2",
    comments = "Energy flux at the surface",
    compute! = compute_hfes!,
)

###
# Surface evaporation (2d)
###
compute_evspsbl!(out, state, cache, time) =
    compute_evspsbl!(out, state, cache, time, cache.atmos.moisture_model)
compute_evspsbl!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("evspsbl", model)

function compute_evspsbl!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    (; ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core

    @lazy @. out = dot(ρ_flux_q_tot, surface_ct3_unit)
end

add_diagnostic_variable!(
    short_name = "evspsbl",
    long_name = "Evaporation Including Sublimation and Transpiration",
    units = "kg m^-2 s^-1",
    comments = "evaporation at the surface",
    compute! = compute_evspsbl!,
)

###
# Precipitation (2d)
###
compute_pr!(out, state, cache, time) =
    compute_pr!(out, state, cache, time, cache.atmos.precip_model)
compute_pr!(_, _, _, _, precip_model::T) where {T} =
    error_diagnostic_variable("pr", precip_model)

function compute_pr!(
    out,
    state,
    cache,
    time,
    precip_model::Union{
        NoPrecipitation,
        Microphysics0Moment,
        Microphysics1Moment,
    },
)
    @lazy @. out =
        cache.precipitation.surface_rain_flux .+
        cache.precipitation.surface_snow_flux
end

add_diagnostic_variable!(
    short_name = "pr",
    long_name = "Precipitation",
    standard_name = "precipitation",
    units = "kg m^-2 s^-1",
    comments = "Total precipitation including rain and snow",
    compute! = compute_pr!,
)

###
# Precipitation (3d)
###
compute_husra!(out, state, cache, time) =
    compute_husra!(out, state, cache, time, cache.atmos.precip_model)
compute_husra!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("husra", model)

function compute_husra!(
    out,
    state,
    cache,
    time,
    precip_model::Microphysics1Moment,
)
    @lazy @. out = state.c.ρq_rai ./ state.c.ρ
end

add_diagnostic_variable!(
    short_name = "husra",
    long_name = "Mass Fraction of Rain",
    standard_name = "mass_fraction_of_rain_in_air",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of rain water in the grid cell divided by
    the mass of air (dry air + water vapor + cloud condensate) in the grid cells.
    """,
    compute! = compute_husra!,
)

compute_hussn!(out, state, cache, time) =
    compute_hussn!(out, state, cache, time, cache.atmos.precip_model)
compute_hussn!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussn", model)

function compute_hussn!(
    out,
    state,
    cache,
    time,
    precip_model::Microphysics1Moment,
)
    @lazy @. out = state.c.ρq_sno / state.c.ρ
end

add_diagnostic_variable!(
    short_name = "hussn",
    long_name = "Mass Fraction of Snow",
    standard_name = "mass_fraction_of_snow_in_air",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of snow in the grid cell divided by
    the mass of air (dry air + water vapor + cloud condensate) in the grid cells.
    """,
    compute! = compute_hussn!,
)

###
# Topography
###
compute_orog!(out, state, cache, time) =
    compute_orog!(out, state, cache, time, axes(state.c).grid.hypsography)

function compute_orog!(out, state, cache, time, hypsography::Grids.Flat)
    # When we have a Flat topography, we just have to return a field of zeros
    if isnothing(out)
        return zeros(Spaces.horizontal_space(axes(state.c.ρ)))
    else
        # There's shouldn't be much point in this branch, but let's leave it here for
        # consistency
        out .= zeros(Spaces.horizontal_space(axes(state.c.ρ)))
    end
end

function compute_orog!(out, state, cache, time, hypsography)
    if isnothing(out)
        return Geometry.tofloat.(hypsography.surface)
    else
        out .= Geometry.tofloat.(hypsography.surface)
    end
end

add_diagnostic_variable!(
    short_name = "orog",
    long_name = "Surface Altitude",
    standard_name = "surface_altitude",
    units = "m",
    comments = "Elevation of the horizontal coordinates",
    compute! = compute_orog!,
)
