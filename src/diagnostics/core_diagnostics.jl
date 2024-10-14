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
#     out .= TD.relative_humidity.(thermo_params, cache.ᶜts)
# end
#
# 2. Define a function that has the correct signature and calls this function
#
# compute_hur!(out, state, cache, time) =
#     compute_hur!(out, state, cache, time, cache.atmos.moisture_model)
#
# 3. Define a function that returns an error when the model is incorrect
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
        if isnothing(out)
            return copy(state.c.ρ)
        else
            out .= state.c.ρ
        end
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
        if isnothing(out)
            return copy(u_component.(Geometry.UVector.(cache.precomputed.ᶜu)))
        else
            out .= u_component.(Geometry.UVector.(cache.precomputed.ᶜu))
        end
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
        if isnothing(out)
            return copy(v_component.(Geometry.VVector.(cache.precomputed.ᶜu)))
        else
            out .= v_component.(Geometry.VVector.(cache.precomputed.ᶜu))
        end
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
        if isnothing(out)
            return copy(w_component.(Geometry.WVector.(cache.precomputed.ᶜu)))
        else
            out .= w_component.(Geometry.WVector.(cache.precomputed.ᶜu))
        end
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
        if isnothing(out)
            return TD.air_temperature.(thermo_params, cache.precomputed.ᶜts)
        else
            out .= TD.air_temperature.(thermo_params, cache.precomputed.ᶜts)
        end
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
        if isnothing(out)
            return TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜts)
        else
            out .= TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜts)
        end
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
        if isnothing(out)
            return TD.specific_enthalpy.(thermo_params, cache.precomputed.ᶜts)
        else
            out .= TD.specific_enthalpy.(thermo_params, cache.precomputed.ᶜts)
        end
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
        if isnothing(out)
            return copy(cache.precomputed.ᶜp)
        else
            out .= cache.precomputed.ᶜp
        end
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
        if isnothing(out)
            return copy(vort)
        else
            out .= vort
        end
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
        if isnothing(out)
            return copy(cache.precomputed.cloud_diagnostics_tuple.cf) .* 100
        else
            out .= cache.precomputed.cloud_diagnostics_tuple.cf .* 100
        end
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
        if isnothing(out)
            return copy(cache.precomputed.ᶜK)
        else
            out .= cache.precomputed.ᶜK
        end
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
        if isnothing(out)
            return copy(cache.precomputed.ᶜmixing_length)
        else
            out .= cache.precomputed.ᶜmixing_length
        end
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
        if isnothing(out)
            return copy(cache.precomputed.ᶜlinear_buoygrad)
        else
            out .= cache.precomputed.ᶜlinear_buoygrad
        end
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
        if isnothing(out)
            return copy(cache.precomputed.ᶜstrain_rate_norm)
        else
            out .= cache.precomputed.ᶜstrain_rate_norm
        end
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
    if isnothing(out)
        return TD.relative_humidity.(thermo_params, cache.precomputed.ᶜts)
    else
        out .= TD.relative_humidity.(thermo_params, cache.precomputed.ᶜts)
    end
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
    if isnothing(out)
        return state.c.ρq_tot ./ state.c.ρ
    else
        out .= state.c.ρq_tot ./ state.c.ρ
    end
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
    if isnothing(out)
        return copy(cache.precomputed.cloud_diagnostics_tuple.q_liq)
    else
        out .= cache.precomputed.cloud_diagnostics_tuple.q_liq
    end
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
    if isnothing(out)
        return copy(cache.precomputed.cloud_diagnostics_tuple.q_ice)
    else
        out .= cache.precomputed.cloud_diagnostics_tuple.q_ice
    end
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
    if isnothing(out)
        return TD.total_specific_humidity.(
            thermo_params,
            cache.precomputed.sfc_conditions.ts,
        )
    else
        out .=
            TD.total_specific_humidity.(
                thermo_params,
                cache.precomputed.sfc_conditions.ts,
            )
    end
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
        if isnothing(out)
            return TD.air_temperature.(
                thermo_params,
                cache.precomputed.sfc_conditions.ts,
            )
        else
            out .=
                TD.air_temperature.(
                    thermo_params,
                    cache.precomputed.sfc_conditions.ts,
                )
        end
    end,
)

###
# Near-surface air temperature (2d)
###
add_diagnostic_variable!(
    short_name = "tas",
    long_name = "Near-Surface Air Temperature",
    standard_name = "air_temperature",
    units = "K",
    comments = "Temperature at the bottom cell center of the atmosphere",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return TD.air_temperature.(
                thermo_params,
                Fields.level(cache.precomputed.ᶜts, 1),
            )
        else
            out .=
                TD.air_temperature.(
                    thermo_params,
                    Fields.level(cache.precomputed.ᶜts, 1),
                )
        end
    end,
)

###
# Near-surface U velocity (2d)
###
add_diagnostic_variable!(
    short_name = "uas",
    long_name = "Eastward Near-Surface Wind",
    standard_name = "eastward_wind",
    units = "m s^-1",
    comments = "Eastward component of the wind at the bottom cell center of the atmosphere",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(
                u_component.(
                    Geometry.UVector.(Fields.level(cache.precomputed.ᶜu, 1))
                ),
            )
        else
            out .=
                u_component.(
                    Geometry.UVector.(Fields.level(cache.precomputed.ᶜu, 1))
                )
        end
    end,
)

###
# Near-surface V velocity (2d)
###
add_diagnostic_variable!(
    short_name = "vas",
    long_name = "Northward Near-Surface Wind",
    standard_name = "northward_wind",
    units = "m s^-1",
    comments = "Northward (meridional) wind component at the bottom cell center of the atmosphere",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(
                v_component.(
                    Geometry.VVector.(Fields.level(cache.precomputed.ᶜu, 1))
                ),
            )
        else
            out .=
                v_component.(
                    Geometry.VVector.(Fields.level(cache.precomputed.ᶜu, 1))
                )
        end
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
    if isnothing(out)
        return dot.(ρ_flux_h_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_h_tot, surface_ct3_unit)
    end
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

    if isnothing(out)
        return dot.(ρ_flux_q_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_q_tot, surface_ct3_unit)
    end
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
    if isnothing(out)
        return cache.precipitation.surface_rain_flux .+
               cache.precipitation.surface_snow_flux
    else
        out .=
            cache.precipitation.surface_rain_flux .+
            cache.precipitation.surface_snow_flux
    end
end

add_diagnostic_variable!(
    short_name = "pr",
    long_name = "Precipitation",
    standard_name = "precipitation",
    units = "kg m^-2 s^-1",
    comments = "Total precipitation including rain and snow",
    compute! = compute_pr!,
)

compute_prra!(out, state, cache, time) =
    compute_prra!(out, state, cache, time, cache.atmos.precip_model)
compute_prra!(_, _, _, _, precip_model::T) where {T} =
    error_diagnostic_variable("prra", precip_model)

function compute_prra!(
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
    if isnothing(out)
        return cache.precipitation.surface_rain_flux
    else
        out .= cache.precipitation.surface_rain_flux
    end
end

add_diagnostic_variable!(
    short_name = "prra",
    long_name = "Rainfall Flux",
    standard_name = "rainfall_flux",
    units = "kg m^-2 s^-1",
    comments = "Precipitation including all forms of water in the liquid phase",
    compute! = compute_prra!,
)

compute_prsn!(out, state, cache, time) =
    compute_prsn!(out, state, cache, time, cache.atmos.precip_model)
compute_prsn!(_, _, _, _, precip_model::T) where {T} =
    error_diagnostic_variable("prsn", precip_model)

function compute_prsn!(
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
    if isnothing(out)
        return cache.precipitation.surface_snow_flux
    else
        out .= cache.precipitation.surface_snow_flux
    end
end

add_diagnostic_variable!(
    short_name = "prsn",
    long_name = "Snowfall Flux",
    standard_name = "snowfall_flux",
    units = "kg m^-2 s^-1",
    comments = "Precipitation including all forms of water in the solid phase",
    compute! = compute_prsn!,
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
    if isnothing(out)
        return state.c.ρq_rai ./ state.c.ρ
    else
        out .= state.c.ρq_rai ./ state.c.ρ
    end
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
    if isnothing(out)
        return state.c.ρq_sno ./ state.c.ρ
    else
        out .= state.c.ρq_sno ./ state.c.ρ
    end
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

###
# Condensed water path (2d)
###
compute_clwvi!(out, state, cache, time) =
    compute_clwvi!(out, state, cache, time, cache.atmos.moisture_model)
compute_clwvi!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clwvi", model)

function compute_clwvi!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        clw = cache.scratch.ᶜtemp_scalar
        @. clw =
            state.c.ρ * (
                cache.precomputed.cloud_diagnostics_tuple.q_liq +
                cache.precomputed.cloud_diagnostics_tuple.q_ice
            )
        Operators.column_integral_definite!(out, clw)
        return out
    else
        clw = cache.scratch.ᶜtemp_scalar
        @. clw =
            state.c.ρ * (
                cache.precomputed.cloud_diagnostics_tuple.q_liq +
                cache.precomputed.cloud_diagnostics_tuple.q_ice
            )
        Operators.column_integral_definite!(out, clw)
    end
end

add_diagnostic_variable!(
    short_name = "clwvi",
    long_name = "Condensed Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_condensed_water",
    units = "kg m-2",
    comments = """
    Mass of condensed (liquid + ice) water in the column divided by the area of the column 
    (not just the area of the cloudy portion of the column). It doesn't include precipitating hydrometeors.
    """,
    compute! = compute_clwvi!,
)

###
# Liquid water path (2d)
###
compute_lwp!(out, state, cache, time) =
    compute_lwp!(out, state, cache, time, cache.atmos.moisture_model)
compute_lwp!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("lwp", model)

function compute_lwp!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        lw = cache.scratch.ᶜtemp_scalar
        @. lw = state.c.ρ * cache.precomputed.cloud_diagnostics_tuple.q_liq
        Operators.column_integral_definite!(out, lw)
        return out
    else
        lw = cache.scratch.ᶜtemp_scalar
        @. lw = state.c.ρ * cache.precomputed.cloud_diagnostics_tuple.q_liq
        Operators.column_integral_definite!(out, lw)
    end
end

add_diagnostic_variable!(
    short_name = "lwp",
    long_name = "Liquid Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_liquid_water",
    units = "kg m-2",
    comments = """
    The total mass of liquid water in cloud per unit area.
    (not just the area of the cloudy portion of the column). It doesn't include precipitating hydrometeors.
    """,
    compute! = compute_lwp!,
)

###
# Ice water path (2d)
###
compute_clivi!(out, state, cache, time) =
    compute_clivi!(out, state, cache, time, cache.atmos.moisture_model)
compute_clivi!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clivi", model)

function compute_clivi!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        cli = cache.scratch.ᶜtemp_scalar
        @. cli = state.c.ρ * cache.precomputed.cloud_diagnostics_tuple.q_ice
        Operators.column_integral_definite!(out, cli)
        return out
    else
        cli = cache.scratch.ᶜtemp_scalar
        @. cli = state.c.ρ * cache.precomputed.cloud_diagnostics_tuple.q_ice
        Operators.column_integral_definite!(out, cli)
    end
end

add_diagnostic_variable!(
    short_name = "clivi",
    long_name = "Ice Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_ice",
    units = "kg m-2",
    comments = """
    The total mass of ice in cloud per unit area.
    (not just the area of the cloudy portion of the column). It doesn't include precipitating hydrometeors.
    """,
    compute! = compute_clivi!,
)


###
# Vertical integrated dry static energy (2d)
###
function compute_dsevi!(out, state, cache, time)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        cp = CAP.cp_d(cache.params)
        dse = cache.scratch.ᶜtemp_scalar
        @. dse =
            state.c.ρ * (
                cp * TD.air_temperature(thermo_params, cache.precomputed.ᶜts) +
                cache.core.ᶜΦ
            )
        Operators.column_integral_definite!(out, dse)
        return out
    else
        cp = CAP.cp_d(cache.params)
        dse = cache.scratch.ᶜtemp_scalar
        @. dse =
            state.c.ρ * (
                cp * TD.air_temperature(thermo_params, cache.precomputed.ᶜts) +
                cache.core.ᶜΦ
            )
        Operators.column_integral_definite!(out, dse)
    end
end

add_diagnostic_variable!(
    short_name = "dsevi",
    long_name = "Dry Static Energy Vertical Integral",
    units = "",
    comments = """
    """,
    compute! = compute_dsevi!,
)

###
# column integrated cloud fraction (2d)
###
compute_clvi!(out, state, cache, time) =
    compute_clvi!(out, state, cache, time, cache.atmos.moisture_model)
compute_clvi!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clvi", model)

function compute_clvi!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        cloud_cover = cache.scratch.ᶜtemp_scalar
        FT = Spaces.undertype(axes(cloud_cover))
        @. cloud_cover = ifelse(
            cache.precomputed.cloud_diagnostics_tuple.cf > zero(FT),
            one(FT),
            zero(FT),
        )
        Operators.column_integral_definite!(out, cloud_cover)
        return out
    else
        cloud_cover = cache.scratch.ᶜtemp_scalar
        FT = Spaces.undertype(axes(cloud_cover))
        @. cloud_cover = ifelse(
            cache.precomputed.cloud_diagnostics_tuple.cf > zero(FT),
            one(FT),
            zero(FT),
        )
        Operators.column_integral_definite!(out, cloud_cover)
    end
end

add_diagnostic_variable!(
    short_name = "clvi",
    long_name = "Vertical Cloud Fraction Integral",
    units = "m",
    comments = """
    The total height of the column occupied at least partially by cloud.
    """,
    compute! = compute_clvi!,
)


###
# Column integrated total specific humidity (2d)
###
compute_prw!(out, state, cache, time) =
    compute_prw!(out, state, cache, time, cache.atmos.moisture_model)
compute_prw!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("prw", model)

function compute_prw!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        Operators.column_integral_definite!(out, state.c.ρq_tot)
        return out
    else
        Operators.column_integral_definite!(out, state.c.ρq_tot)
    end
end

add_diagnostic_variable!(
    short_name = "prw",
    long_name = "Water Vapor Path",
    standard_name = "atmospheric_mass_content_of_water_vapor",
    units = "kg m^-2",
    comments = "Vertically integrated specific humidity",
    compute! = compute_prw!,
)

###
# Column integrated relative humidity (2d)
###
compute_hurvi!(out, state, cache, time) =
    compute_hurvi!(out, state, cache, time, cache.atmos.moisture_model)
compute_hurvi!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hurvi", model)

function compute_hurvi!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        # compute vertical integral of saturation specific humidity
        # note next line currently allocates; currently no correct scratch space
        sat_vi = zeros(axes(Fields.level(state.f, half)))
        sat = cache.scratch.ᶜtemp_scalar
        @. sat =
            state.c.ρ *
            TD.q_vap_saturation(thermo_params, cache.precomputed.ᶜts)
        Operators.column_integral_definite!(sat_vi, sat)
        # compute saturation-weighted vertical integral of specific humidity
        hur = cache.scratch.ᶜtemp_scalar
        @. hur = TD.vapor_specific_humidity(
            CAP.thermodynamics_params(cache.params),
            cache.precomputed.ᶜts,
        )
        hur_weighted = cache.scratch.ᶜtemp_scalar_2
        @. hur_weighted = state.c.ρ * hur / sat_vi
        Operators.column_integral_definite!(out, hur_weighted)
        return out
    else
        # compute vertical integral of saturation specific humidity
        # note next line currently allocates; currently no correct scratch space
        sat_vi = zeros(axes(Fields.level(state.f, half)))
        sat = cache.scratch.ᶜtemp_scalar
        @. sat =
            state.c.ρ *
            TD.q_vap_saturation(thermo_params, cache.precomputed.ᶜts)
        Operators.column_integral_definite!(sat_vi, sat)
        # compute saturation-weighted vertical integral of specific humidity
        hur = cache.scratch.ᶜtemp_scalar
        @. hur = TD.vapor_specific_humidity(
            CAP.thermodynamics_params(cache.params),
            cache.precomputed.ᶜts,
        )
        hur_weighted = cache.scratch.ᶜtemp_scalar_2
        @. hur_weighted = state.c.ρ * hur / sat_vi
        Operators.column_integral_definite!(out, hur_weighted)
    end
end

add_diagnostic_variable!(
    short_name = "hurvi",
    long_name = "Relative Humidity Saturation-Weighted Vertical Integral",
    standard_name = "relative_humidity_vi",
    units = "kg m^-2",
    comments = "Integrated relative humidity over the vertical column",
    compute! = compute_hurvi!,
)


###
# Vapor specific humidity (3d)
###
compute_husv!(out, state, cache, time) =
    compute_husv!(out, state, cache, time, cache.atmos.moisture_model)
compute_husv!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("husv", model)

function compute_husv!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.vapor_specific_humidity.(
            CAP.thermodynamics_params(cache.params),
            cache.precomputed.ᶜts,
        )
    else
        out .=
            TD.vapor_specific_humidity.(
                CAP.thermodynamics_params(cache.params),
                cache.precomputed.ᶜts,
            )
    end
end

add_diagnostic_variable!(
    short_name = "husv",
    long_name = "Vapor Specific Humidity",
    units = "kg kg^-1",
    comments = "Mass of water vapor per mass of air",
    compute! = compute_husv!,
)

###
# Implicit solver
###
function compute_ejac1!(out, state, cache, time)
    FT = eltype(state)
    device = ClimaComms.device(state)
    first_column_jacobian = view(cache.jacobian.cache.column_matrices, 1, :, :)
    if isnothing(out)
        column_length = length(first(column_iterator(state)))
        out = Array{FT}(undef, column_length, column_length)
    end
    ClimaComms.allowscalar(copyto!, device, out, first_column_jacobian)
    view(out, LinearAlgebra.diagind(out)) .+= 1
    out ./= cache.jacobian.cache.dtγ_ref[]
end

add_diagnostic_variable!(
    short_name = "ejac1",
    long_name = "Exact Jacobian matrix of first column",
    standard_name = "exact_jacobian",
    units = "",
    comments = "Exact Jacobian matrix of tendency in first column",
    compute! = compute_ejac1!,
)

tensor_axes_tuple(::Type{T}) where {T} =
    T <: Geometry.AxisTensor ?
    map(axis -> typeof(axis).parameters[1], axes(T)) : ()

# TODO: Move this function to ClimaCore.
# In order for the following function to be used in Field broadcast expressions,
# it has to be type-stable. As of Julia 1.10, this means that it needs to use
# unrolled functions, and that its default recursion limit must be disabled.
primitive_value_at_index(value, (row_axes, col_axes)) =
    if isprimitivetype(typeof(value)) # same as a LinearAlgebra.UniformScaling
        row_axes == col_axes ? value : zero(value)
    elseif value isa Geometry.AxisVector
        @assert isprimitivetype(eltype(value))
        @assert length(row_axes) == 1 && length(col_axes) == 0
        value_axes = tensor_axes_tuple(typeof(value))
        row_axis_index = findfirst(==(row_axes[1]), value_axes[1])
        isnothing(row_axis_index) ? zero(eltype(value)) : value[row_axis_index]
    elseif value isa Geometry.AxisTensor
        @assert isprimitivetype(eltype(value))
        @assert length(row_axes) == 1 && length(col_axes) == 1
        value_axes = tensor_axes_tuple(typeof(value))
        row_axis_index = findfirst(==(row_axes[1]), value_axes[1])
        col_axis_index = findfirst(==(col_axes[1]), value_axes[2])
        isnothing(row_axis_index) || isnothing(col_axis_index) ?
        zero(eltype(value)) : value[row_axis_index, col_axis_index]
    elseif value isa LinearAlgebra.Adjoint
        primitive_value_at_index(parent(value), (col_axes, row_axes))
    else
        sub_names = fieldnames(typeof(value))
        sub_values =
            MatrixFields.unrolled_map(Base.Fix1(getfield, value), sub_names)
        nonempty_sub_values =
            MatrixFields.unrolled_filter(x -> sizeof(x) > 0, sub_values)
        @assert length(nonempty_sub_values) == 1
        primitive_value_at_index(nonempty_sub_values[1], (row_axes, col_axes))
    end

@static if hasfield(Method, :recursion_relation)
    for method in methods(primitive_value_at_index)
        method.recursion_relation = Returns(true)
    end
end

function compute_ajac1!(out, state, cache, time)
    FT = eltype(state)
    device = ClimaComms.device(state)
    field_names = scalar_field_names(state)
    index_ranges = scalar_field_index_ranges(state)
    if isnothing(out)
        column_length = length(first(column_iterator(state)))
        out = Array{FT}(undef, column_length, column_length)
    end
    out .= FT(0)
    for ((block_row, block_col), matrix_block) in cache.jacobian.cache.matrix
        is_child_name_of_row = Base.Fix2(MatrixFields.is_child_name, block_row)
        is_child_name_of_col = Base.Fix2(MatrixFields.is_child_name, block_col)
        subblock_row_indices = findall(is_child_name_of_row, field_names)
        subblock_col_indices = findall(is_child_name_of_col, field_names)
        for (sub_row, subblock_row_index) in enumerate(subblock_row_indices)
            for (sub_col, subblock_col_index) in enumerate(subblock_col_indices)
                row_index_range = index_ranges[subblock_row_index]
                col_index_range = index_ranges[subblock_col_index]
                out_subblock = view(out, row_index_range, col_index_range)
                if matrix_block isa LinearAlgebra.UniformScaling
                    view(out_subblock, LinearAlgebra.diagind(out_subblock)) .=
                        sub_row == sub_col ? matrix_block.λ :
                        zero(matrix_block.λ)
                else
                    block_row_field = MatrixFields.get_field(state, block_row)
                    block_col_field = MatrixFields.get_field(state, block_col)
                    subblock_row_axes = map(
                        Base.Fix2(getindex, sub_row),
                        tensor_axes_tuple(eltype(block_row_field)),
                    )
                    subblock_col_axes = map(
                        Base.Fix2(getindex, sub_col),
                        tensor_axes_tuple(eltype(block_col_field)),
                    )
                    @assert length(subblock_row_axes) in (0, 1)
                    @assert length(subblock_col_axes) in (0, 1)
                    value_in_subblock = Base.Fix2(
                        primitive_value_at_index,
                        (subblock_row_axes, subblock_col_axes),
                    )
                    column_block = Fields.column(matrix_block, 1, 1, 1)
                    column_subblock = map.(value_in_subblock, column_block)
                    ClimaComms.allowscalar(
                        copyto!,
                        device,
                        out_subblock,
                        MatrixFields.column_field2array_view(column_subblock),
                    )
                end
            end
        end
    end
    view(out, LinearAlgebra.diagind(out)) .+= FT(1)
    out ./= cache.jacobian.cache.dtγ_ref[]
end

add_diagnostic_variable!(
    short_name = "ajac1",
    long_name = "Approximate Jacobian matrix of first column",
    standard_name = "approx_jacobian",
    units = "",
    comments = "Approximate Jacobian matrix of tendency in first column",
    compute! = compute_ajac1!,
)

add_diagnostic_variable!(
    short_name = "ajacerr1",
    long_name = "Error of approximate Jacobian matrix of first column",
    standard_name = "approx_jacobian_error",
    units = "",
    comments = "Error of approximate Jacobian matrix of tendency in first column",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            compute_ajac1!(nothing, state, cache, time) .-
            compute_ejac1!(nothing, state, cache, time)
        else
            compute_ajac1!(out, state, cache, time)
            out .-= compute_ejac1!(nothing, state, cache, time)
            # TODO: Rewrite this to avoid allocations.
        end
    end,
)
