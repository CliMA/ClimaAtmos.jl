# This file is included in Diagnostics.jl
#
# README: Adding a new core diagnostic:
#
# In addition to the metadata (names, comments, ...), the most important step in adding a
# new DiagnosticVariable is defining its compute! function. `compute!` has to take four
# arguments: (out, state, cache, time), and as to write the diagnostic in place into the
# `out` variable.
#
# Often, it is possible to compute certain diagnostics only for specific models (e.g.,
# humidity for moist models). For that, it is convenient to adopt the following pattern:
#
# 1. Define a catch base function that does the computation we want to do for the case we know
# how to handle, for example
#
# function compute_relative_humidity!(
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
# compute_relative_humidity!(out, state, cache, time) =
#     compute_relative_humidity!(out, state, cache, time, cache.atmos.moisture_model)
#
# 3. Define a function that returns an error when the model is incorrect
#
# compute_relative_humidity!(_, _, _, _, model::T) where {T} =
#     error_diagnostic_variable("relative_humidity", model)
#
# We can also output a specific error message
#
# compute_relative_humidity!(_, _, _, _, model::T) where {T} =
#     error_diagnostic_variable("relative humidity makes sense only for moist models")

# General helper functions for undefined diagnostics for a particular model
error_diagnostic_variable(
    message = "Cannot compute $variable with model = $T",
) = error(message)
error_diagnostic_variable(variable, model::T) where {T} =
    error_diagnostic_variable("Cannot compute $variable with model = $T")

###
# Rho (3d)
###
add_diagnostic_variable!(
    short_name = "air_density",
    long_name = "Air Density",
    units = "kg m^-3",
    comments = "Density of air, a prognostic variable",
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

# TODO: This velocity might not be defined (e.g., in a column model). Add dispatch to catch
# that.
add_diagnostic_variable!(
    short_name = "eastward_wind",
    long_name = "Eastward Wind",
    units = "m s^-1",
    comments = "Eastward (zonal) wind component, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(Geometry.UVector.(cache.ᶜu))
        else
            out .= Geometry.UVector.(cache.ᶜu)
        end
    end,
)

###
# V velocity (3d)
###

# TODO: This velocity might not be defined (e.g., in a column model). Add dispatch to catch
# that.
add_diagnostic_variable!(
    short_name = "northward_wind",
    long_name = "Northward Wind",
    units = "m s^-1",
    comments = "Northward (meridional) wind component, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(Geometry.VVector.(cache.ᶜu))
        else
            out .= Geometry.VVector.(cache.ᶜu)
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
    short_name = "vertical_wind",
    long_name = "Vertical Wind",
    units = "m s^-1",
    comments = "Vertical wind component, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(Geometry.WVector.(cache.ᶜu))
        else
            out .= Geometry.WVector.(cache.ᶜu)
        end
    end,
)

###
# Temperature (3d)
###
add_diagnostic_variable!(
    short_name = "air_temperature",
    long_name = "Air Temperature",
    units = "K",
    comments = "Temperature of air",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return TD.air_temperature.(thermo_params, cache.ᶜts)
        else
            out .= TD.air_temperature.(thermo_params, cache.ᶜts)
        end
    end,
)

###
# Potential temperature (3d)
###
add_diagnostic_variable!(
    short_name = "air_potential_temperature",
    long_name = "Air potential temperature",
    units = "K",
    comments = "Potential temperature of air",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return TD.dry_pottemp.(thermo_params, cache.ᶜts)
        else
            out .= TD.dry_pottemp.(thermo_params, cache.ᶜts)
        end
    end,
)

###
# Air pressure (3d)
###
add_diagnostic_variable!(
    short_name = "air_pressure",
    long_name = "Air pressure",
    units = "Pa",
    comments = "Pressure of air",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.ᶜp)
        else
            out .= cache.ᶜp
        end
    end,
)

###
# Vorticity (3d)
###
add_diagnostic_variable!(
    short_name = "atmosphere_relative_vorticity",
    long_name = "Vertical component of relative vorticity",
    units = "s^-1",
    comments = "Vertical component of relative vorticity",
    compute! = (out, state, cache, time) -> begin
        ᶜvort = @. Geometry.WVector(curlₕ(state.c.uₕ))
        # We need to ensure smoothness, so we call DSS
        Spaces.weighted_dss!(ᶜvort)
        if isnothing(out)
            return copy(ᶜvort)
        else
            out .= ᶜvort
        end
    end,
)


###
# Relative humidity (3d)
###
compute_relative_humidity!(out, state, cache, time) =
    compute_relative_humidity!(
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )
compute_relative_humidity!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("relative_humidity", model)

function compute_relative_humidity!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.relative_humidity.(thermo_params, cache.ᶜts)
    else
        out .= TD.relative_humidity.(thermo_params, cache.ᶜts)
    end
end

add_diagnostic_variable!(
    short_name = "relative_humidity",
    long_name = "Relative Humidity",
    units = "",
    comments = "Total amount of water vapor in the air relative to the amount achievable by saturation at the current temperature",
    compute! = compute_relative_humidity!,
)

###
# Total specific humidity (3d)
###
compute_specific_humidity!(out, state, cache, time) =
    compute_specific_humidity!(
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )
compute_specific_humidity!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("specific_humidity", model)

function compute_specific_humidity!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.total_specific_humidity.(thermo_params, cache.ᶜts)
    else
        out .= TD.total_specific_humidity.(thermo_params, cache.ᶜts)
    end
end

add_diagnostic_variable!(
    short_name = "specific_humidity",
    long_name = "Specific Humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air, a prognostic variable",
    compute! = compute_specific_humidity!,
)

###
# Surface specific humidity (2d)
###
compute_surface_specific_humidity!(out, state, cache, time) =
    compute_surface_specific_humidity!(
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )
compute_surface_specific_humidity!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("surface_specific_humidity", model)

function compute_surface_specific_humidity!(
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
            cache.sfc_conditions.ts,
        )
    else
        out .=
            TD.total_specific_humidity.(thermo_params, cache.sfc_conditions.ts)
    end
end

add_diagnostic_variable!(
    short_name = "surface_specific_humidity",
    long_name = "Surface Specific Humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air in the near-surface layer",
    compute! = compute_surface_specific_humidity!,
)

###
# Surface temperature (2d)
###
function compute_surface_temperature!(out, state, cache, time)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.air_temperature.(thermo_params, cache.sfc_conditions.ts)
    else
        out .= TD.air_temperature.(thermo_params, cache.sfc_conditions.ts)
    end
end

add_diagnostic_variable!(
    short_name = "surface_temperature",
    long_name = "Surface Temperature",
    units = "K",
    comments = "Temperature of the surface",
    compute! = compute_surface_temperature!,
)

###
# Eastward surface drag component (2d)
###
compute_eastward_drag!(out, state, cache, time) =
    compute_eastward_drag!(out, state, cache, time, cache.atmos.energy_form)
compute_eastward_drag!(_, _, _, _, energy_form::T) where {T} =
    error_diagnostic_variable("eastward_drag", energy_form)

function drag_vector(state, cache)
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    (; ρ_flux_uₕ) = cache.sfc_conditions
    return Geometry.UVVector.(
        adjoint.(ρ_flux_uₕ ./ Spaces.level(ᶠinterp.(state.c.ρ), half)) .*
        surface_ct3_unit
    )
end

function compute_eastward_drag!(
    out,
    state,
    cache,
    time,
    energy_form::T,
) where {T <: TotalEnergy}
    if isnothing(out)
        return drag_vector(state, cache).components.data.:1
    else
        out .= drag_vector(state, cache).components.data.:1
    end
end

add_diagnostic_variable!(
    short_name = "eastward_drag",
    long_name = "Eastward component of the surface drag",
    units = "kg m^-2 s^-2",
    comments = "Eastward component of the surface drag",
    compute! = compute_eastward_drag!,
)

###
# Northward surface drag component (2d)
###
compute_northward_drag!(out, state, cache, time) =
    compute_northward_drag!(out, state, cache, time, cache.atmos.energy_form)
compute_northward_drag!(_, _, _, _, energy_form::T) where {T} =
    error_diagnostic_variable("northward_drag", energy_form)

function compute_northward_drag!(
    out,
    state,
    cache,
    time,
    energy_form::T,
) where {T <: TotalEnergy}
    if isnothing(out)
        return drag_vector(state, cache).components.data.:2
    else
        out .= drag_vector(state, cache).components.data.:2
    end
end

add_diagnostic_variable!(
    short_name = "northward_drag",
    long_name = "Northward component of the surface drag",
    units = "kg m^-2 s^-2",
    comments = "Northward component of the surface drag",
    compute! = compute_northward_drag!,
)

###
# Surface energy flux (2d) - TODO: this may need to be split into sensible and latent heat fluxes
###
function compute_surface_energy_flux!(out, state, cache, time)
    (; ρ_flux_h_tot) = cache.sfc_conditions
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    if isnothing(out)
        return dot.(ρ_flux_h_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_h_tot, surface_ct3_unit)
    end
end

add_diagnostic_variable!(
    short_name = "surface_energy_flux",
    long_name = "Surface energy flux",
    units = "W m^-2",
    comments = "Energy flux at the surface",
    compute! = compute_surface_energy_flux!,
)

###
# Surface evaporation (2d)
###
compute_surface_evaporation!(out, state, cache, time) =
    compute_surface_evaporation!(
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
        cache.atmos.energy_form,
    )
compute_surface_evaporation!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    energy_form::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute surface_evaporation with energy_form = TotalEnergy() and with a moist model",
)

function compute_surface_evaporation!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
    energy_form::TotalEnergy,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    (; ρ_flux_q_tot) = cache.sfc_conditions
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))

    if isnothing(out)
        return dot.(ρ_flux_q_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_q_tot, surface_ct3_unit)
    end
end

add_diagnostic_variable!(
    short_name = "surface_evaporation",
    long_name = "Surface evaporation",
    units = "kg s^-1 m^-2",
    comments = "evaporation at the surface",
    compute! = compute_surface_evaporation!,
)
