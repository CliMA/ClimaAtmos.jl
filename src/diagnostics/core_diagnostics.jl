# This file is included in Diagnostics.jl

# general helper for undefined functions for a particular model (avoids some repetition)
function compute_variable!(
    variable::Val,
    out,
    state,
    cache,
    time,
    model::T,
) where {T}
    error("Cannot compute $variable with model = $T")
end

###
# Rho (3d)
###
add_diagnostic_variable!(
    short_name = "air_density",
    long_name = "Air Density",
    units = "kg m^-3",
    comments = "Density of air, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return copy(state.c.ρ)
    end,
)

###
# U velocity (3d)
###
add_diagnostic_variable!(
    short_name = "eastward_wind",
    long_name = "Eastward Wind",
    units = "m s^-1",
    comments = "Eastward (zonal) wind component, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return copy(Geometry.UVector.(cache.ᶜu))
    end,
)

###
# V velocity (3d)
###
add_diagnostic_variable!(
    short_name = "northward_wind",
    long_name = "Northward Wind",
    units = "m s^-1",
    comments = "Northward (meridional) wind component, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return copy(Geometry.VVector.(cache.ᶜu))
    end,
)

###
# W velocity (3d)
###
# TODO: may want to convert to omega (Lagrangian pressure tendency) as standard output,
# but this is probably more useful for now
add_diagnostic_variable!(
    short_name = "vertical_wind",
    long_name = "Vertical Wind",
    units = "m s^-1",
    comments = "Vertical wind component, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return copy(Geometry.WVector.(cache.ᶜu))
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
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        thermo_params = CAP.thermodynamics_params(cache.params)
        return copy(TD.air_temperature.(thermo_params, cache.ᶜts))
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
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        thermo_params = CAP.thermodynamics_params(cache.params)
        return copy(TD.dry_pottemp.(thermo_params, cache.ᶜts))
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
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return copy(cache.ᶜp)
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
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        ᶜvort = @. Geometry.WVector(curlₕ(state.c.uₕ))
        if cache.do_dss
            Spaces.weighted_dss!(ᶜvort)
        end
        return copy(ᶜvort)
    end,
)


###
# Relative humidity (3d)
###
function compute_variable!(
    Val{:relative_humidity},
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    thermo_params = CAP.thermodynamics_params(cache.params)
    return TD.relative_humidity.(thermo_params, cache.ᶜts)
end

compute_relative_humidity!(out, state, cache, time) =
    compute_variable!(
        Val(:relative_humidity),
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )

add_diagnostic_variable!(
    short_name = "Relative Humidity",
    long_name = "relative_humidity",
    units = "",
    comments = "Total amount of water vapor in the air relative to the amount achievable by saturation at the current temperature",
    compute! = compute_relative_humidity!,
)

###
# Total specific humidity (3d)
###
function compute_variable!(
    ::Val{:specific_humidity},
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    thermo_params = CAP.thermodynamics_params(cache.params)
    return TD.total_specific_humidity.(thermo_params, cache.ᶜts)
end

compute_specific_humidity!(out, state, cache, time) =
    compute_variable!(
        Val(:specific_humidity),
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )

add_diagnostic_variable!(
    short_name = "Specific_humidity",
    long_name = "Specific Humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air, a prognostic variable",
    compute! = compute_specific_humidity!,
)

###
# Surface specific humidity (2d) - although this could be collapsed into the 3d version + reduction, it probably makes sense to keep it separate, since it's really the property of the surface
###
function compute_variable!(
    ::Val{:surface_specific_humidity},
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    thermo_params = CAP.thermodynamics_params(cache.params)
    return TD.total_specific_humidity.(thermo_params, cache.sfc_conditions.ts)
end

compute_surface_specific_humidity!(out, state, cache, time) =
    compute_variable!(
        Val(:surface_specific_humidity),
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )

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
add_diagnostic_variable!(
    short_name = "surface_temperature",
    long_name = "Surface Temperature",
    units = "K",
    comments = "Temperature of the surface",
    compute! = TD.air_temperature.(thermo_params, cache.sfc_conditions.ts),
)

###
# Eastward surface drag component (2d)
###
function drag_vector!(state, cache)
    sfc_local_geometry =
    Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit =
        CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    (; ρ_flux_uₕ) = cache.sfc_conditions
    sfc_flux_momentum =
        Geometry.UVVector.(
            adjoint.(ρ_flux_uₕ ./ Spaces.level(ᶠinterp.(state.c.ρ), half)) .*
            surface_ct3_unit
    )
end

function compute_variable!(
    ::Val{:eastward_drag},
    out,
    state,
    cache,
    time,
    model::T,
) where {T <: TotalEnergy}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ

    return drag_vector!(state, cache).components.data.:1
end

compute_eastward_drag!(out, state, cache, time) =
    compute_variable!(
        Val(:eastward_drag),
        out,
        state,
        cache,
        time,
        cache.atmos.energy_form,
    )

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
function compute_variable!(
    ::Val{:northward_drag},
    out,
    state,
    cache,
    time,
    model::T,
) where {T <: TotalEnergy}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ

    return drag_vector!(state, cache).components.data.:2
end

compute_northward_drag!(out, state, cache, time) =
    compute_variable!(
        Val(:northward_drag),
        out,
        state,
        cache,
        time,
        cache.atmos.energy_form,
    )

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
function compute_variable!(
    ::Val{:surface_energy_flux},
    out,
    state,
    cache,
    time,
    model::T,
) where {T <: TotalEnergy}
    (;ρ_flux_h_tot) = cache.sfc_conditions
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    return dot.(ρ_flux_h_tot, surface_ct3_unit)
end

compute_surface_energy_flux!(out, state, cache, time) =
    compute_variable!(
        Val(:surface_energy_flux),
        out,
        state,
        cache,
        time,
        cache.atmos.energy_form,
    )

add_diagnostic_variable!(
    short_name = "surface_energy_flux",
    long_name = "Surface energy flux",
    units = "W m^-2",
    comments =  "energy flux at the surface",
    compute! = compute_surface_energy_flux!,
)

###
# Surface evaporation (2d)
###
function compute_variable!(
    ::Val{:surface_evaporation},
    out,
    state,
    cache,
    time,
    model::T,
) where {T <: TotalEnergy}

    compute_variable!(
        Val(:surface_evaporation),
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )
end

function compute_variable!(
    ::Val{:surface_evaporation},
    out,
    state,
    cache,
    time,
    model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    (;ρ_flux_q_tot) = cache.sfc_conditions
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    return dot.(ρ_flux_q_tot, surface_ct3_unit)
end

compute_surface_evaporation!(out, state, cache, time) =
    compute_variable!(
        Val(:surface_evaporation),
        out,
        state,
        cache,
        time,
        cache.atmos.energy_form,
    )

add_diagnostic_variable!(
    short_name = "surface_evaporation",
    long_name = "Surface evaporation",
    units = "kg s^-1 m^-2",
    comments =  "evaporation at the surface",
    compute! = compute_surface_evaporation!,
)

# as required, all 3d variables will be sliced to X? (TBD) levels
