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
#     out .= TD.relative_humidity.(thermo_params, cache.precomputed.ᶜT, state.c.ρ, cache.precomputed.ᶜq_tot_safe, cache.precomputed.ᶜq_liq_rai, cache.precomputed.ᶜq_ice_sno))
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
            return copy(w_component.(Geometry.WVector.(cache.precomputed.ᶠu)))
        else
            out .= w_component.(Geometry.WVector.(cache.precomputed.ᶠu))
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
        if isnothing(out)
            return copy(cache.precomputed.ᶜT)
        else
            out .= cache.precomputed.ᶜT
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
        (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = cache.precomputed
        if isnothing(out)
            return TD.potential_temperature.(
                thermo_params,
                ᶜT,
                state.c.ρ,
                ᶜq_tot_safe,
                ᶜq_liq_rai,
                ᶜq_ice_sno,
            )
        else
            out .=
                TD.potential_temperature.(
                    thermo_params,
                    ᶜT,
                    state.c.ρ,
                    ᶜq_tot_safe,
                    ᶜq_liq_rai,
                    ᶜq_ice_sno,
                )
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
        (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = cache.precomputed
        if isnothing(out)
            return TD.enthalpy.(thermo_params, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
        else
            out .= TD.enthalpy.(thermo_params, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
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
        vort = @. w_component.(Geometry.WVector(wcurlₕ(cache.precomputed.ᶜu)))
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
# Geopotential height (3d)
###
add_diagnostic_variable!(
    short_name = "zg",
    long_name = "Geopotential Height",
    standard_name = "geopotential_height",
    units = "m",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return cache.core.ᶜΦ ./ CAP.grav(cache.params)
        else
            out .= cache.core.ᶜΦ ./ CAP.grav(cache.params)
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
            return copy(cache.precomputed.ᶜcloud_fraction) .* 100
        else
            out .= cache.precomputed.ᶜcloud_fraction .* 100
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
        turbconv_model = cache.atmos.turbconv_model
        # TODO: consolidate remaining mixing length types
        # (smagorinsky_lilly, dz) into a single mixing length function
        if isa(turbconv_model, PrognosticEDMFX) ||
           isa(turbconv_model, DiagnosticEDMFX) ||
           isa(turbconv_model, EDOnlyEDMFX)
            ᶜmixing_length_field = ᶜmixing_length(state, cache)
        else
            (; params) = cache
            (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
            ᶜdz = Fields.Δz_field(axes(state.c))
            ᶜprandtl_nvec = @. lazy(
                turbulent_prandtl_number(
                    params,
                    ᶜlinear_buoygrad,
                    ᶜstrain_rate_norm,
                ),
            )
            ᶜmixing_length_field = @. lazy(
                smagorinsky_lilly_length(
                    CAP.c_smag(params),
                    sqrt(max(ᶜlinear_buoygrad, 0)),   # N_eff
                    ᶜdz,
                    ᶜprandtl_nvec,
                    ᶜstrain_rate_norm,
                ),
            )
        end


        if isnothing(out)
            return Base.materialize(ᶜmixing_length_field)
        else
            out .= ᶜmixing_length_field
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
# Smagorinsky Lilly diffusivity
###
add_diagnostic_variable!(
    short_name = "Dh_smag",
    long_name = "Horizontal smagorinsky diffusivity",
    units = "m^2 s^-1",
    compute! = (out, _, cache, _) -> begin
        (; ᶜD_h) = cache.precomputed
        isnothing(out) ? copy(ᶜD_h) : (out .= ᶜD_h)
    end,
)
add_diagnostic_variable!(
    short_name = "Dv_smag",
    long_name = "Vertical smagorinsky diffusivity",
    units = "m^2 s^-1",
    compute! = (out, _, cache, _) -> begin
        (; ᶜD_v) = cache.precomputed
        isnothing(out) ? copy(ᶜD_v) : (out .= ᶜD_v)
    end,
)
add_diagnostic_variable!(
    short_name = "strainh_smag",
    long_name = "Horizontal strain rate magnitude (for Smagorinsky)",
    units = "s",
    compute! = (out, state, cache, _) -> begin
        (; ᶜS_norm_h) = cache.precomputed
        isnothing(out) ? copy(ᶜS_norm_h) : (out .= ᶜS_norm_h)
    end,
)
add_diagnostic_variable!(
    short_name = "strainv_smag",
    long_name = "Vertical strain rate magnitude (for Smagorinsky)",
    units = "s",
    compute! = (out, state, cache, _) -> begin
        (; ᶜS_norm_v) = cache.precomputed
        isnothing(out) ? copy(ᶜS_norm_v) : (out .= ᶜS_norm_v)
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
    (; ᶜT, ᶜp, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = cache.precomputed
    if isnothing(out)
        return TD.relative_humidity.(
            thermo_params,
            ᶜT,
            ᶜp,
            ᶜq_tot_safe,
            ᶜq_liq_rai,
            ᶜq_ice_sno,
        )
    else
        out .=
            TD.relative_humidity.(
                thermo_params,
                ᶜT,
                ᶜp,
                ᶜq_tot_safe,
                ᶜq_liq_rai,
                ᶜq_ice_sno,
            )
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
    moisture_model::EquilMoistModel,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_liq_rai)
    else
        out .= cache.precomputed.ᶜq_liq_rai
    end
end

function compute_clw!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
)
    if isnothing(out)
        return state.c.ρq_liq ./ state.c.ρ
    else
        out .= state.c.ρq_liq ./ state.c.ρ
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
    moisture_model::EquilMoistModel,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_ice_sno)
    else
        out .= cache.precomputed.ᶜq_ice_sno
    end
end

function compute_cli!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
)
    if isnothing(out)
        return state.c.ρq_ice ./ state.c.ρ
    else
        out .= state.c.ρq_ice ./ state.c.ρ
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
    # q_vap_sfc is the total specific humidity at the surface (no liquid/ice)
    if isnothing(out)
        return copy(cache.precomputed.sfc_conditions.q_vap_sfc)
    else
        out .= cache.precomputed.sfc_conditions.q_vap_sfc
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
        if isnothing(out)
            return copy(cache.precomputed.sfc_conditions.T_sfc)
        else
            out .= cache.precomputed.sfc_conditions.T_sfc
        end
    end,
)

#=
###
# Sea ice thickness (2d)
###
add_diagnostic_variable!(
    short_name = "sithick",
    long_name = "Sea Ice Thickness",
    standard_name = "sea_ice_thickness",
    units = "m",
    comments = "Sea ice thickness for Eisenman sea ice model",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return copy(state.sfc.h_ice)
        else
            out .=
                state.sfc.h_ice
        end
    end,
)
=#

###
# Mixed layer temperature
### TODO: verify standard name
add_diagnostic_variable!(
    short_name = "toml",
    long_name = "Ocean Mixed Layer Temperature",
    standard_name = "ocean_mixed_layer_temperature",
    units = "m",
    comments = "(nonstandard) Ocean mixed layer temperature for Eisenman sea ice model",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return copy(state.sfc.T_ml)
        else
            out .=
                state.sfc.T_ml
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
        T_level = Fields.level(cache.precomputed.ᶜT, 1)
        if isnothing(out)
            return copy(T_level)
        else
            out .= T_level
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
                    Geometry.UVector.(Fields.level(cache.precomputed.ᶜu, 1)),
                ),
            )
        else
            out .=
                u_component.(
                    Geometry.UVector.(Fields.level(cache.precomputed.ᶜu, 1)),
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
                    Geometry.VVector.(Fields.level(cache.precomputed.ᶜu, 1)),
                ),
            )
        else
            out .=
                v_component.(
                    Geometry.VVector.(Fields.level(cache.precomputed.ᶜu, 1)),
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
# Latent heat flux (2d)
###
compute_hfls!(out, state, cache, time) =
    compute_hfls!(out, state, cache, time, cache.atmos.moisture_model)
compute_hfls!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hfls", model)

function compute_hfls!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    (; ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    thermo_params = CAP.thermodynamics_params(cache.params)
    LH_v0 = TD.Parameters.LH_v0(thermo_params)

    if isnothing(out)
        return dot.(ρ_flux_q_tot, surface_ct3_unit) .* LH_v0
    else
        out .= dot.(ρ_flux_q_tot, surface_ct3_unit) .* LH_v0
    end
end

add_diagnostic_variable!(
    short_name = "hfls",
    long_name = "Surface Upward Latent Heat Flux",
    standard_name = "surface_upward_latent_heat_flux",
    units = "W m^-2",
    compute! = compute_hfls!,
)

###
# Sensible heat flux (2d)
###
compute_hfss!(out, state, cache, time) =
    compute_hfss!(out, state, cache, time, cache.atmos.moisture_model)
compute_hfss!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hfss", model)

function compute_hfss!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: DryModel}
    (; ρ_flux_h_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core

    if isnothing(out)
        return dot.(ρ_flux_h_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_h_tot, surface_ct3_unit)
    end
end

function compute_hfss!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    (; ρ_flux_h_tot, ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    thermo_params = CAP.thermodynamics_params(cache.params)
    LH_v0 = TD.Parameters.LH_v0(thermo_params)

    if isnothing(out)
        return dot.(ρ_flux_h_tot, surface_ct3_unit) .-
               dot.(ρ_flux_q_tot, surface_ct3_unit) .* LH_v0
    else
        out .=
            dot.(ρ_flux_h_tot, surface_ct3_unit) .-
            dot.(ρ_flux_q_tot, surface_ct3_unit) .* LH_v0
    end
end

add_diagnostic_variable!(
    short_name = "hfss",
    long_name = "Surface Upward Sensible Heat Flux",
    standard_name = "surface_upward_sensible_heat_flux",
    units = "W m^-2",
    compute! = compute_hfss!,
)

###
# Precipitation (2d)
###
compute_pr!(out, state, cache, time) =
    compute_pr!(out, state, cache, time, cache.atmos.microphysics_model)
compute_pr!(_, _, _, _, microphysics_model::T) where {T} =
    error_diagnostic_variable("pr", microphysics_model)

function compute_pr!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        NoPrecipitation,
        Microphysics0Moment,
        QuadratureMicrophysics{Microphysics0Moment},
        Microphysics1Moment,
        QuadratureMicrophysics{Microphysics1Moment},
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
)
    if isnothing(out)
        return cache.precomputed.surface_rain_flux .+
               cache.precomputed.surface_snow_flux
    else
        out .=
            cache.precomputed.surface_rain_flux .+
            cache.precomputed.surface_snow_flux
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
    compute_prra!(out, state, cache, time, cache.atmos.microphysics_model)
compute_prra!(_, _, _, _, microphysics_model::T) where {T} =
    error_diagnostic_variable("prra", microphysics_model)

function compute_prra!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        NoPrecipitation,
        Microphysics0Moment,
        QuadratureMicrophysics{Microphysics0Moment},
        Microphysics1Moment,
        QuadratureMicrophysics{Microphysics1Moment},
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
)
    if isnothing(out)
        return cache.precomputed.surface_rain_flux
    else
        out .= cache.precomputed.surface_rain_flux
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
    compute_prsn!(out, state, cache, time, cache.atmos.microphysics_model)
compute_prsn!(_, _, _, _, microphysics_model::T) where {T} =
    error_diagnostic_variable("prsn", microphysics_model)

function compute_prsn!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        NoPrecipitation,
        Microphysics0Moment,
        QuadratureMicrophysics{Microphysics0Moment},
        Microphysics1Moment,
        QuadratureMicrophysics{Microphysics1Moment},
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
)
    if isnothing(out)
        return cache.precomputed.surface_snow_flux
    else
        out .= cache.precomputed.surface_snow_flux
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
    compute_husra!(out, state, cache, time, cache.atmos.microphysics_model)
compute_husra!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("husra", model)

function compute_husra!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment,
        QuadratureMicrophysics{Microphysics1Moment},
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
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
    compute_hussn!(out, state, cache, time, cache.atmos.microphysics_model)
compute_hussn!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussn", model)

function compute_hussn!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment,
        QuadratureMicrophysics{Microphysics1Moment},
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
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

compute_cdnc!(out, state, cache, time) =
    compute_cdnc!(out, state, cache, time, cache.atmos.microphysics_model)
compute_cdnc!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("cdnc", model)

function compute_cdnc!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
)
    if isnothing(out)
        return state.c.ρn_liq
    else
        out .= state.c.ρn_liq
    end
end

add_diagnostic_variable!(
    short_name = "cdnc",
    long_name = "Cloud Liquid Droplet Number Concentration",
    standard_name = "number_concentration_of_cloud_liquid_water_particles_in_air",
    units = "m^-3",
    comments = """
    This is calculated as the number of cloud liquid water droplets in the grid 
    cell divided by the cell volume.
    """,
    compute! = compute_cdnc!,
)

compute_ncra!(out, state, cache, time) =
    compute_ncra!(out, state, cache, time, cache.atmos.microphysics_model)
compute_ncra!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("ncra", model)

function compute_ncra!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
        Microphysics2MomentP3,
        QuadratureMicrophysics{Microphysics2MomentP3},
    },
)
    if isnothing(out)
        return state.c.ρn_rai
    else
        out .= state.c.ρn_rai
    end
end

add_diagnostic_variable!(
    short_name = "ncra",
    long_name = "Raindrop Number Concentration",
    standard_name = "number_concentration_of_raindrops_in_air",
    units = "m^-3",
    comments = """
    This is calculated as the number of raindrops in the grid cell divided
    by the cell volume.
    """,
    compute! = compute_ncra!,
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
    moisture_model::EquilMoistModel,
)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        clw = cache.scratch.ᶜtemp_scalar
        @. clw =
            state.c.ρ * (
                cache.precomputed.ᶜq_liq_rai +
                cache.precomputed.ᶜq_ice_sno
            )
        Operators.column_integral_definite!(out, clw)
        return out
    else
        clw = cache.scratch.ᶜtemp_scalar
        @. clw =
            state.c.ρ * (
                cache.precomputed.ᶜq_liq_rai +
                cache.precomputed.ᶜq_ice_sno
            )
        Operators.column_integral_definite!(out, clw)
    end
end

function compute_clwvi!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        clw = cache.scratch.ᶜtemp_scalar
        @. clw = state.c.ρq_liq + state.c.ρq_ice
        Operators.column_integral_definite!(out, clw)
        return out
    else
        clw = cache.scratch.ᶜtemp_scalar
        @. clw = state.c.ρq_liq + state.c.ρq_ice
        Operators.column_integral_definite!(out, clw)
    end
end

add_diagnostic_variable!(
    short_name = "clwvi",
    long_name = "Condensed Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_condensed_water",
    units = "kg m^-2",
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
    moisture_model::EquilMoistModel,
)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        lw = cache.scratch.ᶜtemp_scalar
        @. lw = state.c.ρ * cache.precomputed.ᶜq_liq_rai
        Operators.column_integral_definite!(out, lw)
        return out
    else
        lw = cache.scratch.ᶜtemp_scalar
        @. lw = state.c.ρ * cache.precomputed.ᶜq_liq_rai
        Operators.column_integral_definite!(out, lw)
    end
end

function compute_lwp!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        lw = cache.scratch.ᶜtemp_scalar
        @. lw = state.c.ρq_liq
        Operators.column_integral_definite!(out, lw)
        return out
    else
        lw = cache.scratch.ᶜtemp_scalar
        @. lw = state.c.ρq_liq
        Operators.column_integral_definite!(out, lw)
    end
end

add_diagnostic_variable!(
    short_name = "lwp",
    long_name = "Liquid Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_liquid_water",
    units = "kg m^-2",
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
        @. cli = state.c.ρ * cache.precomputed.ᶜq_ice_sno
        Operators.column_integral_definite!(out, cli)
        return out
    else
        cli = cache.scratch.ᶜtemp_scalar
        @. cli = state.c.ρ * cache.precomputed.ᶜq_ice_sno
        Operators.column_integral_definite!(out, cli)
    end
end

add_diagnostic_variable!(
    short_name = "clivi",
    long_name = "Ice Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_ice",
    units = "kg m^-2",
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
    ᶜT = cache.precomputed.ᶜT
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        cp = CAP.cp_d(cache.params)
        dse = cache.scratch.ᶜtemp_scalar
        @. dse = state.c.ρ * (TD.dry_static_energy(thermo_params, ᶜT, cache.core.ᶜΦ))
        Operators.column_integral_definite!(out, dse)
        return out
    else
        cp = CAP.cp_d(cache.params)
        dse = cache.scratch.ᶜtemp_scalar
        @. dse = state.c.ρ * (TD.dry_static_energy(thermo_params, ᶜT, cache.core.ᶜΦ))
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
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        cloud_cover = cache.scratch.ᶜtemp_scalar
        FT = Spaces.undertype(axes(cloud_cover))
        @. cloud_cover = ifelse(
            cache.precomputed.ᶜcloud_fraction > zero(FT),
            one(FT),
            zero(FT),
        )
        Operators.column_integral_definite!(out, cloud_cover)
        return out
    else
        cloud_cover = cache.scratch.ᶜtemp_scalar
        FT = Spaces.undertype(axes(cloud_cover))
        @. cloud_cover = ifelse(
            cache.precomputed.ᶜcloud_fraction > zero(FT),
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
    (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = cache.precomputed
    # Vapor specific humidity = q_tot - q_liq - q_ice
    ᶜq_vap = @. lazy(ᶜq_tot_safe - ᶜq_liq_rai - ᶜq_ice_sno)
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        # compute vertical integral of saturation specific humidity
        # note next line currently allocates; currently no correct scratch space
        sat_vi = zeros(axes(Fields.level(state.f, half)))
        sat = cache.scratch.ᶜtemp_scalar
        @. sat =
            state.c.ρ *
            TD.q_vap_saturation(thermo_params, ᶜT, state.c.ρ, ᶜq_liq_rai, ᶜq_ice_sno)
        Operators.column_integral_definite!(sat_vi, sat)
        # compute saturation-weighted vertical integral of specific humidity
        hur_weighted = cache.scratch.ᶜtemp_scalar_2
        @. hur_weighted = state.c.ρ * ᶜq_vap / sat_vi
        Operators.column_integral_definite!(out, hur_weighted)
        return out
    else
        # compute vertical integral of saturation specific humidity
        # note next line currently allocates; currently no correct scratch space
        sat_vi = zeros(axes(Fields.level(state.f, half)))
        sat = cache.scratch.ᶜtemp_scalar
        @. sat =
            state.c.ρ *
            TD.q_vap_saturation(thermo_params, ᶜT, state.c.ρ, ᶜq_liq_rai, ᶜq_ice_sno)
        Operators.column_integral_definite!(sat_vi, sat)
        # compute saturation-weighted vertical integral of specific humidity
        hur_weighted = cache.scratch.ᶜtemp_scalar_2
        @. hur_weighted = state.c.ρ * ᶜq_vap / sat_vi
        Operators.column_integral_definite!(out, hur_weighted)
    end
end

add_diagnostic_variable!(
    short_name = "hurvi",
    long_name = "Relative Humidity Saturation-Weighted Vertical Integral",
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
    # Vapor specific humidity = q_tot - q_liq - q_ice
    (; ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = cache.precomputed
    if isnothing(out)
        return @. ᶜq_tot_safe - ᶜq_liq_rai - ᶜq_ice_sno
    else
        out .= @. ᶜq_tot_safe - ᶜq_liq_rai - ᶜq_ice_sno
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
# Analytic Steady-State Approximations
###

# These are only available when `check_steady_state` is `true`.

add_diagnostic_variable!(
    short_name = "uapredicted",
    long_name = "Predicted Eastward Wind",
    units = "m s^-1",
    comments = "Predicted steady-state eastward (zonal) wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            u_component.(cache.steady_state_velocity.ᶜu)
        else
            out .= u_component.(cache.steady_state_velocity.ᶜu)
        end
    end,
)

add_diagnostic_variable!(
    short_name = "vapredicted",
    long_name = "Predicted Northward Wind",
    units = "m s^-1",
    comments = "Predicted steady-state northward (meridional) wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            v_component.(cache.steady_state_velocity.ᶜu)
        else
            out .= v_component.(cache.steady_state_velocity.ᶜu)
        end
    end,
)

add_diagnostic_variable!(
    short_name = "wapredicted",
    long_name = "Predicted Upward Air Velocity",
    units = "m s^-1",
    comments = "Predicted steady-state vertical wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            w_component.(cache.steady_state_velocity.ᶠu)
        else
            out .= w_component.(cache.steady_state_velocity.ᶠu)
        end
    end,
)

add_diagnostic_variable!(
    short_name = "uaerror",
    long_name = "Error of Eastward Wind",
    units = "m s^-1",
    comments = "Error of steady-state eastward (zonal) wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            u_component.(Geometry.UVWVector.(cache.precomputed.ᶜu)) .-
            u_component.(cache.steady_state_velocity.ᶜu)
        else
            out .=
                u_component.(Geometry.UVWVector.(cache.precomputed.ᶜu)) .-
                u_component.(cache.steady_state_velocity.ᶜu)
        end
    end,
)

add_diagnostic_variable!(
    short_name = "vaerror",
    long_name = "Error of Northward Wind",
    units = "m s^-1",
    comments = "Error of steady-state northward (meridional) wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            v_component.(Geometry.UVWVector.(cache.precomputed.ᶜu)) .-
            v_component.(cache.steady_state_velocity.ᶜu)
        else
            out .=
                v_component.(Geometry.UVWVector.(cache.precomputed.ᶜu)) .-
                v_component.(cache.steady_state_velocity.ᶜu)
        end
    end,
)

add_diagnostic_variable!(
    short_name = "waerror",
    long_name = "Error of Upward Air Velocity",
    units = "m s^-1",
    comments = "Error of steady-state vertical wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            w_component.(Geometry.UVWVector.(cache.precomputed.ᶠu)) .-
            w_component.(cache.steady_state_velocity.ᶠu)
        else
            out .=
                w_component.(Geometry.UVWVector.(cache.precomputed.ᶠu)) .-
                w_component.(cache.steady_state_velocity.ᶠu)
        end
    end,
)

###
# Convective Available Potential Energy (2d)
###
function compute_cape!(out, state, cache, time)
    thermo_params = lazy.(CAP.thermodynamics_params(cache.params))
    FT = eltype(thermo_params)
    g = lazy.(TD.Parameters.grav(thermo_params))

    # Get surface parcel properties from sfc_conditions
    # At the surface, q_tot ≈ q_vap (no condensate)
    surface_q = lazy.(cache.precomputed.sfc_conditions.q_vap_sfc)
    surface_T = lazy.(cache.precomputed.sfc_conditions.T_sfc)
    # Use lowest level pressure as approximate surface pressure
    surface_p = lazy.(Fields.level(ᶠinterp.(cache.precomputed.ᶜp), half))
    # Compute liquid-ice potential temperature at surface (no condensate, so q_liq=q_ice=0)
    surface_θ_liq_ice =
        lazy.(
            TD.liquid_ice_pottemp_given_pressure.(
                thermo_params,
                surface_T,
                surface_p,
                surface_q,
            ),
        )

    # Helper function to extract just T from saturation_adjustment result
    # (avoids broadcasting issues with NamedTuple containing bool)
    _parcel_T_from_sa(thermo_params, p, θ_liq_ice, q_tot, maxiter, tol) =
        TD.saturation_adjustment(
            thermo_params,
            TD.pθ_li(),
            p,
            θ_liq_ice,
            q_tot;
            maxiter,
            tol,
        ).T

    # Create parcel thermodynamic states at each level based on energy & moisture at surface
    parcel_T =
        lazy.(
            _parcel_T_from_sa.(
                thermo_params,
                cache.precomputed.ᶜp,
                surface_θ_liq_ice,
                surface_q,
                4,
                FT(0),
            ),
        )

    # Calculate virtual temperatures for parcel & environment
    parcel_Tv = lazy.(TD.virtual_temperature.(thermo_params, parcel_T, surface_q))
    (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = cache.precomputed
    env_Tv =
        lazy.(
            TD.virtual_temperature.(thermo_params, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
        )

    # Calculate buoyancy from the difference in virtual temperatures
    ᶜbuoyancy = cache.scratch.ᶜtemp_scalar
    ᶜbuoyancy .= g .* (parcel_Tv .- env_Tv) ./ env_Tv

    # restrict to tropospheric buoyancy (generously below 20km) TODO: integrate from LFC to LNB 
    FT = Spaces.undertype(axes(ᶜbuoyancy))
    ᶜbuoyancy .=
        ᶜbuoyancy .*
        ifelse.(
            Fields.coordinate_field(state.c.ρ).z .< FT(20000.0),
            FT(1.0),
            FT(0.0),
        )

    # Integrate positive buoyancy to get CAPE
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    Operators.column_integral_definite!(out′, lazy.(max.(ᶜbuoyancy, 0)))
    return out′
end

add_diagnostic_variable!(
    short_name = "cape",
    long_name = "Convective Available Potential Energy",
    standard_name = "convective_available_potential_energy",
    units = "J kg^-1",
    comments = "Energy available to a parcel lifted moist adiabatically from the surface. We assume fully reversible phase changes and no precipitation.",
    compute! = compute_cape!,
)

###
# Mean sea level pressure (2d)
###
function compute_mslp!(out, state, cache, time)
    thermo_params = CAP.thermodynamics_params(cache.params)
    g = TD.Parameters.grav(thermo_params)
    q_tot_safe_level = Fields.level(cache.precomputed.ᶜq_tot_safe, 1)
    q_liq_rai_level = Fields.level(cache.precomputed.ᶜq_liq_rai, 1)
    q_ice_sno_level = Fields.level(cache.precomputed.ᶜq_ice_sno, 1)
    R_m_surf = @. lazy(
        TD.gas_constant_air(
            thermo_params,
            q_tot_safe_level,
            q_liq_rai_level,
            q_ice_sno_level,
        ),
    )

    p_level = Fields.level(cache.precomputed.ᶜp, 1)
    t_level = Fields.level(cache.precomputed.ᶜT, 1)
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

    if isnothing(out)
        return p_level .* (1 .+ Γ .* z_level ./ t_level) .^ (g / Γ ./ R_m_surf)
    else
        out .= p_level .* (1 .+ Γ .* z_level ./ t_level) .^ (g / Γ ./ R_m_surf)
    end
end

add_diagnostic_variable!(
    short_name = "mslp",
    long_name = "Mean Sea Level Pressure",
    standard_name = "mean_sea_level_pressure",
    units = "Pa",
    comments = "Mean sea level pressure computed using a lapse-rate-dependent hypsometric reduction (ERA-style; Γ=6.5 K/km with virtual temperature via moist gas constant).",
    compute! = compute_mslp!,
)

###
# Rainwater path (2d)
###
compute_rwp!(out, state, cache, time) =
    compute_rwp!(out, state, cache, time, cache.atmos.microphysics_model)
compute_rwp!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("rwp", model)

function compute_rwp!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3}}
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        rw = cache.scratch.ᶜtemp_scalar
        @. rw = state.c.ρq_rai
        Operators.column_integral_definite!(out, rw)
        return out
    else
        rw = cache.scratch.ᶜtemp_scalar
        @. rw = state.c.ρq_rai
        Operators.column_integral_definite!(out, rw)
    end
end

add_diagnostic_variable!(
    short_name = "rwp",
    long_name = "Rainwater Path",
    standard_name = "atmosphere_mass_content_of_rainwater",
    units = "kg m^-2",
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
    thermo_params = CAP.thermodynamics_params(cache.params)

    # Read T-based variances from cache
    (; ᶜT′T′, ᶜq′q′) = cache.precomputed

    result = if type == :qt_qt
        ᶜq′q′
    elseif type == :T_T
        ᶜT′T′
    elseif type == :T_qt
        corr = correlation_Tq(cache.params)
        @. corr * sqrt(max(0, ᶜT′T′)) * sqrt(max(0, ᶜq′q′))
    else
        error("Unknown variance type")
    end

    if isnothing(out)
        return copy(result)
    else
        out .= result
    end
end

compute_env_q_tot_variance!(out, state, cache, time) =
    compute_covariance_diagnostics!(out, state, cache, time, :qt_qt)
compute_env_temperature_variance!(out, state, cache, time) =
    compute_covariance_diagnostics!(out, state, cache, time, :T_T)
compute_env_q_tot_temperature_covariance!(out, state, cache, time) =
    compute_covariance_diagnostics!(out, state, cache, time, :T_qt)

function compute_env_q_tot_temperature_correlation!(out, state, cache, time)
    corr = correlation_Tq(cache.params)
    if isnothing(out)
        return fill(corr, axes(state.c))
    else
        out .= corr
    end
end

add_diagnostic_variable!(
    short_name = "env_q_tot_variance",
    long_name = "Environment Variance of Total Specific Humidity",
    units = "kg^2 kg^-2",
    compute! = compute_env_q_tot_variance!,
)

add_diagnostic_variable!(
    short_name = "env_temperature_variance",
    long_name = "Environment Variance of Temperature",
    units = "K^2",
    compute! = compute_env_temperature_variance!,
)

add_diagnostic_variable!(
    short_name = "env_q_tot_temperature_covariance",
    long_name = "Environment Covariance of Total Specific Humidity and Temperature",
    units = "kg K kg^-1",
    compute! = compute_env_q_tot_temperature_covariance!,
)

add_diagnostic_variable!(
    short_name = "env_q_tot_temperature_correlation",
    long_name = "Environment Correlation of Total Specific Humidity and Temperature",
    units = "1",
    compute! = compute_env_q_tot_temperature_correlation!,
)
