#=
This file is included in Diagnostics.jl

README: Adding a new core diagnostic:

In addition to the metadata (names, comments, ...), the most important step
in adding a new DiagnosticVariable is defining its compute function. 
`compute` takes three arguments: (state, cache, time), 
and returns the diagnostic value, preferring lazy fields when possible.

Often, it is possible to compute certain diagnostics only for specific models
(e.g., humidity for moist models). For that, it is convenient to adopt the following pattern:

1. Define a function for the case we know how to handle:

function compute_hur(state, cache, time, ::MoistMicrophysics)
    tps = CAP.thermodynamics_params(cache.params)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    @. lazy(TD.relative_humidity(tps, ᶜT, state.c.ρ, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
end

2. Define a dispatch function with the correct signature:

compute_hur(state, cache, time) =
    compute_hur(state, cache, time, cache.atmos.microphysics_model)

3. Define an error fallback for unsupported models:

compute_hur(state, cache, time, model) = error_diagnostic_variable("hur", model)

=#

# General helper functions for undefined diagnostics
error_diagnostic_variable(message = "Cannot compute variable") = error(message)
error_diagnostic_variable(variable, ::T) where {T} =
    error_diagnostic_variable("Cannot compute $variable with model = $T")

###
# Density (3d)
###
add_diagnostic_variable!(short_name = "rhoa", units = "kg m^-3",
    long_name = "Air Density", standard_name = "air_density",
    compute = (state, _, _) -> state.c.ρ,
)

###
# U velocity (3d)
###
add_diagnostic_variable!(short_name = "ua", units = "m s^-1",
    long_name = "Eastward Wind",
    standard_name = "eastward_wind",
    comments = "Eastward (zonal) wind component",
    compute = (_, cache, _) -> @.(lazy(u_component(UVec(cache.precomputed.ᶜu)))),
)

###
# V velocity (3d)
###

add_diagnostic_variable!(short_name = "va", units = "m s^-1",
    long_name = "Northward Wind",
    standard_name = "northward_wind",
    comments = "Northward (meridional) wind component",
    compute = (_, cache, _) -> @.(lazy(v_component(VVec(cache.precomputed.ᶜu)))),
)

###
# W velocity (3d)
###
# TODO: may want to convert to omega (Lagrangian pressure tendency) as standard output,
# but this is probably more useful for now
add_diagnostic_variable!(short_name = "wa", units = "m s^-1",
    long_name = "Upward Air Velocity",
    standard_name = "upward_air_velocity",
    comments = "Vertical wind component",
    compute = (_, cache, _) -> @.(lazy(w_component(WVec(cache.precomputed.ᶠu)))),
)

###
# Temperature (3d)
###
add_diagnostic_variable!(short_name = "ta", units = "K",
    long_name = "Air Temperature",
    standard_name = "air_temperature",
    compute = (_, cache, _) -> cache.precomputed.ᶜT,
)

###
# Potential temperature (3d)
###
function compute_thetaa(state, cache, _)
    tps = CAP.thermodynamics_params(cache.params)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    ᶜρ = state.c.ρ
    return @. lazy(TD.potential_temperature(tps, ᶜT, ᶜρ, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
end

add_diagnostic_variable!(short_name = "thetaa", units = "K",
    long_name = "Air Potential Temperature",
    standard_name = "air_potential_temperature",
    compute = compute_thetaa,
)

###
# Enthalpy (3d)
###
function compute_ha(_, cache, _)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    return @. lazy(TD.enthalpy(thermo_params, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
end

add_diagnostic_variable!(short_name = "ha", units = "m^2 s^-2",
    long_name = "Air Specific Enthalpy",
    compute = compute_ha,
)

###
# Air pressure (3d)
###
add_diagnostic_variable!(short_name = "pfull", units = "Pa",
    long_name = "Pressure at Model Full-Levels",
    compute = (_, cache, _) -> cache.precomputed.ᶜp,
)

###
# Vorticity (3d)
###
function compute_rv(_, cache, _)
    vort = @. w_component(WVec(wcurlₕ(cache.precomputed.ᶜu)))  # TODO: Allocates
    # We need to ensure smoothness, so we call DSS
    Spaces.weighted_dss!(vort)
    return vort
end

add_diagnostic_variable!(short_name = "rv", units = "s^-1",
    long_name = "Relative Vorticity",
    standard_name = "relative_vorticity",
    comments = "Vertical component of relative vorticity",
    compute = compute_rv,
)

###
# Geopotential height (3d)
###
add_diagnostic_variable!(short_name = "zg", units = "m",
    long_name = "Geopotential Height",
    standard_name = "geopotential_height",
    compute = (_, cache, _) -> @.(lazy(cache.core.ᶜΦ / CAP.grav(cache.params))),
)

###
# Cloud fraction (3d)
###
add_diagnostic_variable!(short_name = "cl", units = "%",
    long_name = "Cloud fraction",
    compute = (_, cache, _) -> @.(lazy(cache.precomputed.ᶜcloud_fraction * 100)),
)

###
# Total kinetic energy
###
add_diagnostic_variable!(short_name = "ke", units = "m^2 s^-2",
    long_name = "Total Kinetic Energy",
    standard_name = "total_kinetic_energy",
    comments = "The kinetic energy on cell centers",
    compute = (_, cache, _) -> cache.precomputed.ᶜK,
)

###
# Mixing length (3d)
###
function compute_lmix(state, cache, _)
    turbconv_model = cache.atmos.turbconv_model
    # TODO: consolidate remaining mixing length types
    # (smagorinsky_lilly, dz) into a single function
    if isa(turbconv_model, PrognosticEDMFX) ||
       isa(turbconv_model, DiagnosticEDMFX) ||
       isa(turbconv_model, EDOnlyEDMFX)
        return ᶜmixing_length(state, cache)
    end
    (; params) = cache
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    ᶜdz = Fields.Δz_field(axes(state.c))
    ᶜprandtl_nvec = @. lazy(turbulent_prandtl_number(
        params, ᶜlinear_buoygrad, ᶜstrain_rate_norm,
    ))
    return @. lazy(
        smagorinsky_lilly_length(
            CAP.c_smag(params),
            sqrt(max(ᶜlinear_buoygrad, 0)),   # N_eff
            ᶜdz, ᶜprandtl_nvec, ᶜstrain_rate_norm,
        ),
    )
end

add_diagnostic_variable!(short_name = "lmix", units = "m",
    long_name = "Environment Mixing Length",
    comments = """
    Calculated as smagorinsky length scale without EDMF
    SGS model, or from mixing length closure with EDMF
    SGS model.
    """,
    compute = compute_lmix,
)

###
# Buoyancy gradient (3d)
###
add_diagnostic_variable!(short_name = "bgrad", units = "s^-2",
    long_name = "Linearized Buoyancy Gradient",
    compute = (_, cache, _) -> cache.precomputed.ᶜlinear_buoygrad,
)

###
# Strain rate magnitude (3d)
###
add_diagnostic_variable!(short_name = "strain", units = "s^-2",
    long_name = "String Rate Magnitude",
    compute = (_, cache, _) -> cache.precomputed.ᶜstrain_rate_norm,
)

###
# Smagorinsky Lilly diffusivity
###
add_diagnostic_variable!(short_name = "Dh_smag", units = "m^2 s^-1",
    long_name = "Horizontal smagorinsky diffusivity",
    compute = (_, cache, _) -> cache.precomputed.ᶜD_h,
)
add_diagnostic_variable!(short_name = "Dv_smag", units = "m^2 s^-1",
    long_name = "Vertical smagorinsky diffusivity",
    compute = (_, cache, _) -> cache.precomputed.ᶜD_v,
)
add_diagnostic_variable!(short_name = "strainh_smag", units = "s",
    long_name = "Horizontal strain rate magnitude (for Smagorinsky)",
    compute = (_, cache, _) -> cache.precomputed.ᶜS_norm_h,
)
add_diagnostic_variable!(short_name = "strainv_smag", units = "s",
    long_name = "Vertical strain rate magnitude (for Smagorinsky)",
    compute = (_, cache, _) -> cache.precomputed.ᶜS_norm_v,
)

###
# Relative humidity (3d)
###
compute_hur(state, cache, time) =
    compute_hur(state, cache, time, cache.atmos.microphysics_model)
compute_hur(_, _, _, model) = error_diagnostic_variable("hur", model)

function compute_hur(_, cache, _, ::MoistMicrophysics)
    tps = CAP.thermodynamics_params(cache.params)
    (; ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    return @. lazy(TD.relative_humidity(tps, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
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
    compute_hus(state, cache, time, cache.atmos.microphysics_model)
compute_hus(_, _, _, model) = error_diagnostic_variable("hus", model)

compute_hus(state, _, _, ::MoistMicrophysics) = @. lazy(specific(state.c.ρq_tot, state.c.ρ))

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
    compute_clw(state, cache, time, cache.atmos.microphysics_model)
compute_clw(_, _, _, model) = error_diagnostic_variable("clw", model)

compute_clw(_, cache, _, ::EquilibriumMicrophysics0M) = cache.precomputed.ᶜq_liq

compute_clw(state, _, _, ::NonEquilibriumMicrophysics) =
    @. lazy(specific(state.c.ρq_lcl, state.c.ρ))

add_diagnostic_variable!(short_name = "clw", units = "kg kg^-1",
    long_name = "Mass Fraction of Cloud Liquid Water",
    standard_name = "mass_fraction_of_cloud_liquid_water_in_air",
    comments = """
    Includes both large-scale and convective cloud.
    This is calculated as the mass of cloud liquid water
    in the grid cell divided by the mass of air
    (including water in all phases) in the grid cells.
    """,
    compute = compute_clw,
)

###
# Ice water specific humidity (3d)
###
compute_cli(state, cache, time) =
    compute_cli(state, cache, time, cache.atmos.microphysics_model)
compute_cli(_, _, _, model) = error_diagnostic_variable("cli", model)

compute_cli(_, cache, _, ::EquilibriumMicrophysics0M) = cache.precomputed.ᶜq_ice

compute_cli(state, _, _, ::NonEquilibriumMicrophysics1M) =
    @. lazy(specific(state.c.ρq_icl, state.c.ρ))

compute_cli(state, _, _, ::NonEquilibriumMicrophysics2M) =
    @. lazy(specific(state.c.ρq_ice, state.c.ρ))

add_diagnostic_variable!(short_name = "cli", units = "kg kg^-1",
    long_name = "Mass Fraction of Cloud Ice",
    standard_name = "mass_fraction_of_cloud_ice_in_air",
    comments = """
    Includes both large-scale and convective cloud.
    This is calculated as the mass of cloud ice in the
    grid cell divided by the mass of air (including water
    in all phases) in the grid cell.
    """,
    compute = compute_cli,
)

###
# Surface specific humidity (2d)
###
compute_hussfc(state, cache, time) =
    compute_hussfc(state, cache, time, cache.atmos.microphysics_model)
compute_hussfc(_, _, _, model) = error_diagnostic_variable("hussfc", model)

# q_vap_sfc is the total specific humidity at the surface
compute_hussfc(_, cache, _, ::MoistMicrophysics) =
    cache.precomputed.sfc_conditions.q_vap_sfc

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
    compute = (_, cache, _) -> cache.precomputed.sfc_conditions.T_sfc,
)

###
# Near-surface air temperature (2d)
###
add_diagnostic_variable!(short_name = "tas", units = "K",
    long_name = "Near-Surface Air Temperature",
    standard_name = "air_temperature",
    comments = "Temperature at the bottom cell center of the atmosphere",
    compute = (_, cache, _) -> Fields.level(cache.precomputed.ᶜT, 1),
)

###
# Near-surface U velocity (2d)
###
function compute_uas(_, cache, _)
    ᶜu₁ = Fields.level(cache.precomputed.ᶜu, 1)
    return @. lazy(u_component(UVec(ᶜu₁)))
end

add_diagnostic_variable!(short_name = "uas", units = "m s^-1",
    long_name = "Eastward Near-Surface Wind",
    standard_name = "eastward_wind",
    comments = "Eastward component of the wind at the bottom cell center of the atmosphere",
    compute = compute_uas,
)

###
# Near-surface V velocity (2d)
###
function compute_vas(_, cache, _)
    ᶜu₁ = Fields.level(cache.precomputed.ᶜu, 1)
    return @. lazy(v_component(VVec(ᶜu₁)))
end

add_diagnostic_variable!(short_name = "vas", units = "m s^-1",
    long_name = "Northward Near-Surface Wind",
    standard_name = "northward_wind",
    comments = "Northward (meridional) wind component \
                at the bottom cell center of the atmosphere",
    compute = compute_vas,
)

###
# Eastward and northward surface drag component (2d)
###
function compute_tau(cache, component)
    (; surface_ct3_unit) = cache.core
    (; ρ_flux_uₕ) = cache.precomputed.sfc_conditions
    ρ_flux = @. UV(adjoint(ρ_flux_uₕ) * surface_ct3_unit)  # TODO: this allocates
    return getproperty(ρ_flux.components.data, component)
end

compute_tauu(_, cache, _) = compute_tau(cache, :1)
compute_tauv(_, cache, _) = compute_tau(cache, :2)

add_diagnostic_variable!(short_name = "tauu", units = "Pa",
    long_name = "Surface Downward Eastward Wind Stress",
    standard_name = "downward_eastward_stress",
    comments = "Eastward component of the surface drag",
    compute = compute_tauu,
)
add_diagnostic_variable!(short_name = "tauv", units = "Pa",
    long_name = "Surface Downward Northward Wind Stress",
    standard_name = "downward_northward_stress",
    comments = "Northward component of the surface drag",
    compute = compute_tauv,
)

###
# Surface energy flux (2d)
# TODO: may need to split into sensible and latent heat
###
function compute_hfes(_, cache, _)
    (; ρ_flux_h_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    return @. lazy(dot(ρ_flux_h_tot, surface_ct3_unit))
end

add_diagnostic_variable!(short_name = "hfes", units = "W m^-2",
    long_name = "Surface Upward Energy Flux",
    comments = "Energy flux at the surface",
    compute = compute_hfes,
)

###
# Surface evaporation (2d)
###
compute_evspsbl(state, cache, time) =
    compute_evspsbl(state, cache, time, cache.atmos.microphysics_model)
compute_evspsbl(_, _, _, model) = error_diagnostic_variable("evspsbl", model)

function compute_evspsbl(_, cache, _, ::MoistMicrophysics)
    (; ρ_flux_q_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    return @. lazy(dot(ρ_flux_q_tot, surface_ct3_unit))
end

add_diagnostic_variable!(short_name = "evspsbl", units = "kg m^-2 s^-1",
    long_name = "Evaporation Including Sublimation and Transpiration",
    comments = "Evaporation at the surface",
    compute = compute_evspsbl,
)

###
# Latent heat flux (2d)
###
compute_hfls(state, cache, time) =
    compute_hfls(state, cache, time, cache.atmos.microphysics_model)
compute_hfls(_, _, _, model) = error_diagnostic_variable("hfls", model)

function compute_hfls(_, cache, _, ::MoistMicrophysics)
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
    compute_hfss(state, cache, time, cache.atmos.microphysics_model)
compute_hfss(_, _, _, model) = error_diagnostic_variable("hfss", model)

function compute_hfss(_, cache, _, ::DryModel)
    (; ρ_flux_h_tot) = cache.precomputed.sfc_conditions
    (; surface_ct3_unit) = cache.core
    return @. lazy(dot(ρ_flux_h_tot, surface_ct3_unit))
end

function compute_hfss(_, cache, _, ::MoistMicrophysics)
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

compute_pr(_, cache, _, ::Union{DryModel, MoistMicrophysics}) =
    @. lazy(cache.precomputed.surface_rain_flux + cache.precomputed.surface_snow_flux)

add_diagnostic_variable!(short_name = "pr", units = "kg m^-2 s^-1",
    long_name = "Precipitation",
    standard_name = "precipitation",
    comments = "Total precipitation including rain and snow",
    compute = compute_pr,
)

compute_prra(state, cache, time) =
    compute_prra(state, cache, time, cache.atmos.microphysics_model)
compute_prra(_, _, _, model) = error_diagnostic_variable("prra", model)

compute_prra(_, cache, _, ::Union{DryModel, MoistMicrophysics}) =
    cache.precomputed.surface_rain_flux

add_diagnostic_variable!(short_name = "prra", units = "kg m^-2 s^-1",
    long_name = "Rainfall Flux",
    standard_name = "rainfall_flux",
    comments = "Precipitation including all forms of water in the liquid phase",
    compute = compute_prra,
)

compute_prsn(state, cache, time) =
    compute_prsn(state, cache, time, cache.atmos.microphysics_model)
compute_prsn(_, _, _, model) = error_diagnostic_variable("prsn", model)

compute_prsn(_, cache, _, ::Union{DryModel, MoistMicrophysics}) =
    cache.precomputed.surface_snow_flux

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
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
) = @. lazy(specific(state.c.ρq_rai, state.c.ρ))

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
compute_hussn(_, _, _, model) = error_diagnostic_variable("hussn", model)

compute_hussn(state, _, _, ::NonEquilibriumMicrophysics1M) =
    @. lazy(specific(state.c.ρq_sno, state.c.ρ))

compute_hussn(state, _, _, ::NonEquilibriumMicrophysics2M) =
    @. lazy(specific(state.c.ρq_ice, state.c.ρ))  # TODO This should be `husice`, or something like that

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

compute_cdnc(state, _, _, ::NonEquilibriumMicrophysics2M) = state.c.ρn_lcl

add_diagnostic_variable!(short_name = "cdnc", units = "m^-3",
    long_name = "Cloud Liquid Droplet Number Concentration",
    standard_name = "number_concentration_of_cloud_liquid_water_particles_in_air",
    comments = """
    This is calculated as the number of cloud liquid water droplets in the grid cell
    divided by the cell volume.
    """,
    compute = compute_cdnc,
)

compute_ncra(state, cache, time) =
    compute_ncra(state, cache, time, cache.atmos.microphysics_model)
compute_ncra(_, _, _, model) = error_diagnostic_variable("ncra", model)

compute_ncra(state, _, _, ::NonEquilibriumMicrophysics2M) = state.c.ρn_rai

add_diagnostic_variable!(short_name = "ncra", units = "m^-3",
    long_name = "Raindrop Number Concentration",
    standard_name = "number_concentration_of_raindrops_in_air",
    comments = "This is calculated as the number of raindrops in the grid cell \
                divided by the cell volume.",
    compute = compute_ncra,
)

###
# Topography
###
compute_orog!(out, state, cache, time) =
    compute_orog!(out, state, cache, time, axes(state.c).grid.hypsography)

function compute_orog!(out, state, _, _, ::Grids.Flat)
    hspace = Spaces.horizontal_space(axes(state.c.ρ))
    isnothing(out) ? zeros(hspace) : (out .= 0)
end

function compute_orog!(out, _, _, _, hypsography)
    sfc = @. lazy(Geometry.tofloat(hypsography.surface))
    isnothing(out) ? Base.materialize(sfc) : (out .= sfc)
end

add_diagnostic_variable!(short_name = "orog", units = "m",
    long_name = "Surface Altitude",
    standard_name = "surface_altitude",
    comments = "Elevation of the horizontal coordinates",
    compute! = compute_orog!,
)

###
# Condensed water path (2d)
###
compute_clwvi(state, cache, time) =
    compute_clwvi(state, cache, time, cache.atmos.microphysics_model)
compute_clwvi(_, _, _, model) = error_diagnostic_variable("clwvi", model)

function compute_clwvi(state, cache, _, ::EquilibriumMicrophysics0M)
    out = cache.scratch.ᶠtemp_field_level
    (; ᶜq_liq, ᶜq_ice) = cache.precomputed
    clw = @. lazy(state.c.ρ * (ᶜq_liq + ᶜq_ice))
    Operators.column_integral_definite!(out, clw)
    return out
end

function compute_clwvi(state, cache, _, ::NonEquilibriumMicrophysics1M)
    out = cache.scratch.ᶠtemp_field_level
    clw = @. lazy(state.c.ρq_lcl + state.c.ρq_icl)
    Operators.column_integral_definite!(out, clw)
    return out
end

function compute_clwvi(state, cache, _, ::NonEquilibriumMicrophysics2M)
    out = cache.scratch.ᶠtemp_field_level
    clw = @. lazy(state.c.ρq_lcl + state.c.ρq_ice)
    Operators.column_integral_definite!(out, clw)
    return out
end

add_diagnostic_variable!(short_name = "clwvi", units = "kg m^-2",
    long_name = "Condensed Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_condensed_water",
    comments = """
    Mass of condensed (liquid + ice) water in the column divided by
    the area of the column (not just the cloudy portion).
    Does not include precipitating hydrometeors.
    """,
    compute = compute_clwvi,
)

###
# Liquid water path (2d)
###
compute_lwp(state, cache, time) =
    compute_lwp(state, cache, time, cache.atmos.microphysics_model)
compute_lwp(_, _, _, model) = error_diagnostic_variable("lwp", model)

function compute_lwp(state, cache, _, ::EquilibriumMicrophysics0M)
    out = cache.scratch.ᶠtemp_field_level
    lw = @. lazy(state.c.ρ * cache.precomputed.ᶜq_liq)
    Operators.column_integral_definite!(out, lw)
    return out
end

function compute_lwp(state, cache, _, ::NonEquilibriumMicrophysics)
    out = cache.scratch.ᶠtemp_field_level
    lw = state.c.ρq_lcl
    Operators.column_integral_definite!(out, lw)
    return out
end

add_diagnostic_variable!(short_name = "lwp", units = "kg m^-2",
    long_name = "Liquid Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_liquid_water",
    comments = "The total mass of liquid water in cloud per unit area. \
                Does not include precipitating hydrometeors.",
    compute = compute_lwp,
)

###
# Ice water path (2d)
###
compute_clivi(state, cache, time) =
    compute_clivi(state, cache, time, cache.atmos.microphysics_model)
compute_clivi(_, _, _, model) = error_diagnostic_variable("clivi", model)

function compute_clivi(state, cache, _, ::MoistMicrophysics)
    out = cache.scratch.ᶠtemp_field_level
    cli = @. lazy(state.c.ρ * cache.precomputed.ᶜq_ice)
    Operators.column_integral_definite!(out, cli)
    return out
end

add_diagnostic_variable!(short_name = "clivi", units = "kg m^-2",
    long_name = "Ice Water Path",
    standard_name = "atmosphere_mass_content_of_cloud_ice",
    comments = "The total mass of ice in cloud per unit area. \
                Does not include precipitating hydrometeors.",  # TODO: This comment is not correct for 2M
    compute = compute_clivi,
)


###
# Vertical integrated dry static energy (2d)
###
function compute_dsevi(state, cache, _)
    thermo_params = CAP.thermodynamics_params(cache.params)
    ᶜT = cache.precomputed.ᶜT
    out = cache.scratch.ᶠtemp_field_level
    dse = @. lazy(state.c.ρ * TD.dry_static_energy(thermo_params, ᶜT, cache.core.ᶜΦ))
    Operators.column_integral_definite!(out, dse)
    return out
end

add_diagnostic_variable!(short_name = "dsevi", units = "",
    long_name = "Dry Static Energy Vertical Integral",
    compute = compute_dsevi,
)

###
# Column integrated cloud fraction (2d)
###
compute_clvi(state, cache, time) =
    compute_clvi(state, cache, time, cache.atmos.microphysics_model)
compute_clvi(_, _, _, model) = error_diagnostic_variable("clvi", model)

function compute_clvi(_, cache, _, ::MoistMicrophysics)
    out = cache.scratch.ᶠtemp_field_level
    (; ᶜcloud_fraction) = cache.precomputed
    FT = eltype(ᶜcloud_fraction)
    cloud_cover = @. lazy(ifelse(ᶜcloud_fraction > zero(FT), one(FT), zero(FT)))
    Operators.column_integral_definite!(out, cloud_cover)
    return out
end

add_diagnostic_variable!(short_name = "clvi", units = "m",
    long_name = "Vertical Cloud Fraction Integral",
    comments = "The total height of the column occupied at least partially by cloud.",
    compute = compute_clvi,
)


###
# Column integrated total specific humidity (2d)
###
compute_prw(state, cache, time) =
    compute_prw(state, cache, time, cache.atmos.microphysics_model)
compute_prw(_, _, _, model) = error_diagnostic_variable("prw", model)

function compute_prw(state, cache, _, ::MoistMicrophysics)
    out = cache.scratch.ᶠtemp_field_level
    Operators.column_integral_definite!(out, state.c.ρq_tot)
    return out
end

add_diagnostic_variable!(short_name = "prw", units = "kg m^-2",
    long_name = "Water Vapor Path",
    standard_name = "atmospheric_mass_content_of_water_vapor",
    comments = "Vertically integrated specific humidity",
    compute = compute_prw,
)

###
# Column integrated relative humidity (2d)
###
compute_hurvi(state, cache, time) =
    compute_hurvi(state, cache, time, cache.atmos.microphysics_model)
compute_hurvi(_, _, _, model) = error_diagnostic_variable("hurvi", model)

function compute_hurvi(state, cache, _, ::MoistMicrophysics)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    # Vapor specific humidity = q_tot - q_liq - q_ice
    ᶜq_vap = @. lazy(ᶜq_tot_nonneg - ᶜq_liq - ᶜq_ice)
    # vertical integral of saturation specific humidity
    sat_vi = cache.scratch.ᶠtemp_field_level
    sat = @. lazy(
        state.c.ρ * TD.q_vap_saturation(thermo_params, ᶜT, state.c.ρ, ᶜq_liq, ᶜq_ice),
    )
    Operators.column_integral_definite!(sat_vi, sat)
    # saturation-weighted vertical integral
    hur_weighted = @. lazy(state.c.ρ * ᶜq_vap / sat_vi)
    out = cache.scratch.ᶠtemp_field_level
    Operators.column_integral_definite!(out, hur_weighted)
    return out
end

add_diagnostic_variable!(short_name = "hurvi", units = "kg m^-2",
    long_name = "Relative Humidity Saturation-Weighted Vertical Integral",
    comments = "Integrated relative humidity over the vertical column",
    compute = compute_hurvi,
)


###
# Vapor specific humidity (3d)
###
compute_husv(state, cache, time) =
    compute_husv(state, cache, time, cache.atmos.microphysics_model)
compute_husv(_, _, _, model) = error_diagnostic_variable("husv", model)

function compute_husv(_, cache, _, ::MoistMicrophysics)
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    return @. lazy(ᶜq_tot_nonneg - ᶜq_liq - ᶜq_ice)
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

add_diagnostic_variable!(short_name = "uapredicted", units = "m s^-1",
    long_name = "Predicted Eastward Wind",
    comments = "Predicted steady-state eastward (zonal) wind component",
    compute = (_, cache, _) -> @.(lazy(u_component(cache.steady_state_velocity.ᶜu))),
)

add_diagnostic_variable!(short_name = "vapredicted", units = "m s^-1",
    long_name = "Predicted Northward Wind",
    comments = "Predicted steady-state northward (meridional) wind component",
    compute = (_, cache, _) -> @.(lazy(v_component(cache.steady_state_velocity.ᶜu))),
)

add_diagnostic_variable!(short_name = "wapredicted", units = "m s^-1",
    long_name = "Predicted Upward Air Velocity",
    comments = "Predicted steady-state vertical wind component",
    compute = (_, cache, _) -> @.(lazy(w_component(cache.steady_state_velocity.ᶠu))),
)

compute_uaerror(cache) = @. lazy(
    u_component(UVW(cache.precomputed.ᶜu)) -
    u_component(cache.steady_state_velocity.ᶜu),
)

add_diagnostic_variable!(short_name = "uaerror", units = "m s^-1",
    long_name = "Error of Eastward Wind",
    comments = "Error of steady-state eastward (zonal) wind component",
    compute = (_, cache, _) -> compute_uaerror(cache),
)

compute_vaerror(cache) = @. lazy(
    v_component(UVW(cache.precomputed.ᶜu)) -
    v_component(cache.steady_state_velocity.ᶜu),
)

add_diagnostic_variable!(short_name = "vaerror", units = "m s^-1",
    long_name = "Error of Northward Wind",
    comments = "Error of steady-state northward (meridional) wind component",
    compute = (_, cache, _) -> compute_vaerror(cache),
)

compute_waerror(cache) = @. lazy(
    w_component(UVW(cache.precomputed.ᶠu)) -
    w_component(cache.steady_state_velocity.ᶠu),
)

add_diagnostic_variable!(short_name = "waerror", units = "m s^-1",
    long_name = "Error of Upward Air Velocity",
    comments = "Error of steady-state vertical wind component",
    compute = (_, cache, _) -> compute_waerror(cache),
)

###
# Convective Available Potential Energy (2d)
###
function compute_cape(state, cache, _)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜp, sfc_conditions) = cache.precomputed
    tps = CAP.thermodynamics_params(cache.params)
    FT = eltype(tps)
    g = TD.Parameters.grav(tps)

    # Get surface parcel properties from sfc_conditions
    # At the surface, q_tot ≈ q_vap (no condensate)
    surface_q = sfc_conditions.q_vap_sfc
    surface_T = sfc_conditions.T_sfc
    # Use lowest level pressure as approximate surface pressure
    surface_p = lazy.(Fields.level(ᶠinterp.(ᶜp), half))
    # Compute liquid-ice potential temperature at surface (no condensate, so q_liq=q_ice=0)
    surface_θ_liq_ice =
        lazy.(TD.liquid_ice_pottemp_given_pressure.(tps, surface_T, surface_p, surface_q))

    # Helper function to extract just T from saturation_adjustment result
    # (avoids broadcasting issues with NamedTuple containing bool)
    _parcel_T_from_sa(tps, p, θ_liq_ice, q_tot, maxiter) =
        TD.saturation_adjustment(tps, TD.pθ_li(), p, θ_liq_ice, q_tot; maxiter).T

    # Create parcel thermodynamic states at each level based on energy & moisture at surface
    parcel_T = lazy.(_parcel_T_from_sa.(tps, ᶜp, surface_θ_liq_ice, surface_q, 4))

    # Calculate virtual temperatures for parcel & environment
    parcel_Tv = lazy.(TD.virtual_temperature.(tps, parcel_T, surface_q))
    env_Tv = lazy.(TD.virtual_temperature.(tps, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))

    # Calculate buoyancy from the difference in virtual temperatures
    ᶜbuoyancy = cache.scratch.ᶜtemp_scalar
    @. ᶜbuoyancy = g * (parcel_Tv - env_Tv) / env_Tv

    # restrict to tropospheric buoyancy (generously below 20km) TODO: integrate from LFC to LNB
    z = Fields.coordinate_field(state.c.ρ).z
    @. ᶜbuoyancy = ᶜbuoyancy * ifelse(z < FT(20000), FT(1), FT(0))

    # Integrate positive buoyancy to get CAPE
    out = cache.scratch.ᶠtemp_field_level
    Operators.column_integral_definite!(out, lazy.(max.(ᶜbuoyancy, 0)))
    return out
end

add_diagnostic_variable!(short_name = "cape", units = "J kg^-1",
    long_name = "Convective Available Potential Energy",
    standard_name = "convective_available_potential_energy",
    comments = "Energy available to a parcel lifted moist adiabatically from the surface. \
                Assumes fully reversible phase changes and no precipitation.",
    compute = compute_cape,
)

###
# Mean sea level pressure (2d)
###
function compute_mslp(state, cache, _)
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜp, ᶜT) = cache.precomputed
    tps = CAP.thermodynamics_params(cache.params)
    g = TD.Parameters.grav(tps)
    q_tot_safe_level = Fields.level(ᶜq_tot_nonneg, 1)
    q_liq_level = Fields.level(ᶜq_liq, 1)
    q_ice_level = Fields.level(ᶜq_ice, 1)
    R_m_surf = lazy.(TD.gas_constant_air.(tps, q_tot_safe_level, q_liq_level, q_ice_level))

    p_level = Fields.level(ᶜp, 1)
    t_level = Fields.level(ᶜT, 1)
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

add_diagnostic_variable!(short_name = "mslp", units = "Pa",
    long_name = "Mean Sea Level Pressure",
    standard_name = "mean_sea_level_pressure",
    comments = """
    Mean sea level pressure computed using a lapse-rate-dependent hypsometric reduction 
    (ERA-style; Γ=6.5 K/km with virtual temperature via moist gas constant).""",
    compute = compute_mslp,
)

###
# Rainwater path (2d)
###
compute_rwp(state, cache, time) =
    compute_rwp(state, cache, time, cache.atmos.microphysics_model)
compute_rwp(_, _, _, model) = error_diagnostic_variable("rwp", model)

function compute_rwp(state, cache, _,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
)
    rwp = cache.scratch.ᶠtemp_field_level
    Operators.column_integral_definite!(rwp, state.c.ρq_rai)
    return rwp
end

add_diagnostic_variable!(short_name = "rwp", units = "kg m^-2",
    long_name = "Rainwater Path",
    standard_name = "atmosphere_mass_content_of_rainwater",
    comments = """
    The total mass of rainwater per unit area.
    (not just the area of the cloudy portion of the column).
    """,
    compute = compute_rwp,
)

###
# Covariances (3d)
###
function compute_covariance_diagnostics(_, cache, _, type)
    (; ᶜT′T′, ᶜq′q′) = cache.precomputed
    if type == :qt_qt
        return ᶜq′q′
    elseif type == :T_T
        return ᶜT′T′
    elseif type == :T_qt
        corr = correlation_Tq(cache.params)
        return @. lazy(corr * sqrt(max(0, ᶜT′T′)) * sqrt(max(0, ᶜq′q′)))
    else
        error("Unknown variance type")
    end
end

compute_env_q_tot_variance(state, cache, time) =
    compute_covariance_diagnostics(state, cache, time, :qt_qt)
compute_env_temperature_variance(state, cache, time) =
    compute_covariance_diagnostics(state, cache, time, :T_T)
compute_env_q_tot_temperature_covariance(state, cache, time) =
    compute_covariance_diagnostics(state, cache, time, :T_qt)

function compute_env_q_tot_temperature_correlation(_, cache, _)
    corr = correlation_Tq(cache.params)
    (; ᶜtemp_scalar) = cache.scratch
    return @. lazy(one(ᶜtemp_scalar) * corr)
end

add_diagnostic_variable!(short_name = "env_q_tot_variance", units = "kg^2 kg^-2",
    long_name = "Environment Variance of Total Specific Humidity",
    compute = compute_env_q_tot_variance,
)

add_diagnostic_variable!(
    short_name = "env_temperature_variance", units = "K^2",
    long_name = "Environment Variance of Temperature",
    compute = compute_env_temperature_variance,
)

add_diagnostic_variable!(
    short_name = "env_q_tot_temperature_covariance",
    long_name = "Environment Covariance of Total Specific Humidity and Temperature",
    units = "kg K kg^-1",
    compute = compute_env_q_tot_temperature_covariance,
)

add_diagnostic_variable!(
    short_name = "env_q_tot_temperature_correlation",
    long_name = "Environment Correlation of Total Specific Humidity and Temperature",
    units = "1",
    compute = compute_env_q_tot_temperature_correlation,
)
