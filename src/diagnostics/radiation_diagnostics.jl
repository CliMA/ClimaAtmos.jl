# This file is included in Diagnostics.jl

"""
    geometric_scaling(z, planet_radius)

Compute geometric scaling factor for radiative fluxes at radial height `z`.

Helper function for scaling radiation diagnostics.
"""
geometric_scaling(z, planet_radius) = ((z + planet_radius) / planet_radius)^2

"""
    ᶠradiation_field_3d(state, cache, radiation_mode, flux_array)
    ᶠradiation_field_toa(state, cache, flux_array)
    ᶠradiation_field_sfc(state, flux_array)

Compute radiative fluxes, as a diagnostic field in 3d, at TOA, or at the surface

# Arguments
- `state, cache`: The model state and cache
- `radiation_mode`: The RRTMGP radiation mode
- `flux_array`: The radiative flux array, e.g. `cache.radiation.rrtmgp_model.face_sw_flux_dn`

"""
function ᶠradiation_field_3d(state, cache, radiation_mode, flux_array)
    (; deep_atmosphere) = radiation_mode
    planet_radius = CAP.planet_radius(cache.params)
    z = Fields.coordinate_field(axes(state.f)).z
    flux = Fields.array2field(flux_array, axes(state.f))
    deep_atmosphere ? lazy.(flux .* geometric_scaling.(z, planet_radius)) : flux
end

ᶠradiation_field_sfc(state, flux_array) =
    Fields.level(Fields.array2field(flux_array, axes(state.f)), half)

function ᶠradiation_field_toa(state, cache, radiation_mode, flux_array)
    (; deep_atmosphere) = radiation_mode
    nlevels = Spaces.nlevels(axes(state.c))
    z_max = Spaces.z_max(axes(state.f))
    planet_radius = CAP.planet_radius(cache.params)
    flux = Fields.level(Fields.array2field(flux_array, axes(state.f)), nlevels + half)
    deep_atmosphere ? lazy.(flux .* geometric_scaling.(z_max, planet_radius)) : flux
end

# Radiative fluxes

###
# Downwelling shortwave radiation (3d)
###
compute_rsd(state, cache, time) =
    compute_rsd(state, cache, time, cache.atmos.radiation_mode)
compute_rsd(_, _, _, radiation_mode) = error_diagnostic_variable("rsd", radiation_mode)

function compute_rsd(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_sw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_sw_flux_dn)
end

add_diagnostic_variable!(short_name = "rsd", units = "W m^-2",
    long_name = "Downwelling Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air",
    comments = "Downwelling shortwave radiation (3d)",
    compute = compute_rsd,
)

###
# TOA downwelling shortwave radiation (2d)
###
compute_rsdt(state, cache, time) =
    compute_rsdt(state, cache, time, cache.atmos.radiation_mode)
compute_rsdt(_, _, _, radiation_mode) = error_diagnostic_variable("rsdt", radiation_mode)

function compute_rsdt(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_sw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_toa(state, cache, radiation_mode, face_sw_flux_dn)
end

add_diagnostic_variable!(short_name = "rsdt", units = "W m^-2",
    long_name = "TOA Incident Shortwave Radiation",
    standard_name = "toa_incoming_shortwave_flux",
    comments = "Downward shortwave radiation at the top of the atmosphere",
    compute = compute_rsdt,
)

###
# Surface downwelling shortwave radiation (2d)
###
compute_rsds(state, cache, time) =
    compute_rsds(state, cache, time, cache.atmos.radiation_mode)
compute_rsds(_, _, _, radiation_mode) = error_diagnostic_variable("rsds", radiation_mode)

function compute_rsds(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode)
    (; face_sw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_sw_flux_dn)
end

add_diagnostic_variable!(short_name = "rsds", units = "W m^-2",
    long_name = "Surface Downwelling Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air",
    comments = "Downwelling shortwave radiation at the surface",
    compute = compute_rsds,
)

###
# Upwelling shortwave radiation (3d)
###
compute_rsu(state, cache, time) =
    compute_rsu(state, cache, time, cache.atmos.radiation_mode)
compute_rsu(_, _, _, radiation_mode) = error_diagnostic_variable("rsu", radiation_mode)

function compute_rsu(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_sw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_sw_flux_up)
end

add_diagnostic_variable!(short_name = "rsu", units = "W m^-2",
    long_name = "Upwelling Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air",
    comments = "Upwelling shortwave radiation at the surface",
    compute = compute_rsu,
)

###
# TOA upwelling shortwave radiation (2d)
###
compute_rsut(state, cache, time) =
    compute_rsut(state, cache, time, cache.atmos.radiation_mode)
compute_rsut(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsut", radiation_mode)

function compute_rsut(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_sw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_toa(state, cache, radiation_mode, face_sw_flux_up)
end

add_diagnostic_variable!(short_name = "rsut", units = "W m^-2",
    long_name = "TOA Outgoing Shortwave Radiation",
    standard_name = "toa_outgoing_shortwave_flux",
    comments = "Upwelling shortwave radiation at the top of the atmosphere",
    compute! = compute_rsut,
)

###
# Surface upwelling shortwave radiation (2d)
###
compute_rsus(state, cache, time) =
    compute_rsus(state, cache, time, cache.atmos.radiation_mode)
compute_rsus(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsus", radiation_mode)

function compute_rsus(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_sw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_sw_flux_up)
end

add_diagnostic_variable!(short_name = "rsus", units = "W m^-2",
    long_name = "Surface Upwelling Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air",
    comments = "Upwelling shortwave radiation at the surface",
    compute = compute_rsus,
)

###
# Downwelling longwave radiation (3d)
###
compute_rld(state, cache, time) =
    compute_rld(state, cache, time, cache.atmos.radiation_mode)
compute_rld(_, _, _, radiation_mode) = error_diagnostic_variable("rld", radiation_mode)

function compute_rld(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_lw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_lw_flux_dn)
end

add_diagnostic_variable!(short_name = "rld", units = "W m^-2",
    long_name = "Downwelling Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air",
    comments = "Downwelling longwave radiation",
    compute = compute_rld,
)

###
# Surface downwelling longwave radiation (2d)
###
compute_rlds(state, cache, time) =
    compute_rlds(state, cache, time, cache.atmos.radiation_mode)
compute_rlds(_, _, _, radiation_mode) =
    error_diagnostic_variable("rlds", radiation_mode)

function compute_rlds(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_lw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_lw_flux_dn)
end

add_diagnostic_variable!(short_name = "rlds", units = "W m^-2",
    long_name = "Surface Downwelling Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air",
    comments = "Downwelling longwave radiation at the surface",
    compute = compute_rlds,
)

###
# Upwelling longwave radiation (3d)
###
compute_rlu(state, cache, time) =
    compute_rlu(state, cache, time, cache.atmos.radiation_mode)
compute_rlu(_, _, _, radiation_mode) = error_diagnostic_variable("rlu", radiation_mode)

function compute_rlu(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_lw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_lw_flux_up)
end

add_diagnostic_variable!(short_name = "rlu", units = "W m^-2",
    long_name = "Upwelling Longwave Radiation",
    standard_name = "surface_upwelling_longwave_flux_in_air",
    comments = "Upwelling longwave radiation",
    compute = compute_rlu,
)

###
# TOA upwelling longwave radiation (2d)
###
compute_rlut(state, cache, time) =
    compute_rlut(state, cache, time, cache.atmos.radiation_mode)
compute_rlut(_, _, _, radiation_mode) =
    error_diagnostic_variable("rlut", radiation_mode)

function compute_rlut(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_lw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_toa(state, cache, radiation_mode, face_lw_flux_up)
end

add_diagnostic_variable!(short_name = "rlut", units = "W m^-2",
    long_name = "TOA Outgoing Longwave Radiation",
    standard_name = "toa_outgoing_longwave_flux",
    comments = "Upwelling longwave radiation at the top of the atmosphere",
    compute = compute_rlut,
)

###
# Surface upwelling longwave radiation (2d)
###
compute_rlus(state, cache, time) =
    compute_rlus(state, cache, time, cache.atmos.radiation_mode)
compute_rlus(_, _, _, radiation_mode) =
    error_diagnostic_variable("rlus", radiation_mode)

function compute_rlus(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_lw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_lw_flux_up)
end

add_diagnostic_variable!(short_name = "rlus", units = "W m^-2",
    long_name = "Surface Upwelling Longwave Radiation",
    standard_name = "surface_upwelling_longwave_flux_in_air",
    comments = "Upwelling longwave radiation at the surface",
    compute = compute_rlus,
)

###
# Downelling clear sky shortwave radiation (3d)
###
compute_rsdcs(state, cache, time) =
    compute_rsdcs(state, cache, time, cache.atmos.radiation_mode)
compute_rsdcs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsdcs", radiation_mode)

function compute_rsdcs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    (; face_clear_sw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_clear_sw_flux_dn)
end

add_diagnostic_variable!(short_name = "rsdcs", units = "W m^-2",
    long_name = "Downwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
    comments = "Downwelling clear sky shortwave radiation (3d)",
    compute = compute_rsdcs,
)

###
# Surface downwelling clear sky shortwave radiation (2d)
###
compute_rsdscs(state, cache, time) =
    compute_rsdscs(state, cache, time, cache.atmos.radiation_mode)
compute_rsdscs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsdscs", radiation_mode)

function compute_rsdscs(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_clear_sw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_clear_sw_flux_dn)
end

add_diagnostic_variable!(short_name = "rsdscs", units = "W m^-2",
    long_name = "Surface Downwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
    comments = "Downwelling clear-sky shortwave radiation at the surface",
    compute = compute_rsdscs,
)

###
# Upwelling clear sky shortwave radiation (3d)
###
compute_rsucs(state, cache, time) =
    compute_rsucs(state, cache, time, cache.atmos.radiation_mode)
compute_rsucs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsucs", radiation_mode)

function compute_rsucs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    (; face_clear_sw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_clear_sw_flux_up)
end

add_diagnostic_variable!(short_name = "rsucs", units = "W m^-2",
    long_name = "Upwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air_assuming_clear_sky",
    comments = "Upwelling clear sky shortwave radiation (3d)",
    compute = compute_rsucs,
)

###
# TOA upwelling clear sky shortwave radiation (2d)
###
compute_rsutcs(state, cache, time) =
    compute_rsutcs(state, cache, time, cache.atmos.radiation_mode)
compute_rsutcs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsutcs", radiation_mode)

function compute_rsutcs(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_clear_sw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_toa(state, cache, radiation_mode, face_clear_sw_flux_up)
end

add_diagnostic_variable!(short_name = "rsutcs", units = "W m^-2",
    long_name = "TOA Outgoing Clear-Sky Shortwave Radiation",
    standard_name = "toa_outgoing_shortwave_flux_assuming_clear_sky",
    comments = "Upwelling clear-sky shortwave radiation at the top of the atmosphere",
    compute = compute_rsutcs,
)

###
# Surface clear sky upwelling shortwave radiation (2d)
###
compute_rsuscs(state, cache, time) =
    compute_rsuscs(state, cache, time, cache.atmos.radiation_mode)
compute_rsuscs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsuscs", radiation_mode)

function compute_rsuscs(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode)
    (; face_clear_sw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_sfc(state, face_clear_sw_flux_up)
end

add_diagnostic_variable!(short_name = "rsuscs", units = "W m^-2",
    long_name = "Surface Upwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air_assuming_clear_sky",
    comments = "Upwelling clear-sky shortwave radiation at the surface",
    compute = compute_rsuscs,
)


###
# Downwelling clear sky longwave radiation (3d)
###
compute_rldcs(state, cache, time) =
    compute_rldcs(state, cache, time, cache.atmos.radiation_mode)
compute_rldcs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rldcs", radiation_mode)

function compute_rldcs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    (; face_clear_lw_flux_dn) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_clear_lw_flux_dn)
end

add_diagnostic_variable!(short_name = "rldcs", units = "W m^-2",
    long_name = "Downwelling Clear-Sky Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air_assuming_clear_sky",
    comments = "Downwelling clear sky longwave radiation (3d)",
    compute = compute_rldcs,
)

###
# Surface clear sky downwelling longwave radiation (2d)
###
compute_rldscs(state, cache, time) =
    compute_rldscs(state, cache, time, cache.atmos.radiation_mode)
compute_rldscs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rldscs", radiation_mode)

function compute_rldscs(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode)
    (; face_clear_lw_flux_dn) = cache.radiation_mode.rrtmgp_model
    ᶠradiation_field_sfc(state, face_clear_lw_flux_dn)
end

add_diagnostic_variable!(short_name = "rldscs", units = "W m^-2",
    long_name = "Surface Downwelling Clear-Sky Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air_assuming_clear_sky",
    comments = "Downwelling clear-sky longwave radiation at the surface",
    compute = compute_rldscs,
)

###
# Upwelling clear sky longwave radiation (3d)
###
compute_rlucs(state, cache, time) =
    compute_rlucs(state, cache, time, cache.atmos.radiation_mode)
compute_rlucs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rlucs", radiation_mode)

function compute_rlucs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    (; face_clear_lw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_3d(state, cache, radiation_mode, face_clear_lw_flux_up)
end

add_diagnostic_variable!(short_name = "rlucs", units = "W m^-2",
    long_name = "Upwelling Clear-Sky Longwave Radiation",
    standard_name = "surface_upwelling_longwave_flux_in_air_assuming_clear_sky",
    comments = "Upwelling clear sky longwave radiation",
    compute = compute_rlucs,
)

###
# TOA clear sky upwelling longwave radiation (2d)
###
compute_rlutcs(state, cache, time) =
    compute_rlutcs(state, cache, time, cache.atmos.radiation_mode)
compute_rlutcs(_, _, _, radiation_mode) =
    error_diagnostic_variable("rlutcs", radiation_mode)

function compute_rlutcs(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode)
    (; face_clear_lw_flux_up) = cache.radiation.rrtmgp_model
    ᶠradiation_field_toa(state, cache, radiation_mode, face_clear_lw_flux_up)
end

add_diagnostic_variable!(short_name = "rlutcs", units = "W m^-2",
    long_name = "TOA Outgoing Clear-Sky Longwave Radiation",
    standard_name = "toa_outgoing_longwave_flux_assuming_clear_sky",
    comments = "Upwelling clear-sky longwave radiation at the top of the atmosphere",
    compute = compute_rlutcs,
)


###
# Effective radius for liquid clouds (3d)
###
compute_reffclw(state, cache, time) =
    compute_reffclw(state, cache, time, cache.atmos.radiation_mode)
compute_reffclw(_, _, _, radiation_mode) =
    error_diagnostic_variable("reffclw", radiation_mode)

function compute_reffclw(state, cache, _,
    ::Union{RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics, RRTMGPI.AllSkyRadiation},
)
    (; center_cloud_liquid_effective_radius) = cache.radiation.rrtmgp_model
    reff = Fields.array2field(center_cloud_liquid_effective_radius, axes(state.c))
    FT = eltype(state)
    μm_to_m = FT(1e-6)  # RRTMGP stores r_eff in microns
    return @. lazy(reff * μm_to_m)
end

add_diagnostic_variable!(short_name = "reffclw", units = "m",
    long_name = "Effective radius for liquid clouds",
    standard_name = "effective_radius_of_cloud_liquid_particles",
    comments = "In-cloud ratio of the third moment over the second moment \
                of the particle size distribution. Set to zero outside of clouds.",
    compute = compute_reffclw,
)

###
# Effective radius for ice clouds (3d)
###
compute_reffcli(state, cache, time) =
    compute_reffcli(state, cache, time, cache.atmos.radiation_mode)
compute_reffcli(_, _, _, radiation_mode) =
    error_diagnostic_variable("reffcli", radiation_mode)

function compute_reffcli(state, cache, _,
    ::Union{RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics, RRTMGPI.AllSkyRadiation},
)
    (; center_cloud_ice_effective_radius) = cache.radiation.rrtmgp_model
    reff = Fields.array2field(center_cloud_ice_effective_radius, axes(state.c))
    FT = eltype(state)
    μm_to_m = FT(1e-6)  # RRTMGP stores r_eff in microns
    return @. lazy(reff * μm_to_m)
end

add_diagnostic_variable!(short_name = "reffcli", units = "m",
    long_name = "Effective radius for ice clouds",
    standard_name = "effective_radius_of_cloud_ice_particles",
    comments = "In-cloud ratio of the third moment over the second moment \
                of the particle size distribution. Set to zero outside of clouds.",
    compute = compute_reffcli,
)

###
# Aerosol extinction optical depth (2d)
###
compute_od550aer(state, cache, time) =
    compute_od550aer(state, cache, time, cache.atmos.radiation_mode)
compute_od550aer(_, _, _, radiation_mode) =
    error_diagnostic_variable("od550aer", radiation_mode)

function compute_od550aer(state, cache, _,
    radiation_mode::Union{
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics, RRTMGPI.AllSkyRadiation,
        RRTMGPI.ClearSkyRadiation,
    },
)
    @assert cache.atmos.radiation_mode.aerosol_radiation "aerosol_radiation must be true to enable aerosol optical depth diagnostics"
    (; aod_sw_extinction) = cache.radiation.rrtmgp_model
    return Fields.array2field(aod_sw_extinction, axes(Fields.level(state.f, half)))
end

add_diagnostic_variable!(short_name = "od550aer", units = "",
    long_name = "Ambient Aerosol Optical Thickness at 550nm",
    standard_name = "atmosphere_optical_thickness_due_to_ambient_aerosol_particles",
    comments = "Aerosol optical depth from the ambient aerosols at wavelength 550 nm",
    compute = compute_od550aer,
)

###
# Aerosol scattering optical depth (2d)
###
compute_odsc550aer(state, cache, time) =
    compute_odsc550aer(state, cache, time, cache.atmos.radiation_mode)
compute_odsc550aer(_, _, _, radiation_mode) =
    error_diagnostic_variable("odsc550aer", radiation_mode)

function compute_odsc550aer(state, cache, _,
    radiation_mode::Union{
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics, RRTMGPI.AllSkyRadiation,
        RRTMGPI.ClearSkyRadiation,
    },
)
    @assert cache.atmos.radiation_mode.aerosol_radiation "aerosol_radiation must be true to enable aerosol optical depth diagnostics"
    (; aod_sw_scattering) = cache.radiation.rrtmgp_model
    return Fields.array2field(aod_sw_scattering, axes(Fields.level(state.f, half)))
end

add_diagnostic_variable!(short_name = "odsc550aer", units = "",
    long_name = "Ambient Scattering Aerosol Optical Thickness at 550nm",
    comments = "Aerosol scattering optical depth from the ambient aerosols at wavelength 550 nm",
    compute = compute_odsc550aer,
)
