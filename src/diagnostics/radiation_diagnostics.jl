# This file is included in Diagnostics.jl

"""
    geometric_scaling(z, planet_radius)

Compute geometric scaling factor for radiative fluxes at radial height `z`.

Helper function for scaling radiation diagnostics.
"""
geometric_scaling(z, planet_radius) = ((z + planet_radius) / planet_radius)^2

"""
    ᶠradiation_field_3d(state, cache, radiation_mode, field_name::Symbol)
    ᶠradiation_field_toa(state, cache, radiation_mode, field_name::Symbol)
    ᶠradiation_field_sfc(state, cache, field_name::Symbol)

Compute radiative fluxes as a diagnostic field in 3d, at TOA, or at the surface.

# Arguments
- `state, cache`: The model state and cache
- `radiation_mode`: The RRTMGP radiation mode
- `field_name`: Symbol naming the flux field on `cache.radiation.rrtmgp_model`,
  e.g. `:face_sw_flux_dn`

"""
function ᶠradiation_field_3d(state, cache, radiation_mode, field_name::Symbol)
    (; deep_atmosphere) = radiation_mode
    planet_radius = CAP.planet_radius(cache.params)
    z = Fields.coordinate_field(axes(state.f)).z
    field = getproperty(cache.radiation.rrtmgp_model, field_name)
    flux = Fields.array2field(field, axes(state.f))
    deep_atmosphere ? lazy.(flux .* geometric_scaling.(z, planet_radius)) : flux
end

function ᶠradiation_field_sfc(state, cache, field_name::Symbol)
    field = getproperty(cache.radiation.rrtmgp_model, field_name)
    Fields.level(Fields.array2field(field, axes(state.f)), half)
end

function ᶠradiation_field_toa(state, cache, radiation_mode, field_name::Symbol)
    (; deep_atmosphere) = radiation_mode
    nlevels = Spaces.nlevels(axes(state.c))
    z_max = Spaces.z_max(axes(state.f))
    planet_radius = CAP.planet_radius(cache.params)
    field = getproperty(cache.radiation.rrtmgp_model, field_name)
    flux = Fields.level(Fields.array2field(field, axes(state.f)), nlevels + half)
    deep_atmosphere ? lazy.(flux .* geometric_scaling.(z_max, planet_radius)) : flux
end

# Radiative fluxes

###
# Downwelling shortwave radiation (3d)
###
compute_rsd(state, cache, time) =
    compute_rsd(state, cache, time, cache.atmos.radiation_mode)
compute_rsd(_, _, _, radiation_mode) = error_diagnostic_variable("rsd", radiation_mode)

compute_rsd(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_3d(state, cache, radiation_mode, :face_sw_flux_dn)

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

compute_rsdt(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_toa(state, cache, radiation_mode, :face_sw_flux_dn)

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

compute_rsds(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_sw_flux_dn)

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

compute_rsu(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_3d(state, cache, radiation_mode, :face_sw_flux_up)

add_diagnostic_variable!(short_name = "rsu", units = "W m^-2",
    long_name = "Upwelling Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air",
    comments = "Upwelling shortwave radiation (3d)",
    compute = compute_rsu,
)

###
# TOA upwelling shortwave radiation (2d)
###
compute_rsut(state, cache, time) =
    compute_rsut(state, cache, time, cache.atmos.radiation_mode)
compute_rsut(_, _, _, radiation_mode) =
    error_diagnostic_variable("rsut", radiation_mode)

compute_rsut(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_toa(state, cache, radiation_mode, :face_sw_flux_up)

add_diagnostic_variable!(short_name = "rsut", units = "W m^-2",
    long_name = "TOA Outgoing Shortwave Radiation",
    standard_name = "toa_outgoing_shortwave_flux",
    comments = "Upwelling shortwave radiation at the top of the atmosphere",
    compute = compute_rsut,
)

###
# Surface upwelling shortwave radiation (2d)
###
compute_rsus(state, cache, time) =
    compute_rsus(state, cache, time, cache.atmos.radiation_mode)
compute_rsus(_, _, _, radiation_mode) = error_diagnostic_variable("rsus", radiation_mode)

compute_rsus(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_sw_flux_up)

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

compute_rld(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_3d(state, cache, radiation_mode, :face_lw_flux_dn)

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

compute_rlds(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_lw_flux_dn)

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

compute_rlu(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_3d(state, cache, radiation_mode, :face_lw_flux_up)

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

compute_rlut(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_toa(state, cache, radiation_mode, :face_lw_flux_up)

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

compute_rlus(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_lw_flux_up)

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

compute_rsdcs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
) = ᶠradiation_field_3d(state, cache, radiation_mode, :face_clear_sw_flux_dn)

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

compute_rsdscs(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_clear_sw_flux_dn)

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

compute_rsucs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
) = ᶠradiation_field_3d(state, cache, radiation_mode, :face_clear_sw_flux_up)

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

compute_rsutcs(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_toa(state, cache, radiation_mode, :face_clear_sw_flux_up)

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

compute_rsuscs(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_clear_sw_flux_up)

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

compute_rldcs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
) = ᶠradiation_field_3d(state, cache, radiation_mode, :face_clear_lw_flux_dn)

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

compute_rldscs(state, cache, _, ::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_sfc(state, cache, :face_clear_lw_flux_dn)

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
compute_rlucs(_, _, _, radiation_mode) = error_diagnostic_variable("rlucs", radiation_mode)

compute_rlucs(state, cache, _,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
) = ᶠradiation_field_3d(state, cache, radiation_mode, :face_clear_lw_flux_up)

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

compute_rlutcs(state, cache, _, radiation_mode::RRTMGPI.AbstractRRTMGPMode) =
    ᶠradiation_field_toa(state, cache, radiation_mode, :face_clear_lw_flux_up)

add_diagnostic_variable!(short_name = "rlutcs", units = "W m^-2",
    long_name = "TOA Outgoing Clear-Sky Longwave Radiation",
    standard_name = "toa_outgoing_longwave_flux_assuming_clear_sky",
    comments = "Upwelling clear-sky longwave radiation at the top of the atmosphere",
    compute = compute_rlutcs,
)


###
# Effective radius for liquid/ice clouds (3d)
###

# RRTMGP stores effective radii in microns; convert to SI metres.
function ᶜreff_field(state, cache, field_name::Symbol)
    field = getproperty(cache.radiation.rrtmgp_model, field_name)
    reff = Fields.array2field(field, axes(state.c))
    return @. lazy(reff / 1_000_000)  # μm -> m
end

compute_reffclw(state, cache, time) =
    compute_reffclw(state, cache, time, cache.atmos.radiation_mode)
compute_reffclw(_, _, _, radiation_mode) =
    error_diagnostic_variable("reffclw", radiation_mode)

compute_reffclw(state, cache, _,
    ::Union{RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics, RRTMGPI.AllSkyRadiation},
) = ᶜreff_field(state, cache, :center_cloud_liquid_effective_radius)

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

compute_reffcli(state, cache, _,
    ::Union{RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics, RRTMGPI.AllSkyRadiation},
) = ᶜreff_field(state, cache, :center_cloud_ice_effective_radius)

add_diagnostic_variable!(short_name = "reffcli", units = "m",
    long_name = "Effective radius for ice clouds",
    standard_name = "effective_radius_of_cloud_ice_particles",
    comments = "In-cloud ratio of the third moment over the second moment \
                of the particle size distribution. Set to zero outside of clouds.",
    compute = compute_reffcli,
)

###
# Aerosol optical depth diagnostics (2d)
###

# Requires aerosol_radiation = true; field is defined on the surface face level.
function ᶠaod_field(state, cache, field_name::Symbol)
    @assert cache.atmos.radiation_mode.aerosol_radiation "aerosol_radiation must be true to enable aerosol optical depth diagnostics"
    field = getproperty(cache.radiation.rrtmgp_model, field_name)
    return Fields.array2field(field, axes(Fields.level(state.f, half)))
end

const _AerosolRadiationModes = Union{
    RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
    RRTMGPI.AllSkyRadiation,
    RRTMGPI.ClearSkyRadiation,
}

compute_od550aer(state, cache, time) =
    compute_od550aer(state, cache, time, cache.atmos.radiation_mode)
compute_od550aer(_, _, _, radiation_mode) =
    error_diagnostic_variable("od550aer", radiation_mode)

compute_od550aer(state, cache, _, ::_AerosolRadiationModes) =
    ᶠaod_field(state, cache, :aod_sw_extinction)

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

compute_odsc550aer(state, cache, _, ::_AerosolRadiationModes) =
    ᶠaod_field(state, cache, :aod_sw_scattering)

add_diagnostic_variable!(short_name = "odsc550aer", units = "",
    long_name = "Ambient Scattering Aerosol Optical Thickness at 550nm",
    comments = "Aerosol scattering optical depth from the ambient aerosols at wavelength 550 nm",
    compute = compute_odsc550aer,
)
