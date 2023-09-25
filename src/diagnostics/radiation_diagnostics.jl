# This file is included in Diagnostics.jl

# Radiative fluxes

###
# Downwelling shortwave radiation (3d)
###
compute_rsd!(out, state, cache, time) =
    compute_rsd!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsd!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsd", radiation_mode)

function compute_rsd!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_sw_flux_dn),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_sw_flux_dn),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsd",
    long_name = "Downwelling Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air",
    units = "W m^-2",
    comments = "Downwelling shortwave radiation",
    compute! = compute_rsd!,
)

###
# Upwelling shortwave radiation (3d)
###
compute_rsu!(out, state, cache, time) =
    compute_rsu!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsu!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsu", radiation_mode)

function compute_rsu!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_sw_flux_up),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_sw_flux_up),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsu",
    long_name = "Upwelling Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air",
    units = "W m^-2",
    comments = "Upwelling shortwave radiation",
    compute! = compute_rsu!,
)

###
# Downwelling longwave radiation (3d)
###
compute_rld!(out, state, cache, time) =
    compute_rld!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rld!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rld", radiation_mode)

function compute_rld!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_lw_flux_dn),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_lw_flux_dn),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rld",
    long_name = "Downwelling Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air",
    units = "W m^-2",
    comments = "Downwelling longwave radiation",
    compute! = compute_rld!,
)

###
# Upwelling longwave radiation (3d)
###
compute_rlu!(out, state, cache, time) =
    compute_rlu!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rlu!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rlu", radiation_mode)

function compute_rlu!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_lw_flux_up),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_lw_flux_up),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rlu",
    long_name = "Upwelling Longwave Radiation",
    standard_name = "surface_upwelling_longwave_flux_in_air",
    units = "W m^-2",
    comments = "Upwelling longwave radiation",
    compute! = compute_rlu!,
)

###
# Downelling clear sky shortwave radiation (3d)
###
compute_rsdcs!(out, state, cache, time) =
    compute_rsdcs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsdcs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsdcs", radiation_mode)

function compute_rsdcs!(
    out,
    state,
    cache,
    time,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_sw_flux_dn),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_sw_flux_dn),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsdcs",
    long_name = "Downwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Downwelling clear sky shortwave radiation",
    compute! = compute_rsdcs!,
)

###
# Upwelling clear sky shortwave radiation (3d)
###
compute_rsucs!(out, state, cache, time) =
    compute_rsucs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsucs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsucs", radiation_mode)

function compute_rsucs!(
    out,
    state,
    cache,
    time,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_sw_flux_up),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_sw_flux_up),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsucs",
    long_name = "Upwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Upwelling clear sky shortwave radiation",
    compute! = compute_rsucs!,
)

###
# Downwelling clear sky longwave radiation (3d)
###
compute_rldcs!(out, state, cache, time) =
    compute_rldcs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rldcs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rldcs", radiation_mode)

function compute_rldcs!(
    out,
    state,
    cache,
    time,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_lw_flux_dn),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_lw_flux_dn),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rldcs",
    long_name = "Downwelling Clear-Sky Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Downwelling clear sky longwave radiation",
    compute! = compute_rldcs!,
)

###
# Upwelling clear sky longwave radiation (3d)
###
compute_rlucs!(out, state, cache, time) =
    compute_rlucs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rlucs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rlucs", radiation_mode)

function compute_rlucs!(
    out,
    state,
    cache,
    time,
    radiation_mode::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
)
    FT = eltype(state)
    if isnothing(out)
        return RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_lw_flux_up),
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            FT.(cache.radiation_model.face_clear_lw_flux_up),
            axes(state.f),
        )
    end
end

add_diagnostic_variable!(
    short_name = "rlucs",
    long_name = "Upwelling Clear-Sky Longwave Radiation",
    standard_name = "surface_upwelling_longwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Upwelling clear sky longwave radiation",
    compute! = compute_rlucs!,
)
