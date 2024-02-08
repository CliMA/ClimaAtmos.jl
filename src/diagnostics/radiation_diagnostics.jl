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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_sw_flux_dn,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_sw_flux_dn,
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
# TOA downwelling shortwave radiation (2d)
###
compute_rsdt!(out, state, cache, time) =
    compute_rsdt!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsdt!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsdt", radiation_mode)

function compute_rsdt!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    nlevels = Spaces.nlevels(axes(state.c))
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_dn,
                axes(state.f),
            ),
            nlevels + half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_dn,
                axes(state.f),
            ),
            nlevels + half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsdt",
    long_name = "TOA Incident Shortwave Radiation",
    standard_name = "toa_incoming_shortwave_flux",
    units = "W m^-2",
    comments = "Downward shortwave radiation at the top of the atmosphere",
    compute! = compute_rsdt!,
)

###
# Surface downwelling shortwave radiation (2d)
###
compute_rsds!(out, state, cache, time) =
    compute_rsds!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsds!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsds", radiation_mode)

function compute_rsds!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsds",
    long_name = "Surface Downwelling Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air",
    units = "W m^-2",
    comments = "Downwelling shortwave radiation at the surface",
    compute! = compute_rsds!,
)

###
# Upwelling shortwave radiation (2d)
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_sw_flux_up,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_sw_flux_up,
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
# TOA upwelling shortwave radiation (2d)
###
compute_rsut!(out, state, cache, time) =
    compute_rsut!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsut!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsut", radiation_mode)

function compute_rsut!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    nlevels = Spaces.nlevels(axes(state.c))
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsut",
    long_name = "TOA Outgoing Shortwave Radiation",
    standard_name = "toa_outgoing_shortwave_flux",
    units = "W m^-2",
    comments = "Upwelling shortwave radiation at the top of the atmosphere",
    compute! = compute_rsut!,
)

###
# Surface upwelling shortwave radiation (2d)
###
compute_rsus!(out, state, cache, time) =
    compute_rsus!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsus!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsus", radiation_mode)

function compute_rsus!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_up,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_sw_flux_up,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsus",
    long_name = "Surface Upwelling Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air",
    units = "W m^-2",
    comments = "Upwelling shortwave radiation at the surface",
    compute! = compute_rsus!,
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_lw_flux_dn,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_lw_flux_dn,
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
# Surface downwelling longwave radiation (2d)
###
compute_rlds!(out, state, cache, time) =
    compute_rlds!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rlds!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rlds", radiation_mode)

function compute_rlds!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_lw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_lw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rlds",
    long_name = "Surface Downwelling Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air",
    units = "W m^-2",
    comments = "Downwelling longwave radiation at the surface",
    compute! = compute_rlds!,
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_lw_flux_up,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_lw_flux_up,
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
# TOA upwelling longwave radiation (2d)
###
compute_rlut!(out, state, cache, time) =
    compute_rlut!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rlut!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rlut", radiation_mode)

function compute_rlut!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    nlevels = Spaces.nlevels(axes(state.c))
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_lw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_lw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rlut",
    long_name = "TOA Outgoing Longwave Radiation",
    standard_name = "toa_outgoing_longwave_flux",
    units = "W m^-2",
    comments = "Upwelling longwave radiation at the top of the atmosphere",
    compute! = compute_rlut!,
)

###
# Surface upwelling longwave radiation (2d)
###
compute_rlus!(out, state, cache, time) =
    compute_rlus!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rlus!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rlus", radiation_mode)

function compute_rlus!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_lw_flux_up,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_lw_flux_up,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rlus",
    long_name = "Surface Upwelling Longwave Radiation",
    standard_name = "surface_upwelling_longwave_flux_in_air",
    units = "W m^-2",
    comments = "Upwelling longwave radiation at the surface",
    compute! = compute_rlus!,
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_sw_flux_dn,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_sw_flux_dn,
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
# Surface downwelling clear sky shortwave radiation (2d)
###
compute_rsdscs!(out, state, cache, time) =
    compute_rsdscs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsdscs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsdscs", radiation_mode)

function compute_rsdscs!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_sw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_sw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsdscs",
    long_name = "Surface Downwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Downwelling clear-sky shortwave radiation at the surface",
    compute! = compute_rsdscs!,
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_sw_flux_up,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_sw_flux_up,
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
# TOA upwelling clear sky shortwave radiation (2d)
###
compute_rsutcs!(out, state, cache, time) =
    compute_rsutcs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsutcs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsutcs", radiation_mode)

function compute_rsutcs!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    nlevels = Spaces.nlevels(axes(state.c))
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_sw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_sw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsutcs",
    long_name = "TOA Outgoing Clear-Sky Shortwave Radiation",
    standard_name = "toa_outgoing_shortwave_flux_assuming_clear_sky",
    units = "W m^-2",
    comments = "Upwelling clear-sky shortwave radiation at the top of the atmosphere",
    compute! = compute_rsutcs!,
)

###
# Surface clear sky upwelling shortwave radiation (2d)
###
compute_rsuscs!(out, state, cache, time) =
    compute_rsuscs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rsuscs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rsuscs", radiation_mode)

function compute_rsuscs!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_sw_flux_up,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_sw_flux_up,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rsuscs",
    long_name = "Surface Upwelling Clear-Sky Shortwave Radiation",
    standard_name = "surface_upwelling_shortwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Upwelling clear-sky shortwave radiation at the surface",
    compute! = compute_rsuscs!,
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_lw_flux_dn,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_lw_flux_dn,
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
# Surface clear sky downwelling longwave radiation (2d)
###
compute_rldscs!(out, state, cache, time) =
    compute_rldscs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rldscs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rldscs", radiation_mode)

function compute_rldscs!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_lw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_lw_flux_dn,
                axes(state.f),
            ),
            half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rldscs",
    long_name = "Surface Downwelling Clear-Sky Longwave Radiation",
    standard_name = "surface_downwelling_longwave_flux_in_air_assuming_clear_sky",
    units = "W m^-2",
    comments = "Downwelling clear-sky longwave radiation at the surface",
    compute! = compute_rldscs!,
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
    if isnothing(out)
        return RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_lw_flux_up,
            axes(state.f),
        )
    else
        out .= RRTMGPI.array2field(
            cache.radiation.radiation_model.face_clear_lw_flux_up,
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

###
# TOA clear sky upwelling longwave radiation (2d)
###
compute_rlutcs!(out, state, cache, time) =
    compute_rlutcs!(out, state, cache, time, cache.atmos.radiation_mode)
compute_rlutcs!(_, _, _, _, radiation_mode::T) where {T} =
    error_diagnostic_variable("rlutcs", radiation_mode)

function compute_rlutcs!(
    out,
    state,
    cache,
    time,
    radiation_mode::T,
) where {T <: RRTMGPI.AbstractRRTMGPMode}
    nlevels = Spaces.nlevels(axes(state.c))
    if isnothing(out)
        return Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_lw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    else
        out .= Fields.level(
            RRTMGPI.array2field(
                cache.radiation.radiation_model.face_clear_lw_flux_up,
                axes(state.f),
            ),
            nlevels + half,
        )
    end
end

add_diagnostic_variable!(
    short_name = "rlutcs",
    long_name = "TOA Outgoing Clear-Sky Longwave Radiation",
    standard_name = "toa_outgoing_longwave_flux_assuming_clear_sky",
    units = "W m^-2",
    comments = "Upwelling clear-sky longwave radiation at the top of the atmosphere",
    compute! = compute_rlutcs!,
)
