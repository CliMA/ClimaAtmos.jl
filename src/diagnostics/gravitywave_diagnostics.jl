# This file is included in diagnostics.jl
# Gravity-wave related diagnostics

###
# Eastward Non-Orographic Gravity-Wave Tendency
###
compute_utendnogw(state, cache, time) = compute_utendnogw(
    state, cache, time, cache.atmos.non_orographic_gravity_wave,
)
compute_utendnogw(_, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("utendnogw", non_orographic_gravity_wave)

compute_utendnogw(_, cache, _, ::NonOrographicGravityWave) =
    cache.non_orographic_gravity_wave.uforcing

add_diagnostic_variable!(short_name = "utendnogw", units = "m s^-2",
    long_name = "Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",
    standard_name = "tendency_of_eastward_wind_due_to_nonorographic_gravity_wave_drag",
    comments = "Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",
    compute = compute_utendnogw,
)

###
# Northward Non-Orographic Gravity-Wave Tendency
###
compute_vtendnogw(state, cache, time) = compute_vtendnogw(
    state, cache, time, cache.atmos.non_orographic_gravity_wave,
)
compute_vtendnogw(_, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("vtendnogw", non_orographic_gravity_wave)

compute_vtendnogw(_, cache, _, ::NonOrographicGravityWave) =
    cache.non_orographic_gravity_wave.vforcing

add_diagnostic_variable!(short_name = "vtendnogw", units = "m s^-2",
    long_name = "Northward Acceleration Due to Non-Orographic Gravity Wave Drag",
    standard_name = "tendency_of_northward_wind_due_to_nonorographic_gravity_wave_drag",
    comments = "Northward Acceleration Due to Non-Orographic Gravity Wave Drag",
    compute = compute_vtendnogw,
)

###
# Beres convective GW diagnostics (2D surface fields)
# These dispatch on the BS type parameter: Nothing = no Beres, BeresSourceParams = Beres enabled
###

# Max convective heating rate Q0
compute_nogw_Q0!(out, state, cache, time) = compute_nogw_Q0!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_Q0!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_Q0", non_orographic_gravity_wave)
compute_nogw_Q0!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_Q0 requires Beres source enabled")

function compute_nogw_Q0!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    if isnothing(out)
        return copy(cache.non_orographic_gravity_wave.gw_Q0)
    else
        out .= cache.non_orographic_gravity_wave.gw_Q0
    end
end

add_diagnostic_variable!(
    short_name = "nogw_Q0",
    long_name = "NOGW Max Convective Heating Rate",
    units = "K s-1",
    comments = "Column-maximum convective heating rate used by the Beres NOGW source spectrum",
    compute! = compute_nogw_Q0!,
)

# Convective heating depth
compute_nogw_h_heat!(out, state, cache, time) = compute_nogw_h_heat!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_h_heat!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_h_heat", non_orographic_gravity_wave)
compute_nogw_h_heat!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_h_heat requires Beres source enabled")

function compute_nogw_h_heat!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    if isnothing(out)
        return copy(cache.non_orographic_gravity_wave.gw_h_heat)
    else
        out .= cache.non_orographic_gravity_wave.gw_h_heat
    end
end

add_diagnostic_variable!(
    short_name = "nogw_h_heat",
    long_name = "NOGW Convective Heating Depth",
    units = "m",
    comments = "Vertical extent of significant convective heating used by the Beres NOGW source spectrum",
    compute! = compute_nogw_h_heat!,
)
