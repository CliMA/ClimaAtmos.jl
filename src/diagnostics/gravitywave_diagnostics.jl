# This file is included in diagnostics.jl

# Gravity-wave related diagnostics

###
# Eastward Non-Orographic Gravity-Wave Tendency 
###
compute_utendnogw!(out, state, cache, time) =
    compute_utendnogw!(out, state, cache, time, cache.atmos.non_orographic_gravity_wave)
compute_utendnogw!(_, _, _, _, non_orographic_gravity_wave::T) where {T} =
    error_diagnostic_variable("utendnogw", non_orographic_gravity_wave)

function compute_utendnogw!(
    out,
    state,
    cache,
    time,
    ::T,
) where {T <: NonOrographicGravityWave}
    if isnothing(out)
        # u_waveforcing is not a field:
        # return Fields.array2field(
        #     cache.non_orographic_gravity_wave.u_waveforcing,
        #     axes(state.c),
        # )
        return copy(cache.non_orographic_gravity_wave.u_waveforcing)
    else
        out .= cache.non_orographic_gravity_wave.u_waveforcing
    end
end

add_diagnostic_variable!(
    short_name = "utendnogw",
    long_name = "Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",
    standard_name = "tendency_of_eastward_wind_due_to_nonorographic_gravity_wave_drag",
    units = "m s-2",
    comments = "Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",
    compute! = compute_utendnogw!,
)


###
# Northward Non-Orographic Gravity-Wave Tendency 
###
compute_vtendnogw!(out, state, cache, time) =
    compute_vtendnogw!(out, state, cache, time, cache.atmos.non_orographic_gravity_wave)
compute_vtendnogw!(_, _, _, _, non_orographic_gravity_wave::T) where {T} =
    error_diagnostic_variable("vtendnogw", non_orographic_gravity_wave)

function compute_vtendnogw!(
    out,
    state,
    cache,
    time,
    ::T,
) where {T <: NonOrographicGravityWave}
    if isnothing(out)
        # v_waveforcing is not a Field:
        # return Fields.array2field(
        #     cache.non_orographic_gravity_wave.v_waveforcing,
        #     axes(state.c),
        # )
        return copy(cache.non_orographic_gravity_wave.v_waveforcing)
    else
        out .= cache.non_orographic_gravity_wave.v_waveforcing
    end
end

add_diagnostic_variable!(
    short_name = "vtendnogw",
    long_name = "Northward Acceleration Due to Non-Orographic Gravity Wave Drag",
    standard_name = "tendency_of_northward_wind_due_to_nonorographic_gravity_wave_drag",
    units = "m s-2",
    comments = "Northward Acceleration Due to Non-Orographic Gravity Wave Drag",
    compute! = compute_vtendnogw!,
)