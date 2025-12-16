# This file is included in diagnostics.jl
# Gravity-wave related diagnostics

###
# Eastward Non-Orographic Gravity-Wave Tendency 
###
compute_utendnogw(state, cache, time) = 
    compute_utendnogw(state, cache, time, cache.atmos.non_orographic_gravity_wave)
compute_utendnogw(_, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("utendnogw", non_orographic_gravity_wave)

compute_utendnogw(_, cache, _, ::NonOrographicGravityWave) = 
    cache.non_orographic_gravity_wave.uforcing

add_diagnostic_variable!(short_name = "utendnogw", units = "m s-2",
    long_name = "Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",
    standard_name = "tendency_of_eastward_wind_due_to_nonorographic_gravity_wave_drag",
    comments = "Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",
    compute = compute_utendnogw,
)

###
# Northward Non-Orographic Gravity-Wave Tendency 
###
compute_vtendnogw(state, cache, time) = 
    compute_vtendnogw(state, cache, time, cache.atmos.non_orographic_gravity_wave)
compute_vtendnogw(_, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("vtendnogw", non_orographic_gravity_wave)

compute_vtendnogw(_, cache, _, ::NonOrographicGravityWave) = 
    cache.non_orographic_gravity_wave.vforcing

add_diagnostic_variable!(short_name = "vtendnogw", units = "m s-2",
    long_name = "Northward Acceleration Due to Non-Orographic Gravity Wave Drag",
    standard_name = "tendency_of_northward_wind_due_to_nonorographic_gravity_wave_drag",
    comments = "Northward Acceleration Due to Non-Orographic Gravity Wave Drag",
    compute = compute_vtendnogw,
)
