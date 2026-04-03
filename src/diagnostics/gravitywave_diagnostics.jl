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
# All 2D Beres diagnostics are gated by beres_active: output 0 when inactive.
###

# Helper: apply beres_active gate to a 2D field
function _gated_copy!(out, field, active)
    if isnothing(out)
        result = copy(field)
        result .= ifelse.(active .> 0, field, eltype(field)(0))
        return result
    else
        out .= ifelse.(active .> 0, field, eltype(field)(0))
    end
end

# Beres activation flag
compute_nogw_beres_active!(out, state, cache, time) = compute_nogw_beres_active!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_beres_active!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_beres_active", non_orographic_gravity_wave)
compute_nogw_beres_active!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_beres_active requires Beres source enabled")

function compute_nogw_beres_active!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    if isnothing(out)
        return copy(cache.non_orographic_gravity_wave.gw_beres_active)
    else
        out .= cache.non_orographic_gravity_wave.gw_beres_active
    end
end

add_diagnostic_variable!(
    short_name = "nogw_beres_active",
    long_name = "NOGW Beres Source Activation Flag",
    units = "1",
    comments = "1 where Beres convective GW source is active (Q0 > threshold and h_heat > minimum), 0 otherwise",
    compute! = compute_nogw_beres_active!,
)

# Max convective heating rate Q0 (gated by beres_active)
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
    (; gw_Q0, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_Q0, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_Q0",
    long_name = "NOGW Max Convective Heating Rate",
    units = "K s-1",
    comments = "Column-maximum convective heating rate used by the Beres NOGW source spectrum (zero when inactive)",
    compute! = compute_nogw_Q0!,
)

# Convective heating depth (gated by beres_active)
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
    (; gw_h_heat, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_h_heat, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_h_heat",
    long_name = "NOGW Convective Heating Depth",
    units = "m",
    comments = "Vertical extent of significant convective heating used by the Beres NOGW source spectrum (zero when inactive)",
    compute! = compute_nogw_h_heat!,
)

# Convective envelope bottom height (gated by beres_active)
compute_nogw_zbot!(out, state, cache, time) = compute_nogw_zbot!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_zbot!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_zbot", non_orographic_gravity_wave)
compute_nogw_zbot!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_zbot requires Beres source enabled")

function compute_nogw_zbot!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    (; gw_zbot, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_zbot, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_zbot",
    long_name = "NOGW Convective Envelope Bottom Height",
    units = "m",
    comments = "Bottom of the convective heating envelope used by the Beres NOGW source (zero when inactive)",
    compute! = compute_nogw_zbot!,
)

# Convective envelope top height (gated by beres_active)
compute_nogw_ztop!(out, state, cache, time) = compute_nogw_ztop!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_ztop!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_ztop", non_orographic_gravity_wave)
compute_nogw_ztop!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_ztop requires Beres source enabled")

function compute_nogw_ztop!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    (; gw_ztop, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_ztop, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_ztop",
    long_name = "NOGW Convective Envelope Top Height",
    units = "m",
    comments = "Top of the convective heating envelope used by the Beres NOGW source (zero when inactive)",
    compute! = compute_nogw_ztop!,
)

# 3D convective heating rate (Q_conv) — NOT gated (3D field, beres_active is 2D)
compute_nogw_Q_conv!(out, state, cache, time) = compute_nogw_Q_conv!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_Q_conv!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_Q_conv", non_orographic_gravity_wave)
compute_nogw_Q_conv!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_Q_conv requires Beres source enabled")

function compute_nogw_Q_conv!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    if isnothing(out)
        return copy(cache.non_orographic_gravity_wave.gw_Q_conv)
    else
        out .= cache.non_orographic_gravity_wave.gw_Q_conv
    end
end

add_diagnostic_variable!(
    short_name = "nogw_Q_conv",
    long_name = "NOGW Convective Heating Rate",
    units = "K s-1",
    comments = "3D convective heating rate from EDMF mass-flux divergence used by the Beres NOGW source",
    compute! = compute_nogw_Q_conv!,
)

# Deep convection fraction: gw_deep_count / gw_cb_count
compute_nogw_deep_frac!(out, state, cache, time) = compute_nogw_deep_frac!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_deep_frac!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_deep_frac", non_orographic_gravity_wave)
compute_nogw_deep_frac!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_deep_frac requires Beres source enabled")

function compute_nogw_deep_frac!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    (; gw_deep_count, gw_cb_count) = cache.non_orographic_gravity_wave
    frac = @. ifelse(gw_cb_count > FT(0), gw_deep_count / gw_cb_count, FT(0))
    if isnothing(out)
        return copy(frac)
    else
        out .= frac
    end
end

add_diagnostic_variable!(
    short_name = "nogw_deep_frac",
    long_name = "NOGW Fraction of Callbacks with Deep Convection (z_top > 10km)",
    units = "1",
    comments = "Fraction of Beres callback invocations where z_top exceeded 10 km (cumulative since simulation start)",
    compute! = compute_nogw_deep_frac!,
)
