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
# Beres convective GW diagnostics (source-internal verification fields)
# These dispatch on the BS type parameter: Nothing = no Beres, BeresSourceParams = Beres enabled
#
# Every diagnostic below is verification/debug-only — it exposes the convective
# source internals and is toggled by the Beres `detailed_diagnostics` flag
# (config `nogw_beres_detailed_diagnostics`).
###

# Error unless Beres detailed (verification) diagnostics are enabled.
# Reached only from the BeresSourceParams methods, so `beres_source` is non-nothing.
function _require_beres_detailed(short_name, cache)
    cache.atmos.non_orographic_gravity_wave.beres_source.detailed_diagnostics ||
        error_diagnostic_variable(
            "$short_name requires nogw_beres_detailed_diagnostics = true " *
            "(Beres source-internal diagnostics are verification/debug-only)",
        )
    return nothing
end

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

# --- Activation & envelope geometry (2D, gated by beres_active) ---
# nogw_beres_active, nogw_Q0, nogw_h_heat, nogw_zbot, nogw_ztop

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
    _require_beres_detailed("nogw_beres_active", cache)
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

# Convective heating amplitude Q0 (gated by beres_active)
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
    _require_beres_detailed("nogw_Q0", cache)
    (; gw_Q0, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_Q0, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_Q0",
    long_name = "NOGW Convective Heating Amplitude",
    units = "K s-1",
    comments = "Half-sine in-cloud convective heating amplitude ((pi/2) times the depth-mean of the in-cloud Q_conv_ic over the convective envelope) forcing the Beres NOGW source spectrum (zero when inactive)",
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
    _require_beres_detailed("nogw_h_heat", cache)
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
    _require_beres_detailed("nogw_zbot", cache)
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
    _require_beres_detailed("nogw_ztop", cache)
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

# --- 3D heating & source-profile fields (NOT gated; 3D, already zero
# outside the envelope / in inactive columns) ---
# nogw_Q_conv, nogw_Q_conv_ic, nogw_halfsine

# 3D convective heating rate (Q_conv)
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
    _require_beres_detailed("nogw_Q_conv", cache)
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
    comments = "3D grid-mean convective heating rate from EDMF mass-flux divergence; used for Beres envelope detection and activation gating",
    compute! = compute_nogw_Q_conv!,
)

# 3D in-cloud convective heating rate (Q_conv_ic)
compute_nogw_Q_conv_ic!(out, state, cache, time) = compute_nogw_Q_conv_ic!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_Q_conv_ic!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_Q_conv_ic", non_orographic_gravity_wave)
compute_nogw_Q_conv_ic!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_Q_conv_ic requires Beres source enabled")

function compute_nogw_Q_conv_ic!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    _require_beres_detailed("nogw_Q_conv_ic", cache)
    if isnothing(out)
        return copy(cache.non_orographic_gravity_wave.gw_Q_conv_ic)
    else
        out .= cache.non_orographic_gravity_wave.gw_Q_conv_ic
    end
end

add_diagnostic_variable!(
    short_name = "nogw_Q_conv_ic",
    long_name = "NOGW In-Cloud Convective Heating Rate",
    units = "K s-1",
    comments = "3D in-cloud (per-draft conditional-mean) convective heating rate from EDMF mass-flux divergence without the area-fraction dilution; sets the Beres NOGW source spectrum amplitude",
    compute! = compute_nogw_Q_conv_ic!,
)

# 3D native launched half-sine source profile Q0·sin(π(z−z_bot)/h)
compute_nogw_halfsine!(out, state, cache, time) = compute_nogw_halfsine!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_halfsine!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_halfsine", non_orographic_gravity_wave)
compute_nogw_halfsine!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_halfsine requires Beres source enabled")

function compute_nogw_halfsine!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    _require_beres_detailed("nogw_halfsine", cache)
    if isnothing(out)
        return copy(cache.non_orographic_gravity_wave.gw_halfsine)
    else
        out .= cache.non_orographic_gravity_wave.gw_halfsine
    end
end

add_diagnostic_variable!(
    short_name = "nogw_halfsine",
    long_name = "NOGW Beres Launched Half-Sine Source Profile",
    units = "K s-1",
    comments = "3D native half-sine source profile Q0*sin(pi*(z-z_bot)/h) over the convective heating depth, computed in-column so it is remap-consistent with nogw_Q_conv_ic (the moment-matched envelope it approximates); zero outside the envelope and in inactive columns",
    compute! = compute_nogw_halfsine!,
)

# --- Launched-spectrum summaries (2D, gated by beres_active) ---
# nogw_launch_flux, nogw_c_centroid, nogw_a_cover

# Launched source momentum flux magnitude (2D, gated by beres_active)
compute_nogw_launch_flux!(out, state, cache, time) = compute_nogw_launch_flux!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_launch_flux!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_launch_flux", non_orographic_gravity_wave)
compute_nogw_launch_flux!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_launch_flux requires Beres source enabled")

function compute_nogw_launch_flux!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    _require_beres_detailed("nogw_launch_flux", cache)
    (; gw_launch_flux, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_launch_flux, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_launch_flux",
    long_name = "NOGW Beres Launched Source Flux Magnitude",
    units = "Pa",
    comments = "Total launched Beres source momentum flux magnitude, a_cover * sum_c |B0(c)|, summed over the zonal-direction phase-speed spectrum at the launch level (zero when inactive)",
    compute! = compute_nogw_launch_flux!,
)

# Launched-spectrum flux-weighted phase-speed centroid (2D, gated by beres_active)
compute_nogw_c_centroid!(out, state, cache, time) = compute_nogw_c_centroid!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_c_centroid!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_c_centroid", non_orographic_gravity_wave)
compute_nogw_c_centroid!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_c_centroid requires Beres source enabled")

function compute_nogw_c_centroid!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    _require_beres_detailed("nogw_c_centroid", cache)
    (; gw_c_centroid, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_c_centroid, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_c_centroid",
    long_name = "NOGW Beres Launched Spectrum Phase-Speed Centroid",
    units = "m s-1",
    comments = "Flux-weighted phase-speed centroid of the launched Beres source spectrum, sum_c c*|B0(c)| / sum_c |B0(c)| in the zonal direction; characterizes the spectral shape for the Beres-c vs Beres-G comparison (zero when inactive)",
    compute! = compute_nogw_c_centroid!,
)

# Convective coverage (envelope-mean updraft area fraction, gated by beres_active)
compute_nogw_a_cover!(out, state, cache, time) = compute_nogw_a_cover!(
    out,
    state,
    cache,
    time,
    cache.atmos.non_orographic_gravity_wave,
)
compute_nogw_a_cover!(_, _, _, _, non_orographic_gravity_wave) =
    error_diagnostic_variable("nogw_a_cover", non_orographic_gravity_wave)
compute_nogw_a_cover!(_, _, _, _, ::NonOrographicGravityWave{FT, Nothing}) where {FT} =
    error_diagnostic_variable("nogw_a_cover requires Beres source enabled")

function compute_nogw_a_cover!(
    out,
    state,
    cache,
    time,
    ::NonOrographicGravityWave{FT, <:BeresSourceParams},
) where {FT}
    _require_beres_detailed("nogw_a_cover", cache)
    (; gw_a_cover, gw_beres_active) = cache.non_orographic_gravity_wave
    return _gated_copy!(out, gw_a_cover, gw_beres_active)
end

add_diagnostic_variable!(
    short_name = "nogw_a_cover",
    long_name = "NOGW Convective Coverage",
    units = "1",
    comments = "Mass-weighted envelope-mean updraft area fraction; dilutes the deposited Beres NOGW momentum flux (intermittency analog, zero when inactive)",
    compute! = compute_nogw_a_cover!,
)
