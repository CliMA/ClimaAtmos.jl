# plot_varanal.jl
#
# ARM VARANAL single-column observation comparison plots.
#
# The standard EDMF column diagnostics (summary.pdf) are handled by the
# EDMFColumnPlots union in ci_plots.jl.  This file adds a more-specific
# make_plots method that:
#   1. Runs the standard EDMF column plots  →  summary.pdf
#   2. Compares model output against ARM observations (sonde, BEATM, CLDRAD)
#      →  obs_comparison.pdf  (skipped when data is unavailable)
#
# Included from ci_plots.jl.  CairoMakie, ClimaAnalysis, Poppler_jll,
# Statistics.mean, and ClimaAtmos (CA) are already in scope.

import NCDatasets as NC
import Dates
import Interpolations as Intp
import Statistics: std, quantile, median

# ────────────────────────────────────────────────────────────────────────────
# Paths (overridable via environment variables)
# ────────────────────────────────────────────────────────────────────────────

const _ARM_OBS_BASE = get(
    ENV,
    "ARM_OBS_DIR",
    "/net/sampo/data1/cchristo/ARM_data/arm_265588",
)
const _SONDE_DIR = get(
    ENV,
    "ARM_SONDE_DIR",
    joinpath(_ARM_OBS_BASE, "sgpinterpolatedsondeC1.c1"),
)
const _BEATM_DIR = get(
    ENV,
    "ARM_BEATM_DIR",
    joinpath(_ARM_OBS_BASE, "sgparmbeatmC1.c1"),
)
const _CLDRAD_DIR = get(
    ENV,
    "ARM_CLDRAD_DIR",
    joinpath(_ARM_OBS_BASE, "sgparmbecldradC1.c1"),
)

const _OBS_MAX_HEIGHT = 5000.0
const _OBS_N_VERT = 100

# SGP Central Facility elevation (m above sea level).
# Sonde heights are ASL; model/BEATM heights are AGL.
const _SGP_ELEVATION_M = 315.0

# UTC offset for the observation site (hours behind UTC).
# SGP is in US Central = UTC-6.  Used to select a local-day
# window for diurnal profile snapshots.
const _SITE_UTC_OFFSET_H = -6

# Day to use for diurnal-cycle profile snapshots.
# :last  → last full day of the simulation
# integer → 0-based day offset from sim start (0 = first day)
const _PROFILE_SNAPSHOT_DAY = :last

# ────────────────────────────────────────────────────────────────────────────
# Variable mapping tables
# ────────────────────────────────────────────────────────────────────────────

# Profile variables with per-variable colormaps
const _SONDE_MAP = [
    (model = "ta",  obs = "temp",   mfn = x -> x .- 273.15, ofn = identity,
     label = "Temperature",       units = "°C",
     cmap = Makie.Reverse(:RdYlBu), divergent = false),
    (model = "hur", obs = "rh",     mfn = x -> x .* 100.0,  ofn = identity,
     label = "Relative Humidity", units = "%",
     cmap = :YlGnBu,                divergent = false),
    (model = "hus", obs = "sh",     mfn = x -> x .* 1000.0, ofn = x -> x .* 1000.0,
     label = "Specific Humidity", units = "g/kg",
     cmap = :YlGnBu,                divergent = false),
    (model = "ua",  obs = "u_wind", mfn = identity,         ofn = identity,
     label = "U Wind",            units = "m/s",
     cmap = Makie.Reverse(:RdBu),   divergent = true),
    (model = "va",  obs = "v_wind", mfn = identity,         ofn = identity,
     label = "V Wind",            units = "m/s",
     cmap = Makie.Reverse(:RdBu),   divergent = true),
    (model = "thetaa", obs = "potential_temp", mfn = identity, ofn = identity,
     label = "Potential Temperature", units = "K",
     cmap = Makie.Reverse(:RdYlBu), divergent = false),
]

# BEATM profile variables (height-coordinate, sparse ~3-hourly)
const _BEATM_PROFILE_MAP = [
    (model = "ta",  obs = "T_z",  mfn = identity,          ofn = identity,
     label = "Temperature",       units = "K",
     cmap = Makie.Reverse(:RdYlBu), divergent = false),
    (model = "hur", obs = "rh_z", mfn = x -> x .* 100.0,   ofn = identity,
     label = "Relative Humidity", units = "%",
     cmap = :YlGnBu,                divergent = false),
    (model = "ua",  obs = "u_z",  mfn = identity,          ofn = identity,
     label = "U Wind",            units = "m/s",
     cmap = Makie.Reverse(:RdBu),   divergent = true),
    (model = "va",  obs = "v_z",  mfn = identity,          ofn = identity,
     label = "V Wind",            units = "m/s",
     cmap = Makie.Reverse(:RdBu),   divergent = true),
    (model = "thetaa", obs = "theta_z", mfn = identity,   ofn = identity,
     label = "Potential Temperature", units = "K",
     cmap = Makie.Reverse(:RdYlBu), divergent = false),
]

# Surface time-series: multi-source overlays grouped by physical quantity
# Each entry can have obs from multiple products
const _SURFACE_OVERLAY_MAP = [
    (model = "ts", label = "Surface Temperature", units = "K",
     obs = [
         (src = :beatm, var = "T_sfc", ofn = identity, label = "BEATM"),
     ]),
    (model = "pr", label = "Precipitation Rate", units = "mm/hr",
     obs = [
         (src = :beatm, var = "prec_sfc", ofn = identity, label = "BEATM"),
     ]),
    (model = "lwp", label = "Liquid Water Path", units = "g/m²",
     obs = [
         (src = :cldrad, var = "lwp", ofn = identity, label = "CLDRAD"),
     ]),
]

# Radiation: model surface diagnostics vs CLDRAD surface obs
const _RADIATION_MAP = [
    (model = "rsds", label = "SW Down (surface)", units = "W/m²",
     obs = [
         (src = :cldrad, var = "swdn", ofn = identity, label = "CLDRAD"),
     ]),
    (model = "rsus", label = "SW Up (surface)", units = "W/m²",
     obs = [
         (src = :cldrad, var = "swup", ofn = identity, label = "CLDRAD"),
     ]),
    (model = "rlds", label = "LW Down (surface)", units = "W/m²",
     obs = [
         (src = :cldrad, var = "lwdn", ofn = identity, label = "CLDRAD"),
     ]),
    (model = "rlus", label = "LW Up (surface)", units = "W/m²",
     obs = [
         (src = :cldrad, var = "lwup", ofn = identity, label = "CLDRAD"),
     ]),
]

# Surface fluxes: BEATM has observed SH/LH
const _FLUX_MAP = [
    (label = "Sensible Heat Flux", units = "W/m²",
     obs = [
         (src = :beatm, var = "SH_baebbr", ofn = identity, label = "BEATM BAEBBR"),
         (src = :beatm, var = "SH_qcecor", ofn = identity, label = "BEATM QCECOR"),
     ]),
    (label = "Latent Heat Flux", units = "W/m²",
     obs = [
         (src = :beatm, var = "LH_baebbr", ofn = identity, label = "BEATM BAEBBR"),
         (src = :beatm, var = "LH_qcecor", ofn = identity, label = "BEATM QCECOR"),
     ]),
]

# Cloud fraction: CLDRAD profile + surface totals
const _CLOUD_MAP = [
    (label = "Total Cloud Fraction", units = "%",
     obs = [
         (src = :cldrad, var = "tot_cld",     ofn = identity, label = "CLDRAD ARSCL"),
         (src = :cldrad, var = "tot_cld_tsi",  ofn = identity, label = "CLDRAD TSI"),
     ]),
]

# ────────────────────────────────────────────────────────────────────────────
# NaN-safe helpers
# ────────────────────────────────────────────────────────────────────────────

_nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
_nanstd(x)  = (v = filter(!isnan, x); length(v) < 2 ? NaN : std(v))
_nanquantile(x, q) = (v = filter(!isnan, x); isempty(v) ? NaN : quantile(v, q))

function _to_float_nan(arr; sentinel = -9999.0)
    out = Float64[ismissing(v) ? NaN : Float64(v) for v in arr]
    out[out .< sentinel + 1] .= NaN
    return out
end

# ────────────────────────────────────────────────────────────────────────────
# OutputVar → raw arrays
# ────────────────────────────────────────────────────────────────────────────

function _slice_column(var)
    haskey(var.dims, "x") && (var = slice(var, x = var.dims["x"][1]))
    haskey(var.dims, "y") && (var = slice(var, y = var.dims["y"][1]))
    return var
end

function _extract_zt(var)
    z_key = z_dim_name(var)
    z_vals = Float64.(collect(var.dims[z_key]))
    t_vals = Float64.(collect(var.dims["time"]))
    raw = Float64.(var.data)
    if var.index2dim[1] == "time"
        raw = permutedims(raw)
    end
    return raw, z_vals, t_vals
end

function _extract_timeseries(var)
    t_vals = Float64.(collect(var.dims["time"]))
    return Float64.(vec(var.data)), t_vals
end

# ────────────────────────────────────────────────────────────────────────────
# ARM data loading
# ────────────────────────────────────────────────────────────────────────────

function _load_arm_sonde(sonde_dir::String, start_date::Dates.Date, n_days::Int)
    isdir(sonde_dir) || return nothing
    obs_vars = ["temp", "rh", "sh", "u_wind", "v_wind", "wspd",
                 "bar_pres", "potential_temp"]
    all_times = Dates.DateTime[]
    all_data = Dict(v => Vector{Float64}[] for v in obs_vars)
    heights_km = nothing

    for d in 0:n_days
        date = start_date + Dates.Day(d)
        date_str = Dates.format(date, "yyyymmdd")
        prefix = "sgpinterpolatedsondeC1.c1.$(date_str)"
        fnames = filter(
            f -> startswith(f, prefix) && endswith(f, ".nc"),
            readdir(sonde_dir),
        )
        for fname in fnames
            ds = nothing
            try
                ds = NC.NCDataset(joinpath(sonde_dir, fname))
                if isnothing(heights_km)
                    heights_km = _to_float_nan(ds["height"][:])
                end
                times = ds["time"][:]
                for i in 1:length(times)
                    push!(all_times, times[i])
                    for vn in obs_vars
                        haskey(ds, vn) || continue
                        arr = ds[vn]
                        raw_maybe = if ndims(arr) == 2
                            arr[:, i]
                        elseif ndims(arr) == 1
                            arr[:]
                        else
                            continue
                        end
                        push!(all_data[vn], _to_float_nan(raw_maybe))
                    end
                end
            catch e
                @warn "Could not load sonde $fname" exception = e
            finally
                !isnothing(ds) && close(ds)
            end
        end
    end
    isempty(all_times) && return nothing
    heights_m = heights_km .* 1000.0 .- _SGP_ELEVATION_M
    matrices = Dict{String, Matrix{Float64}}()
    for (vn, vecs) in all_data
        isempty(vecs) && continue
        matrices[vn] = reduce(hcat, vecs)'
    end
    return (data = matrices, times = all_times, heights_m = heights_m)
end

function _load_arm_yearly_cdf(
    data_dir::String, file_prefix::String,
    start_date::Dates.Date, n_days::Int,
    profile_vars::Vector{String}, surface_vars::Vector{String};
    height_key::String = "z",
)
    isdir(data_dir) || return nothing
    year = Dates.year(start_date)
    pattern = "$(file_prefix).$(year)0101.000000.cdf"
    fpath = joinpath(data_dir, pattern)
    isfile(fpath) || return nothing

    ds = nothing
    try
        ds = NC.NCDataset(fpath)
        times_raw = ds["time"][:]
        t_start = Dates.DateTime(start_date)
        t_end = t_start + Dates.Day(n_days)
        tidx = findall(t -> !ismissing(t) && t_start <= t <= t_end, times_raw)
        isempty(tidx) && return nothing
        times = Dates.DateTime[times_raw[i] for i in tidx]

        heights_m = haskey(ds, height_key) ?
            _to_float_nan(ds[height_key][:]) : nothing

        profiles = Dict{String, Matrix{Float64}}()
        for vn in profile_vars
            haskey(ds, vn) || continue
            arr = ds[vn]
            ndims(arr) >= 2 || continue
            mat = _to_float_nan(arr[:, tidx])  # (height, time_sel) in Julia
            profiles[vn] = mat'  # → (n_time, n_height)
        end

        surface = Dict{String, Vector{Float64}}()
        for vn in surface_vars
            haskey(ds, vn) || continue
            arr = ds[vn]
            ndims(arr) >= 1 || continue
            surface[vn] = _to_float_nan(
                ndims(arr) == 1 ? arr[tidx] : arr[1, tidx],
            )
        end

        return (
            profiles = profiles, surface = surface,
            times = times, heights_m = heights_m,
        )
    catch e
        @warn "Could not load $file_prefix" exception = e
        return nothing
    finally
        !isnothing(ds) && close(ds)
    end
end

const _P0_HPA = 1000.0  # reference pressure for θ computation
const _RD_CP  = 287.04 / 1004.0  # R_d / c_p (dry air)

"""Compute dry potential temperature: θ = T × (p₀/p)^(R/cₚ)."""
function _compute_theta(T_K::AbstractArray, p_hPa::AbstractArray)
    return @. T_K * (_P0_HPA / max(p_hPa, 1.0))^_RD_CP
end

function _load_arm_beatm(beatm_dir, start_date, n_days)
    result = _load_arm_yearly_cdf(
        beatm_dir, "sgparmbeatmC1.c1", start_date, n_days,
        ["T_z", "rh_z", "u_z", "v_z", "Td_z", "p_z"],
        ["T_sfc", "rh_sfc", "p_sfc", "prec_sfc",
         "SH_baebbr", "LH_baebbr", "SH_qcecor", "LH_qcecor"];
        height_key = "z",
    )
    isnothing(result) && return nothing
    # Derive potential temperature from T_z and p_z if both present
    if haskey(result.profiles, "T_z") && haskey(result.profiles, "p_z")
        result.profiles["theta_z"] = _compute_theta(
            result.profiles["T_z"], result.profiles["p_z"],
        )
    end
    return result
end

function _load_arm_cldrad(cldrad_dir, start_date, n_days)
    _load_arm_yearly_cdf(
        cldrad_dir, "sgparmbecldradC1.c1", start_date, n_days,
        ["cld_frac", "cld_frac_MMCR", "cld_frac_MPL"],
        ["tot_cld", "tot_cld_tsi", "swdn", "swup", "swdif",
         "lwdn", "lwup", "pwv", "lwp",
         "cld_low", "cld_mid", "cld_high"];
        height_key = "height",
    )
end

# ────────────────────────────────────────────────────────────────────────────
# Interpolation & time matching
# ────────────────────────────────────────────────────────────────────────────

function _interp_profile(src_z, src_data, target_z)
    valid = .!isnan.(src_data)
    sum(valid) < 3 && return fill(NaN, length(target_z))
    zv, dv = src_z[valid], src_data[valid]
    idx = sortperm(zv)
    itp = Intp.interpolate((zv[idx],), dv[idx], Intp.Gridded(Intp.Linear()))
    etp = Intp.extrapolate(itp, NaN)
    return Float64[etp(z) for z in target_z]
end

function _interp_and_match(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start,
)
    n_z, n_t = length(target_z), length(model_t)
    model_out = Matrix{Float64}(undef, n_z, n_t)
    for t in 1:n_t
        model_out[:, t] = _interp_profile(model_z, model_zt[:, t], target_z)
    end
    model_dt = [sim_start + Dates.Millisecond(round(Int, s * 1000)) for s in model_t]
    obs_ms = Float64[Dates.value(t) for t in obs_times]
    obs_out = Matrix{Float64}(undef, n_z, n_t)
    for t in 1:n_t
        mms = Float64(Dates.value(model_dt[t]))
        _, idx = findmin(abs.(obs_ms .- mms))
        obs_out[:, t] = _interp_profile(obs_z, obs_th[idx, :], target_z)
    end
    return model_out, obs_out
end

"""Convert model seconds-since-start to hours-since-start for obs DateTimes."""
function _obs_hours(obs_times::Vector{Dates.DateTime}, sim_start::Dates.DateTime)
    return Float64[
        Dates.value(t - sim_start) / (3600.0 * 1000.0) for t in obs_times
    ]
end

function _match_timeseries(model_data, model_t, obs_data, obs_times, sim_start)
    model_dt = [sim_start + Dates.Millisecond(round(Int, s * 1000)) for s in model_t]
    obs_ms = Float64[Dates.value(t) for t in obs_times]
    obs_matched = Vector{Float64}(undef, length(model_t))
    for t in eachindex(model_t)
        mms = Float64(Dates.value(model_dt[t]))
        _, idx = findmin(abs.(obs_ms .- mms))
        obs_matched[t] = obs_data[idx]
    end
    return model_data, obs_matched
end

# ────────────────────────────────────────────────────────────────────────────
# Figures: dense-obs profile comparison (sonde — heatmap panels)
# ────────────────────────────────────────────────────────────────────────────

function _fig_time_height(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start, label, units;
    cmap = :viridis, divergent = false, obs_source = "Obs",
)
    mi, oi = _interp_and_match(model_zt, model_z, model_t,
        obs_th, obs_z, obs_times, target_z, sim_start)
    t_hrs = model_t ./ 3600.0
    z_km = target_z ./ 1000.0
    bias = mi .- oi

    all_v = filter(!isnan, vcat(vec(mi), vec(oi)))
    isempty(all_v) && return CairoMakie.Figure()
    vlo = _nanquantile(all_v, 0.02)
    vhi = _nanquantile(all_v, 0.98)
    if divergent
        va = max(abs(vlo), abs(vhi)); vlo, vhi = -va, va
    end
    bv = filter(!isnan, vec(bias))
    bmax = isempty(bv) ? 1.0 : _nanquantile(abs.(bv), 0.95)

    fig = CairoMakie.Figure(size = (1100, 1000))
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "Model: $label ($units)", ylabel = "Height (km)")
    hm1 = CairoMakie.heatmap!(ax1, t_hrs, z_km, mi'; colormap = cmap, colorrange = (vlo, vhi))
    CairoMakie.Colorbar(fig[1, 2], hm1; label = units)

    ax2 = CairoMakie.Axis(fig[2, 1]; title = "$obs_source: $label ($units)", ylabel = "Height (km)")
    hm2 = CairoMakie.heatmap!(ax2, t_hrs, z_km, oi'; colormap = cmap, colorrange = (vlo, vhi))
    CairoMakie.Colorbar(fig[2, 2], hm2; label = units)

    ax3 = CairoMakie.Axis(fig[3, 1]; title = "Bias (Model − $obs_source): $label",
        xlabel = "Time (hours)", ylabel = "Height (km)")
    hm3 = CairoMakie.heatmap!(ax3, t_hrs, z_km, bias';
        colormap = Makie.Reverse(:RdBu), colorrange = (-bmax, bmax))
    CairoMakie.Colorbar(fig[3, 2], hm3; label = "Δ $units")

    CairoMakie.linkxaxes!(ax1, ax2, ax3)
    return fig
end

function _fig_profiles(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start, label, units;
    obs_source = "Obs",
)
    mi, oi = _interp_and_match(model_zt, model_z, model_t,
        obs_th, obs_z, obs_times, target_z, sim_start)
    z_km = target_z ./ 1000.0
    nz = length(target_z)
    m_mean = [_nanmean(mi[i, :]) for i in 1:nz]
    m_std  = [_nanstd(mi[i, :])  for i in 1:nz]
    o_mean = [_nanmean(oi[i, :]) for i in 1:nz]
    o_std  = [_nanstd(oi[i, :])  for i in 1:nz]
    bias_profile = m_mean .- o_mean
    rmse_profile = [sqrt(_nanmean((mi[i, :] .- oi[i, :]) .^ 2)) for i in 1:nz]

    fig = CairoMakie.Figure(size = (1200, 500))
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "Mean Profiles (± 1σ)",
        xlabel = "$label ($units)", ylabel = "Height (km)")
    CairoMakie.band!(ax1, CairoMakie.Point2f.(m_mean .- m_std, z_km),
        CairoMakie.Point2f.(m_mean .+ m_std, z_km); color = (:blue, 0.2))
    CairoMakie.lines!(ax1, m_mean, z_km; color = :blue, linewidth = 2, label = "Model")
    CairoMakie.band!(ax1, CairoMakie.Point2f.(o_mean .- o_std, z_km),
        CairoMakie.Point2f.(o_mean .+ o_std, z_km); color = (:red, 0.2))
    CairoMakie.lines!(ax1, o_mean, z_km; color = :red, linewidth = 2, label = obs_source)
    CairoMakie.axislegend(ax1)

    ax2 = CairoMakie.Axis(fig[1, 2]; title = "Mean Bias (Model − $obs_source)",
        xlabel = "Bias ($units)", ylabel = "Height (km)")
    CairoMakie.lines!(ax2, bias_profile, z_km; color = :black, linewidth = 2)
    CairoMakie.vlines!(ax2, [0.0]; color = :gray, linestyle = :dash)

    ax3 = CairoMakie.Axis(fig[1, 3]; title = "RMSE",
        xlabel = "RMSE ($units)", ylabel = "Height (km)")
    CairoMakie.lines!(ax3, rmse_profile, z_km; color = :green, linewidth = 2)
    CairoMakie.linkyaxes!(ax1, ax2, ax3)
    return fig
end

# ────────────────────────────────────────────────────────────────────────────
# Figures: diurnal-cycle profile snapshots (Model + BEATM + Sonde)
# ────────────────────────────────────────────────────────────────────────────

"""
    _pick_snapshot_day(sim_start, n_days) → Date

Resolve `_PROFILE_SNAPSHOT_DAY` to a concrete Date.
"""
function _pick_snapshot_day(sim_start::Dates.DateTime, n_days::Int)
    d0 = Dates.Date(sim_start)
    if _PROFILE_SNAPSHOT_DAY === :last
        return d0 + Dates.Day(max(n_days - 1, 0))
    else
        return d0 + Dates.Day(Int(_PROFILE_SNAPSHOT_DAY))
    end
end

"""
Find BEATM observation indices on local `day` that contain valid data for
`obs_var`.  The window spans local midnight to local midnight, i.e.
UTC (00:00 − offset) to UTC (24:00 − offset).
"""
function _valid_beatm_indices(beatm, obs_var::String, day::Dates.Date)
    utc_start = Dates.DateTime(day) - Dates.Hour(_SITE_UTC_OFFSET_H)
    utc_end   = utc_start + Dates.Day(1)
    idxs = Int[]
    for i in eachindex(beatm.times)
        t = beatm.times[i]
        (utc_start <= t < utc_end) || continue
        prof = beatm.profiles[obs_var][i, :]
        any(x -> !isnan(x), prof) || continue
        push!(idxs, i)
    end
    return idxs
end

"""
Find the nearest sonde profile to a given DateTime. Returns the profile
vector (on sonde heights) or nothing if no sonde within 90 min.
"""
function _nearest_sonde_profile(
    sonde, obs_var::String, target_time::Dates.DateTime;
    max_gap_min = 90,
)
    isnothing(sonde) && return nothing
    haskey(sonde.data, obs_var) || return nothing
    sonde_ms = Float64[Dates.value(t) for t in sonde.times]
    target_ms = Float64(Dates.value(target_time))
    _, idx = findmin(abs.(sonde_ms .- target_ms))
    gap = abs(sonde_ms[idx] - target_ms) / 60_000.0  # minutes
    gap > max_gap_min && return nothing
    return sonde.data[obs_var][idx, :]
end

"""
Diurnal-cycle profile snapshots for one variable on one day.
Each panel shows up to three curves: Model (blue), BEATM (red), Sonde (green).
Panels are placed at every valid BEATM time on that day.
"""
function _fig_diurnal_profiles(
    model_zt, model_z, model_t,
    beatm, beatm_obs_var,
    sonde, sonde_obs_var,
    target_z, sim_start, day,
    label, units;
    beatm_ofn = identity, sonde_ofn = identity,
)
    z_km = target_z ./ 1000.0
    panel_idx = _valid_beatm_indices(beatm, beatm_obs_var, day)
    isempty(panel_idx) && return nothing

    model_dt = [sim_start + Dates.Millisecond(round(Int, s * 1000)) for s in model_t]
    model_ms = Float64[Dates.value(t) for t in model_dt]

    ncols = min(4, length(panel_idx))
    nrows = cld(length(panel_idx), ncols)
    fig = CairoMakie.Figure(size = (320 * ncols, 400 * nrows))

    for (p, bi) in enumerate(panel_idx)
        row = cld(p, ncols)
        col = mod1(p, ncols)
        ax = CairoMakie.Axis(fig[row, col];
            xlabel = "$label ($units)",
            ylabel = col == 1 ? "Height (km)" : "")

        # Model at nearest time
        bt_ms = Float64(Dates.value(beatm.times[bi]))
        _, midx = findmin(abs.(model_ms .- bt_ms))
        mp = _interp_profile(model_z, model_zt[:, midx], target_z)
        CairoMakie.lines!(ax, mp, z_km; color = :blue, linewidth = 2, label = "Model")

        # BEATM
        bp_raw = beatm_ofn(beatm.profiles[beatm_obs_var][bi, :])
        bp = _interp_profile(beatm.heights_m, bp_raw, target_z)
        CairoMakie.lines!(ax, bp, z_km; color = :red, linewidth = 2, label = "BEATM")

        # Sonde (if available near this time)
        sp_raw = _nearest_sonde_profile(sonde, sonde_obs_var, beatm.times[bi])
        if !isnothing(sp_raw)
            sp = _interp_profile(sonde.heights_m, sonde_ofn(sp_raw), target_z)
            CairoMakie.lines!(ax, sp, z_km;
                color = :green, linewidth = 1.5, linestyle = :dash, label = "Sonde")
        end

        local_t = beatm.times[bi] + Dates.Hour(_SITE_UTC_OFFSET_H)
        ax.title = Dates.format(local_t, "HH:MM") * " LT"
        if p == 1
            CairoMakie.axislegend(ax; position = :rt, labelsize = 9)
        end
    end
    day_str = Dates.format(day, "yyyy-mm-dd")
    CairoMakie.Label(fig[0, :],
        "$label — Diurnal Profiles ($day_str, local time)"; fontsize = 16)
    return fig
end

# ────────────────────────────────────────────────────────────────────────────
# Figures: CLDRAD cloud fraction profile (heatmap + model heatmap)
# ────────────────────────────────────────────────────────────────────────────

function _fig_cloud_profile(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start;
    max_height = _OBS_MAX_HEIGHT,
)
    mi, oi = _interp_and_match(model_zt, model_z, model_t,
        obs_th, obs_z, obs_times, target_z, sim_start)
    t_hrs = model_t ./ 3600.0
    z_km = target_z ./ 1000.0

    fig = CairoMakie.Figure(size = (1100, 700))
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "Model: Cloud Fraction (%)", ylabel = "Height (km)")
    hm1 = CairoMakie.heatmap!(ax1, t_hrs, z_km, (mi .* 100)';
        colormap = :Blues, colorrange = (0, 100))
    CairoMakie.Colorbar(fig[1, 2], hm1; label = "%")

    ax2 = CairoMakie.Axis(fig[2, 1]; title = "CLDRAD: Cloud Fraction (%)",
        xlabel = "Time (hours)", ylabel = "Height (km)")
    hm2 = CairoMakie.heatmap!(ax2, t_hrs, z_km, oi';
        colormap = :Blues, colorrange = (0, 100))
    CairoMakie.Colorbar(fig[2, 2], hm2; label = "%")
    CairoMakie.linkxaxes!(ax1, ax2)
    return fig
end

# ────────────────────────────────────────────────────────────────────────────
# Figures: multi-source surface time series overlay
# ────────────────────────────────────────────────────────────────────────────

const _OBS_COLORS = [:red, :orange, :green, :purple, :brown, :cyan]

"""
Multi-source surface time-series comparison.
`obs_entries`: vector of `(hours, data, label, is_sparse)` tuples.
Model is plotted as a continuous blue line; dense obs as lines, sparse obs as
scatter markers.
"""
function _fig_surface_multisource(
    model_data, model_t, obs_entries,
    label, units,
)
    t_hrs = model_t ./ 3600.0

    fig = CairoMakie.Figure(size = (1100, 500))
    ax = CairoMakie.Axis(fig[1, 1];
        title = "$label", xlabel = "Time (hours)", ylabel = "$label ($units)")
    CairoMakie.lines!(ax, t_hrs, model_data;
        color = :blue, linewidth = 2, label = "Model")

    for (i, entry) in enumerate(obs_entries)
        c = _OBS_COLORS[mod1(i, length(_OBS_COLORS))]
        if entry.is_sparse
            CairoMakie.scatter!(ax, entry.hours, entry.data;
                color = c, markersize = 5, label = entry.label)
        else
            CairoMakie.lines!(ax, entry.hours, entry.data;
                color = c, linewidth = 1.5, label = entry.label)
        end
    end
    CairoMakie.axislegend(ax; position = :rt)
    return fig
end

"""
Surface time-series overlay where there is no model variable — obs-only plot.
"""
function _fig_surface_obs_only(
    obs_entries, sim_start, label, units,
)
    fig = CairoMakie.Figure(size = (1100, 400))
    ax = CairoMakie.Axis(fig[1, 1];
        title = "$label (observations)", xlabel = "Time (hours)",
        ylabel = "$label ($units)")
    for (i, entry) in enumerate(obs_entries)
        c = _OBS_COLORS[mod1(i, length(_OBS_COLORS))]
        if entry.is_sparse
            CairoMakie.scatter!(ax, entry.hours, entry.data;
                color = c, markersize = 5, label = entry.label)
        else
            CairoMakie.lines!(ax, entry.hours, entry.data;
                color = c, linewidth = 1.5, label = entry.label)
        end
    end
    CairoMakie.axislegend(ax; position = :rt)
    return fig
end

# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

function _varanal_start_date(output_path)
    for f in readdir(output_path)
        endswith(f, ".yml") || continue
        try
            content = read(joinpath(output_path, f), String)
            m = match(r"start_date:\s*[\"']?(\d{8})[\"']?", content)
            !isnothing(m) && return Dates.DateTime(m[1], "yyyymmdd")
        catch
        end
    end
    return nothing
end

function _varanal_n_days(output_path; default = 7)
    for f in readdir(output_path)
        endswith(f, ".yml") || continue
        try
            content = read(joinpath(output_path, f), String)
            m = match(r"t_end:\s*[\"']?(\d+)days[\"']?", content)
            !isnothing(m) && return parse(Int, m[1])
        catch
        end
    end
    return default
end

function _write_obs_stats(output_path, entries)
    fpath = joinpath(output_path, "obs_comparison_statistics.txt")
    open(fpath, "w") do io
        println(io, "ARM VARANAL SCM vs Observations Comparison Statistics")
        println(io, repeat("=", 60))
        for (name, stats) in entries
            println(io, "\n$name:")
            println(io, repeat("-", 40))
            for (k, v) in stats
                println(io, "  $(rpad(k, 20)) $(round(v; sigdigits = 4))")
            end
        end
    end
    @info "Obs comparison statistics written to $fpath"
end

"""Detect whether an obs product is sparse (<= 10 samples/day)."""
_is_sparse(times, n_days) = length(times) / max(n_days, 1) <= 10

"""Look up obs data from loaded products. Returns (hours, data, is_sparse) or nothing."""
function _get_obs_surface(src::Symbol, var::String, ofn, data_products, sim_start, n_days)
    prod = get(data_products, src, nothing)
    isnothing(prod) && return nothing
    haskey(prod.surface, var) || return nothing
    raw = ofn(prod.surface[var])
    hrs = _obs_hours(prod.times, sim_start)
    is_sparse = _is_sparse(prod.times, n_days)
    return (hours = hrs, data = raw, is_sparse = is_sparse)
end

# ────────────────────────────────────────────────────────────────────────────
# Main observation comparison driver
# ────────────────────────────────────────────────────────────────────────────

function _make_arm_obs_comparison(output_path::AbstractString)
    sim_start = _varanal_start_date(output_path)
    if isnothing(sim_start)
        @warn "Could not determine start_date; skipping obs comparison"
        return
    end
    n_days = _varanal_n_days(output_path)

    simdir = SimDir(output_path)
    reduction = "inst"
    avail = ClimaAnalysis.available_periods(simdir; short_name = "ta", reduction)
    avail_vec = collect(avail)
    period = avail_vec[argmin(CA.time_to_seconds.(avail_vec))]

    target_z = collect(range(0.0, _OBS_MAX_HEIGHT; length = _OBS_N_VERT))
    obs_pages = String[]
    stats_entries = Tuple{String, Vector{Pair{String, Float64}}}[]

    # Load all products
    sonde  = _load_arm_sonde(_SONDE_DIR, Dates.Date(sim_start), n_days)
    beatm  = _load_arm_beatm(_BEATM_DIR, Dates.Date(sim_start), n_days)
    cldrad = _load_arm_cldrad(_CLDRAD_DIR, Dates.Date(sim_start), n_days)

    data_products = Dict{Symbol, Any}()
    !isnothing(beatm)  && (data_products[:beatm] = beatm)
    !isnothing(cldrad) && (data_products[:cldrad] = cldrad)

    # ── 1. Sonde profile comparisons (dense — heatmaps) ─────────────────

    if !isnothing(sonde)
        @info "Sonde data loaded" n_times = length(sonde.times)
        for m in _SONDE_MAP
            m.model in keys(simdir.vars) || continue
            haskey(sonde.data, m.obs) || continue
            @info "  Sonde: $(m.label)"

            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            data_zt, mz, mt = _extract_zt(var)
            mc = m.mfn(data_zt)
            oc = m.ofn(sonde.data[m.obs])

            fig = _fig_time_height(mc, mz, mt, oc, sonde.heights_m,
                sonde.times, target_z, sim_start, m.label, m.units;
                cmap = m.cmap, divergent = m.divergent, obs_source = "Sonde")
            fp = joinpath(output_path, "obs_sonde_$(m.model)_th.pdf")
            CairoMakie.save(fp, fig); push!(obs_pages, fp)

            fig2 = _fig_profiles(mc, mz, mt, oc, sonde.heights_m,
                sonde.times, target_z, sim_start, m.label, m.units;
                obs_source = "Sonde")
            fp2 = joinpath(output_path, "obs_sonde_$(m.model)_prof.pdf")
            CairoMakie.save(fp2, fig2); push!(obs_pages, fp2)

            mi, oi = _interp_and_match(mc, mz, mt, oc, sonde.heights_m,
                sonde.times, target_z, sim_start)
            valid = .!isnan.(mi) .& .!isnan.(oi)
            if sum(valid) > 10
                push!(stats_entries, ("Sonde $(m.label)", [
                    "Mean Bias" => _nanmean(mi[valid] .- oi[valid]),
                    "RMSE" => sqrt(_nanmean((mi[valid] .- oi[valid]) .^ 2)),
                    "Model Mean" => _nanmean(mi[valid]),
                    "Obs Mean" => _nanmean(oi[valid]),
                ]))
            end
        end
    end

    # ── 2. Diurnal profile snapshots (Model + BEATM + Sonde) ───────────

    snap_day = _pick_snapshot_day(sim_start, n_days)

    if !isnothing(beatm) && !isnothing(beatm.heights_m)
        @info "BEATM data loaded" n_times = length(beatm.times)

        # Mapping from BEATM profile var → matching sonde var + conversions
        _beatm_sonde_link = Dict(
            "T_z"     => (sonde_var = "temp",           beatm_ofn = identity,
                          sonde_ofn = x -> x .+ 273.15, m_label = "ta"),
            "rh_z"    => (sonde_var = "rh",             beatm_ofn = identity,
                          sonde_ofn = identity, m_label = "hur"),
            "u_z"     => (sonde_var = "u_wind",         beatm_ofn = identity,
                          sonde_ofn = identity, m_label = "ua"),
            "v_z"     => (sonde_var = "v_wind",         beatm_ofn = identity,
                          sonde_ofn = identity, m_label = "va"),
            "theta_z" => (sonde_var = "potential_temp",  beatm_ofn = identity,
                          sonde_ofn = identity, m_label = "thetaa"),
        )

        for m in _BEATM_PROFILE_MAP
            m.model in keys(simdir.vars) || continue
            haskey(beatm.profiles, m.obs) || continue
            @info "  Diurnal profiles: $(m.label) on $(snap_day)"

            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            data_zt, mz, mt = _extract_zt(var)
            mc = m.mfn(data_zt)

            link = get(_beatm_sonde_link, m.obs, nothing)
            sonde_var = isnothing(link) ? "" : link.sonde_var
            sonde_ofn_fn = isnothing(link) ? identity : link.sonde_ofn

            # Diurnal snapshot panels (Model + BEATM + Sonde)
            fig = _fig_diurnal_profiles(
                mc, mz, mt,
                beatm, m.obs,
                sonde, sonde_var,
                target_z, sim_start, snap_day,
                m.label, m.units;
                beatm_ofn = m.ofn, sonde_ofn = sonde_ofn_fn,
            )
            if !isnothing(fig)
                fp = joinpath(output_path, "obs_diurnal_$(m.model).pdf")
                CairoMakie.save(fp, fig); push!(obs_pages, fp)
            end

            # Mean profile comparison (all BEATM times, aggregated)
            oc = m.ofn(beatm.profiles[m.obs])
            fig2 = _fig_profiles(mc, mz, mt, oc, beatm.heights_m,
                beatm.times, target_z, sim_start, m.label, m.units;
                obs_source = "BEATM")
            fp2 = joinpath(output_path, "obs_beatm_$(m.model)_prof.pdf")
            CairoMakie.save(fp2, fig2); push!(obs_pages, fp2)

            mi, oi = _interp_and_match(mc, mz, mt, oc, beatm.heights_m,
                beatm.times, target_z, sim_start)
            valid = .!isnan.(mi) .& .!isnan.(oi)
            if sum(valid) > 10
                push!(stats_entries, ("BEATM $(m.label)", [
                    "Mean Bias" => _nanmean(mi[valid] .- oi[valid]),
                    "RMSE" => sqrt(_nanmean((mi[valid] .- oi[valid]) .^ 2)),
                    "Model Mean" => _nanmean(mi[valid]),
                    "Obs Mean" => _nanmean(oi[valid]),
                ]))
            end
        end
    end

    # ── 3. CLDRAD cloud fraction profile ─────────────────────────────────

    if !isnothing(cldrad) && !isnothing(cldrad.heights_m) &&
       haskey(cldrad.profiles, "cld_frac") && "cl" in keys(simdir.vars)
        @info "  CLDRAD cloud fraction profile"
        var = _slice_column(get(simdir; short_name = "cl", reduction, period))
        data_zt, mz, mt = _extract_zt(var)
        oc = cldrad.profiles["cld_frac"]

        fig = _fig_cloud_profile(data_zt, mz, mt, oc,
            cldrad.heights_m, cldrad.times, target_z, sim_start)
        fp = joinpath(output_path, "obs_cldrad_clfrac.pdf")
        CairoMakie.save(fp, fig); push!(obs_pages, fp)
    end

    # ── 4. Surface time series: multi-source overlays ────────────────────

    for m in _SURFACE_OVERLAY_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(entries, (hours = result.hours, data = result.data,
                label = o.label, is_sparse = result.is_sparse))
        end
        isempty(entries) && continue
        if m.model in keys(simdir.vars)
            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            md, mt = _extract_timeseries(var)
            fig = _fig_surface_multisource(md, mt, entries, m.label, m.units)
        else
            fig = _fig_surface_obs_only(entries, sim_start, m.label, m.units)
        end
        fp = joinpath(output_path, "obs_sfc_$(m.model).pdf")
        CairoMakie.save(fp, fig); push!(obs_pages, fp)
    end

    # ── 5. Radiation fields ──────────────────────────────────────────────

    for m in _RADIATION_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(entries, (hours = result.hours, data = result.data,
                label = o.label, is_sparse = result.is_sparse))
        end
        isempty(entries) && continue
        if m.model in keys(simdir.vars)
            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            md, mt = _extract_timeseries(var)
            fig = _fig_surface_multisource(md, mt, entries, m.label, m.units)
        else
            fig = _fig_surface_obs_only(entries, sim_start, m.label, m.units)
        end
        fp = joinpath(output_path, "obs_rad_$(m.model).pdf")
        CairoMakie.save(fp, fig); push!(obs_pages, fp)
    end

    # ── 6. Surface fluxes (obs-only, no direct model diagnostic) ─────────

    for m in _FLUX_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(entries, (hours = result.hours, data = result.data,
                label = o.label, is_sparse = result.is_sparse))
        end
        isempty(entries) && continue
        fig = _fig_surface_obs_only(entries, sim_start, m.label, m.units)
        safe_name = replace(lowercase(m.label), " " => "_")
        fp = joinpath(output_path, "obs_flux_$(safe_name).pdf")
        CairoMakie.save(fp, fig); push!(obs_pages, fp)
    end

    # ── 7. Cloud totals (multi-source overlay) ───────────────────────────

    for m in _CLOUD_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(entries, (hours = result.hours, data = result.data,
                label = o.label, is_sparse = result.is_sparse))
        end
        isempty(entries) && continue
        fig = _fig_surface_obs_only(entries, sim_start, m.label, m.units)
        safe_name = replace(lowercase(m.label), " " => "_")
        fp = joinpath(output_path, "obs_cloud_$(safe_name).pdf")
        CairoMakie.save(fp, fig); push!(obs_pages, fp)
    end

    # ── Assemble ─────────────────────────────────────────────────────────

    !isempty(stats_entries) && _write_obs_stats(output_path, stats_entries)
    existing = filter(isfile, obs_pages)
    if !isempty(existing)
        output_file = joinpath(output_path, "obs_comparison.pdf")
        run(`$(pdfunite()) $(Cmd(vcat(existing, [output_file])))`)
        Filesystem.rm.(existing, force = true)
        @info "Saved $output_file"
    end
end

# ────────────────────────────────────────────────────────────────────────────
# make_plots entry point
# ────────────────────────────────────────────────────────────────────────────

function make_plots(
    ::Val{:prognostic_edmfx_armvaranal_column},
    output_paths::Vector{<:AbstractString},
)
    # 1. Standard EDMF column plots → summary.pdf
    simdirs = SimDir.(output_paths)
    is_box = true

    short_names = [
        "wa", "waup", "ta", "taup", "hus", "husup", "arup", "tke", "ua",
        "thetaa", "thetaaup", "ha", "haup", "hur", "hurup", "lmix",
        "cl", "clw", "clwup", "cli", "cliup",
    ]
    reduction = "inst"

    available_periods = ClimaAnalysis.available_periods(
        simdirs[1]; short_name = short_names[1], reduction,
    )
    available_periods = collect(available_periods)
    period = available_periods[argmin(CA.time_to_seconds.(available_periods))]

    short_name_tuples = pair_edmf_names(short_names)
    var_groups_zt =
        map_comparison(simdirs, short_name_tuples) do simdir, name_tuple
            vars = map(
                short_name -> get(simdir; short_name, reduction, period),
                name_tuple,
            )
            is_box ? map(var -> slice(var, x = 0.0, y = 0.0), vars) : vars
        end

    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,)
        for group in var_groups_zt
    ]

    tmp_file = make_plots_generic(
        output_paths, output_name = "tmp", var_groups_z;
        plot_fn = plot_edmf_vert_profile!, MAX_NUM_COLS = 2, MAX_NUM_ROWS = 4,
    )
    make_plots_generic(
        output_paths, vcat((var_groups_zt...)...),
        plot_fn = plot_parsed_attribute_title!, summary_files = [tmp_file],
        MAX_NUM_COLS = 2, MAX_NUM_ROWS = 4,
    )

    # 2. ARM observation comparison → obs_comparison.pdf
    _make_arm_obs_comparison(first(output_paths))
end
