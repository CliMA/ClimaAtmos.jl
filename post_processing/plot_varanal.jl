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
import Statistics: mean, std, quantile

# ────────────────────────────────────────────────────────────────────────────
# Paths (`arm_sgp_varanal_obs` artifact, via AtmosArtifacts.jl)
# ────────────────────────────────────────────────────────────────────────────

function _arm_varanal_obs_dirs()
    return (
        sonde = CA.AA.arm_sgp_varanal_obs_dir("sonde"),
        beatm = CA.AA.arm_sgp_varanal_obs_dir("beatm"),
        cldrad = CA.AA.arm_sgp_varanal_obs_dir("cldrad"),
    )
end

# Vertical grid for obs interpolation.  Full domain covers the model top;
# obs data will be NaN above their instrument ceiling (~5 km for sonde/BEATM).
const _Z_FULL_M = 15_000.0   # full model domain top
const _Z_ZOOM_M = 4_000.0   # zoomed boundary-layer view
const _OBS_N_VERT = 200        # interpolation points on the full grid

# SGP Central Facility elevation (m above sea level).
# Sonde heights are ASL; model/BEATM heights are AGL.
const _SGP_ELEVATION_M = 315.0

# ────────────────────────────────────────────────────────────────────────────
# Variable mapping tables
# ────────────────────────────────────────────────────────────────────────────

# Profile variables with per-variable colormaps.
# Sonde has potential_temp directly, so we use that instead of raw temperature
# for the time-height comparison.
const _SONDE_MAP = [
    (model = "thetaa", obs = "potential_temp", mfn = identity, ofn = identity,
        label = "Potential Temperature", units = "K",
        cmap = Makie.Reverse(:RdYlBu), divergent = false),
    (model = "hur", obs = "rh", mfn = x -> x .* 100.0, ofn = identity,
        label = "Relative Humidity", units = "%",
        cmap = :YlGnBu, divergent = false),
    (model = "hus", obs = "sh", mfn = x -> x .* 1000.0, ofn = x -> x .* 1000.0,
        label = "Specific Humidity", units = "g/kg",
        cmap = :YlGnBu, divergent = false),
    (model = "va", obs = "v_wind", mfn = identity, ofn = identity,
        label = "V Wind", units = "m/s",
        cmap = Makie.Reverse(:RdBu), divergent = true),
]

# U and V shown as profile-only (mean + RMSE) against sonde
const _SONDE_WIND_PROFILES = [
    (model = "ua", obs = "u_wind", mfn = identity, ofn = identity,
        label = "U Wind", units = "m/s"),
    (model = "va", obs = "v_wind", mfn = identity, ofn = identity,
        label = "V Wind", units = "m/s"),
]

# Surface time-series: multi-source overlays grouped by physical quantity
# Each entry can have obs from multiple products
const _SURFACE_OVERLAY_MAP = [
    (model = "ts", label = "Surface Temperature", units = "K",
        mfn = identity,
        obs = [
            (src = :beatm, var = "T_sfc", ofn = identity, label = "BEATM"),
        ]),
    (model = "pr", label = "Precipitation Rate", units = "mm/hr",
        mfn = x -> -x .* 3600,  # kg/m²/s -> mm/hr, flip sign
        obs = [
            (src = :beatm, var = "prec_sfc", ofn = identity, label = "BEATM"),
        ]),
    (model = "lwp", label = "Liquid Water Path", units = "g/m²",
        mfn = x -> x .* 1000,  # kg/m² -> g/m²
        obs = [
            (src = :cldrad, var = "lwp", ofn = identity, label = "CLDRAD"),
        ]),
]

# Radiation: model diagnostics vs CLDRAD obs (surface + TOA)
const _RADIATION_MAP = [
    (model = "rsds", label = "SW Down (surface)", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "swdn", ofn = identity, label = "CLDRAD"),
        ]),
    (model = "rsus", label = "SW Up (surface)", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "swup", ofn = identity, label = "CLDRAD"),
        ]),
    (model = "rlds", label = "LW Down (surface)", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "lwdn", ofn = identity, label = "CLDRAD"),
        ]),
    (model = "rlus", label = "LW Up (surface)", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "lwup", ofn = identity, label = "CLDRAD"),
        ]),
    (model = "rlut", label = "LW Out (TOA)", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "lw_net_TOA", ofn = identity, label = "CLDRAD"),
        ]),
    (model = "rsdt", label = "SW Down (TOA)", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "sw_dn_TOA", ofn = identity, label = "CLDRAD"),
        ]),
]

# Surface fluxes: BEATM obs + model diagnostics
const _FLUX_MAP = [
    (model = "hfss", label = "Sensible Heat Flux", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :beatm, var = "SH_baebbr", ofn = identity, label = "BEATM BAEBBR"),
            (src = :beatm, var = "SH_qcecor", ofn = identity, label = "BEATM QCECOR"),
        ]),
    (model = "hfls", label = "Latent Heat Flux", units = "W/m²",
        mfn = identity,
        obs = [
            (src = :beatm, var = "LH_baebbr", ofn = identity, label = "BEATM BAEBBR"),
            (src = :beatm, var = "LH_qcecor", ofn = identity, label = "BEATM QCECOR"),
        ]),
]

# Cloud fraction: CLDRAD obs + model total cloud
const _CLOUD_MAP = [
    (model = "clt", label = "Total Cloud Fraction", units = "%",
        mfn = identity,
        obs = [
            (src = :cldrad, var = "tot_cld", ofn = x -> x .* 100, label = "CLDRAD ARSCL"),
            (src = :cldrad, var = "tot_cld_tsi", ofn = x -> x .* 100, label = "CLDRAD TSI"),
        ]),
]

# ────────────────────────────────────────────────────────────────────────────
# NaN-safe helpers
# ────────────────────────────────────────────────────────────────────────────

_nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
_nanstd(x) = (v = filter(!isnan, x); length(v) < 2 ? NaN : std(v))
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
    obs_vars = ["temp", "rh", "sh", "u_wind", "v_wind", "potential_temp"]
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
    # Find the CDF file: subset artifacts use e.g. "*.201009.cdf",
    # full artifacts use "*.20100101.000000.cdf". Glob for any match.
    candidates = filter(
        f -> startswith(f, file_prefix) && endswith(f, ".cdf"),
        readdir(data_dir),
    )
    isempty(candidates) && return nothing
    fpath = joinpath(data_dir, first(candidates))

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


function _load_arm_beatm(beatm_dir, start_date, n_days)
    _load_arm_yearly_cdf(
        beatm_dir, "sgparmbeatmC1.c1", start_date, n_days,
        String[],
        ["T_sfc", "prec_sfc",
            "SH_baebbr", "LH_baebbr", "SH_qcecor", "LH_qcecor"];
        height_key = "z",
    )
end

function _load_arm_cldrad(cldrad_dir, start_date, n_days)
    _load_arm_yearly_cdf(
        cldrad_dir, "sgparmbecldradC1.c1", start_date, n_days,
        ["cld_frac"],
        ["tot_cld", "tot_cld_tsi", "swdn", "swup",
            "lwdn", "lwup", "lwp",
            "lw_net_TOA", "sw_net_TOA", "sw_dn_TOA"];
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

    # Skip model interpolation when target_z already matches model_z exactly.
    on_model_grid =
        (length(model_z) == n_z) &&
        maximum(abs.(sort(model_z) .- sort(target_z))) < 1.0   # 1 m tolerance
    model_out = if on_model_grid
        Float64.(model_zt)   # already on the right grid; no-op
    else
        out = Matrix{Float64}(undef, n_z, n_t)
        for t in 1:n_t
            out[:, t] = _interp_profile(model_z, model_zt[:, t], target_z)
        end
        out
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

"""
Convert model seconds-since-start to hours-since-start for obs DateTimes.
"""
function _obs_hours(obs_times::Vector{Dates.DateTime}, sim_start::Dates.DateTime)
    return Float64[
        Dates.value(t - sim_start) / (3600.0 * 1000.0) for t in obs_times
    ]
end

# ────────────────────────────────────────────────────────────────────────────
# Figures: dense-obs profile comparison (sonde — heatmap panels)
# ────────────────────────────────────────────────────────────────────────────

function _fig_time_height(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start, label, units;
    cmap = :viridis, divergent = false, obs_source = "Obs",
    z_top_km = nothing,
)
    mi, oi = _interp_and_match(model_zt, model_z, model_t,
        obs_th, obs_z, obs_times, target_z, sim_start)
    t_hrs = model_t ./ 3600.0
    z_km = target_z ./ 1000.0
    bias = mi .- oi

    z_mask = isnothing(z_top_km) ? trues(length(z_km)) : z_km .<= z_top_km
    all_v = filter(!isnan, vcat(vec(mi[z_mask, :]), vec(oi[z_mask, :])))
    isempty(all_v) && return CairoMakie.Figure()
    vlo = _nanquantile(all_v, 0.02)
    vhi = _nanquantile(all_v, 0.98)
    if divergent
        va = max(abs(vlo), abs(vhi))
        vlo, vhi = -va, va
    end
    bv = filter(!isnan, vec(bias[z_mask, :]))
    bmax = isempty(bv) ? 1.0 : _nanquantile(abs.(bv), 0.95)

    ylims = isnothing(z_top_km) ? (0.0, maximum(z_km)) : (0.0, z_top_km)
    zoom_tag = isnothing(z_top_km) ? "" : " (zoom)"

    fig = CairoMakie.Figure(size = (1100, 1000))
    ax1 = CairoMakie.Axis(
        fig[1, 1];
        title = "Model: $label ($units)$zoom_tag",
        ylabel = "Height (km)",
    )
    hm1 =
        CairoMakie.heatmap!(ax1, t_hrs, z_km, mi'; colormap = cmap, colorrange = (vlo, vhi))
    CairoMakie.Colorbar(fig[1, 2], hm1; label = units)
    CairoMakie.ylims!(ax1, ylims)

    ax2 = CairoMakie.Axis(
        fig[2, 1];
        title = "$obs_source: $label ($units)$zoom_tag",
        ylabel = "Height (km)",
    )
    hm2 =
        CairoMakie.heatmap!(ax2, t_hrs, z_km, oi'; colormap = cmap, colorrange = (vlo, vhi))
    CairoMakie.Colorbar(fig[2, 2], hm2; label = units)
    CairoMakie.ylims!(ax2, ylims)

    ax3 = CairoMakie.Axis(fig[3, 1]; title = "Bias (Model − $obs_source): $label$zoom_tag",
        xlabel = "Time (hours)", ylabel = "Height (km)")
    hm3 = CairoMakie.heatmap!(ax3, t_hrs, z_km, bias';
        colormap = Makie.Reverse(:RdBu), colorrange = (-bmax, bmax))
    CairoMakie.Colorbar(fig[3, 2], hm3; label = "Δ $units")
    CairoMakie.ylims!(ax3, ylims)

    CairoMakie.linkxaxes!(ax1, ax2, ax3)
    return fig
end

function _compute_profile_stats(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start;
    z_top_km = nothing,
)
    mi, oi = _interp_and_match(model_zt, model_z, model_t,
        obs_th, obs_z, obs_times, target_z, sim_start)
    z_km = target_z ./ 1000.0
    nz = length(target_z)
    m_mean = [_nanmean(mi[i, :]) for i in 1:nz]
    m_std = [_nanstd(mi[i, :]) for i in 1:nz]
    o_mean = [_nanmean(oi[i, :]) for i in 1:nz]
    o_std = [_nanstd(oi[i, :]) for i in 1:nz]
    rmse_profile = [sqrt(_nanmean((mi[i, :] .- oi[i, :]) .^ 2)) for i in 1:nz]
    ylims = isnothing(z_top_km) ? (0.0, maximum(z_km)) : (0.0, z_top_km)
    return (; z_km, m_mean, m_std, o_mean, o_std, rmse_profile, ylims, mi, oi)
end

function _plot_profile_mean_panel!(ax, stats, label, units, obs_source)
    (; z_km, m_mean, m_std, o_mean, o_std, ylims) = stats
    CairoMakie.band!(ax, CairoMakie.Point2f.(m_mean .- m_std, z_km),
        CairoMakie.Point2f.(m_mean .+ m_std, z_km); color = (:blue, 0.2))
    CairoMakie.lines!(ax, m_mean, z_km; color = :blue, linewidth = 2, label = "Model")
    CairoMakie.band!(ax, CairoMakie.Point2f.(o_mean .- o_std, z_km),
        CairoMakie.Point2f.(o_mean .+ o_std, z_km); color = (:red, 0.2))
    CairoMakie.lines!(ax, o_mean, z_km; color = :red, linewidth = 2, label = obs_source)
    CairoMakie.ylims!(ax, ylims)
    ax.xlabel = "$label ($units)"
    ax.ylabel = "Height (km)"
end

function _plot_profile_rmse_panel!(ax, stats, units)
    (; z_km, rmse_profile, ylims) = stats
    CairoMakie.lines!(ax, rmse_profile, z_km; color = :green, linewidth = 2)
    CairoMakie.ylims!(ax, ylims)
    ax.xlabel = "RMSE ($units)"
    ax.ylabel = "Height (km)"
end

"""
Consolidated mean-profile + RMSE figure. Each entry is a NamedTuple with
`label`, `units`, `obs_source`, and the arguments for `_compute_profile_stats`.
"""
function _fig_profiles_consolidated(
    entries;
    title = "Mean Profiles",
    ncol = 2,
    z_top_km = nothing,
)
    isempty(entries) && return CairoMakie.Figure()
    zoom_tag = isnothing(z_top_km) ? "" : " (zoom)"
    nrow = ceil(Int, length(entries) / ncol)
    fig = CairoMakie.Figure(size = (520 * ncol, 320 * nrow))
    !isempty(title) && CairoMakie.Label(fig[0, :], "$title$zoom_tag"; fontsize = 16)

    y_axes = CairoMakie.Axis[]
    for (i, entry) in enumerate(entries)
        row = div(i - 1, ncol) + 1
        col = mod(i - 1, ncol) + 1
        sub = fig[row, col] = CairoMakie.GridLayout()
        stats = _compute_profile_stats(
            entry.model_zt, entry.model_z, entry.model_t,
            entry.obs_th, entry.obs_z, entry.obs_times,
            entry.target_z, entry.sim_start;
            z_top_km,
        )
        ax_mean = CairoMakie.Axis(sub[1, 1]; title = entry.label)
        ax_rmse = CairoMakie.Axis(sub[1, 2]; title = "RMSE")
        _plot_profile_mean_panel!(
            ax_mean,
            stats,
            entry.label,
            entry.units,
            entry.obs_source,
        )
        _plot_profile_rmse_panel!(ax_rmse, stats, entry.units)
        if i == 1
            CairoMakie.axislegend(ax_mean; position = :rt, labelsize = 9)
        end
        push!(y_axes, ax_mean, ax_rmse)
    end
    !isempty(y_axes) && CairoMakie.linkyaxes!(y_axes...)
    return fig
end

# ────────────────────────────────────────────────────────────────────────────
# Figures: CLDRAD cloud fraction profile (heatmap + model heatmap)
# ────────────────────────────────────────────────────────────────────────────

function _fig_cloud_profile(
    model_zt, model_z, model_t,
    obs_th, obs_z, obs_times,
    target_z, sim_start;
    z_top_km = nothing,
)
    mi, oi = _interp_and_match(model_zt, model_z, model_t,
        obs_th, obs_z, obs_times, target_z, sim_start)
    t_hrs = model_t ./ 3600.0
    z_km = target_z ./ 1000.0
    ylims = isnothing(z_top_km) ? (0.0, maximum(z_km)) : (0.0, z_top_km)
    zoom_tag = isnothing(z_top_km) ? "" : " (zoom)"

    fig = CairoMakie.Figure(size = (1100, 700))
    ax1 = CairoMakie.Axis(
        fig[1, 1];
        title = "Model: Cloud Fraction (%)$zoom_tag",
        ylabel = "Height (km)",
    )
    hm1 = CairoMakie.heatmap!(ax1, t_hrs, z_km, (mi .* 100)';
        colormap = :Blues, colorrange = (0, 100))
    CairoMakie.Colorbar(fig[1, 2], hm1; label = "%")
    CairoMakie.ylims!(ax1, ylims)

    ax2 = CairoMakie.Axis(fig[2, 1]; title = "CLDRAD: Cloud Fraction (%)$zoom_tag",
        xlabel = "Time (hours)", ylabel = "Height (km)")
    hm2 = CairoMakie.heatmap!(ax2, t_hrs, z_km, oi';
        colormap = :Blues, colorrange = (0, 100))
    CairoMakie.Colorbar(fig[2, 2], hm2; label = "%")
    CairoMakie.ylims!(ax2, ylims)
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
function _fig_surface_obs_only(obs_entries, label, units)
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

"""
Detect whether an obs product is sparse (<= 10 samples/day).
"""
_is_sparse(times, n_days) = length(times) / max(n_days, 1) <= 10

"""
Look up obs data from loaded products. Returns (hours, data, is_sparse) or nothing.
"""
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

    # Use the model's own vertical levels as the target grid so model data
    # passes through without interpolation.  Fall back to a uniform grid if
    # no 3D variable is available yet.
    function _model_z_levels(simdir, reduction, period)
        for sn in ["ta", "thetaa", "hur", "hus", "ua"]
            sn in keys(simdir.vars) || continue
            try
                v = _slice_column(get(simdir; short_name = sn, reduction, period))
                zk = z_dim_name(v)
                return sort(Float64.(collect(v.dims[zk])))
            catch
            end
        end
        return collect(range(0.0, _Z_FULL_M; length = _OBS_N_VERT))
    end
    target_z = _model_z_levels(simdir, reduction, period)

    # (z_top_km=nothing → full domain, z_top_km=value → zoomed)
    _zoom_levels = [
        (sfx = "full", z_top_km = nothing),
        (sfx = "zoom", z_top_km = _Z_ZOOM_M / 1000.0),
    ]

    obs_pages = String[]
    stats_entries = Tuple{String, Vector{Pair{String, Float64}}}[]

    obs_dirs = try
        _arm_varanal_obs_dirs()
    catch e
        @warn "ARM VARANAL obs artifact unavailable; skipping obs comparison" exception = e
        return
    end
    (; sonde, beatm, cldrad) = obs_dirs

    # Load all products
    sonde = _load_arm_sonde(sonde, Dates.Date(sim_start), n_days)
    beatm = _load_arm_beatm(beatm, Dates.Date(sim_start), n_days)
    cldrad = _load_arm_cldrad(cldrad, Dates.Date(sim_start), n_days)

    data_products = Dict{Symbol, Any}()
    !isnothing(beatm) && (data_products[:beatm] = beatm)
    !isnothing(cldrad) && (data_products[:cldrad] = cldrad)

    # ── 1. Sonde profile comparisons (dense — heatmaps) ─────────────────

    if !isnothing(sonde)
        @info "Sonde data loaded" n_times = length(sonde.times)
        sonde_profile_entries = typeof((
            label = "",
            units = "",
            obs_source = "",
            model_zt = zeros(0, 0),
            model_z = zeros(0),
            model_t = zeros(0),
            obs_th = zeros(0, 0),
            obs_z = zeros(0),
            obs_times = Dates.DateTime[],
            target_z = zeros(0),
            sim_start = sim_start,
        ))[]

        for m in _SONDE_MAP
            m.model in keys(simdir.vars) || continue
            haskey(sonde.data, m.obs) || continue
            @info "  Sonde: $(m.label)"

            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            data_zt, mz, mt = _extract_zt(var)
            mc = m.mfn(data_zt)
            oc = m.ofn(sonde.data[m.obs])

            for zl in _zoom_levels
                fig = _fig_time_height(mc, mz, mt, oc, sonde.heights_m,
                    sonde.times, target_z, sim_start, m.label, m.units;
                    cmap = m.cmap, divergent = m.divergent, obs_source = "Sonde",
                    z_top_km = zl.z_top_km)
                fp = joinpath(output_path, "obs_sonde_$(m.model)_th_$(zl.sfx).pdf")
                CairoMakie.save(fp, fig)
                push!(obs_pages, fp)
            end

            push!(
                sonde_profile_entries,
                (;
                    label = m.label,
                    units = m.units,
                    obs_source = "Sonde",
                    model_zt = mc,
                    model_z = mz,
                    model_t = mt,
                    obs_th = oc,
                    obs_z = sonde.heights_m,
                    obs_times = sonde.times,
                    target_z,
                    sim_start,
                ),
            )

            stats = _compute_profile_stats(
                mc, mz, mt, oc, sonde.heights_m, sonde.times, target_z, sim_start,
            )
            valid = .!isnan.(stats.mi) .& .!isnan.(stats.oi)
            if sum(valid) > 10
                push!(
                    stats_entries,
                    (
                        "Sonde $(m.label)",
                        [
                            "RMSE" =>
                                sqrt(_nanmean((stats.mi[valid] .- stats.oi[valid]) .^ 2)),
                            "Model Mean" => _nanmean(stats.mi[valid]),
                            "Obs Mean" => _nanmean(stats.oi[valid]),
                        ],
                    ),
                )
            end
        end

        # U and V wind: profile-only comparisons (mean + RMSE)
        for wp in _SONDE_WIND_PROFILES
            wp.model in keys(simdir.vars) || continue
            haskey(sonde.data, wp.obs) || continue
            @info "  Sonde profiles: $(wp.label)"

            var = _slice_column(get(simdir; short_name = wp.model, reduction, period))
            data_zt, mz, mt = _extract_zt(var)
            mc = wp.mfn(data_zt)
            oc = wp.ofn(sonde.data[wp.obs])

            any(e -> e.label == wp.label, sonde_profile_entries) && continue
            push!(
                sonde_profile_entries,
                (;
                    label = wp.label,
                    units = wp.units,
                    obs_source = "Sonde",
                    model_zt = mc,
                    model_z = mz,
                    model_t = mt,
                    obs_th = oc,
                    obs_z = sonde.heights_m,
                    obs_times = sonde.times,
                    target_z,
                    sim_start,
                ),
            )
        end

        for zl in _zoom_levels
            isempty(sonde_profile_entries) && break
            fig = _fig_profiles_consolidated(
                sonde_profile_entries;
                title = "Model vs Sonde Mean Profiles",
                ncol = 2,
                z_top_km = zl.z_top_km,
            )
            fp = joinpath(output_path, "obs_sonde_profiles_$(zl.sfx).pdf")
            CairoMakie.save(fp, fig)
            push!(obs_pages, fp)
        end
    end

    # ── 2. CLDRAD cloud fraction profile ─────────────────────────────────

    if !isnothing(cldrad) && !isnothing(cldrad.heights_m) &&
       haskey(cldrad.profiles, "cld_frac") && "cl" in keys(simdir.vars)
        @info "  CLDRAD cloud fraction profile"
        var = _slice_column(get(simdir; short_name = "cl", reduction, period))
        data_zt, mz, mt = _extract_zt(var)
        oc = cldrad.profiles["cld_frac"]

        for zl in _zoom_levels
            fig = _fig_cloud_profile(data_zt, mz, mt, oc,
                cldrad.heights_m, cldrad.times, target_z, sim_start;
                z_top_km = zl.z_top_km)
            fp = joinpath(output_path, "obs_cldrad_clfrac_$(zl.sfx).pdf")
            CairoMakie.save(fp, fig)
            push!(obs_pages, fp)
        end
    end

    # ── 3. Surface time series: multi-source overlays ────────────────────

    for m in _SURFACE_OVERLAY_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(
                entries,
                (hours = result.hours, data = result.data,
                    label = o.label, is_sparse = result.is_sparse),
            )
        end
        isempty(entries) && continue
        if m.model in keys(simdir.vars)
            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            md, mt = _extract_timeseries(var)
            md = m.mfn(md)
            fig = _fig_surface_multisource(md, mt, entries, m.label, m.units)
        else
            fig = _fig_surface_obs_only(entries, m.label, m.units)
        end
        fp = joinpath(output_path, "obs_sfc_$(m.model).pdf")
        CairoMakie.save(fp, fig)
        push!(obs_pages, fp)
    end

    # ── 4. Radiation fields ──────────────────────────────────────────────

    for m in _RADIATION_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(
                entries,
                (hours = result.hours, data = result.data,
                    label = o.label, is_sparse = result.is_sparse),
            )
        end
        isempty(entries) && continue
        if m.model in keys(simdir.vars)
            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            md, mt = _extract_timeseries(var)
            md = m.mfn(md)
            fig = _fig_surface_multisource(md, mt, entries, m.label, m.units)
        else
            fig = _fig_surface_obs_only(entries, m.label, m.units)
        end
        fp = joinpath(output_path, "obs_rad_$(m.model).pdf")
        CairoMakie.save(fp, fig)
        push!(obs_pages, fp)
    end

    # ── 5. Surface fluxes (model + obs overlay) ──────────────────────────

    for m in _FLUX_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(
                entries,
                (hours = result.hours, data = result.data,
                    label = o.label, is_sparse = result.is_sparse),
            )
        end
        isempty(entries) && continue
        if m.model in keys(simdir.vars)
            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            md, mt = _extract_timeseries(var)
            md = m.mfn(md)
            fig = _fig_surface_multisource(md, mt, entries, m.label, m.units)
        else
            fig = _fig_surface_obs_only(entries, m.label, m.units)
        end
        safe_name = replace(lowercase(m.label), " " => "_")
        fp = joinpath(output_path, "obs_flux_$(safe_name).pdf")
        CairoMakie.save(fp, fig)
        push!(obs_pages, fp)
    end

    # ── 6. Cloud totals (model + obs overlay) ─────────────────────────────

    for m in _CLOUD_MAP
        entries = NamedTuple{(:hours, :data, :label, :is_sparse),
            Tuple{Vector{Float64}, Vector{Float64}, String, Bool}}[]
        for o in m.obs
            result = _get_obs_surface(o.src, o.var, o.ofn,
                data_products, sim_start, n_days)
            isnothing(result) && continue
            push!(
                entries,
                (hours = result.hours, data = result.data,
                    label = o.label, is_sparse = result.is_sparse),
            )
        end
        isempty(entries) && continue
        if m.model in keys(simdir.vars)
            var = _slice_column(get(simdir; short_name = m.model, reduction, period))
            md, mt = _extract_timeseries(var)
            md = m.mfn(md)
            fig = _fig_surface_multisource(md, mt, entries, m.label, m.units)
        else
            fig = _fig_surface_obs_only(entries, m.label, m.units)
        end
        safe_name = replace(lowercase(m.label), " " => "_")
        fp = joinpath(output_path, "obs_cloud_$(safe_name).pdf")
        CairoMakie.save(fp, fig)
        push!(obs_pages, fp)
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
            map(_slice_column, vars)
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
