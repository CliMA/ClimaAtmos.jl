# beres_plot_helpers.jl — shared utilities for Beres NOGW plotter scripts
#
# Usage: include("beres_plot_helpers.jl") at the top of each plotter script.

import ClimaAnalysis
import ClimaAnalysis: slice
using Statistics

const DAY_S = 86400.0

# --- Mode ---
@enum PlotMode MODE_AVG MODE_INST

"""
    parse_mode(args) -> (mode, remaining_args)

Strip `--inst` flag from CLI args. Returns `(MODE_INST, filtered_args)` or
`(MODE_AVG, args)`.
"""
function parse_mode(args)
    if "--inst" in args
        return MODE_INST, filter(a -> a != "--inst", args)
    else
        return MODE_AVG, collect(args)
    end
end

# --- Variable loading ---

"""
    _load_var(simdir, short_name; reduction_order, period_order, prefer_period)

Core variable loader shared by `load_var` (avg) and `load_var_inst` (inst).

  - `reduction_order`: reductions to try in order; the first one the simdir offers
    is used, falling back to `first(available_reductions)`.
  - `period_order`: periods to try in order when `prefer_period` is unset/unavailable.
  - `prefer_period`: if given and available, overrides `period_order` so callers can
    force every field onto the same temporal grid (avoids mismatch between e.g.
    hourly Q₀ and daily arup profiles).
"""
function _load_var(
    simdir,
    short_name;
    reduction_order,
    period_order,
    prefer_period = nothing,
)
    reds = ClimaAnalysis.available_reductions(simdir; short_name)
    red = first(reds)
    for r in reduction_order
        if r in reds
            red = r
            break
        end
    end
    periods = ClimaAnalysis.available_periods(simdir; short_name, reduction = red)
    period = nothing
    if !isnothing(prefer_period) && prefer_period in periods
        period = prefer_period
    else
        for p in period_order
            if p in periods
                period = p
                break
            end
        end
    end
    var =
        isnothing(period) ? get(simdir; short_name, reduction = red) :
        get(simdir; short_name, reduction = red, period)
    ntimes = haskey(var.dims, "time") ? length(var.dims["time"]) : 0
    trange =
        ntimes > 0 ? "($(var.dims["time"][1])s – $(var.dims["time"][end])s)" :
        "(no time dim)"
    println(
        "  Loaded $short_name: reduction=$red, period=$period, $ntimes snapshots $trange",
    )
    return var
end

"""
Load a variable preferring the "average" reduction (daily period first).
See `_load_var` for `prefer_period`.
"""
load_var(simdir, short_name; prefer_period = nothing) = _load_var(
    simdir,
    short_name;
    reduction_order = ("average",),
    period_order = ("1d", "12h", "1h", "10s", "10d", "30d", "1M"),
    prefer_period,
)

"""
Load a variable preferring the "inst" reduction (hourly period first), falling
back to "average". See `_load_var` for `prefer_period`.
"""
load_var_inst(simdir, short_name; prefer_period = nothing) = _load_var(
    simdir,
    short_name;
    reduction_order = ("inst", "average"),
    period_order = ("1h", "10s", "1d", "12h", "10d", "30d", "1M"),
    prefer_period,
)

"""
Mode-aware variable loading: avg mode uses `load_var`, inst mode uses
`load_var_inst`. `prefer_period` is honored in both modes.
"""
function load_var_for_mode(simdir, short_name, mode::PlotMode; prefer_period = nothing)
    return mode == MODE_AVG ? load_var(simdir, short_name; prefer_period) :
           load_var_inst(simdir, short_name; prefer_period)
end

# --- Snapshot selectors ---

"""
Average over the last `avg_window_days` ending at `t_end_days`.
"""
function last_snapshot(var; t_end_days = Inf, avg_window_days = 10.0)
    haskey(var.dims, "time") || return var
    times = var.dims["time"]
    t_end = t_end_days == Inf ? times[end] : t_end_days * DAY_S
    t_start = t_end - avg_window_days * DAY_S
    idx = findall(t -> t >= t_start && t <= t_end, times)
    if length(idx) <= 1
        return slice(var; time = t_end)
    end
    slices = [slice(var; time = times[i]) for i in idx]
    avg = deepcopy(slices[1])
    avg.data .= mean([s.data for s in slices])
    return avg
end

"""
Single snapshot at (or nearest to) `t_end_days`.
"""
function single_snapshot(var; t_end_days = Inf)
    haskey(var.dims, "time") || return var
    times = var.dims["time"]
    t_end = t_end_days == Inf ? times[end] : t_end_days * DAY_S
    return slice(var; time = t_end)
end

"""
Snap to the nearest available time to `target_time` (seconds).
"""
function snapshot_at_time(var, target_time)
    haskey(var.dims, "time") || return var
    times = var.dims["time"]
    _, idx = findmin(abs.(times .- target_time))
    return slice(var; time = times[idx])
end

"""
    get_snapshot(var, mode; t_end_days, avg_window_days, inst_time)

Mode-aware snapshot:

  - MODE_AVG: windowed average via `last_snapshot`
  - MODE_INST: single snapshot at `inst_time` (or last time if nothing)
"""
function get_snapshot(
    var,
    mode::PlotMode;
    t_end_days = Inf,
    avg_window_days = 10.0,
    inst_time = nothing,
)
    if mode == MODE_AVG
        return last_snapshot(var; t_end_days, avg_window_days)
    else
        if isnothing(inst_time)
            return single_snapshot(var; t_end_days)
        else
            return snapshot_at_time(var, inst_time)
        end
    end
end

"""
Mode-aware vertical profile at (lon, lat).
"""
function profile_at(var, lon, lat, mode::PlotMode; kwargs...)
    s = get_snapshot(var, mode; kwargs...)
    col = slice(s; lon = lon, lat = lat)
    return col.dims["z"], col.data
end

# --- Peak active timestep finder ---

"""
    find_peak_active_time(simdir; Q0_threshold, lat_max, candidate_times)

Scan inst `nogw_Q0` to find the timestep with the most active tropical columns.
Returns `(best_time_seconds, best_count)`.

If `candidate_times` is provided, only search those times (snapped to nearest
available Q₀ time). Use this to restrict the search to times where profile
diagnostics (arup, waup, etc.) are also available, avoiding temporal mismatch
between hourly 2D fields and daily 3D profiles.
"""
function find_peak_active_time(
    simdir;
    Q0_threshold = 1.0e-5,
    lat_max = 30.0,
    candidate_times = nothing,
)
    Q0_inst = load_var_inst(simdir, "nogw_Q0")
    haskey(Q0_inst.dims, "time") || error("No time dimension in inst nogw_Q0")
    Q0_times = Q0_inst.dims["time"]
    lats = Q0_inst.dims["lat"]

    # Determine which times to search
    if isnothing(candidate_times)
        search_times = Q0_times
    else
        # Snap each candidate to the nearest available Q₀ time
        search_times =
            unique([Q0_times[argmin(abs.(Q0_times .- ct))] for ct in candidate_times])
    end

    best_time = search_times[end]
    best_count = 0

    for t in search_times
        snap = slice(Q0_inst; time = t)
        count_active = 0
        for (ilat, lat) in enumerate(lats)
            abs(lat) >= lat_max && continue
            for ilon in axes(snap.data, 1)
                v = snap.data[ilon, ilat]
                if !isnan(v) && v > Q0_threshold
                    count_active += 1
                end
            end
        end
        if count_active > best_count
            best_count = count_active
            best_time = t
        end
    end

    println(
        "Peak active timestep: t=$(best_time)s ($(round(best_time / DAY_S, digits=2)) days), $best_count active columns",
    )
    return best_time, best_count
end

# --- Hotspot finder ---

"""
Pick columns at given percentiles of a 2D field (positive, well-separated),
restricted to `|lat| < lat_max`.
"""
function find_at_percentiles(field_2d, percentiles; lat_max = 30.0)
    data = field_2d.data
    lats = field_2d.dims["lat"]
    lons = field_2d.dims["lon"]

    vals = Float64[]
    coords = Tuple{Float64, Float64}[]
    for (ilat, lat) in enumerate(lats)
        abs(lat) >= lat_max && continue
        for (ilon, lon) in enumerate(lons)
            v = data[ilon, ilat]
            (isnan(v) || v <= 0) && continue
            push!(vals, v)
            push!(coords, (lon, lat))
        end
    end
    isempty(vals) && return Tuple{Float64, Float64}[]

    perm = sortperm(vals)  # ascending
    n = length(vals)

    selected = Tuple{Float64, Float64}[]
    for p in percentiles
        t = clamp(round(Int, n * p), 1, n)
        found = false
        for offset in 0:div(n, 4)
            for idx in (t + offset, t - offset)
                (idx < 1 || idx > n) && continue
                lon, lat = coords[perm[idx]]
                too_close = any(s -> abs(s[1] - lon) < 10 && abs(s[2] - lat) < 10, selected)
                if !too_close
                    push!(selected, (lon, lat))
                    found = true
                    break
                end
            end
            found && break
        end
    end
    return selected
end

# --- Output filename helper ---

"""
Append `_inst` to filename stem when in inst mode.
"""
function output_filename(output_dir, basename, mode::PlotMode; suffix = ".png")
    stem = mode == MODE_INST ? "$(basename)_inst" : basename
    return joinpath(output_dir, stem * suffix)
end
