"""
    plotter_beres_verification.jl

Verification plots for the Beres (2004) convective GW source with mass-flux divergence heating.
Produces a 3×4 multi-panel figure.

Usage:
    julia --project=.buildkite plotter_beres_verification.jl [--inst] <output_dir> [t_end_days] [avg_days]

Pass --inst to use a single instantaneous snapshot (peak beres_active timestep)
instead of time-averaged diagnostics.
"""

import ClimaAnalysis
import ClimaAnalysis: slice
import ClimaAnalysis.Visualize as viz
import CairoMakie
using Statistics

include("beres_plot_helpers.jl")

# --- Parse CLI ---
mode, remaining_args = parse_mode(ARGS)
output_dir = length(remaining_args) >= 1 ? remaining_args[1] : "output_active"
T_END_DAYS = length(remaining_args) >= 2 ? parse(Float64, remaining_args[2]) : Inf
AVG_WINDOW_DAYS = length(remaining_args) >= 3 ? parse(Float64, remaining_args[3]) : 10.0

# Auto-extract nc_files.tar if .nc files are missing
tar_path = joinpath(output_dir, "nc_files.tar")
if isfile(tar_path) && isempty(filter(f -> endswith(f, ".nc"), readdir(output_dir)))
    println("Extracting $tar_path ...")
    run(`tar xf $tar_path -C $output_dir`)
end

# --- Configuration ---
figsize = (2800, 2400)
z_max_edmf = 20000.0
z_max_gw = 60000.0
n_hotspots = 11
hotspot_percentiles = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
hotspot_colors = [
    :darkred,
    :red,
    :orangered,
    :orange,
    :goldenrod,
    :olive,
    :teal,
    :steelblue,
    :blue,
    :slateblue,
    :purple,
]
hotspot_labels =
    ["p100", "p95", "p90", "p85", "p80", "p75", "p70", "p65", "p60", "p55", "p50"]

simdir = ClimaAnalysis.SimDir(output_dir)

# Snapshot kwargs used throughout
snap_kw = Dict(:t_end_days => T_END_DAYS, :avg_window_days => AVG_WINDOW_DAYS)

# Use daily resolution for all fields to avoid temporal mismatch between
# hourly 2D fields (Q₀) and daily 3D profiles (arup, waup, etc.)
const PREFER_PERIOD = "1d"

# In inst mode, find peak active timestep restricted to daily timestamps
inst_time = nothing
if mode == MODE_INST
    _ref_var = load_var_inst(simdir, "arup"; prefer_period = PREFER_PERIOD)
    _ref_times = haskey(_ref_var.dims, "time") ? _ref_var.dims["time"] : nothing
    inst_time, n_active = find_peak_active_time(simdir; candidate_times = _ref_times)
    snap_kw[:inst_time] = inst_time
end

mode_str =
    mode == MODE_INST ? "INST (t=$(round(inst_time / DAY_S, digits=2))d)" :
    "AVG (last $(AVG_WINDOW_DAYS)d)"
println("Mode: $mode_str")

# --- Load fields ---
println("Loading diagnostics from: $output_dir")
println("Available variables: ", ClimaAnalysis.available_vars(simdir))

pp = PREFER_PERIOD
pr_var = load_var_for_mode(simdir, "pr", mode; prefer_period = pp)
Q0_var = load_var_for_mode(simdir, "nogw_Q0", mode; prefer_period = pp)
h_heat_var = load_var_for_mode(simdir, "nogw_h_heat", mode; prefer_period = pp)
ua_var = load_var_for_mode(simdir, "ua", mode; prefer_period = pp)
ta_var = load_var_for_mode(simdir, "ta", mode; prefer_period = pp)
ha_var = load_var_for_mode(simdir, "ha", mode; prefer_period = pp)
arup_var = load_var_for_mode(simdir, "arup", mode; prefer_period = pp)
waup_var = load_var_for_mode(simdir, "waup", mode; prefer_period = pp)
taup_var = load_var_for_mode(simdir, "taup", mode; prefer_period = pp)
haup_var = load_var_for_mode(simdir, "haup", mode; prefer_period = pp)
rhoa_var = load_var_for_mode(simdir, "rhoa", mode; prefer_period = pp)
utend_var = load_var_for_mode(simdir, "utendnogw", mode; prefer_period = pp)
vtend_var = load_var_for_mode(simdir, "vtendnogw", mode; prefer_period = pp)

# Snapshots for 2D fields
pr_last = get_snapshot(pr_var, mode; snap_kw...)
Q0_last = get_snapshot(Q0_var, mode; snap_kw...)
h_heat_last = get_snapshot(h_heat_var, mode; snap_kw...)

# Mask below the Beres activation thresholds + restrict to tropics
Q0_threshold = 1.0e-5
h_heat_min = 1000.0
lat_max = 89.0
println(
    "Max Q0: ",
    maximum(filter(!isnan, Q0_last.data)),
    " (threshold: ",
    Q0_threshold,
    ")",
)
println(
    "Max h_heat: ",
    maximum(filter(!isnan, h_heat_last.data)),
    " (threshold: ",
    h_heat_min,
    ")",
)
lats_2d = [lat for _ in Q0_last.dims["lon"], lat in Q0_last.dims["lat"]]
tropical_mask = abs.(lats_2d) .< lat_max
active_mask =
    (Q0_last.data .> Q0_threshold) .& (h_heat_last.data .> h_heat_min) .& tropical_mask
println("Active columns: ", count(active_mask), " / ", length(active_mask))
Q0_masked = deepcopy(Q0_last)
Q0_masked.data .= ifelse.(active_mask, Q0_masked.data, NaN)
h_heat_masked = deepcopy(h_heat_last)
h_heat_masked.data .= ifelse.(active_mask, h_heat_masked.data, NaN)

pr_pos = deepcopy(pr_last)
pr_pos.data .= .-pr_pos.data .* DAY_S

# Find hotspot columns
println("Finding convective hotspots (p99–p50 precip)...")
hotspots = find_at_percentiles(pr_pos, hotspot_percentiles)
println("Selected hotspots (lon, lat): ", hotspots)

# Helper: profile at a hotspot using current mode
_profile(var, lon, lat) = profile_at(var, lon, lat, mode; snap_kw...)

# --- Create figure ---
fig = CairoMakie.Figure(size = figsize, fontsize = 14)
CairoMakie.Label(fig[0, 1:4], mode_str; fontsize = 16, tellwidth = false)

# ========== ROW 1: Lat-lon maps ==========
println("Plotting Row 1: lat-lon maps...")
viz.plot!(
    fig[1, 1],
    pr_pos;
    more_kwargs = Dict(
        :plot => Dict(:colormap => :Blues),
        :axis => Dict(:title => "Precipitation (mm/day)"),
    ),
)
if any(active_mask)
    viz.plot!(
        fig[1, 2],
        Q0_masked;
        more_kwargs = Dict(
            :plot => Dict(:colormap => :Reds),
            :axis => Dict(:title => "Max Heating Q₀ (K/s) [above threshold]"),
        ),
    )
    viz.plot!(
        fig[1, 3],
        h_heat_masked;
        more_kwargs = Dict(
            :plot => Dict(:colormap => :viridis),
            :axis => Dict(:title => "Heating Depth h (m) [active cols]"),
        ),
    )
else
    CairoMakie.Axis(fig[1, 2]; title = "Q₀ — no active columns")
    CairoMakie.Axis(fig[1, 3]; title = "h_heat — no active columns")
end

# Panel (1,4): Q0 vs max(a·w·Δh) above 3 km
waup_snap = get_snapshot(waup_var, mode; snap_kw...)
arup_snap = get_snapshot(arup_var, mode; snap_kw...)
haup_snap = get_snapshot(haup_var, mode; snap_kw...)
ha_snap = get_snapshot(ha_var, mode; snap_kw...)
z_3d = waup_snap.dims["z"]
z_mask_above_3km = z_3d .>= 3000.0
lons_3d = waup_snap.dims["lon"]
lats_3d = waup_snap.dims["lat"]
mf_dh_max = zeros(length(lons_3d), length(lats_3d))
for (ilat, lat) in enumerate(lats_3d)
    for (ilon, lon) in enumerate(lons_3d)
        w_col = waup_snap.data[ilon, ilat, :]
        a_col = arup_snap.data[ilon, ilat, :]
        haup_col = haup_snap.data[ilon, ilat, :]
        ha_col = ha_snap.data[ilon, ilat, :]
        n = min(length(w_col), length(a_col), length(haup_col), length(ha_col))
        mf_dh = abs.(a_col[1:n] .* w_col[1:n] .* (haup_col[1:n] .- ha_col[1:n]))
        vals_above = mf_dh[z_mask_above_3km[1:n]]
        valid_vals = filter(x -> !isnan(x), vals_above)
        mf_dh_max[ilon, ilat] = isempty(valid_vals) ? 0.0 : maximum(valid_vals)
    end
end

Q0_data = Q0_last.data[:]
mf_data = mf_dh_max[:]
active_flat = active_mask[:]
valid = .!isnan.(Q0_data) .& .!isnan.(mf_data) .& active_flat .& (mf_data .> 0)
mf_hi = any(valid) ? quantile(mf_data[valid], 0.99) : 50.0
Q0_hi = any(valid) ? quantile(Q0_data[valid], 0.99) : 1.5e-4
ax14 = CairoMakie.Axis(fig[1, 4];
    title = "Q₀ vs max(a·w·Δh) z>3km",
    xlabel = "max arup·waup·Δh (W/m²)",
    ylabel = "Q₀ (K/s)",
    limits = ((0, mf_hi * 1.1), (0, Q0_hi * 1.1)),
)
if any(valid)
    CairoMakie.scatter!(ax14, mf_data[valid], Q0_data[valid];
        markersize = 5, alpha = 0.4, color = :steelblue)
end
for (ih, (lon, lat)) in enumerate(hotspots)
    w_col = slice(waup_snap; lon = lon, lat = lat).data
    a_col = slice(arup_snap; lon = lon, lat = lat).data
    haup_col = slice(haup_snap; lon = lon, lat = lat).data
    ha_col = slice(ha_snap; lon = lon, lat = lat).data
    n = min(length(w_col), length(a_col), length(haup_col), length(ha_col))
    mf_dh = abs.(a_col[1:n] .* w_col[1:n] .* (haup_col[1:n] .- ha_col[1:n]))
    vals_above = mf_dh[z_mask_above_3km[1:n]]
    valid_vals = filter(x -> !isnan(x), vals_above)
    mf_val = isempty(valid_vals) ? 0.0 : maximum(valid_vals)
    Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
    CairoMakie.scatter!(ax14, [mf_val], [Q0_val];
        markersize = 18, color = hotspot_colors[ih], marker = :xcross)
end

# ========== ROW 2: EDMF vertical profiles ==========
println("Plotting Row 2: EDMF profiles...")

ax21 = CairoMakie.Axis(fig[2, 1];
    title = "Updraft Area Frac", xlabel = "arup", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = _profile(arup_var, lon, lat)
    CairoMakie.lines!(ax21, data, z; color = hotspot_colors[ih],
        label = "$(hotspot_labels[ih])")
end
!isempty(hotspots) && CairoMakie.axislegend(ax21; position = :rt, labelsize = 10)

ax22 = CairoMakie.Axis(fig[2, 2];
    title = "Updraft Velocity", xlabel = "waup (m/s)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = _profile(waup_var, lon, lat)
    CairoMakie.lines!(ax22, data, z; color = hotspot_colors[ih])
end

ax23 = CairoMakie.Axis(fig[2, 3];
    title = "Enthalpy Anomaly", xlabel = "haup - ha (J/kg)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z_up, haup_data = _profile(haup_var, lon, lat)
    z_ha, ha_data = _profile(ha_var, lon, lat)
    n = min(length(haup_data), length(ha_data))
    delta_h = haup_data[1:n] .- ha_data[1:n]
    CairoMakie.lines!(ax23, delta_h, z_up[1:n]; color = hotspot_colors[ih])
    h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
    CairoMakie.hlines!(ax23, [0.0, h_val];
        color = hotspot_colors[ih], linestyle = :dash, linewidth = 0.5)
end

ax24 = CairoMakie.Axis(fig[2, 4];
    title = "Temp Anomaly", xlabel = "ΔT (K)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z_up, taup_data = _profile(taup_var, lon, lat)
    z_ta, ta_data = _profile(ta_var, lon, lat)
    n = min(length(taup_data), length(ta_data))
    delta_T = taup_data[1:n] .- ta_data[1:n]
    CairoMakie.lines!(ax24, delta_T, z_up[1:n]; color = hotspot_colors[ih])
end

# ========== ROW 3: Beres / GW outputs ==========
println("Plotting Row 3: Beres outputs...")

ax31 = CairoMakie.Axis(fig[3, 1];
    title = "Heating: sin ref (dash) vs EDMF (solid)",
    xlabel = "Q₁ (K/s)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
    Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
    if h_val > 1000 && Q0_val > 0
        z_ref = range(0, h_val; length = 100)
        sin_ref = Q0_val .* sin.(π .* z_ref ./ h_val)
        CairoMakie.lines!(ax31, sin_ref, collect(z_ref);
            color = hotspot_colors[ih], linestyle = :dash, linewidth = 2,
            label = ih == 1 ? "sin(πz/h)" : nothing)
    end

    z_up, haup_data = _profile(haup_var, lon, lat)
    z_ha, ha_data = _profile(ha_var, lon, lat)
    z_ar, arup_data = _profile(arup_var, lon, lat)
    z_w, waup_data = _profile(waup_var, lon, lat)
    z_rho, rho_data = _profile(rhoa_var, lon, lat)

    z_q, Q1 = reconstruct_Qconv(z_up, rho_data, arup_data, waup_data, haup_data, ha_data)
    CairoMakie.lines!(ax31, abs.(Q1), z_q; color = hotspot_colors[ih], alpha = 0.5,
        linewidth = 1,
        label = ih == 1 ? "EDMF |∂flux/∂z|" : nothing)
end
!isempty(hotspots) && CairoMakie.axislegend(ax31; position = :rt, labelsize = 9)

ax32 = CairoMakie.Axis(fig[3, 2];
    title = "u-GW Drag (×1e5)", xlabel = "m/s² ×1e5", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_gw)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = _profile(utend_var, lon, lat)
    CairoMakie.lines!(ax32, data .* 1e5, z; color = hotspot_colors[ih])
end

ax33 = CairoMakie.Axis(fig[3, 3];
    title = "v-GW Drag (×1e5)", xlabel = "m/s² ×1e5", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_gw)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = _profile(vtend_var, lon, lat)
    CairoMakie.lines!(ax33, data .* 1e5, z; color = hotspot_colors[ih])
end

# Panel (3,4): Beres source spectrum B(c) — offline computation
ax34 = CairoMakie.Axis(fig[3, 4];
    title = "Beres Spectrum B(c)", xlabel = "Phase speed c (m/s)", ylabel = "MF (arb.)",
)
try
    @eval using ClimaAtmos

    dc = 4.0
    cmax = 100.0
    nc_bins = Int(floor(2 * cmax / dc + 1))
    c_tuple = ntuple(nn -> (nn - 1) * dc - cmax, nc_bins)
    gw_ncval = Val(nc_bins)

    beres = ClimaAtmos.BeresSourceParams{Float64}(;
        Q0_threshold = 1.0e-5, beres_scale_factor = 1.0,
        σ_x = 4000.0, ν_min = 8.727e-4, ν_max = 1.047e-2, n_ν = 9,
    )

    for (ih, (lon, lat)) in enumerate(hotspots)
        Q0_val = Float64(slice(Q0_last; lon = lon, lat = lat).data[1])
        h_val = Float64(slice(h_heat_last; lon = lon, lat = lat).data[1])
        z_ua, ua_data = _profile(ua_var, lon, lat)
        z_ta, ta_data = _profile(ta_var, lon, lat)

        h_mask = z_ua .<= h_val
        u_heat = any(h_mask) ? Float64(mean(ua_data[h_mask])) : 0.0

        g = 9.81
        cp_d = 1004.0
        N_vals = Float64[]
        for k in 2:(length(z_ta) - 1)
            dTdz = (ta_data[k + 1] - ta_data[k - 1]) / (z_ta[k + 1] - z_ta[k - 1])
            N2 = (g / ta_data[k]) * (dTdz + g / cp_d)
            N2 > 0 && push!(N_vals, sqrt(N2))
        end
        N_source = isempty(N_vals) ? 0.012 : mean(N_vals)

        B = ClimaAtmos.wave_source(
            c_tuple, u_heat, Q0_val, h_val, N_source, beres, gw_ncval,
        )
        CairoMakie.lines!(ax34, collect(c_tuple), collect(B); color = hotspot_colors[ih])
    end
catch e
    println("Spectrum plot skipped: $e")
    CairoMakie.text!(ax34, 0.5, 0.5;
        text = "ClimaAtmos\nnot loaded", space = :relative, align = (:center, :center))
end

# --- Save ---
outfile = output_filename(output_dir, "beres_verification", mode)
CairoMakie.save(outfile, fig)
println("Saved figure to: $outfile")
