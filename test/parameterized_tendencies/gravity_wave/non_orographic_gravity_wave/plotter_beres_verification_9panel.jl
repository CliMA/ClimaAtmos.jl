"""
    plotter_beres_verification_9panel.jl

Trimmed 3×3 verification plot for the Beres (2004) convective GW source.
Panels: precip, Q₀, h_depth | arup, waup, scatter | heating, u-drag, v-drag

Usage:
    julia --project=.buildkite plotter_beres_verification_9panel.jl [--inst] <output_dir> [t_end_days] [avg_days]
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
figsize = (1200, 800)
z_max_edmf = 20000.0
z_max_gw = 60000.0
n_hotspots = 3
hotspot_percentiles = [1.00, 0.85, 0.70]
hotspot_colors = [:darkred, :orange, :teal]
hotspot_labels = ["p100", "p85", "p70"]

simdir = ClimaAnalysis.SimDir(output_dir)

# Snapshot kwargs used throughout
snap_kw = Dict(:t_end_days => T_END_DAYS, :avg_window_days => AVG_WINDOW_DAYS)

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
arup_var = load_var_for_mode(simdir, "arup", mode; prefer_period = pp)
waup_var = load_var_for_mode(simdir, "waup", mode; prefer_period = pp)
haup_var = load_var_for_mode(simdir, "haup", mode; prefer_period = pp)
ha_var = load_var_for_mode(simdir, "ha", mode; prefer_period = pp)
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

# Find hotspot columns — rank by Q0 itself so every hotspot is guaranteed
# Beres-active (positive Q0 and h_heat) for the sine-reference / scatter panels.
println("Finding convective hotspots among Beres-active columns (ranked by Q0)...")
hotspots = find_at_percentiles(Q0_masked, hotspot_percentiles)
println("Selected hotspots (lon, lat): ", hotspots)

# Helper: profile at a hotspot using current mode
_profile(var, lon, lat) = profile_at(var, lon, lat, mode; snap_kw...)

# --- Create figure ---
fig = CairoMakie.Figure(size = figsize, fontsize = 10)
CairoMakie.Label(fig[0, 1:3], mode_str; fontsize = 12, tellwidth = false)

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

# ========== ROW 2: EDMF profiles + scatter ==========
println("Plotting Row 2: EDMF profiles + scatter...")

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

# Panel (2,3): Q0 vs max(a·w·Δh) above 3 km
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

# Pre-compute hotspot scatter coordinates so axis limits can include them
hotspot_mf = Float64[]
hotspot_Q0 = Float64[]
for (lon, lat) in hotspots
    w_col = slice(waup_snap; lon = lon, lat = lat).data
    a_col = slice(arup_snap; lon = lon, lat = lat).data
    haup_col = slice(haup_snap; lon = lon, lat = lat).data
    ha_col = slice(ha_snap; lon = lon, lat = lat).data
    n = min(length(w_col), length(a_col), length(haup_col), length(ha_col))
    mf_dh = abs.(a_col[1:n] .* w_col[1:n] .* (haup_col[1:n] .- ha_col[1:n]))
    vals_above = mf_dh[z_mask_above_3km[1:n]]
    valid_vals = filter(x -> !isnan(x), vals_above)
    push!(hotspot_mf, isempty(valid_vals) ? 0.0 : maximum(valid_vals))
    push!(hotspot_Q0, slice(Q0_last; lon = lon, lat = lat).data[1])
end

# Expand axis limits to include hotspot markers (with 10% padding)
xlim_hi = max(mf_hi, isempty(hotspot_mf) ? 0.0 : maximum(hotspot_mf)) * 1.1
ylim_hi = max(Q0_hi, isempty(hotspot_Q0) ? 0.0 : maximum(hotspot_Q0)) * 1.1

ax23 = CairoMakie.Axis(fig[2, 3];
    title = "Q₀ vs max(a·w·Δh) z>3km",
    xlabel = "max arup·waup·Δh (W/m²)",
    ylabel = "Q₀ (K/s)",
    limits = ((0, xlim_hi), (0, ylim_hi)),
)
if any(valid)
    CairoMakie.scatter!(ax23, mf_data[valid], Q0_data[valid];
        markersize = 5, alpha = 0.4, color = :steelblue)
end
for ih in eachindex(hotspots)
    CairoMakie.scatter!(ax23, [hotspot_mf[ih]], [hotspot_Q0[ih]];
        markersize = 18, color = hotspot_colors[ih], marker = :xcross)
end

# ========== ROW 3: Beres / GW outputs ==========
println("Plotting Row 3: Beres outputs...")

# Compute y-limit from max heating depth across hotspots (with 20% padding)
h_vals_hotspots =
    [slice(h_heat_last; lon = lon, lat = lat).data[1] for (lon, lat) in hotspots]
z_max_heating = maximum(filter(x -> x > 0, h_vals_hotspots); init = z_max_edmf) * 1.2

ax31 = CairoMakie.Axis(fig[3, 1];
    title = "Heating: sin ref (dash) vs EDMF (solid)",
    xlabel = "Q₁ (K/s)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_heating)),
)
let sin_label_done = false, edmf_label_done = false
    for (ih, (lon, lat)) in enumerate(hotspots)
        h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
        Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
        println(
            "  Hotspot $(hotspot_labels[ih]) ($(lon), $(lat)): h_val=$h_val, Q0_val=$Q0_val",
        )
        if h_val > 1000 && Q0_val > 0
            z_ref = range(0, h_val; length = 100)
            sin_ref = Q0_val .* sin.(π .* z_ref ./ h_val)
            lbl_sin = sin_label_done ? nothing : "sin(πz/h)"
            sin_label_done = true
            CairoMakie.lines!(ax31, sin_ref, collect(z_ref);
                color = hotspot_colors[ih], linestyle = :dash, linewidth = 2,
                label = lbl_sin)
        end

        z_up, haup_data = _profile(haup_var, lon, lat)
        z_ha, ha_data = _profile(ha_var, lon, lat)
        z_ar, arup_data = _profile(arup_var, lon, lat)
        z_w, waup_data = _profile(waup_var, lon, lat)
        z_rho, rho_data = _profile(rhoa_var, lon, lat)

        z_q, Q1 =
            reconstruct_Qconv(z_up, rho_data, arup_data, waup_data, haup_data, ha_data)
        lbl_edmf = edmf_label_done ? nothing : "EDMF |∂flux/∂z|"
        edmf_label_done = true
        CairoMakie.lines!(ax31, abs.(Q1), z_q; color = hotspot_colors[ih], alpha = 0.5,
            linewidth = 1,
            label = lbl_edmf)
    end
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

# --- Save ---
outfile = output_filename(output_dir, "beres_verification_9panel", mode)
CairoMakie.save(outfile, fig)
println("Saved figure to: $outfile")
