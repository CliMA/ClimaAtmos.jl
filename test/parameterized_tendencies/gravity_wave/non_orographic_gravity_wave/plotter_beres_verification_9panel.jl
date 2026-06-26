"""
    plotter_beres_verification_9panel.jl

Trimmed 3×3 verification plot for the Beres (2004) convective GW source.
The figure follows a where / why / what story:
  Row 1 (WHERE): precip, Q₀ map, h_heat map
  Row 2 (WHY — gating decision): arup + z_top gate, Q_conv/Q_conv_ic envelope
         with z_bot threshold/floor lines, (Q₀, h_heat) activation-gate scatter
  Row 3 (WHAT — source & response): in-cloud heating vs half-sine, u-drag, v-drag

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
# Optional single-hotspot flag (--p100 | --p85 | --p70): all three hotspots are
# still found (so the selected point is identical to the 3-hotspot run), but the
# per-hotspot panels plot only the chosen one. The row-1 maps stay global.
const _HS_FLAGS = ("--p100", "--p85", "--p70")
_hs_present = filter(t -> t in remaining_args, collect(_HS_FLAGS))
single_hotspot =
    isempty(_hs_present) ? nothing : replace(_hs_present[1], "--" => "")
remaining_args = filter(a -> !(a in _HS_FLAGS), remaining_args)
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

# In inst mode: if T_END_DAYS is provided (finite), use that day directly;
# otherwise find the peak-active timestep restricted to daily timestamps.
inst_time = nothing
if mode == MODE_INST
    if isfinite(T_END_DAYS)
        inst_time = T_END_DAYS * DAY_S
    else
        _ref_var = load_var_inst(simdir, "arup"; prefer_period = PREFER_PERIOD)
        _ref_times = haskey(_ref_var.dims, "time") ? _ref_var.dims["time"] : nothing
        inst_time, _ = find_peak_active_time(simdir; candidate_times = _ref_times)
    end
    snap_kw[:inst_time] = inst_time
end

mode_str =
    mode == MODE_INST ? "INST (t=$(round(inst_time / DAY_S, digits=2))d)" :
    "AVG (last $(AVG_WINDOW_DAYS)d)"
println("Mode: $mode_str")

# --- Load fields ---
println("Loading diagnostics from: $output_dir")
avail = ClimaAnalysis.available_vars(simdir)
println("Available variables: ", avail)

# Fail loudly if a required diagnostic is missing (e.g. an older run written
# before nogw_a_cover / nogw_Q_conv_ic existed). These must be in the run's
# diagnostics list to produce this figure.
required = [
    "pr",
    "nogw_Q0",
    "nogw_h_heat",
    "nogw_a_cover",
    "arup",
    "nogw_Q_conv_ic",
    "utendnogw",
    "vtendnogw",
]
missing_vars = filter(v -> !(v in avail), required)
isempty(missing_vars) || error(
    "Missing required diagnostics in $output_dir: $(missing_vars). " *
    "Re-run with these in the diagnostics list (nogw_a_cover and nogw_Q_conv_ic " *
    "are newer than some older Beres runs).",
)
# nogw_zbot/nogw_ztop are optional: the envelope band is taken from the native
# nogw_halfsine support when present (the remap-clean source), and these remapped
# 2D scalars are only a fallback for older runs that lack the half-sine.
has_zbot_ztop = ("nogw_zbot" in avail) && ("nogw_ztop" in avail)

pp = PREFER_PERIOD
pr_var = load_var_for_mode(simdir, "pr", mode; prefer_period = pp)
Q0_var = load_var_for_mode(simdir, "nogw_Q0", mode; prefer_period = pp)
h_heat_var = load_var_for_mode(simdir, "nogw_h_heat", mode; prefer_period = pp)
zbot_var =
    has_zbot_ztop ?
    load_var_for_mode(simdir, "nogw_zbot", mode; prefer_period = pp) : nothing
ztop_var =
    has_zbot_ztop ?
    load_var_for_mode(simdir, "nogw_ztop", mode; prefer_period = pp) : nothing
a_cover_var = load_var_for_mode(simdir, "nogw_a_cover", mode; prefer_period = pp)
arup_var = load_var_for_mode(simdir, "arup", mode; prefer_period = pp)
Qconv_ic_var = load_var_for_mode(simdir, "nogw_Q_conv_ic", mode; prefer_period = pp)
utend_var = load_var_for_mode(simdir, "utendnogw", mode; prefer_period = pp)
vtend_var = load_var_for_mode(simdir, "vtendnogw", mode; prefer_period = pp)

# Optional: the native launched half-sine profile (3D, remap-clean). When
# present, panel (3,1) overlays it instead of the offline Q0·sin reconstruction
# from the three independently-remapped 2D fields (Q0, zbot, h_heat) — the
# product/ratio of interpolated quantities is NOT the interpolation of the
# product, so the offline curve carries a remap artifact the native field avoids
# (see docs/beres_p70_moment_match_discussion_prompt.md). Older runs lack it; the
# script then falls back to the reconstruction.
has_halfsine = "nogw_halfsine" in avail
halfsine_var =
    has_halfsine ?
    load_var_for_mode(simdir, "nogw_halfsine", mode; prefer_period = pp) : nothing
println(
    has_halfsine ?
    "  nogw_halfsine present → panel (3,1) uses the native (remap-clean) half-sine" :
    "  nogw_halfsine absent → panel (3,1) falls back to offline Q0·sin reconstruction",
)

# Snapshots for 2D fields
pr_last = get_snapshot(pr_var, mode; snap_kw...)
Q0_last = get_snapshot(Q0_var, mode; snap_kw...)
h_heat_last = get_snapshot(h_heat_var, mode; snap_kw...)
zbot_last = has_zbot_ztop ? get_snapshot(zbot_var, mode; snap_kw...) : nothing
ztop_last = has_zbot_ztop ? get_snapshot(ztop_var, mode; snap_kw...) : nothing
a_cover_last = get_snapshot(a_cover_var, mode; snap_kw...)

# Beres gating parameters (default_config.yml defaults). The script cannot read
# the run's config; update these if a run overrides the gates.
#   Activation gates (applied to the GRID-MEAN Q₀ and the heating depth):
Q0_threshold = 1.0e-5      # beres_Q0_threshold (K/s)
h_heat_min = 1000.0        # beres_h_heat_min (m)
#   Envelope-bottom (lower-boundary) detection thresholds:
z_bot_Q_threshold = 1.157e-5  # beres_z_bot_Q_threshold (K/s)
z_bot_floor = 2000.0          # beres_z_bot_floor (m)
#   Envelope-top updraft-area gate:
a_thresh = 1.0e-3          # updraft area fraction gate for z_top
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
hotspots = find_at_percentiles(Q0_masked, hotspot_percentiles; lat_max = lat_max)
println("Selected hotspots (lon, lat): ", hotspots)
if !isnothing(single_hotspot)
    idx = findfirst(==(single_hotspot), hotspot_labels)
    if !isnothing(idx) && idx <= length(hotspots)
        hotspots = [hotspots[idx]]
        hotspot_labels = [hotspot_labels[idx]]
        hotspot_colors = [hotspot_colors[idx]]
        println("Single-hotspot mode: only $(single_hotspot) at $(hotspots[1])")
    else
        @warn "Requested $single_hotspot not among selected hotspots; plotting all."
    end
end

# Helper: profile at a hotspot using current mode
_profile(var, lon, lat) = profile_at(var, lon, lat, mode; snap_kw...)

# Plot one mode-aware profile line per hotspot of `var` (x scaled by `xscale`).
function _plot_hotspot_profiles!(ax, var; xscale = 1.0)
    for (ih, (lon, lat)) in enumerate(hotspots)
        z, data = _profile(var, lon, lat)
        CairoMakie.lines!(ax, data .* xscale, z; color = hotspot_colors[ih])
    end
end

# Envelope band [z_bot, z_top] for the EDMF panels. When the native half-sine
# field is available, take its support (z where nogw_halfsine > 1% of its peak)
# so the shaded band in (2,2) and the marker lines in (2,1) agree with the dashed
# native half-sine plotted in (3,1). The remapped 2D scalars nogw_zbot/nogw_ztop
# do NOT agree with that curve — the half-sine is a nonlinear function of
# (Q0, z_bot, h), so the separately-remapped scalars and the remapped 3D profile
# disagree (the remap-(c) inconsistency). Fall back to the scalars for older runs.
function _envelope_band(lon, lat)
    if has_halfsine
        z_hs, hs = _profile(halfsine_var, lon, lat)
        pk = maximum(abs, hs)
        if pk > 0
            idx = findall(v -> abs(v) > 0.01 * pk, hs)
            !isempty(idx) && return (z_hs[minimum(idx)], z_hs[maximum(idx)])
        end
    end
    # Fallback for older runs without nogw_halfsine: the remapped 2D scalars.
    # Absent both, signal "no band" with (0, 0) (callers guard on > 0).
    has_zbot_ztop || return (0.0, 0.0)
    return (
        slice(zbot_last; lon = lon, lat = lat).data[1],
        slice(ztop_last; lon = lon, lat = lat).data[1],
    )
end

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
            :axis => Dict(:title => "Q₀ half-sine amplitude (K/s) [above threshold]"),
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

# Panel (2,1): updraft area fraction with each hotspot's Beres envelope.
# arup is EDMF context; dotted = model z_top, dashed = z_bot. The area-gate line
# (arup > a_thresh) is intentionally omitted: in moment_matched mode z_top is NOT
# the arup-crossing level (it is z_c + h/2), so the gate line would mislead; in
# area_threshold mode z_top ≈ where arup crosses ~1e-3.
ax21 = CairoMakie.Axis(fig[2, 1];
    title = "Updraft Area Frac + envelope", xlabel = "arup",
    ylabel = "Height (m)", limits = (nothing, (0, z_max_edmf)),
)
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = _profile(arup_var, lon, lat)
    CairoMakie.lines!(ax21, data, z; color = hotspot_colors[ih],
        label = "$(hotspot_labels[ih])")
    z_bot_val, z_top_val = _envelope_band(lon, lat)
    z_top_val > 0 && CairoMakie.hlines!(ax21, [z_top_val];
        color = hotspot_colors[ih], linestyle = :dot, linewidth = 1)
    z_bot_val > 0 && CairoMakie.hlines!(ax21, [z_bot_val];
        color = hotspot_colors[ih], linestyle = :dash, linewidth = 0.7)
end
!isempty(hotspots) && CairoMakie.axislegend(ax21; position = :rt, labelsize = 9)

# Panel (2,2): the in-cloud heating Q_conv_ic (the profile whose half-sine
# amplitude defines Q₀) with each hotspot's Beres envelope shaded. z_bot_floor
# (the lower bound of the moment integration / envelope) is kept; the grid-mean
# z_bot_Q_threshold line is omitted because in moment_matched mode z_bot = z_c −
# h/2 from the Q_conv_ic moments, NOT the grid-mean Q_conv crossing — so that
# reference would mislead. The grid-mean Q_conv overlay is intentionally dropped:
# it is the gating field, not the source amplitude, and is no longer emitted as a
# diagnostic. Shared height scale with (2,1) and (3,1): all use (0, z_max_edmf).
ax22 = CairoMakie.Axis(fig[2, 2];
    title = "In-cloud heating Q_conv_ic + envelope",
    xlabel = "Q (K/s)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
CairoMakie.hlines!(ax22, [z_bot_floor]; color = :black, linestyle = :dot,
    label = "z_bot floor")
let qic_label_done = false, env_label_done = false
    for (ih, (lon, lat)) in enumerate(hotspots)
        z_qic, Qic = _profile(Qconv_ic_var, lon, lat)
        lbl_qic = qic_label_done ? nothing : "Q_conv_ic (in-cloud)"
        qic_label_done = true
        CairoMakie.lines!(ax22, abs.(Qic), z_qic; color = hotspot_colors[ih],
            linewidth = 2, label = lbl_qic)
        # Shade the half-sine envelope — sourced from the native nogw_halfsine
        # support so it matches the dashed half-sine in (3,1) (see _envelope_band).
        z_bot_val, z_top_val = _envelope_band(lon, lat)
        if z_top_val > z_bot_val > 0
            lbl_env =
                env_label_done ? nothing :
                (has_halfsine ? "half-sine support" : "[z_bot, z_top]")
            env_label_done = true
            CairoMakie.hspan!(ax22, z_bot_val, z_top_val;
                color = (hotspot_colors[ih], 0.08), label = lbl_env)
        end
    end
end
!isempty(hotspots) && CairoMakie.axislegend(ax22; position = :rt, labelsize = 8)

# Panel (2,3): activation-gate scatter. Every tropical column in the (Q₀, h_heat)
# plane, colored by a_cover; the active set is the upper-right quadrant past both
# threshold lines. Makes the two activation gates visible.
Q0_data = Q0_last.data[:]
h_data = h_heat_last.data[:]
ac_data = a_cover_last.data[:]
trop_flat = tropical_mask[:]
valid =
    .!isnan.(Q0_data) .& .!isnan.(h_data) .& trop_flat .& (Q0_data .> 0) .&
    (h_data .> 0)
ylim_hi = any(valid) ? quantile(Q0_data[valid], 0.99) * 1.1 : 1.5e-4
xlim_hi = any(valid) ? quantile(h_data[valid], 0.99) * 1.1 : 20000.0

ax23 = CairoMakie.Axis(fig[2, 3][1, 1];
    title = "Activation gate: (h_heat, Q₀)",
    xlabel = "h_heat (m)", ylabel = "Q₀ (K/s)",
    limits = ((0, xlim_hi), (0, ylim_hi)),
)
if any(valid)
    sc = CairoMakie.scatter!(ax23, h_data[valid], Q0_data[valid];
        markersize = 5, color = ac_data[valid], colormap = :plasma)
    CairoMakie.Colorbar(fig[2, 3][1, 2], sc; label = "a_cover", width = 10)
end
CairoMakie.vlines!(ax23, [h_heat_min]; color = :gray, linestyle = :dash)
CairoMakie.hlines!(ax23, [Q0_threshold]; color = :gray, linestyle = :dash)
for (ih, (lon, lat)) in enumerate(hotspots)
    h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
    Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
    CairoMakie.scatter!(ax23, [h_val], [Q0_val];
        markersize = 18, color = hotspot_colors[ih], marker = :xcross)
end

# ========== ROW 3: Beres / GW outputs ==========
println("Plotting Row 3: Beres outputs...")

# (3,1) shares the height scale with (2,1) and (2,2): all use (0, z_max_edmf).
ax31 = CairoMakie.Axis(fig[3, 1];
    title = has_halfsine ?
            "In-cloud heating (solid) vs native half-sine (dash)" :
            "In-cloud heating (solid) vs half-sine (dash, offline-recon)",
    xlabel = "Q₁ (K/s)", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_edmf)),
)
let sin_label_done = false, edmf_label_done = false
    for (ih, (lon, lat)) in enumerate(hotspots)
        h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
        Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
        # z_bot scalar is only needed for the offline (no-halfsine) reconstruction.
        z_bot_val =
            has_zbot_ztop ? slice(zbot_last; lon = lon, lat = lat).data[1] : NaN
        println(
            "  Hotspot $(hotspot_labels[ih]) ($(lon), $(lat)): zbot=$z_bot_val, h_val=$h_val, Q0_val=$Q0_val",
        )
        if has_halfsine
            # Native launched half-sine: a single remapped 3D field, so it is
            # consistent with the (also single-field) nogw_Q_conv_ic below —
            # apples-to-apples, no offline nonlinear reconstruction.
            z_hs, hs = _profile(halfsine_var, lon, lat)
            lbl_sin = sin_label_done ? nothing : "nogw_halfsine (native)"
            sin_label_done = true
            CairoMakie.lines!(ax31, abs.(hs), z_hs;
                color = hotspot_colors[ih], linestyle = :dash, linewidth = 2,
                label = lbl_sin)
        elseif has_zbot_ztop && h_val > 1000 && Q0_val > 0
            # Fallback for older runs: reconstruct Q0·sin from the three
            # independently-remapped 2D fields (carries a remap artifact).
            z_ref = range(z_bot_val, z_bot_val + h_val; length = 100)
            sin_ref = Q0_val .* sin.(π .* (z_ref .- z_bot_val) ./ h_val)
            lbl_sin = sin_label_done ? nothing : "Q₀·sin(π(z-z_bot)/h) [offline]"
            sin_label_done = true
            CairoMakie.lines!(ax31, sin_ref, collect(z_ref);
                color = hotspot_colors[ih], linestyle = :dash, linewidth = 2,
                label = lbl_sin)
        end

        # Compare against the IN-CLOUD heating Q_conv_ic — this is the profile
        # whose half-sine amplitude defines Q₀, so the convention is consistent.
        z_q, Q1 = _profile(Qconv_ic_var, lon, lat)
        lbl_edmf = edmf_label_done ? nothing : "nogw_Q_conv_ic (model)"
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
_plot_hotspot_profiles!(ax32, utend_var; xscale = 1e5)

ax33 = CairoMakie.Axis(fig[3, 3];
    title = "v-GW Drag (×1e5)", xlabel = "m/s² ×1e5", ylabel = "Height (m)",
    limits = (nothing, (0, z_max_gw)),
)
_plot_hotspot_profiles!(ax33, vtend_var; xscale = 1e5)

# --- Save ---
_base =
    isnothing(single_hotspot) ? "beres_verification_9panel" :
    "beres_verification_9panel_$(single_hotspot)"
outfile = output_filename(output_dir, _base, mode)
CairoMakie.save(outfile, fig)
println("Saved figure to: $outfile")
