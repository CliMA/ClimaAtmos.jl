"""
    plotter_beres_verification.jl

Verification plots for the Beres (2004) convective GW source with mass-flux divergence heating.
Produces a 3×4 multi-panel figure.

Usage:
    julia --project=.buildkite plotter_beres_verification.jl <output_dir>

where <output_dir> is the ClimaAtmos output directory (containing nc_files.tar or .nc files).
"""

import ClimaAnalysis
import ClimaAnalysis: slice
import ClimaAnalysis.Visualize as viz
import CairoMakie
using Statistics

# --- Configuration ---
output_dir = length(ARGS) >= 1 ? ARGS[1] : "output_active"

# Auto-extract nc_files.tar if .nc files are missing
tar_path = joinpath(output_dir, "nc_files.tar")
if isfile(tar_path) && isempty(filter(f -> endswith(f, ".nc"), readdir(output_dir)))
    println("Extracting $tar_path ...")
    run(`tar xf $tar_path -C $output_dir`)
end
figsize = (2800, 2000)
z_max_edmf = 20000.0   # y-axis limit for EDMF profiles (Row 2)
z_max_gw = 60000.0     # y-axis limit for GW drag profiles (Row 3)
n_hotspots = 3
hotspot_colors = [:red, :blue, :green]

simdir = ClimaAnalysis.SimDir(output_dir)

# --- Helper: load a variable with auto-detected reduction/period ---
function load_var(simdir, short_name)
    reds = ClimaAnalysis.available_reductions(simdir; short_name)
    # Prefer "inst" over "average" for our custom diagnostics
    red = "inst" in reds ? "inst" : first(reds)
    return get(simdir; short_name, reduction = red)
end

# --- Helper: get last time snapshot ---
function last_snapshot(var)
    times = var.dims["time"]
    return slice(var; time = times[end])
end

# --- Helper: extract vertical profile at a point ---
function profile_at(var, lon, lat)
    s = last_snapshot(var)
    col = slice(s; lon = lon, lat = lat)
    return col.dims["z"], col.data
end

# --- Helper: find hotspot lat/lon from a 2D field ---
function find_hotspots(field_2d, n)
    data = field_2d.data
    lats = field_2d.dims["lat"]
    lons = field_2d.dims["lon"]

    # Restrict to tropics: |lat| < 30
    vals = Float64[]
    coords = Tuple{Float64, Float64}[]
    for (ilat, lat) in enumerate(lats)
        abs(lat) >= 30 && continue
        for (ilon, lon) in enumerate(lons)
            push!(vals, abs(data[ilon, ilat]))
            push!(coords, (lon, lat))
        end
    end
    perm = sortperm(vals; rev = true)

    # Pick n well-separated hotspots (at least 10° apart)
    selected = Tuple{Float64, Float64}[]
    for idx in perm
        lon, lat = coords[idx]
        too_close = any(s -> abs(s[1] - lon) < 10 && abs(s[2] - lat) < 10, selected)
        if !too_close
            push!(selected, (lon, lat))
        end
        length(selected) >= n && break
    end
    return selected
end

# --- Load fields ---
println("Loading diagnostics from: $output_dir")
println("Available variables: ", ClimaAnalysis.available_vars(simdir))

pr_var = load_var(simdir, "pr")
Q0_var = load_var(simdir, "nogw_Q0")
h_heat_var = load_var(simdir, "nogw_h_heat")
ua_var = load_var(simdir, "ua")
ta_var = load_var(simdir, "ta")
ha_var = load_var(simdir, "ha")
arup_var = load_var(simdir, "arup")
waup_var = load_var(simdir, "waup")
taup_var = load_var(simdir, "taup")
haup_var = load_var(simdir, "haup")
utend_var = load_var(simdir, "utendnogw")
vtend_var = load_var(simdir, "vtendnogw")

# Last snapshots for 2D fields
pr_last = last_snapshot(pr_var)
Q0_last = last_snapshot(Q0_var)
h_heat_last = last_snapshot(h_heat_var)

# Mask h_heat where Beres is inactive: Q0 below threshold OR outside tropics (|lat| >= 30°)
Q0_threshold = 1.157e-4  # same as BeresSourceParams default
h_heat_masked = deepcopy(h_heat_last)
lats_h = h_heat_masked.dims["lat"]
for (ilat, lat) in enumerate(lats_h)
    if abs(lat) >= 30
        h_heat_masked.data[:, ilat] .= NaN
    end
end
# Only mask extratropics; keep all tropical columns regardless of Q0

# Find hotspot columns
println("Finding convective hotspots...")
hotspots = find_hotspots(Q0_last, n_hotspots)
println("Selected hotspots (lon, lat): ", hotspots)

# --- Create figure ---
fig = CairoMakie.Figure(size = figsize, fontsize = 14)

# ========== ROW 1: Lat-lon maps ==========
println("Plotting Row 1: lat-lon maps...")

pr_pos = deepcopy(pr_last)
pr_pos.data .= .-pr_pos.data  # flip sign: downward flux → positive precip
viz.plot!(fig[1, 1], pr_pos; more_kwargs = Dict(
    :plot => Dict(:colormap => :Blues),
    :axis => Dict(:title => "Precipitation (kg/m²/s)"),
))
viz.plot!(fig[1, 2], Q0_last; more_kwargs = Dict(
    :plot => Dict(:colormap => :Reds),
    :axis => Dict(:title => "Max Heating Q₀ (K/s)"),
))
viz.plot!(fig[1, 3], h_heat_masked; more_kwargs = Dict(
    :plot => Dict(:colormap => :viridis),
    :axis => Dict(:title => "Heating Depth h (m) [active cols]"),
))

# Panel (1,4): Q0 vs pr scatter
ax14 = CairoMakie.Axis(fig[1, 4];
    title = "Q₀ vs Precipitation",
    xlabel = "Precip",
    ylabel = "Q₀ (K/s)",
)
pr_data = .-pr_last.data[:]  # flip sign
Q0_data = Q0_last.data[:]
mask = Q0_data .> 0
if any(mask)
    CairoMakie.scatter!(ax14, pr_data[mask], Q0_data[mask];
        markersize = 3, alpha = 0.3, color = :steelblue)
end
for (ih, (lon, lat)) in enumerate(hotspots)
    pr_val = -slice(pr_last; lon = lon, lat = lat).data[1]  # flip sign
    Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
    CairoMakie.scatter!(ax14, [pr_val], [Q0_val];
        markersize = 12, color = hotspot_colors[ih], marker = :xcross)
end

# ========== ROW 2: EDMF vertical profiles ==========
println("Plotting Row 2: EDMF profiles...")

# Panel (2,1): Updraft area fraction
ax21 = CairoMakie.Axis(fig[2, 1]; title = "Updraft Area Frac", xlabel = "arup", ylabel = "Height (m)", limits = (nothing, (0, z_max_edmf)))
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = profile_at(arup_var, lon, lat)
    CairoMakie.lines!(ax21, data, z; color = hotspot_colors[ih],
        label = "$(round(lat; digits=1))°, $(round(lon; digits=1))°")
end
CairoMakie.axislegend(ax21; position = :rt, labelsize = 10)

# Panel (2,2): Updraft vertical velocity
ax22 = CairoMakie.Axis(fig[2, 2]; title = "Updraft Velocity", xlabel = "waup (m/s)", ylabel = "Height (m)", limits = (nothing, (0, z_max_edmf)))
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = profile_at(waup_var, lon, lat)
    CairoMakie.lines!(ax22, data, z; color = hotspot_colors[ih])
end

# Panel (2,3): Enthalpy anomaly haup - ha
ax23 = CairoMakie.Axis(fig[2, 3]; title = "Enthalpy Anomaly", xlabel = "haup - ha (J/kg)", ylabel = "Height (m)", limits = (nothing, (0, z_max_edmf)))
for (ih, (lon, lat)) in enumerate(hotspots)
    z_up, haup_data = profile_at(haup_var, lon, lat)
    z_ha, ha_data = profile_at(ha_var, lon, lat)
    n = min(length(haup_data), length(ha_data))
    delta_h = haup_data[1:n] .- ha_data[1:n]
    CairoMakie.lines!(ax23, delta_h, z_up[1:n]; color = hotspot_colors[ih])
    # Overlay h_heat bounds
    h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
    CairoMakie.hlines!(ax23, [0.0, h_val]; color = hotspot_colors[ih], linestyle = :dash, linewidth = 0.5)
end

# Panel (2,4): Temperature anomaly
ax24 = CairoMakie.Axis(fig[2, 4]; title = "Temp Anomaly", xlabel = "ΔT (K)", ylabel = "Height (m)", limits = (nothing, (0, z_max_edmf)))
for (ih, (lon, lat)) in enumerate(hotspots)
    z_up, taup_data = profile_at(taup_var, lon, lat)
    z_ta, ta_data = profile_at(ta_var, lon, lat)
    n = min(length(taup_data), length(ta_data))
    delta_T = taup_data[1:n] .- ta_data[1:n]
    CairoMakie.lines!(ax24, delta_T, z_up[1:n]; color = hotspot_colors[ih])
end

# ========== ROW 3: Beres / GW outputs ==========
println("Plotting Row 3: Beres outputs...")

# Panel (3,1): Beres assumed Q₀·sin(πz/h) vs EDMF mass-flux divergence
ax31 = CairoMakie.Axis(fig[3, 1]; title = "Heating: sin ref (dash) vs EDMF (solid)", xlabel = "Q₁ (K/s)", ylabel = "Height (m)", limits = (nothing, (0, z_max_edmf)))
for (ih, (lon, lat)) in enumerate(hotspots)
    # Beres assumed profile: Q₀·sin(πz/h) — this is what wave_source uses
    h_val = slice(h_heat_last; lon = lon, lat = lat).data[1]
    Q0_val = slice(Q0_last; lon = lon, lat = lat).data[1]
    if h_val > 1000 && Q0_val > 0
        z_ref = range(0, h_val; length = 100)
        sin_ref = Q0_val .* sin.(π .* z_ref ./ h_val)
        CairoMakie.lines!(ax31, sin_ref, collect(z_ref);
            color = hotspot_colors[ih], linestyle = :dash, linewidth = 2,
            label = ih == 1 ? "sin(πz/h)" : nothing)
    end

    # EDMF reconstructed: -d(a·w·Δh)/dz / cp
    z_up, haup_data = profile_at(haup_var, lon, lat)
    z_ha, ha_data = profile_at(ha_var, lon, lat)
    z_ar, arup_data = profile_at(arup_var, lon, lat)
    z_w, waup_data = profile_at(waup_var, lon, lat)

    cp_d = 1004.0
    n = min(length(haup_data), length(ha_data), length(arup_data), length(waup_data))
    z = z_up[1:n]
    delta_h = haup_data[1:n] .- ha_data[1:n]
    flux = arup_data[1:n] .* waup_data[1:n] .* delta_h

    Q1 = zeros(n)
    for k in 2:(n - 1)
        dz = z[k + 1] - z[k - 1]
        Q1[k] = -(flux[k + 1] - flux[k - 1]) / (dz * cp_d)
    end
    CairoMakie.lines!(ax31, Q1, z; color = hotspot_colors[ih], alpha = 0.5, linewidth = 1,
        label = ih == 1 ? "EDMF ∂flux/∂z" : nothing)
end
CairoMakie.axislegend(ax31; position = :rt, labelsize = 9)

# Panel (3,2): utendnogw
ax32 = CairoMakie.Axis(fig[3, 2]; title = "u-GW Drag (×1e5)", xlabel = "m/s² ×1e5", ylabel = "Height (m)", limits = (nothing, (0, z_max_gw)))
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = profile_at(utend_var, lon, lat)
    CairoMakie.lines!(ax32, data .* 1e5, z; color = hotspot_colors[ih])
end

# Panel (3,3): vtendnogw
ax33 = CairoMakie.Axis(fig[3, 3]; title = "v-GW Drag (×1e5)", xlabel = "m/s² ×1e5", ylabel = "Height (m)", limits = (nothing, (0, z_max_gw)))
for (ih, (lon, lat)) in enumerate(hotspots)
    z, data = profile_at(vtend_var, lon, lat)
    CairoMakie.lines!(ax33, data .* 1e5, z; color = hotspot_colors[ih])
end

# Panel (3,4): Beres source spectrum B(c) — offline computation
ax34 = CairoMakie.Axis(fig[3, 4]; title = "Beres Spectrum B(c)", xlabel = "Phase speed c (m/s)", ylabel = "MF (arb.)")
try
    @eval using ClimaAtmos

    dc = 4.0; cmax = 100.0
    nc_bins = Int(floor(2 * cmax / dc + 1))
    c_tuple = ntuple(nn -> (nn - 1) * dc - cmax, nc_bins)
    gw_ncval = Val(nc_bins)

    beres = ClimaAtmos.BeresSourceParams{Float64}(;
        Q0_threshold = 1.157e-4, beres_scale_factor = 1.0,
        σ_x = 4000.0, ν_min = 8.727e-4, ν_max = 1.047e-2, n_ν = 9,
    )

    for (ih, (lon, lat)) in enumerate(hotspots)
        Q0_val = Float64(slice(Q0_last; lon = lon, lat = lat).data[1])
        h_val = Float64(slice(h_heat_last; lon = lon, lat = lat).data[1])
        z_ua, ua_data = profile_at(ua_var, lon, lat)
        z_ta, ta_data = profile_at(ta_var, lon, lat)

        h_mask = z_ua .<= h_val
        u_heat = any(h_mask) ? Float64(mean(ua_data[h_mask])) : 0.0

        g = 9.81; cp_d = 1004.0
        N_vals = Float64[]
        for k in 2:(length(z_ta) - 1)
            dTdz = (ta_data[k + 1] - ta_data[k - 1]) / (z_ta[k + 1] - z_ta[k - 1])
            N2 = (g / ta_data[k]) * (dTdz + g / cp_d)
            N2 > 0 && push!(N_vals, sqrt(N2))
        end
        N_source = isempty(N_vals) ? 0.012 : mean(N_vals)

        B = ClimaAtmos.wave_source(c_tuple, u_heat, Q0_val, h_val, N_source, beres, gw_ncval)
        CairoMakie.lines!(ax34, collect(c_tuple), collect(B); color = hotspot_colors[ih])
    end
catch e
    println("Spectrum plot skipped: $e")
    CairoMakie.text!(ax34, 0.5, 0.5; text = "ClimaAtmos\nnot loaded", space = :relative, align = (:center, :center))
end

# --- Save ---
outfile = "beres_verification.png"
CairoMakie.save(outfile, fig)
println("Saved figure to: $outfile")
