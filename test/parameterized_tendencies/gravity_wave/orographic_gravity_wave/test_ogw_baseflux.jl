using CUDA

import ClimaComms
import ClimaComms.@import_required_backends

using ClimaCore
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
import ClimaAtmos.Thermodynamics as TD
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, Operators, Spaces, Grids, Utilities, to_cpu, InputOutput

include("../gw_remap_plot_utils.jl")
include("ogw_test_utils.jl")

FT = Float64
ᶜgradᵥ = Operators.GradientF2C()
ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)

h_elem = 8
(; config_file, job_id) = CA.commandline_kwargs()

"""
    compute_base_flux(ogw_mode, comms_ctx, config_file, job_id)

Compute base flux for a given orographic gravity wave mode.

Returns a named tuple with:
- tau_x: zonal base flux (on CPU)
- tau_y: meridional base flux (on CPU)
- topo_info: named tuple of tensor fields (t11, t12, t21, t22, hmax, hmin) on CPU
- Y_cpu: state vector (on CPU)
"""
function compute_base_flux(ogw_mode::String, comms_ctx, config_file, job_id; h_elem = 8)
    @info "Computing base flux for ogw_mode = $ogw_mode"

    simulation, config = if ogw_mode == "raw_topo"
        create_ogw_simulation(config_file, job_id, comms_ctx; h_elem)
    else
        config = CA.AtmosConfig(config_file; job_id, comms_ctx)
        config.parsed_args["h_elem"] = h_elem
        config.parsed_args["orographic_gravity_wave"] = ogw_mode
        config.parsed_args["topography"] = "Earth"
        CA.get_simulation(config), config
    end
    p = simulation.integrator.p
    Y = simulation.integrator.u

    # prepare physical uv input variables for gravity_wave_forcing()
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    ᶜz = Fields.coordinate_field(Y.c).z

    # Simple latitude-based wind profile (from ogwd_baseflux_gpu.jl / Garner05 Figure 1)
    # Tropics (|lat| <= 23.5): u = -7 m/s, v = 0
    # Extratropics: u = 13 m/s, v = 0
    ᶜlocal_geometry = Fields.local_geometry_field(axes(Y.c))
    ᶜlat = ᶜlocal_geometry.coordinates.lat
    @. u_phy = ifelse(abs(ᶜlat) <= FT(23.5), FT(-7), FT(13))
    @. v_phy = FT(0)

    # Unpack cache
    (; topo_ᶜz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
        p.orographic_gravity_wave
    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; values_at_z_pbl, topo_info) =
        p.orographic_gravity_wave
    (; ᶜdTdz, ᶜbuoyancy_frequency) = p.orographic_gravity_wave

    # Extract parameters
    ogw_params = p.orographic_gravity_wave.ogw_params

    # Use constant PBL height of 16 km
    @. topo_ᶜz_pbl = FT(16000)

    # Use constant buoyancy frequency N = 0.01
    @. ᶜbuoyancy_frequency = FT(0.01)

    # Use constant density ρ = 1.0 (like ogwd_baseflux_gpu.jl)
    @. Y.c.ρ = FT(1.0)

    # Compute base flux
    CA.calc_base_flux!(
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_p,
        topo_τ_np,
        #
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜz_pbl,
        #
        values_at_z_pbl,
        #
        ogw_params,
        topo_info,
        #
        Y.c.ρ,
        u_phy,
        v_phy,
        ᶜz,
        ᶜbuoyancy_frequency,
    )

    # Move GPU arrays back to CPU for plotting
    topo_info_cpu = (
        t11 = to_cpu(topo_info.t11),
        t12 = to_cpu(topo_info.t12),
        t21 = to_cpu(topo_info.t21),
        t22 = to_cpu(topo_info.t22),
        hmax = to_cpu(topo_info.hmax),
        hmin = to_cpu(topo_info.hmin),
    )
    return (
        tau_x = to_cpu(topo_τ_x),
        tau_y = to_cpu(topo_τ_y),
        topo_info = topo_info_cpu,
        Y_cpu = to_cpu(Y),
    )
end

#######################################
# COMPUTE BASE FLUX FOR BOTH MODES
#######################################

raw_topo_results =
    compute_base_flux("raw_topo", comms_ctx, config_file, job_id, h_elem = h_elem)
gfdl_results =
    compute_base_flux("gfdl_restart", comms_ctx, config_file, job_id, h_elem = h_elem)

# Use space from one of the results (they should be identical)
ᶜspace = axes(raw_topo_results.Y_cpu.c)

#######################################
# PLOTTING SECTION - 2x2 COMPARISON
#######################################

ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_baseflux"

# Prepare field data dictionary with all 4 fields
field_data = Dict(
    "tau_x_raw" => raw_topo_results.tau_x,
    "tau_y_raw" => raw_topo_results.tau_y,
    "tau_x_gfdl" => gfdl_results.tau_x,
    "tau_y_gfdl" => gfdl_results.tau_y,
)

# 2x2 comparison layout:
# Row 1: zonal flux (tau_x) - raw_topo (left) vs gfdl_restart (right)
# Row 2: meridional flux (tau_y) - raw_topo (left) vs gfdl_restart (right)
comparison_panels = [
    PlotPanel("tau_x_raw", "raw_topo (zonal)", (1, 1); colorrange = (-5.0, 5.0)),
    PlotPanel("tau_x_gfdl", "gfdl_restart (zonal)", (1, 2); colorrange = (-5.0, 5.0)),
    PlotPanel("tau_y_raw", "raw_topo (meridional)", (2, 1); colorrange = (-1.0, 1.0)),
    PlotPanel("tau_y_gfdl", "gfdl_restart (meridional)", (2, 2); colorrange = (-1.0, 1.0)),
]

# Configure plots
plot_config = PlotConfig(
    plot_mode = :horizontal_slice,
    contour_levels = range(-5, 5; length = 20),
    nlat = 90,
    nlon = 180,
    yreversed = false,
    figure_size = (4000, 2000),  # Double width for 2x2 side-by-side layout
    output_format = "pdf",
)

figure_specs = Dict(
    "baseflux_comparison" => comparison_panels,
)

create_figure_set(
    output_dir,
    collect(keys(field_data)),
    field_data,
    raw_topo_results.Y_cpu,
    ᶜspace,
    figure_specs,
    plot_config;
    remap_dir = joinpath(@__DIR__, "ogwd_3d", "remap_data/"),
    FT = FT,
)

#######################################
# ZONAL MEAN DIAGNOSTICS
#######################################

using Statistics: mean

# Include tensor fields alongside baseflux for remapping
all_field_data = Dict{String, Any}(
    "tau_x_raw" => raw_topo_results.tau_x,
    "tau_y_raw" => raw_topo_results.tau_y,
    "tau_x_gfdl" => gfdl_results.tau_x,
    "tau_y_gfdl" => gfdl_results.tau_y,
    "t11_raw" => raw_topo_results.topo_info.t11,
    "t12_raw" => raw_topo_results.topo_info.t12,
    "t21_raw" => raw_topo_results.topo_info.t21,
    "t22_raw" => raw_topo_results.topo_info.t22,
    "t11_gfdl" => gfdl_results.topo_info.t11,
    "t12_gfdl" => gfdl_results.topo_info.t12,
    "t21_gfdl" => gfdl_results.topo_info.t21,
    "t22_gfdl" => gfdl_results.topo_info.t22,
)

# Remap all fields to lat-lon grid
zonal_plot_config = PlotConfig(
    plot_mode = :horizontal_slice,
    nlat = 90,
    nlon = 180,
    cleanup_remap_files = false,
)
datafile_rll = remap_to_latlon(
    joinpath(@__DIR__, "ogwd_3d", "remap_data/"),
    collect(keys(all_field_data)),
    all_field_data,
    raw_topo_results.Y_cpu,
    ᶜspace;
    config = zonal_plot_config,
    FT = FT,
)

# Read remapped lat-lon data
rll_data = NCDataset(datafile_rll) do ds
    lat = Array(ds["lat"])
    d = Dict{String, Any}("lat" => lat)
    for k in keys(all_field_data)
        raw = Array(ds[k])
        # Handle [lon, lat, time] → [lon, lat]
        d[k] = size(raw, 3) > 0 ? raw[:, :, 1] : raw[:, :]
    end
    d
end
lat = rll_data["lat"]

# Zonal statistics helpers
zonal_mean_fn(d) = dropdims(mean(d; dims = 1); dims = 1)
zonal_max_abs_fn(d) = dropdims(maximum(abs.(d); dims = 1); dims = 1)

# Create 3-panel zonal diagnostics figure
fig = CairoMakie.Figure(; size = (1600, 1500))

# Panel 1: Zonal mean τ_x
ax1 = CairoMakie.Axis(fig[1, 1]; title = "Zonal mean τ_x", xlabel = "lat", ylabel = "τ_x")
CairoMakie.lines!(ax1, lat, zonal_mean_fn(rll_data["tau_x_raw"]); label = "raw_topo")
CairoMakie.lines!(ax1, lat, zonal_mean_fn(rll_data["tau_x_gfdl"]); label = "gfdl_restart")
CairoMakie.axislegend(ax1)

# Panel 2: Zonal mean τ_y
ax2 = CairoMakie.Axis(fig[2, 1]; title = "Zonal mean τ_y", xlabel = "lat", ylabel = "τ_y")
CairoMakie.lines!(ax2, lat, zonal_mean_fn(rll_data["tau_y_raw"]); label = "raw_topo")
CairoMakie.lines!(ax2, lat, zonal_mean_fn(rll_data["tau_y_gfdl"]); label = "gfdl_restart")
CairoMakie.axislegend(ax2)

# Panel 3: Zonal mean difference (raw - gfdl)
ax3 = CairoMakie.Axis(
    fig[3, 1];
    title = "Zonal mean difference (raw_topo − gfdl_restart)",
    xlabel = "lat",
    ylabel = "difference",
)
CairoMakie.lines!(
    ax3,
    lat,
    zonal_mean_fn(rll_data["tau_x_raw"]) .- zonal_mean_fn(rll_data["tau_x_gfdl"]);
    label = "Δτ_x",
)
CairoMakie.lines!(
    ax3,
    lat,
    zonal_mean_fn(rll_data["tau_y_raw"]) .- zonal_mean_fn(rll_data["tau_y_gfdl"]);
    label = "Δτ_y",
)
CairoMakie.axislegend(ax3)

mkpath(output_dir)
CairoMakie.save(joinpath(output_dir, "zonal_diagnostics.pdf"), fig)
@info "Saved zonal diagnostics to $(joinpath(output_dir, "zonal_diagnostics.pdf"))"

# Create 4-panel drag tensor diagnostics figure
fig_tensor = CairoMakie.Figure(; size = (1600, 2000))

for (idx, tname) in enumerate(["t11", "t12", "t21", "t22"])
    ax = CairoMakie.Axis(
        fig_tensor[idx, 1];
        title = "Max |$(tname)| per latitude",
        xlabel = "lat",
        ylabel = "|$(tname)|",
    )
    CairoMakie.lines!(
        ax,
        lat,
        zonal_max_abs_fn(rll_data["$(tname)_raw"]);
        label = "raw_topo",
    )
    CairoMakie.lines!(
        ax,
        lat,
        zonal_max_abs_fn(rll_data["$(tname)_gfdl"]);
        label = "gfdl_restart",
    )
    CairoMakie.axislegend(ax)
end

CairoMakie.save(joinpath(output_dir, "drag_tensor_diagnostics.pdf"), fig_tensor)
@info "Saved drag tensor diagnostics to $(joinpath(output_dir, "drag_tensor_diagnostics.pdf"))"
