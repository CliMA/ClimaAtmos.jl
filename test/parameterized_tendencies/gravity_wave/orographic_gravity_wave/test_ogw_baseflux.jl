using CUDA

import ClimaComms
import ClimaComms.@import_required_backends

using ClimaCore
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
import ClimaAtmos.Thermodynamics as TD
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, Operators, Spaces, Grids, Utilities, to_cpu

include("../gw_remap_plot_utils.jl")

const FT = Float64
ᶜgradᵥ = Operators.GradientF2C()
ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)

(; config_file, job_id) = CA.commandline_kwargs()

"""
    compute_base_flux(ogw_mode, comms_ctx, config_file, job_id)

Compute base flux for a given orographic gravity wave mode.

Returns a named tuple with:
- tau_x: zonal base flux (on CPU)
- tau_y: meridional base flux (on CPU)
- Y_cpu: state vector (on CPU)
"""
function compute_base_flux(ogw_mode::String, comms_ctx, config_file, job_id)
    @info "Computing base flux for ogw_mode = $ogw_mode"

    config = CA.AtmosConfig(config_file; job_id, comms_ctx)
    config.parsed_args["h_elem"] = 8
    config.parsed_args["orographic_gravity_wave"] = ogw_mode
    config.parsed_args["topography"] = "Earth"

    simulation = CA.get_simulation(config)
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
    return (
        tau_x = to_cpu(topo_τ_x),
        tau_y = to_cpu(topo_τ_y),
        Y_cpu = to_cpu(Y),
    )
end

#######################################
# COMPUTE BASE FLUX FOR BOTH MODES
#######################################

raw_topo_results = compute_base_flux("raw_topo", comms_ctx, config_file, job_id)
gfdl_results = compute_base_flux("gfdl_restart", comms_ctx, config_file, job_id)

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
