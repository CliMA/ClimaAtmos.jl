"""
Reproduce Figures 1 and 2 from Garner (2005):
"A Topographic Drag Closure Built on an Analytical Base Flux"
Journal of the Atmospheric Sciences, Vol. 62, pp. 2302-2315

Figure 1: Drag over North and South America with tropical wind = -7 m/s,
          mid-latitude wind = 13 m/s
Figure 2: Drag over Asia with uniform zonal wind = 10 m/s

The drag formula from Garner (2005) is:
    τ = (ρ̄N̄)/(ρᵣNᵣ) ⟨T⟩V̄

Where:
- τ: drag vector (τ_u, τ_v)
- ρ̄, N̄: surface density and buoyancy frequency
- ρᵣ, Nᵣ: reference density and buoyancy frequency
- ⟨T⟩: orographic stress tensor (t11, t12, t21, t22)
- V̄: surface wind vector (u, v)
"""

import CUDA
ENV["CLIMACOMMS_DEVICE"] = "CUDA"

import ClimaComms
import ClimaComms.@import_required_backends

using ClimaCore
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
import Thermodynamics as TD
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, DataLayouts, Operators, Spaces, Grids, Utilities, to_cpu

using CairoMakie
using NCDatasets

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

# Setup simulation
(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id, comms_ctx)
config.parsed_args["orographic_gravity_wave"] = "raw_topo"
config.parsed_args["topography"] = "Earth"
(; parsed_args) = config

simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u

# Prepare physical uv input variables
u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2
ᶜz = Fields.coordinate_field(Y.c).z

# Unpack cache and scratch vars
ᶜT = p.scratch.ᶜtemp_scalar
(; topo_ᶜz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
    p.orographic_gravity_wave
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
    p.orographic_gravity_wave
(; values_at_z_pbl, topo_info) = p.orographic_gravity_wave
(; ᶜdTdz, ᶜbuoyancy_frequency) = p.orographic_gravity_wave
(; ᶜts) = p.precomputed
(; params) = p

# Extract parameters
ogw_params = p.orographic_gravity_wave.ogw_params
grav = CAP.grav(params)
thermo_params = CAP.thermodynamics_params(params)

#######################################
# FIGURE 1: Americas
# Wind: -7 m/s in tropics (|lat| <= 23.5°), +13 m/s elsewhere
# ρ = 1.0 kg/m³, N = 0.01 s⁻¹
#######################################

println("\n" * "="^70)
println("FIGURE 1: Drag over Americas")
println("="^70)

# Set wind profile for Figure 1 (Garner 2005, Fig 1)
function garner_fig1_wind!(u_phy, v_phy)
    FT = eltype(u_phy)

    # Set winds based on latitude
    # Get coordinate field that matches the space of u_phy
    coord_field = Fields.coordinate_field(axes(u_phy))

    @. u_phy = ifelse(
        abs(coord_field.lat) <= FT(23.5),
        FT(-7.0),   # Tropical: -7 m/s
        FT(13.0),   # Mid-latitudes: +13 m/s
    )
    @. v_phy = FT(0.0)  # No meridional component
end

# Apply Figure 1 wind pattern
garner_fig1_wind!(u_phy, v_phy)

# Set constant density and buoyancy frequency for idealized test
# In the full model, these would come from thermodynamics
# For Garner reproduction: ρ = 1.0 kg/m³, N = 0.01 s⁻¹
@. Y.c.ρ = FT(1.0)
@. ᶜbuoyancy_frequency = FT(0.01)

# Compute base flux (this extracts values at PBL and computes τ_x, τ_y)
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

# Note: calc_base_flux! already computes the drag components:
#   τ_x = ρ_pbl * N_pbl * (t11 * u_pbl + t21 * v_pbl)
#   τ_y = ρ_pbl * N_pbl * (t12 * u_pbl + t22 * v_pbl)
#
# This is exactly the Garner (2005) formula: τ = (ρ̄N̄)/(ρᵣNᵣ) ⟨T⟩V̄
# with ρᵣ = Nᵣ = 1 in our case (since we set ρ = 1.0, N = 0.01)

#######################################
# FIGURE 2: Asia
# Wind: 10 m/s everywhere (zonal)
# ρ = 1.0 kg/m³, N = 0.01 s⁻¹
#######################################

println("\n" * "="^70)
println("FIGURE 2: Drag over Asia")
println("="^70)

# Set wind profile for Figure 2 (Garner 2005, Fig 2)
@. u_phy = FT(10.0)  # Uniform 10 m/s zonal wind
@. v_phy = FT(0.0)   # No meridional component

# Density and N remain the same (1.0 and 0.01)

# Compute base flux for Figure 2
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

#######################################
# PLOTTING SECTION
#######################################

println("\n" * "="^70)
println("Creating plots...")
println("="^70)

# Store Figure 2 data (already computed above)
τ_x_fig2_cpu = to_cpu(topo_τ_x)
τ_y_fig2_cpu = to_cpu(topo_τ_y)

# Re-run Figure 1 to get its data
garner_fig1_wind!(u_phy, v_phy)
CA.calc_base_flux!(
    topo_τ_x,
    topo_τ_y,
    topo_τ_l,
    topo_τ_p,
    topo_τ_np,
    topo_U_sat,
    topo_FrU_sat,
    topo_FrU_clp,
    topo_FrU_max,
    topo_FrU_min,
    topo_ᶜz_pbl,
    values_at_z_pbl,
    ogw_params,
    topo_info,
    Y.c.ρ,
    u_phy,
    v_phy,
    ᶜz,
    ᶜbuoyancy_frequency,
)
τ_x_fig1_cpu = to_cpu(topo_τ_x)
τ_y_fig1_cpu = to_cpu(topo_τ_y)

Y_cpu = to_cpu(Y)
ᶜspace = axes(Y_cpu.c)

ENV["GKSwstype"] = "nul"

# Use existing remap infrastructure
output_dir = "garner2005_reproduction"
mkpath(output_dir)

# Prepare field data for Figure 1
field_data_fig1 = Dict("τ_x" => τ_x_fig1_cpu, "τ_y" => τ_y_fig1_cpu)

# Prepare field data for Figure 2
field_data_fig2 = Dict("τ_x" => τ_x_fig2_cpu, "τ_y" => τ_y_fig2_cpu)

# Remap to lat/lon (reuse existing utilities)
# For now, let's create simple plots using the existing plotting infrastructure

# Configure plots
config_americas = PlotConfig(
    plot_mode = :horizontal_slice,
    contour_levels = 20,
    nlat = 180,
    nlon = 360,
    yreversed = false,
)

config_asia = PlotConfig(
    plot_mode = :horizontal_slice,
    contour_levels = 20,
    nlat = 180,
    nlon = 360,
    yreversed = false,
)

# Plot components separately first to verify
panels_fig1 = [
    PlotPanel("τ_x", "Figure 1: Zonal drag (Americas)", (1, 1)),
    PlotPanel("τ_y", "Figure 1: Meridional drag (Americas)", (1, 2)),
]

panels_fig2 = [
    PlotPanel("τ_x", "Figure 2: Zonal drag (Asia)", (1, 1)),
    PlotPanel("τ_y", "Figure 2: Meridional drag (Asia)", (1, 2)),
]

figure_specs_fig1 = Dict("fig1_components" => panels_fig1)
figure_specs_fig2 = Dict("fig2_components" => panels_fig2)

println("\nCreating Figure 1 (Americas) component plots...")
create_figure_set(
    output_dir,
    collect(keys(field_data_fig1)),
    field_data_fig1,
    Y_cpu,
    ᶜspace,
    figure_specs_fig1,
    config_americas;
    remap_dir = joinpath(output_dir, "remap_fig1/"),
    FT = FT,
)

println("\nCreating Figure 2 (Asia) component plots...")
create_figure_set(
    output_dir,
    collect(keys(field_data_fig2)),
    field_data_fig2,
    Y_cpu,
    ᶜspace,
    figure_specs_fig2,
    config_asia;
    remap_dir = joinpath(output_dir, "remap_fig2/"),
    FT = FT,
)

println("\n" * "="^70)
println("Garner 2005 reproduction complete!")
println("Output directory: $output_dir")
println("="^70)
