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
ENV["CLIMACOMMS_DEVICE"] = "CPU"

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
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz, read_var, window
using GeoMakie
import GeometryBasics

include("../gw_remap_plot_utils.jl")

FT = Float64
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

    # @Main.infiltrate
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

#######################################
# Custom quiver plotting function
#######################################

"""
    plot_garner_quiver(
        output_file, datafile_rll, τ_x_name, τ_y_name;
        lon_range, lat_range, title, arrow_skip, arrow_scale
    )

Create a Garner-style quiver plot with coastlines showing drag vectors.

# Arguments
- `output_file`: Path to save the figure
- `datafile_rll`: Path to remapped NetCDF file
- `τ_x_name`: Variable name for zonal drag component
- `τ_y_name`: Variable name for meridional drag component
- `lon_range`: Tuple (lon_min, lon_max) in degrees
- `lat_range`: Tuple (lat_min, lat_max) in degrees
- `title`: Plot title
- `arrow_skip`: Downsample factor for arrows (plot every Nth point)
- `arrow_scale`: Scale factor for arrow lengths
"""
function plot_garner_quiver(
    output_file::String,
    datafile_rll::String,
    τ_x_name::String,
    τ_y_name::String;
    lon_range::Tuple{Real,Real} = (-180.0, 180.0),
    lat_range::Tuple{Real,Real} = (-90.0, 90.0),
    title::String = "Orographic Drag",
    arrow_skip::Int = 1,
    arrow_scale::Real = 50.0,
    reference_arrow::Real = 2.0,  # Pa
)
    println("Creating quiver plot: $output_file")

    # Load data directly using NCDatasets (simpler than ClimaAnalysis for remapped files)
    lon, lat, τ_x_data, τ_y_data = NCDatasets.NCDataset(datafile_rll, "r") do ds
        # Read coordinates
        lon_all = Array(ds["lon"][:])
        lat_all = Array(ds["lat"][:])

        # Read data (squeeze time dimension)
        # NCDatasets reads as (lon, lat, time) even though ncdump shows (time, lat, lon)
        # So index [:, :, 1] to get all lon, all lat, first time
        # Result is (lon, lat) = (360, 180)
        τ_x_all = Array(ds[τ_x_name][:, :, 1])
        τ_y_all = Array(ds[τ_y_name][:, :, 1])

        # Apply geographic windowing
        lon_mask = (lon_all .>= lon_range[1]) .& (lon_all .<= lon_range[2])
        lat_mask = (lat_all .>= lat_range[1]) .& (lat_all .<= lat_range[2])

        lon = lon_all[lon_mask]
        lat = lat_all[lat_mask]
        # Data is (lon, lat), so index with [lon_mask, lat_mask]
        τ_x_data = τ_x_all[lon_mask, lat_mask]
        τ_y_data = τ_y_all[lon_mask, lat_mask]

        return lon, lat, τ_x_data, τ_y_data
    end

    # Create figure with white background
    fig = Figure(size = (1200, 800), fontsize = 14, backgroundcolor = :white)
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)",
        title = title,
        aspect = DataAspect(),
        backgroundcolor = :white,
    )

    # Downsample for quiver plot
    lon_skip = 1:arrow_skip:length(lon)
    lat_skip = 1:arrow_skip:length(lat)

    lon_arrows = lon[lon_skip]
    lat_arrows = lat[lat_skip]
    τ_x_arrows = τ_x_data[lon_skip, lat_skip]
    τ_y_arrows = τ_y_data[lon_skip, lat_skip]

    # Create meshgrid for arrows
    lon_grid = repeat(lon_arrows, 1, length(lat_arrows))
    lat_grid = repeat(lat_arrows', length(lon_arrows), 1)

    # Plot quiver arrows (updated to arrows2d! with new parameter names)
    arrows2d!(
        ax,
        vec(lon_grid),
        vec(lat_grid),
        vec(τ_x_arrows) * arrow_scale,
        vec(τ_y_arrows) * arrow_scale,
        tipwidth = 8,
        tiplength = 10,
        lengthscale = 1.0,
        tipcolor = :black,
        shaftcolor = :black,
        shaftwidth = 1.5,
    )

    # Add coastlines using GeoMakie
    try
        coastline_data = GeoMakie.coastlines()

        # Transform coastline coordinates to match lon_range format
        # GeoMakie.coastlines() returns lon in -180:180, but we might use 0:360
        if lon_range[1] > 180  # Using 0-360 format
            # Convert coastlines from -180:180 to 0:360
            # Filter and split segments to avoid dateline artifacts
            transformed_coastlines = []
            for geom in coastline_data
                coords = GeometryBasics.coordinates(geom)
                if isempty(coords)
                    continue
                end

                # Build segments, splitting at dateline crossings and filtering by range
                current_segment = GeometryBasics.Point2f[]
                for i in eachindex(coords)
                    lon = coords[i][1]
                    lat = coords[i][2]
                    lon_shifted = lon < 0 ? lon + 360 : lon

                    # Check for dateline crossing between consecutive points
                    if i > 1
                        prev_lon_shifted = coords[i-1][1] < 0 ? coords[i-1][1] + 360 : coords[i-1][1]
                        # If shifted longitude jumps more than 180°, we crossed the dateline
                        if abs(lon_shifted - prev_lon_shifted) > 180
                            # End current segment and start new one
                            if length(current_segment) > 1
                                push!(transformed_coastlines, GeometryBasics.LineString(current_segment))
                            end
                            current_segment = GeometryBasics.Point2f[]
                        end
                    end

                    # Only add points within or near our longitude range (with 10° buffer)
                    if (lon_shifted >= lon_range[1] - 10) && (lon_shifted <= lon_range[2] + 10)
                        push!(current_segment, GeometryBasics.Point2f(lon_shifted, lat))
                    elseif !isempty(current_segment)
                        # Point is outside range, end segment
                        if length(current_segment) > 1
                            push!(transformed_coastlines, GeometryBasics.LineString(current_segment))
                        end
                        current_segment = GeometryBasics.Point2f[]
                    end
                end

                # Add final segment
                if length(current_segment) > 1
                    push!(transformed_coastlines, GeometryBasics.LineString(current_segment))
                end
            end
            lines!(ax, transformed_coastlines, color = :black, linewidth = 1)
        else
            lines!(ax, coastline_data, color = :black, linewidth = 1)
        end
    catch e
        @warn "Could not plot coastlines with GeoMakie: $e"
        # Fallback: just add grid lines
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xgridcolor = (:gray, 0.3)
        ax.ygridcolor = (:gray, 0.3)
    end

    # Add reference arrow in bottom-left corner
    ref_lon = lon_range[1] + 0.1 * (lon_range[2] - lon_range[1])
    ref_lat = lat_range[1] + 0.1 * (lat_range[2] - lat_range[1])

    arrows2d!(
        ax,
        [ref_lon],
        [ref_lat],
        [reference_arrow * arrow_scale],
        [0.0],
        tipwidth = 12,
        tiplength = 15,
        lengthscale = 1.0,
        tipcolor = :red,
        shaftcolor = :red,
        shaftwidth = 2.5,
    )

    text!(
        ax,
        ref_lon,
        ref_lat - 0.05 * (lat_range[2] - lat_range[1]),
        text = "$(reference_arrow) Pa",
        fontsize = 16,
        color = :red,
        font = :bold,
    )

    # Set axis limits
    xlims!(ax, lon_range...)
    ylims!(ax, lat_range...)

    save(output_file, fig)
    println("  Saved: $output_file")

    return fig
end

# Use existing remap infrastructure
output_dir = "garner2005_reproduction"
mkpath(output_dir)

# Prepare field data for Figure 1
field_data_fig1 = Dict("τ_x" => τ_x_fig1_cpu, "τ_y" => τ_y_fig1_cpu)

# Prepare field data for Figure 2
field_data_fig2 = Dict("τ_x" => τ_x_fig2_cpu, "τ_y" => τ_y_fig2_cpu)

# Remap to lat/lon (reuse existing utilities)
# For now, let's create simple plots using the existing plotting infrastructure

# Configure remapping
config_remap = PlotConfig(
    plot_mode = :horizontal_slice,
    contour_levels = 20,
    nlat = 180,
    nlon = 360,
    yreversed = false,
)

# Remap Figure 1 data
println("\nRemapping Figure 1 (Americas) data...")
remap_dir_fig1 = joinpath(output_dir, "remap_fig1/")
datafile_fig1 = remap_to_latlon(
    remap_dir_fig1,
    ["τ_x", "τ_y"],
    field_data_fig1,
    Y_cpu,
    ᶜspace;
    config = config_remap,
    FT = FT,
)

# Remap Figure 2 data
println("\nRemapping Figure 2 (Asia) data...")
remap_dir_fig2 = joinpath(output_dir, "remap_fig2/")
datafile_fig2 = remap_to_latlon(
    remap_dir_fig2,
    ["τ_x", "τ_y"],
    field_data_fig2,
    Y_cpu,
    ᶜspace;
    config = config_remap,
    FT = FT,
)

# Create Garner-style quiver plots
println("\n" * "="^70)
println("Creating Garner-style quiver plots...")
println("="^70)

# Figure 1: Americas (160°W to 30°W = 200° to 330° in 0-360 notation)
println("\nFigure 1: Drag over Americas")
plot_garner_quiver(
    joinpath(output_dir, "garner_fig1_americas_quiver.png"),
    datafile_fig1,
    "τ_x",
    "τ_y";
    lon_range = (200.0, 330.0),  # 160°W to 30°W in 0-360 format
    lat_range = (-90.0, 90.0),
    title = "Figure 1: Orographic Drag over Americas (Garner 2005)",
    arrow_skip = 2,
    arrow_scale = 5.0,
    reference_arrow = 2.0,
)

# Figure 2: Asia (45°E to 135°E)
println("\nFigure 2: Drag over Asia")
plot_garner_quiver(
    joinpath(output_dir, "garner_fig2_asia_quiver.png"),
    datafile_fig2,
    "τ_x",
    "τ_y";
    lon_range = (45.0, 135.0),
    lat_range = (0.0, 90.0),
    title = "Figure 2: Orographic Drag over Asia (Garner 2005)",
    arrow_skip = 2,
    arrow_scale = 5.0,
    reference_arrow = 2.0,
)

println("\n" * "="^70)
println("Garner 2005 reproduction complete!")
println("Output directory: $output_dir")
println("="^70)
