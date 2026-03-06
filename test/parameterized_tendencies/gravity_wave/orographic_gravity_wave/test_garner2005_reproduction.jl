"""
Reproduce Figures 1, 2, 4, 5, 6, and 7 from Garner (2005):
"A Topographic Drag Closure Built on an Analytical Base Flux"
Journal of the Atmospheric Sciences, Vol. 62, pp. 2302-2315

Figure 1: Drag over North and South America with tropical wind = -7 m/s,
          mid-latitude wind = 13 m/s
Figure 2: Drag over Asia with uniform zonal wind = 10 m/s
Figure 4: Linear drag comparison over Northern Hemisphere:
          - Top panel: Analytical drag D = (ρ̄/ρr) V̄ max(|T₁|, |T₂|)
          - Middle panel: τ_l from GFDL restart (traditional dimensional estimate)
          - Bottom panel: τ_l from raw topography (new dimensional estimate)
Figure 5: Normalized drag curves showing D/D* vs h_max/h_c:
          - Propagating drag ⟨D_p⟩/D* (dashed lines)
          - Total drag (D_p + D_np)/D* (solid lines)
          - Two cases: h_min = 0 and h_min = h_max
Figure 6: Total drag and nonlinear fraction over Americas:
          - Left panel: Total base flux normalized by D* + 0.05
          - Right panel: Nonpropagating fraction τ_np / (τ_p + τ_np)
          - Uniform zonal wind = 10 m/s, γ = 0.4, β = 0.5, ε = 0, ĥ_c = 0.7
Figure 7: Vertical propagation of momentum flux and forcing:
          - Solid lines: Normalized flux ⟨D_p⟩/D* vs height
          - Dashed lines: Normalized momentum forcing ρ∂V/∂t
          - Two cases: h̃_min = 0 and h̃_min = h̃_max
          - Wind profile with jets at 38 m/s (9 km) and 58 m/s (25 km)
          - Stability: N = 0.011 s⁻¹ below 11 km, 0.022 s⁻¹ above
          - Density scale height H = 8 km

The drag formula from Garner (2005) is:
    τ = (ρ̄N̄)/(ρᵣNᵣ) ⟨T⟩V̄

Where:
- τ: drag vector (τ_u, τ_v)
- ρ̄, N̄: surface density and buoyancy frequency
- ρᵣ, Nᵣ: reference density and buoyancy frequency
- ⟨T⟩: orographic stress tensor (t11, t12, t21, t22)
- V̄: surface wind vector (u, v)

For Figure 4, the scalar drag uses equation (19):
    D = (ρ̄/ρr) V̄ max(|T₁|, |T₂|)
where T₁ and T₂ are eigenvalues of the tensor T.

For Figure 5, the dimensional estimate D* from equation (16) is:
    D* = τ_l (linear drag, used for normalization)
with parameters γ = 0.4, β = 0.5, ε = 0, a_1/a_0 = 9.0
"""

import CUDA

import ClimaComms
import ClimaComms.@import_required_backends

using ClimaCore
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore:
    Fields, Geometry, DataLayouts, Operators, Spaces, Grids, Utilities, to_cpu, InputOutput

using CairoMakie
using Statistics
using NCDatasets
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz, read_var, window

include("../gw_remap_plot_utils.jl")
include("ogw_test_utils.jl")

#######################################
# TEST SELECTION
# Set to true/false to enable/disable individual tests
#######################################
const RUN_TESTS = (
    figure1 = true,    # Figure 1: Drag over Americas
    figure2 = true,    # Figure 2: Drag over Asia
    quiver_plots = true,   # Create quiver plots for Figures 1 & 2
    figure4 = true,    # Figure 4: Linear Drag Comparison (tripanel)
    figure5 = true,    # Figure 5: Normalized Drag Curves
    figure6 = true,    # Figure 6: Total Drag and Nonlinear Fraction over Americas
    figure7 = true,     # Figure 7: Vertical Propagation Profile
)

# Helper to check if a test should run
should_run(test::Symbol) = getfield(RUN_TESTS, test)

# Debug print control - set to true to enable verbose debug output
const DEBUG_PRINTS = false

FT = Float64
ᶜgradᵥ = Operators.GradientF2C()
ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)

#######################################
# Helper function: Compute max absolute eigenvalue of 2x2 tensor
#######################################

"""
    compute_tensor_eigenvalue_max(t11, t12, t21, t22)

Compute the maximum absolute eigenvalue of a 2×2 tensor field [[t11, t12], [t21, t22]].

For a 2×2 matrix, eigenvalues are:
    λ = (t11 + t22)/2 ± sqrt(((t11 - t22)/2)² + t12 * t21)

Returns a field containing max(|λ₁|, |λ₂|) at each point.
"""
function compute_tensor_eigenvalue_max(t11, t12, t21, t22)
    FT = eltype(t11)

    # Compute eigenvalues of 2x2 tensor
    # For [[a, b], [c, d]], eigenvalues are: (a+d)/2 ± sqrt(((a-d)/2)^2 + bc)
    trace_half = @. (t11 + t22) / FT(2)
    discriminant = @. ((t11 - t22) / FT(2))^2 + t12 * t21

    # Handle potential numerical issues with sqrt of negative values
    sqrt_disc = @. sqrt(max(discriminant, FT(0)))

    λ1 = @. trace_half + sqrt_disc
    λ2 = @. trace_half - sqrt_disc

    # Return max absolute eigenvalue
    return @. max(abs(λ1), abs(λ2))
end

#######################################
# Figure 7 Profile Functions
#######################################

"""
    garner_fig7_wind_profile(z::FT) where {FT}

Create idealized wind profile V(z) for Figure 7 from Garner (2005).
Two jets: 38 m/s peak at 9 km, 58 m/s peak at 25 km.

The profile is constructed as a base wind plus two Gaussian jets.
"""
function garner_fig7_wind_profile(z::FT) where {FT}
    # Base wind
    V_base = FT(8.0)   # Background wind (m/s)

    # Jet 1: 38 m/s peak at 9 km
    V1 = FT(30.0)      # Jet 1 amplitude (30 + 8 = 38)
    z1 = FT(9000.0)    # Jet 1 center (m)
    σ1 = FT(3000.0)    # Jet 1 width (m)

    # Jet 2: 58 m/s peak at 25 km
    V2 = FT(50.0)      # Jet 2 amplitude (50 + 8 = 58)
    z2 = FT(25000.0)   # Jet 2 center (m)
    σ2 = FT(4000.0)    # Jet 2 width (m)

    return V_base + V1 * exp(-((z - z1) / σ1)^2) + V2 * exp(-((z - z2) / σ2)^2)
end

"""
    garner_fig7_stability_profile(z::FT) where {FT}

Create idealized stability profile N(z) for Figure 7 from Garner (2005).
N = 0.011 s⁻¹ below 11 km (troposphere), N = 0.022 s⁻¹ above (stratosphere).

Step function: N = 0.011 s⁻¹ below 11 km, N = 0.022 s⁻¹ above.
"""
function garner_fig7_stability_profile(z::FT) where {FT}
    z_tropopause = FT(11000.0)       # Tropopause height (m)
    N_troposphere = FT(0.011)        # Tropospheric N (s⁻¹)
    N_stratosphere = FT(0.022)       # Stratospheric N (s⁻¹)

    # Step function at tropopause (using ifelse for broadcasting)
    return ifelse(z < z_tropopause, N_troposphere, N_stratosphere)
end

"""
    garner_fig7_density_profile(z::FT) where {FT}

Create idealized density profile ρ(z) for Figure 7 from Garner (2005).
Exponential profile with scale height H = 8 km.
"""
function garner_fig7_density_profile(z::FT) where {FT}
    ρ_0 = FT(1.2)       # Surface density (kg/m³)
    H = FT(8000.0)      # Scale height (m)
    return ρ_0 * exp(-z / H)
end

"""
    garner_fig7_pressure_profile(z::FT) where {FT}

Create idealized pressure profile p(z) for Figure 7 from Garner (2005).
Exponential profile consistent with isothermal atmosphere, H = 8 km.
"""
function garner_fig7_pressure_profile(z::FT) where {FT}
    p_0 = FT(101325.0)  # Surface pressure (Pa)
    H = FT(8000.0)      # Scale height (m)
    return p_0 * exp(-z / H)
end

# Setup simulation - try local raw_topo first, fall back to artifact
(; config_file, job_id) = CA.commandline_kwargs()
simulation, config = create_ogw_simulation(
    config_file,
    job_id,
    comms_ctx;
    extra_parsed_args = Dict("z_elem" => 10, "z_max" => 45000.0),
)
(; parsed_args) = config
p = simulation.integrator.p
Y = simulation.integrator.u

# Prepare physical uv input variables
u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2
ᶜz = Fields.coordinate_field(Y.c).z

# Unpack cache
(; topo_ᶜz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
    p.orographic_gravity_wave
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
    p.orographic_gravity_wave
(; values_at_z_pbl, topo_info) = p.orographic_gravity_wave
(; ᶜdTdz, ᶜbuoyancy_frequency) = p.orographic_gravity_wave
# (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
(; params) = p

# Extract parameters
ogw_params = p.orographic_gravity_wave.ogw_params
grav = CAP.grav(params)
thermo_params = CAP.thermodynamics_params(params)

# Compute PBL height from initial-condition pressure/temperature profiles
cp_d = CAP.cp_d(params)
ᶜp = p.precomputed.ᶜp
ᶜT = p.precomputed.ᶜT
CA.get_pbl_z!(topo_ᶜz_pbl, ᶜp, ᶜT, ᶜz, grav, cp_d)

#######################################
# FIGURE 1: Americas
# Wind: -7 m/s in tropics (|lat| <= 30.0°), +13 m/s elsewhere
# ρᵣ = 1.0 kg/m³, Nᵣ = 0.01 s⁻¹
#######################################

# Set wind profile for Figure 1 (Garner 2005, Fig 1)
function garner_fig1_wind!(u_phy, v_phy)
    FT = eltype(u_phy)

    # Set winds based on latitude
    # Get coordinate field that matches the space of u_phy
    coord_field = Fields.coordinate_field(axes(u_phy))

    @. u_phy = ifelse(
        abs(coord_field.lat) <= FT(30.0),
        FT(-7.0),   # Tropical: -7 m/s
        FT(13.0),   # Mid-latitudes: +13 m/s
    )
    @. v_phy = FT(0.0)  # No meridional component
end

if should_run(:figure1)
    println("\n" * "="^70)
    println("FIGURE 1: Drag over Americas")
    println("="^70)

    # Apply Figure 1 wind pattern
    garner_fig1_wind!(u_phy, v_phy)

    # Set constant density and buoyancy frequency for idealized test
    # In the full model, these would come from thermodynamics
    # For Garner reproduction: ρᵣ = 1.0 kg/m³, Nᵣ = 0.01 s⁻¹
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
end

#######################################
# FIGURE 2: Asia
# Wind: 10 m/s everywhere (zonal)
# ρ = 1.0 kg/m³, N = 0.01 s⁻¹
#######################################

if should_run(:figure2)
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
end

#######################################
# PLOTTING SECTION
#######################################

# Common setup for plotting (needed by quiver_plots and figure4)
Y_cpu = to_cpu(Y)
ᶜspace = axes(Y_cpu.c)
ENV["GKSwstype"] = "nul"

# Data storage for quiver plots (Figure 1 and 2)
if should_run(:quiver_plots)
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
end

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
    lon_range::Tuple{Real, Real} = (-180.0, 180.0),
    lat_range::Tuple{Real, Real} = (-90.0, 90.0),
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
                        prev_lon_shifted =
                            coords[i - 1][1] < 0 ? coords[i - 1][1] + 360 : coords[i - 1][1]
                        # If shifted longitude jumps more than 180°, we crossed the dateline
                        if abs(lon_shifted - prev_lon_shifted) > 180
                            # End current segment and start new one
                            if length(current_segment) > 1
                                push!(
                                    transformed_coastlines,
                                    GeometryBasics.LineString(current_segment),
                                )
                            end
                            current_segment = GeometryBasics.Point2f[]
                        end
                    end

                    # Only add points within or near our longitude range (with 10° buffer)
                    if (lon_shifted >= lon_range[1] - 10) &&
                       (lon_shifted <= lon_range[2] + 10)
                        push!(current_segment, GeometryBasics.Point2f(lon_shifted, lat))
                    elseif !isempty(current_segment)
                        # Point is outside range, end segment
                        if length(current_segment) > 1
                            push!(
                                transformed_coastlines,
                                GeometryBasics.LineString(current_segment),
                            )
                        end
                        current_segment = GeometryBasics.Point2f[]
                    end
                end

                # Add final segment
                if length(current_segment) > 1
                    push!(
                        transformed_coastlines,
                        GeometryBasics.LineString(current_segment),
                    )
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

"""
    plot_garner_contour(
        output_file, datafile_rll, var_name;
        lon_range, lat_range, title, colormap, normalize
    )

Create a Garner-style contour plot with coastlines showing scalar field.

# Arguments
- `output_file`: Path to save the figure
- `datafile_rll`: Path to remapped NetCDF file
- `var_name`: Variable name for scalar field
- `lon_range`: Tuple (lon_min, lon_max) in degrees
- `lat_range`: Tuple (lat_min, lat_max) in degrees
- `title`: Plot title
- `colormap`: Colormap for contours (default :grays)
- `normalize`: Whether to normalize by global maximum (default true)
"""
function plot_garner_contour(
    output_file::String,
    datafile_rll::String,
    var_name::String;
    lon_range::Tuple{Real, Real} = (-180.0, 180.0),
    lat_range::Tuple{Real, Real} = (-90.0, 90.0),
    title::String = "Linear Drag",
    colormap::Symbol = :grays,
    normalize::Bool = true,
)
    println("Creating contour plot: $output_file")

    # Load data directly using NCDatasets
    lon, lat, data = NCDatasets.NCDataset(datafile_rll, "r") do ds
        # Read coordinates
        lon_all = Array(ds["lon"][:])
        lat_all = Array(ds["lat"][:])

        # Read data (squeeze time dimension)
        data_all = Array(ds[var_name][:, :, 1])

        # Apply geographic windowing
        lon_mask = (lon_all .>= lon_range[1]) .& (lon_all .<= lon_range[2])
        lat_mask = (lat_all .>= lat_range[1]) .& (lat_all .<= lat_range[2])

        lon = lon_all[lon_mask]
        lat = lat_all[lat_mask]
        data = data_all[lon_mask, lat_mask]

        return lon, lat, data
    end

    # Normalize by global maximum if requested
    if normalize
        global_max = maximum(abs.(data))
        if global_max > 0
            data = data ./ global_max
        end
    end

    # Create figure with white background
    fig = Figure(size = (1000, 600), fontsize = 14, backgroundcolor = :white)
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = title,
        aspect = DataAspect(),
        backgroundcolor = :white,
    )

    # Create contour plot
    # data is (lon, lat), contourf expects z[i,j] where i indexes x (lon) and j indexes y (lat)
    hm = contourf!(
        ax,
        lon,
        lat,
        data,
        colormap = colormap,
        levels = range(0, 1, length = 11),
    )

    # Add colorbar
    Colorbar(fig[1, 2], hm, label = normalize ? "Normalized" : var_name)

    # Add coastlines using GeoMakie
    try
        coastline_data = GeoMakie.coastlines()

        # Transform coastline coordinates if needed
        if lon_range[1] > 180  # Using 0-360 format
            transformed_coastlines = []
            for geom in coastline_data
                coords = GeometryBasics.coordinates(geom)
                if isempty(coords)
                    continue
                end

                current_segment = GeometryBasics.Point2f[]
                for i in eachindex(coords)
                    clon = coords[i][1]
                    clat = coords[i][2]
                    lon_shifted = clon < 0 ? clon + 360 : clon

                    if i > 1
                        prev_lon_shifted =
                            coords[i - 1][1] < 0 ? coords[i - 1][1] + 360 : coords[i - 1][1]
                        if abs(lon_shifted - prev_lon_shifted) > 180
                            if length(current_segment) > 1
                                push!(
                                    transformed_coastlines,
                                    GeometryBasics.LineString(current_segment),
                                )
                            end
                            current_segment = GeometryBasics.Point2f[]
                        end
                    end

                    if (lon_shifted >= lon_range[1] - 10) &&
                       (lon_shifted <= lon_range[2] + 10)
                        push!(current_segment, GeometryBasics.Point2f(lon_shifted, clat))
                    elseif !isempty(current_segment)
                        if length(current_segment) > 1
                            push!(
                                transformed_coastlines,
                                GeometryBasics.LineString(current_segment),
                            )
                        end
                        current_segment = GeometryBasics.Point2f[]
                    end
                end

                if length(current_segment) > 1
                    push!(
                        transformed_coastlines,
                        GeometryBasics.LineString(current_segment),
                    )
                end
            end
            lines!(ax, transformed_coastlines, color = :black, linewidth = 1)
        else
            lines!(ax, coastline_data, color = :black, linewidth = 1)
        end
    catch e
        @warn "Could not plot coastlines with GeoMakie: $e"
    end

    # Set axis limits
    xlims!(ax, lon_range...)
    ylims!(ax, lat_range...)

    save(output_file, fig)
    println("  Saved: $output_file")

    return fig
end

"""
    plot_garner_fig4_tripanel(
        output_file, datafile_rll, var_names, titles;
        lon_range, lat_range, colormap
    )

Create a 3-panel Garner Figure 4 style plot showing linear drag from three methods.
Each panel is normalized by its respective global maximum.

# Arguments
- `output_file`: Path to save the figure
- `datafile_rll`: Path to remapped NetCDF file
- `var_names`: Vector of 3 variable names [analytical, gfdl, raw_topo]
- `titles`: Vector of 3 panel titles
- `lon_range`: Tuple (lon_min, lon_max) in degrees
- `lat_range`: Tuple (lat_min, lat_max) in degrees
- `colormap`: Colormap for contours (default :grays)
"""
function plot_garner_fig4_tripanel(
    output_file::String,
    datafile_rll::String,
    var_names::Vector{String},
    titles::Vector{String};
    lon_range::Tuple{Real, Real} = (0.0, 360.0),
    lat_range::Tuple{Real, Real} = (-90.0, 90.0),
    colormap::Symbol = :viridis,
)
    println("Creating Figure 4 tripanel plot: $output_file")

    # Load all three datasets
    lon, lat, data_all = NCDatasets.NCDataset(datafile_rll, "r") do ds
        lon_all = Array(ds["lon"][:])
        lat_all = Array(ds["lat"][:])

        # Apply geographic windowing
        lon_mask = (lon_all .>= lon_range[1]) .& (lon_all .<= lon_range[2])
        lat_mask = (lat_all .>= lat_range[1]) .& (lat_all .<= lat_range[2])

        lon = lon_all[lon_mask]
        lat = lat_all[lat_mask]

        data_all = []
        for var_name in var_names
            data = Array(ds[var_name][:, :, 1])
            push!(data_all, data[lon_mask, lat_mask])
        end

        return lon, lat, data_all
    end

    # Normalize each dataset by its global maximum
    data_normalized = []
    for data in data_all
        global_max = maximum(abs.(data))
        if global_max > 0
            push!(data_normalized, data ./ global_max)
        else
            push!(data_normalized, data)
        end
    end

    # Create figure with 3 panels (vertical stack)
    fig = Figure(size = (1000, 1200), fontsize = 12, backgroundcolor = :white)

    for (i, (data, panel_title)) in enumerate(zip(data_normalized, titles))
        ax = Axis(
            fig[i, 1],
            xlabel = i == 3 ? "Longitude" : "",
            ylabel = "Latitude",
            title = panel_title,
            aspect = DataAspect(),
            backgroundcolor = :white,
        )

        # Create contour plot
        # data is (lon, lat), contourf expects z[i,j] where i indexes x (lon) and j indexes y (lat)
        hm = contourf!(
            ax,
            lon,
            lat,
            data,
            colormap = colormap,
            levels = range(0, 1, length = 11),
        )

        # Add coastlines
        try
            coastline_data = GeoMakie.coastlines()

            if lon_range[1] >= 0 && lon_range[2] <= 180
                # Standard -180:180 format works
                lines!(ax, coastline_data, color = :black, linewidth = 0.5)
            else
                # Need to transform for 0-360 format
                transformed_coastlines = []
                for geom in coastline_data
                    coords = GeometryBasics.coordinates(geom)
                    if isempty(coords)
                        continue
                    end

                    current_segment = GeometryBasics.Point2f[]
                    for j in eachindex(coords)
                        clon = coords[j][1]
                        clat = coords[j][2]
                        lon_shifted = clon < 0 ? clon + 360 : clon

                        if j > 1
                            prev_lon_shifted =
                                coords[j - 1][1] < 0 ? coords[j - 1][1] + 360 :
                                coords[j - 1][1]
                            if abs(lon_shifted - prev_lon_shifted) > 180
                                if length(current_segment) > 1
                                    push!(
                                        transformed_coastlines,
                                        GeometryBasics.LineString(current_segment),
                                    )
                                end
                                current_segment = GeometryBasics.Point2f[]
                            end
                        end

                        if (lon_shifted >= lon_range[1] - 10) &&
                           (lon_shifted <= lon_range[2] + 10) &&
                           (clat >= lat_range[1]) && (clat <= lat_range[2])
                            push!(
                                current_segment,
                                GeometryBasics.Point2f(lon_shifted, clat),
                            )
                        elseif !isempty(current_segment)
                            if length(current_segment) > 1
                                push!(
                                    transformed_coastlines,
                                    GeometryBasics.LineString(current_segment),
                                )
                            end
                            current_segment = GeometryBasics.Point2f[]
                        end
                    end

                    if length(current_segment) > 1
                        push!(
                            transformed_coastlines,
                            GeometryBasics.LineString(current_segment),
                        )
                    end
                end
                lines!(ax, transformed_coastlines, color = :black, linewidth = 0.5)
            end
        catch e
            @warn "Could not plot coastlines: $e"
        end

        xlims!(ax, lon_range...)
        ylims!(ax, lat_range...)

        # Add colorbar only to the last panel
        if i == 3
            Colorbar(fig[1:3, 2], hm, label = "Normalized by global max")
        end
    end

    save(output_file, fig)
    println("  Saved: $output_file")

    return fig
end

"""
    compute_theoretical_drag(h_max, h_min, V, N, ρ; params...)

Compute theoretical drag values for given parameters, implementing Garner (2005) formulas.
Returns (τ_l, τ_p, τ_np) - linear, propagating, and non-propagating drag components.

This function implements the same formulas as calc_base_flux! but for scalar inputs,
bypassing the column_reduce! machinery.

# Arguments
- `h_max`: Maximum subgrid terrain height (m)
- `h_min`: Minimum subgrid terrain height (m)
- `V`: Wind speed (m/s)
- `N`: Buoyancy frequency (s⁻¹)
- `ρ`: Air density (kg/m³)

# Keyword Arguments
- `γ`: Exponent parameter (default 0.4)
- `β`: Saturation parameter (default 0.5)
- `ϵ`: Additional exponent (default 0.0)
- `Fr_crit`: Critical Froude number (default 0.7)
- `a0`: Propagating drag coefficient (default 0.9)
- `a1`: Non-propagating drag coefficient (default 8.1)
- `ρscale`: Reference density scale (default 1.2)
- `L0`: Length scale (default 80000.0)
"""
function compute_theoretical_drag(
    h_max::FT,
    h_min::FT,
    V::FT,
    N::FT,
    ρ::FT;
    γ::FT = FT(0.4),
    β::FT = FT(0.5),
    ϵ::FT = FT(0.0),
    Fr_crit::FT = FT(0.7),
    a0::FT = FT(0.9),
    a1::FT = FT(8.1),
    ρscale::FT = FT(1.2),
    L0::FT = FT(80000.0),
) where {FT}
    # Compute Froude numbers
    # Fr = h * N / V (inverse of the non-dimensional wind speed)
    Fr_max = max(FT(0), h_max) * N / V
    Fr_min = max(FT(0), h_min) * N / V

    # Compute U_sat - saturation velocity scale
    # U_sat = sqrt(ρ/ρscale * V^3 / N / L0)
    U_sat = sqrt(ρ / ρscale * V^3 / N / L0)

    # Compute FrU values (Froude number times U_sat)
    FrU_sat = Fr_crit * U_sat
    FrU_min = Fr_min * U_sat
    FrU_max = Fr_max * U_sat

    # Exponents
    exp1 = 2 + γ - ϵ  # = 2.4 for default params
    exp2 = γ - ϵ - β  # = -0.1 for default params
    exp3 = 1 + γ - ϵ  # = 1.4 for default params
    exp4 = β + 2      # = 2.5 for default params
    exp5 = β + 1      # = 1.5 for default params

    # Check if this is the monochromatic case (h_min ≈ h_max)
    # In this case, use point-evaluation formulas instead of integrals
    # The monochromatic formulas are the INTEGRANDS (not integrals) of the broadband case
    # multiplied by exp1 to account for the distribution normalization
    if abs(h_max - h_min) < FT(1e-6) * max(abs(h_max), FT(1.0))
        # Monochromatic case: single mountain height h = h_max
        FrU = FrU_max

        # Linear drag for monochromatic: point value of integrand
        # Broadband: τ_l = ∫ FrU^exp1 dFrU / exp1 = FrU^exp1 / exp1
        # Monochromatic: evaluates to FrU^exp1 (the integrand, times d(FrU))
        # The exp1 factor accounts for derivative of the integral
        τ_l = FrU^exp1  # Point value

        if Fr_max <= Fr_crit
            # Unsaturated regime: all drag is propagating
            # Broadband: τ_p = a0 * FrU^exp1 / exp1
            # Monochromatic: τ_p = a0 * FrU^exp1 (integrand value)
            τ_p = a0 * FrU^exp1
            τ_np = FT(0.0)
        else
            # Saturated regime: Fr_max > Fr_crit
            # For saturated heights, the propagating drag formula changes
            # From Garner Eq (14): D_p ~ FrU_sat^(β+2) * FrU^(γ-β)
            τ_p = a0 * FrU_sat^exp4 * FrU^exp2

            # Non-propagating drag appears for heights above critical
            # From Garner: D_np ~ [FrU^(1+γ) - FrU_sat^(1+β) * FrU^(γ-β)] / Fr
            τ_np = a1 * U_sat / (1 + β) * (FrU^exp3 - FrU_sat^exp5 * FrU^exp2)
            τ_np = τ_np / Fr_max
        end
    else
        # Broadband case: distribution of mountain heights from h_min to h_max
        FrU_max = max(FrU_max, FrU_min + eps(FT))
        FrU_clp = clamp(FrU_sat, FrU_min, FrU_max)

        # Linear drag (D* for normalization)
        τ_l = (FrU_max^exp1 - FrU_min^exp1) / exp1

        # Propagating drag
        τ_p =
            a0 * (
                (FrU_clp^exp1 - FrU_min^exp1) / exp1 +
                FrU_sat^exp4 * (FrU_max^exp2 - FrU_clp^exp2) / exp2
            )

        # Non-propagating drag
        τ_np =
            a1 * U_sat / (1 + β) * (
                (FrU_max^exp3 - FrU_clp^exp3) / exp3 -
                FrU_sat^exp5 * (FrU_max^exp2 - FrU_clp^exp2) / exp2
            )

        # Scale non-propagating drag
        τ_np = τ_np / max(Fr_crit, Fr_max)
    end

    return (τ_l, τ_p, τ_np)
end

"""
    plot_garner_fig5(
        output_file, x_values, Dp_hmin0, Dp_hmin_eq_hmax,
        total_hmin0, total_hmin_eq_hmax; params...
    )

Create a Garner Figure 5 style plot showing normalized drag curves D/D* vs h_max/h_c.

# Arguments
- `output_file`: Path to save the figure
- `x_values`: Vector of h_max/h_c values (x-axis)
- `Dp_hmin0`: Vector of ⟨D_p⟩/D* for h_min = 0 case
- `Dp_hmin_eq_hmax`: Vector of ⟨D_p⟩/D* for h_min = h_max case
- `total_hmin0`: Vector of (D_p + D_np)/D* for h_min = 0 case
- `total_hmin_eq_hmax`: Vector of (D_p + D_np)/D* for h_min = h_max case
- `γ`, `β`: Parameters shown in plot annotation (default 0.4, 0.5)
"""
function plot_garner_fig5(
    output_file::String,
    x_values::Vector,
    Dp_hmin0::Vector,
    Dp_hmin_eq_hmax::Vector,
    total_hmin0::Vector,
    total_hmin_eq_hmax::Vector;
    γ::Real = 0.4,
    β::Real = 0.5,
)
    println("Creating Figure 5 plot: $output_file")

    # Create figure matching paper style
    fig = Figure(size = (600, 500), fontsize = 14, backgroundcolor = :white)
    ax = Axis(
        fig[1, 1],
        xlabel = "hₘₐₓ/hc",
        ylabel = "D/D*",
        title = "Figure 5: Normalized Drag vs Mountain Height (Garner 2005)",
        backgroundcolor = :white,
    )

    # Plot total drag (solid lines) - different colors
    lines!(ax, x_values, total_hmin_eq_hmax, color = :blue, linewidth = 2,
        label = "Total (hₘᵢₙ = hₘₐₓ)")
    lines!(ax, x_values, total_hmin0, color = :red, linewidth = 2,
        linestyle = :solid, label = "Total (hₘᵢₙ = 0)")

    # Plot propagating drag (dashed lines) - different colors
    lines!(ax, x_values, Dp_hmin_eq_hmax, color = :blue, linewidth = 1.5,
        linestyle = :dash, label = "⟨Dₚ⟩ (hₘᵢₙ = hₘₐₓ)")
    lines!(ax, x_values, Dp_hmin0, color = :red, linewidth = 1.5,
        linestyle = :dash, label = "⟨Dₚ⟩ (hₘᵢₙ = 0)")

    # Add parameter annotations (matching paper style)
    text!(ax, 4.0, 2.3, text = "γ = $γ", fontsize = 12)
    text!(ax, 4.0, 2.1, text = "β = $β", fontsize = 12)

    # Add legend
    axislegend(ax, position = :lt, framevisible = true, labelsize = 10)

    # Set axis limits to match paper
    xlims!(ax, 0, 5)
    ylims!(ax, 0, 2.5)

    save(output_file, fig)
    println("  Saved: $output_file")

    return fig
end

"""
    plot_garner_fig6_twopanel(
        output_file, datafile_rll, var_names, titles, colorbars;
        lon_range, lat_range, colormap
    )

Create a 2-panel Garner Figure 6 style plot showing total drag and nonlinear fraction.

# Arguments
- `output_file`: Path to save the figure
- `datafile_rll`: Path to remapped NetCDF file
- `var_names`: Vector of 2 variable names [total_drag, nonlinear_frac]
- `titles`: Vector of 2 panel titles
- `colorbar_ranges`: Vector of 2 tuples for colorbar ranges [(min1, max1), (min2, max2)]
- `lon_range`: Tuple (lon_min, lon_max) in degrees
- `lat_range`: Tuple (lat_min, lat_max) in degrees
- `colormap`: Colormap for contours (default :grays)
"""
function plot_garner_fig6_twopanel(
    output_file::String,
    datafile_rll::String,
    var_names::Vector{String},
    titles::Vector{String},
    colorbar_ranges::Vector;
    lon_range = (200.0, 330.0),  # Americas: 160°W to 30°W in 0-360 format
    lat_range = (-40.0, 70.0),
    colormap::Symbol = :grays,
)
    println("Creating Figure 6 twopanel plot: $output_file")

    # Load both datasets
    lon, lat, data_all = NCDatasets.NCDataset(datafile_rll, "r") do ds
        lon_all = Array(ds["lon"][:])
        lat_all = Array(ds["lat"][:])

        # Apply geographic windowing
        lon_mask = (lon_all .>= lon_range[1]) .& (lon_all .<= lon_range[2])
        lat_mask = (lat_all .>= lat_range[1]) .& (lat_all .<= lat_range[2])

        lon = lon_all[lon_mask]
        lat = lat_all[lat_mask]

        data_all = []
        for var_name in var_names
            data = Array(ds[var_name][:, :, 1])
            push!(data_all, data[lon_mask, lat_mask])
        end

        return lon, lat, data_all
    end

    # Create figure with 2 side-by-side panels (matching paper layout)
    fig = Figure(size = (1200, 600), fontsize = 12, backgroundcolor = :white)

    for (i, (data, panel_title, cb_range)) in
        enumerate(zip(data_all, titles, colorbar_ranges))
        ax = Axis(
            fig[1, i],
            xlabel = "LONGITUDE",
            ylabel = i == 1 ? "LATITUDE" : "",
            title = panel_title,
            aspect = DataAspect(),
            backgroundcolor = :white,
        )

        # Create contour plot with specified colorbar range
        # Reverse colormap so higher values are darker (matching paper)
        hm = contourf!(
            ax,
            lon,
            lat,
            data,
            colormap = Reverse(colormap),
            levels = range(cb_range[1], cb_range[2], length = 11),
        )

        # Add colorbar below each panel
        Colorbar(fig[2, i], hm, vertical = false, flipaxis = false,
            label = "", ticks = range(cb_range[1], cb_range[2], length = 6))

        # Add coastlines (handling 0-360 longitude format)
        try
            coastline_data = GeoMakie.coastlines()

            if lon_range[1] >= 180  # Using 0-360 format, need to transform
                transformed_coastlines = []
                for geom in coastline_data
                    coords = GeometryBasics.coordinates(geom)
                    if isempty(coords)
                        continue
                    end

                    current_segment = GeometryBasics.Point2f[]
                    for j in eachindex(coords)
                        clon = coords[j][1]
                        clat = coords[j][2]
                        lon_shifted = clon < 0 ? clon + 360 : clon

                        if j > 1
                            prev_lon_shifted =
                                coords[j - 1][1] < 0 ? coords[j - 1][1] + 360 :
                                coords[j - 1][1]
                            if abs(lon_shifted - prev_lon_shifted) > 180
                                if length(current_segment) > 1
                                    push!(
                                        transformed_coastlines,
                                        GeometryBasics.LineString(current_segment),
                                    )
                                end
                                current_segment = GeometryBasics.Point2f[]
                            end
                        end

                        if (lon_shifted >= lon_range[1] - 10) &&
                           (lon_shifted <= lon_range[2] + 10) &&
                           (clat >= lat_range[1]) && (clat <= lat_range[2])
                            push!(
                                current_segment,
                                GeometryBasics.Point2f(lon_shifted, clat),
                            )
                        elseif !isempty(current_segment)
                            if length(current_segment) > 1
                                push!(
                                    transformed_coastlines,
                                    GeometryBasics.LineString(current_segment),
                                )
                            end
                            current_segment = GeometryBasics.Point2f[]
                        end
                    end

                    if length(current_segment) > 1
                        push!(
                            transformed_coastlines,
                            GeometryBasics.LineString(current_segment),
                        )
                    end
                end
                lines!(ax, transformed_coastlines, color = :black, linewidth = 0.5)
            else
                lines!(ax, coastline_data, color = :black, linewidth = 0.5)
            end
        catch e
            @warn "Could not plot coastlines: $e"
        end

        xlims!(ax, lon_range...)
        ylims!(ax, lat_range...)
    end

    save(output_file, fig)
    println("  Saved: $output_file")

    return fig
end

"""
    plot_garner_fig7(
        output_file, z_km, flux_hmin0, flux_hmin_eq_hmax,
        forcing_hmin0, forcing_hmin_eq_hmax, wind_profile
    )

Create a Garner Figure 7 style plot showing vertical propagation of momentum flux
and forcing for two cases: h̃_min = 0 and h̃_min = h̃_max.

# Arguments
- `output_file`: Path to save the figure
- `z_km`: Height array in kilometers (y-axis)
- `flux_hmin0`: Normalized flux D/D* for h_min = 0 case (solid line)
- `flux_hmin_eq_hmax`: Normalized flux D/D* for h_min = h_max case (solid line)
- `forcing_hmin0`: Normalized forcing for h_min = 0 case (dashed line)
- `forcing_hmin_eq_hmax`: Normalized forcing for h_min = h_max case (dashed line)
- `wind_profile`: Wind speed V(z) in m/s (right panel)
"""
function plot_garner_fig7(
    output_file::String,
    z_km::Vector,
    flux_hmin0::Vector,
    flux_hmin_eq_hmax::Vector,
    forcing_hmin0::Vector,
    forcing_hmin_eq_hmax::Vector,
    wind_profile::Vector,
)
    println("Creating Figure 7 plot: $output_file")

    # Create figure matching paper style (two-panel: flux/forcing on left, wind on right)
    fig = Figure(size = (800, 600), fontsize = 14, backgroundcolor = :white)

    # Left panel: Flux and forcing profiles
    ax1 = Axis(
        fig[1, 1],
        xlabel = "D/D*",
        ylabel = "height (km)",
        title = "",
        backgroundcolor = :white,
    )

    # Plot normalized flux (solid lines)
    lines!(ax1, flux_hmin0, z_km, color = :black, linewidth = 2,
        label = "D/D* (h̃ₘᵢₙ = 0)")
    lines!(ax1, flux_hmin_eq_hmax, z_km, color = :black, linewidth = 2,
        linestyle = :solid, label = "D/D* (h̃ₘᵢₙ = h̃ₘₐₓ)")

    # Create second x-axis for forcing (top)
    ax1_top = Axis(
        fig[1, 1],
        xlabel = "ρ∂V/∂t",
        xaxisposition = :top,
        yaxisposition = :right,
        backgroundcolor = :transparent,
        yticklabelsvisible = false,
        yticksvisible = false,
        ylabelvisible = false,
    )

    # Plot normalized forcing (dashed lines)
    lines!(ax1_top, forcing_hmin0, z_km, color = :black, linewidth = 1.5,
        linestyle = :dash, label = "Forcing (h̃ₘᵢₙ = 0)")
    lines!(ax1_top, forcing_hmin_eq_hmax, z_km, color = :black, linewidth = 1.5,
        linestyle = :dash, label = "Forcing (h̃ₘᵢₙ = h̃ₘₐₓ)")

    # Link y-axes
    linkyaxes!(ax1, ax1_top)

    # Set axis limits for main panel
    xlims!(ax1, 0.0, 1.1)
    ylims!(ax1, 0.0, 40.0)

    # Right panel: Wind profile
    ax2 = Axis(
        fig[1, 2],
        xlabel = "V (m/s)",
        ylabel = "",
        title = "",
        backgroundcolor = :white,
        yticklabelsvisible = false,
    )

    lines!(ax2, wind_profile, z_km, color = :black, linewidth = 2, label = "V(z)")

    # Set axis limits for wind panel
    xlims!(ax2, 0.0, 60.0)
    ylims!(ax2, 0.0, 40.0)

    # Link y-axes between panels
    linkyaxes!(ax1, ax2)

    # Add annotations for the two cases
    text!(ax1, 0.5, 14.0, text = "h̃ₘᵢₙ = h̃ₘₐₓ", fontsize = 10)
    text!(ax1, 0.1, 14.0, text = "h̃ₘᵢₙ = 0", fontsize = 10)
    text!(ax1, 0.5, 25.0, text = "h̃ₘᵢₙ = h̃ₘₐₓ", fontsize = 10)
    text!(ax1, 0.05, 29.0, text = "h̃ₘᵢₙ = 0", fontsize = 10)

    # Add V(z) label on wind panel
    text!(ax2, 45.0, 12.0, text = "V(z)", fontsize = 12)

    save(output_file, fig)
    println("  Saved: $output_file")

    return fig
end

# Use existing remap infrastructure
output_dir = "orographic_gravity_wave_test_garner2005"
mkpath(output_dir)

# Configure remapping (shared by quiver_plots and figure4)
config_remap = PlotConfig(
    plot_mode = :horizontal_slice,
    contour_levels = 20,
    nlat = 180,
    nlon = 360,
    yreversed = false,
)

if should_run(:quiver_plots)
    # Prepare field data for Figure 1
    field_data_fig1 = Dict("τ_x" => τ_x_fig1_cpu, "τ_y" => τ_y_fig1_cpu)

    # Prepare field data for Figure 2
    field_data_fig2 = Dict("τ_x" => τ_x_fig2_cpu, "τ_y" => τ_y_fig2_cpu)

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
end

#######################################
# FIGURE 4: Linear Drag Comparison
# Three panels showing:
#   Top: Analytical drag D = (ρ̄/ρr) * V̄ * max(|T₁|, |T₂|)
#   Middle: Linear drag τ_l from GFDL restart topo_info
#   Bottom: Linear drag τ_l from raw_topo
# All with uniform wind V̄ = 10 m/s, ρ̄/ρr = 1.0
#######################################

if should_run(:figure4)
    println("\n" * "="^70)
    println("FIGURE 4: Linear Drag Comparison")
    println("="^70)

    # Set uniform wind for Figure 4 (10 m/s zonal)
    @. u_phy = FT(10.0)
    @. v_phy = FT(0.0)
    @. Y.c.ρ = FT(1.0)
    @. ᶜbuoyancy_frequency = FT(0.01)

    #--- Panel 1: Analytical scalar drag from eigenvalues ---
    println("\nComputing analytical scalar drag (eigenvalue method)...")

    # Compute max absolute eigenvalue of the tensor from raw_topo
    max_eigenvalue = compute_tensor_eigenvalue_max(
        topo_info.t11,
        topo_info.t12,
        topo_info.t21,
        topo_info.t22,
    )

    # Analytical drag: D = (ρ̄/ρr) * V̄ * max(|T₁|, |T₂|)
    # With ρ̄/ρr = 1.0 and V̄ = 10.0 m/s
    V_mag = FT(10.0)  # Wind magnitude
    ρ_ratio = FT(1.0)  # Density ratio
    analytical_drag = similar(max_eigenvalue)
    @. analytical_drag = ρ_ratio * V_mag * max_eigenvalue

    analytical_drag_cpu = to_cpu(analytical_drag)
    println(
        "  Analytical drag range: $(minimum(parent(analytical_drag_cpu))) to $(maximum(parent(analytical_drag_cpu)))",
    )

    #--- Panel 2: Linear drag from GFDL restart ---
    println("\nComputing linear drag from GFDL restart...")

    # Create OGW configuration for GFDL restart
    γ_gfdl = FT(0.4)
    ϵ_gfdl = FT(0.0)
    β_gfdl = FT(0.5)
    h_frac_gfdl = FT(0.1)
    ρscale_gfdl = FT(1.2)
    L0_gfdl = FT(80000.0)
    a0_gfdl = FT(0.9)
    a1_gfdl = FT(3.0)
    Fr_crit_gfdl = FT(0.7)
    topo_info_type_gfdl = Val(:gfdl_restart)
    topography_type_gfdl = Val(:Earth)

    ogw_gfdl = CA.FullOrographicGravityWave{
        FT,
        typeof(topo_info_type_gfdl),
        typeof(topography_type_gfdl),
    }(;
        γ = γ_gfdl,
        ϵ = ϵ_gfdl,
        β = β_gfdl,
        h_frac = h_frac_gfdl,
        ρscale = ρscale_gfdl,
        L0 = L0_gfdl,
        a0 = a0_gfdl,
        a1 = a1_gfdl,
        Fr_crit = Fr_crit_gfdl,
        topo_info = topo_info_type_gfdl,
        topography = topography_type_gfdl,
    )

    # Load GFDL topo_info
    topo_info_gfdl = CA.get_topo_info(Y, ogw_gfdl)

    # Create scratch fields for GFDL computation
    τ_x_gfdl = similar(topo_τ_x)
    τ_y_gfdl = similar(topo_τ_y)
    τ_l_gfdl = similar(topo_τ_l)
    τ_p_gfdl = similar(topo_τ_p)
    τ_np_gfdl = similar(topo_τ_np)
    U_sat_gfdl = similar(topo_U_sat)
    FrU_sat_gfdl = similar(topo_FrU_sat)
    FrU_clp_gfdl = similar(topo_FrU_clp)
    FrU_max_gfdl = similar(topo_FrU_max)
    FrU_min_gfdl = similar(topo_FrU_min)

    # Compute linear drag from GFDL data
    CA.calc_base_flux!(
        τ_x_gfdl,
        τ_y_gfdl,
        τ_l_gfdl,
        τ_p_gfdl,
        τ_np_gfdl,
        U_sat_gfdl,
        FrU_sat_gfdl,
        FrU_clp_gfdl,
        FrU_max_gfdl,
        FrU_min_gfdl,
        topo_ᶜz_pbl,
        values_at_z_pbl,
        ogw_params,
        topo_info_gfdl,
        Y.c.ρ,
        u_phy,
        v_phy,
        ᶜz,
        ᶜbuoyancy_frequency,
    )

    τ_l_gfdl_cpu = to_cpu(τ_l_gfdl)
    println(
        "  GFDL τ_l range: $(minimum(parent(τ_l_gfdl_cpu))) to $(maximum(parent(τ_l_gfdl_cpu)))",
    )

    #--- Panel 3: Linear drag from raw_topo ---
    println("\nComputing linear drag from raw_topo...")

    # Setup simulation - try local raw_topo first, fall back to artifact
    (; config_file, job_id) = CA.commandline_kwargs()
    simulation, config = create_ogw_simulation(config_file, job_id, comms_ctx)
    (; parsed_args) = config
    p = simulation.integrator.p
    Y = simulation.integrator.u

    # Prepare physical uv input variables
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2
    ᶜz = Fields.coordinate_field(Y.c).z

    # Unpack cache
    (; topo_ᶜz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
        p.orographic_gravity_wave
    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; values_at_z_pbl, topo_info) = p.orographic_gravity_wave
    (; ᶜdTdz, ᶜbuoyancy_frequency) = p.orographic_gravity_wave
    # (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    (; params) = p

    # Extract parameters
    ogw_params = p.orographic_gravity_wave.ogw_params
    grav = CAP.grav(params)
    thermo_params = CAP.thermodynamics_params(params)

    # Compute PBL height from initial-condition pressure/temperature profiles
    cp_d = CAP.cp_d(params)
    ᶜp = p.precomputed.ᶜp
    ᶜT = p.precomputed.ᶜT
    CA.get_pbl_z!(topo_ᶜz_pbl, ᶜp, ᶜT, ᶜz, grav, cp_d)

    # Set idealized values for Figure 4 on the new simulation's fields
    @. u_phy = FT(10.0)
    @. v_phy = FT(0.0)
    @. Y.c.ρ = FT(1.0)
    @. ᶜbuoyancy_frequency = FT(0.01)

    # Compute linear drag using raw_topo (already loaded as topo_info)
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

    τ_l_raw_cpu = to_cpu(topo_τ_l)
    println(
        "  Raw topo τ_l range: $(minimum(parent(τ_l_raw_cpu))) to $(maximum(parent(τ_l_raw_cpu)))",
    )

    #--- Prepare data for Figure 4 plotting ---
    println("\nRemapping Figure 4 data...")

    field_data_fig4 = Dict(
        "D_analytical" => analytical_drag_cpu,
        "τ_l_gfdl" => τ_l_gfdl_cpu,
        "τ_l_raw" => τ_l_raw_cpu,
    )

    remap_dir_fig4 = joinpath(output_dir, "remap_fig4/")
    datafile_fig4 = remap_to_latlon(
        remap_dir_fig4,
        ["D_analytical", "τ_l_gfdl", "τ_l_raw"],
        field_data_fig4,
        Y_cpu,
        ᶜspace;
        config = config_remap,
        FT = FT,
    )

    #--- Create Figure 4 tripanel plot ---
    println("\n" * "="^70)
    println("Creating Figure 4 tripanel plot...")
    println("="^70)

    plot_garner_fig4_tripanel(
        joinpath(output_dir, "garner_fig4_linear_drag.png"),
        datafile_fig4,
        ["D_analytical", "τ_l_gfdl", "τ_l_raw"],
        [
            "Analytical: D = V̄ max(|T₁|, |T₂|)",
            "GFDL Restart: τ_l",
            "Raw Topography: τ_l",
        ];
        lon_range = (0.0, 360.0),
        lat_range = (-90.0, 90.0),
        colormap = :cividis,
    )
end

#######################################
# FIGURE 5: Normalized Drag Curves
# Shows D/D* vs h_max/h_c for:
#   - Propagating drag ⟨D_p⟩ (dashed lines)
#   - Total drag (D_p + D_np) (solid lines)
#   - Two cases: h_min = 0 and h_min = h_max
# Parameters: γ = 0.4, β = 0.5, ε = 0, a_1/a_0 = 9.0
#######################################

if should_run(:figure5)
    println("\n" * "="^70)
    println("FIGURE 5: Normalized Drag Curves")
    println("="^70)

    # Physical parameters for Figure 5
    V_fig5 = FT(10.0)   # Wind speed (m/s)
    N_fig5 = FT(0.01)   # Buoyancy frequency (s⁻¹)
    ρ_fig5 = FT(1.0)    # Density (kg/m³)

    # OGW parameters for Figure 5 (from paper)
    γ_fig5 = FT(0.4)
    β_fig5 = FT(0.5)
    ϵ_fig5 = FT(0.0)
    Fr_crit_fig5 = FT(0.7)
    a0_fig5 = FT(0.9)
    a1_fig5 = FT(9.0) * a0_fig5  # a_1/a_0 = 9.0, so a_1 = 8.1

    # Other OGW parameters (use existing values)
    h_frac_fig5 = FT(0.1)
    ρscale_fig5 = FT(1.2)
    L0_fig5 = FT(1.0)

    # Critical height: h_c = Fr_crit * V / N
    h_c = Fr_crit_fig5 * V_fig5 / N_fig5  # = 0.7 * 10 / 0.01 = 700 m
    println("  Critical height h_c = $(h_c) m")

    # Range of h_max/h_c values (x-axis)
    n_points = 100
    x_range = range(FT(0.05), FT(5.0), length = n_points)  # Start from 0.05 to avoid div by zero

    # Storage arrays for the four curves
    Dp_hmin0 = zeros(FT, n_points)
    Dp_hmin_eq_hmax = zeros(FT, n_points)
    total_hmin0 = zeros(FT, n_points)
    total_hmin_eq_hmax = zeros(FT, n_points)

    println("  Computing drag curves for $(n_points) points using calc_base_flux!...")

    # Create ogw_params for Figure 5 (override default parameters)
    ogw_params_fig5 = (;
        Fr_crit = Fr_crit_fig5,
        topo_γ = γ_fig5,
        topo_β = β_fig5,
        topo_ϵ = ϵ_fig5,
        topo_a0 = a0_fig5,
        topo_a1 = a1_fig5,
        topo_ρscale = ρscale_fig5,
        topo_L0 = L0_fig5,
    )

    # Save original topo_info values (to restore later)
    orig_hmax = copy(parent(topo_info.hmax))
    orig_hmin = copy(parent(topo_info.hmin))
    orig_t11 = copy(parent(topo_info.t11))
    orig_t12 = copy(parent(topo_info.t12))
    orig_t21 = copy(parent(topo_info.t21))
    orig_t22 = copy(parent(topo_info.t22))

    # Set identity tensor so Vτ = V (wind projects directly without rotation)
    @. topo_info.t11 = FT(-1.0)
    @. topo_info.t22 = FT(-1.0)
    @. topo_info.t12 = FT(0.0)
    @. topo_info.t21 = FT(0.0)

    # Set constant atmospheric state for Figure 5
    @. u_phy = V_fig5
    @. v_phy = FT(0.0)
    @. Y.c.ρ = ρ_fig5
    @. ᶜbuoyancy_frequency = N_fig5

    # Set values_at_z_pbl directly for Figure 5 (bypasses column_reduce!)
    # Format: (ρ_pbl, u_pbl, v_pbl, N_pbl)
    @. values_at_z_pbl.:1 = ρ_fig5      # ρ = 1.0
    @. values_at_z_pbl.:2 = V_fig5      # u = 10.0
    @. values_at_z_pbl.:3 = FT(0.0)     # v = 0.0
    @. values_at_z_pbl.:4 = N_fig5      # N = 0.01

    # Loop over h_max/h_c values using calc_base_flux!
    for (i, x) in enumerate(x_range)
        local h_max_val = x * h_c

        #--- Case A: h_min = 0 (broadband) ---
        @. topo_info.hmax = h_max_val
        @. topo_info.hmin = FT(0.0)
        if DEBUG_PRINTS
            println(
                "hmax range after set: ",
                extrema(parent(topo_info.hmax)),
                " expected: ",
                h_max_val,
            )
            println(
                "hmin range after set: ",
                extrema(parent(topo_info.hmin)),
                " expected: 0",
            )
        end

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
            ogw_params_fig5,
            topo_info,
            Y.c.ρ,
            u_phy,
            v_phy,
            ᶜz,
            ᶜbuoyancy_frequency,
        )

        if DEBUG_PRINTS
            println("FrU_max range: ", extrema(parent(topo_FrU_max)))
            println("FrU_min range: ", extrema(parent(topo_FrU_min)))
            println("U_sat range: ", extrema(parent(topo_U_sat)))
            println("topo_τ_x range: ", extrema(parent(topo_τ_x)))
            println("values_at_z_pbl: ", values_at_z_pbl)
            println("topo_ᶜz_pbl range: ", extrema(parent(topo_ᶜz_pbl)))
            println("τ_l range: ", extrema(parent(topo_τ_l)))
            println("u_phy range: ", extrema(parent(u_phy)))
            println("Y.c.ρ range: ", extrema(parent(Y.c.ρ)))
        end

        # Extract mean values (should be uniform across all columns)
        τ_l_val = mean(parent(topo_τ_l))
        τ_p_val = mean(parent(topo_τ_p))
        τ_np_val = mean(parent(topo_τ_np))

        # Normalize by D* = τ_l
        D_star = τ_l_val
        if D_star > eps(FT)
            Dp_hmin0[i] = τ_p_val / D_star
            total_hmin0[i] = (τ_p_val + τ_np_val) / D_star
        else
            Dp_hmin0[i] = FT(0.0)
            total_hmin0[i] = FT(0.0)
        end

        #--- Case B: h_min = h_max (monochromatic) ---
        @. topo_info.hmin = h_max_val - FT(100.0)

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
            ogw_params_fig5,
            topo_info,
            Y.c.ρ,
            u_phy,
            v_phy,
            ᶜz,
            ᶜbuoyancy_frequency,
        )

        # Extract mean values
        τ_l_mono = mean(parent(topo_τ_l))
        τ_p_mono = mean(parent(topo_τ_p))
        τ_np_mono = mean(parent(topo_τ_np))

        # Normalize by D*_mono = τ_l_mono for monochromatic case
        D_star_mono = τ_l_mono
        if D_star_mono > eps(FT)
            Dp_hmin_eq_hmax[i] = τ_p_mono / D_star_mono
            total_hmin_eq_hmax[i] = (τ_p_mono + τ_np_mono) / D_star_mono
        else
            Dp_hmin_eq_hmax[i] = FT(0.0)
            total_hmin_eq_hmax[i] = FT(0.0)
        end
    end

    # Restore original topo_info values
    parent(topo_info.hmax) .= orig_hmax
    parent(topo_info.hmin) .= orig_hmin
    parent(topo_info.t11) .= orig_t11
    parent(topo_info.t12) .= orig_t12
    parent(topo_info.t21) .= orig_t21
    parent(topo_info.t22) .= orig_t22

    if DEBUG_PRINTS
        println("  Done computing drag curves.")
        println("  ⟨Dp⟩/D* range (h_min=0): $(minimum(Dp_hmin0)) to $(maximum(Dp_hmin0))")
        println(
            "  ⟨Dp⟩/D* range (h_min=h_max): $(minimum(Dp_hmin_eq_hmax)) to $(maximum(Dp_hmin_eq_hmax))",
        )
        println(
            "  Total/D* range (h_min=0): $(minimum(total_hmin0)) to $(maximum(total_hmin0))",
        )
        println(
            "  Total/D* range (h_min=h_max): $(minimum(total_hmin_eq_hmax)) to $(maximum(total_hmin_eq_hmax))",
        )
    end

    # Create Figure 5 plot
    println("\n" * "="^70)
    println("Creating Figure 5 plot...")
    println("="^70)

    plot_garner_fig5(
        joinpath(output_dir, "garner_fig5_normalized_drag.png"),
        collect(x_range),
        Dp_hmin0,
        Dp_hmin_eq_hmax,
        total_hmin0,
        total_hmin_eq_hmax;
        γ = γ_fig5,
        β = β_fig5,
    )

    println("\n" * "="^70)
    println("Garner 2005 reproduction complete!")
    println("Output directory: $output_dir")
    println("="^70)
end

#######################################
# FIGURE 6: Total Drag and Nonlinear Fraction over Americas
# Two panels showing:
#   Left: Total base flux normalized by (D* + 0.05)
#   Right: Nonpropagating fraction τ_np / (τ_p + τ_np)
# Parameters: γ = 0.4, β = 0.5, ε = 0, ĥ_c = 0.7
# Wind: uniform zonal 10 m/s
# Density/stability: ρ = 1.0 kg/m³, N = 0.01 s⁻¹
#######################################

if should_run(:figure6)
    println("\n" * "="^70)
    println("FIGURE 6: Total Drag and Nonlinear Fraction over Americas")
    println("="^70)

    # Set wind profile: uniform 10 m/s zonal (same as Figure 2)
    @. u_phy = FT(10.0)
    @. v_phy = FT(0.0)

    # Set density and buoyancy frequency (same as Figure 1)
    @. Y.c.ρ = FT(1.0)
    @. ᶜbuoyancy_frequency = FT(0.01)

    # Compute base flux
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

    # Compute normalized total drag: (τ_p + τ_np) / (D* + 0.05)
    # D* = τ_l (small-amplitude/linear drag limit)
    # The +0.05 offset masks regions of weak forcing
    total_drag_normalized = similar(topo_τ_l)
    @. total_drag_normalized = (topo_τ_p + topo_τ_np) / (topo_τ_l + FT(0.05))

    # Compute nonlinear (nonpropagating) fraction: τ_np / (τ_p + τ_np)
    nonlinear_fraction = similar(topo_τ_l)
    total_flux = @. topo_τ_p + topo_τ_np
    @. nonlinear_fraction = topo_τ_np / (topo_τ_l + FT(0.05))#max(total_flux, eps(FT))

    # Print diagnostic info
    total_drag_cpu = to_cpu(total_drag_normalized)
    nonlinear_frac_cpu = to_cpu(nonlinear_fraction)
    println(
        "  Total drag normalized range: $(minimum(parent(total_drag_cpu))) to $(maximum(parent(total_drag_cpu)))",
    )
    println(
        "  Nonlinear fraction range: $(minimum(parent(nonlinear_frac_cpu))) to $(maximum(parent(nonlinear_frac_cpu)))",
    )

    # Prepare field data for remapping
    field_data_fig6 = Dict(
        "total_drag" => total_drag_cpu,
        "nonlinear_frac" => nonlinear_frac_cpu,
    )

    # Remap Figure 6 data
    println("\nRemapping Figure 6 data...")
    remap_dir_fig6 = joinpath(output_dir, "remap_fig6/")
    datafile_fig6 = remap_to_latlon(
        remap_dir_fig6,
        ["total_drag", "nonlinear_frac"],
        field_data_fig6,
        Y_cpu,
        ᶜspace;
        config = config_remap,
        FT = FT,
    )

    # Create Figure 6 twopanel plot
    println("\n" * "="^70)
    println("Creating Figure 6 twopanel plot...")
    println("="^70)

    plot_garner_fig6_twopanel(
        joinpath(output_dir, "garner_fig6_drag_and_nonlinear.png"),
        datafile_fig6,
        ["total_drag", "nonlinear_frac"],
        ["Total Drag", "Nonlinear Fraction"],
        [(0.0, 2.0), (0.0, 1.0)];  # Colorbar ranges matching paper
        lon_range = (200.0, 330.0),  # Americas region: 160°W to 30°W in 0-360 format
        lat_range = (-40.0, 70.0),
        colormap = :grays,
    )

    println("\n" * "="^70)
    println("Garner 2005 Figure 6 reproduction complete!")
    println("Output directory: $output_dir")
    println("="^70)
end

#######################################
# FIGURE 7: Vertical Propagation Profile
# Shows D/D* and ρ∂V/∂t vs height for:
#   - h_min = 0 (broadband)
#   - h_min = h_max (monochromatic)
# Parameters: γ = 0.4, β = 0.5, ε = 0, h_max = 1.2h_c
# Wind: jets at 38 m/s (9 km) and 58 m/s (25 km)
# Stability: N = 0.011 below 11 km, 0.022 above
# Density scale height: H = 8 km
#######################################

if should_run(:figure7)
    println("\n" * "="^70)
    println("FIGURE 7: Vertical Propagation Profile")
    println("="^70)

    # Setup simulation with higher vertical resolution for Figure 7
    # Try local raw_topo first, fall back to artifact
    (; config_file, job_id) = CA.commandline_kwargs()
    simulation, config = create_ogw_simulation(
        config_file,
        job_id,
        comms_ctx;
        extra_parsed_args = Dict("z_elem" => 64, "z_max" => 45000.0),
    )
    (; parsed_args) = config
    p = simulation.integrator.p
    Y = simulation.integrator.u

    # Prepare physical uv input variables
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2
    ᶜz = Fields.coordinate_field(Y.c).z

    # Unpack cache
    (; topo_ᶜz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
        p.orographic_gravity_wave
    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; values_at_z_pbl, topo_info) = p.orographic_gravity_wave
    (; ᶜdTdz, ᶜbuoyancy_frequency) = p.orographic_gravity_wave
    # (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    (; params) = p

    # Extract parameters
    ogw_params = p.orographic_gravity_wave.ogw_params
    grav = CAP.grav(params)
    thermo_params = CAP.thermodynamics_params(params)

    # Parameters from paper
    H_scale = FT(8000.0)    # Density scale height (m)
    Fr_crit_fig7 = FT(0.7)  # Critical Froude number
    γ_fig7 = FT(0.4)
    β_fig7 = FT(0.5)
    ϵ_fig7 = FT(0.0)
    a0_fig7 = FT(0.9)
    a1_fig7 = FT(8.1)       # a_1/a_0 = 9.0

    # Surface conditions for h_c calculation
    V_surf = garner_fig7_wind_profile(FT(0.0))
    N_surf = garner_fig7_stability_profile(FT(0.0))
    h_c = Fr_crit_fig7 * V_surf / N_surf
    h_max_val = FT(1.2) * h_c  # h̃_max = 1.2h_c

    println("  Surface wind V_surf = $(V_surf) m/s")
    println("  Surface stability N_surf = $(N_surf) s⁻¹")
    println("  Critical height h_c = $(h_c) m")
    println("  Mountain height h_max = $(h_max_val) m (1.2 × h_c)")

    # Get coordinate fields
    ᶜz_field = Fields.coordinate_field(Y.c).z
    ᶠz_field = Fields.coordinate_field(Y.f).z

    # Set up idealized profiles in 3D fields
    # Note: broadcast over the z coordinate field
    @. u_phy = garner_fig7_wind_profile(ᶜz_field)
    @. v_phy = FT(0.0)
    @. Y.c.ρ = garner_fig7_density_profile(ᶜz_field)
    @. ᶜbuoyancy_frequency = garner_fig7_stability_profile(ᶜz_field)

    # Set identity tensor so Vτ = V (wind projects directly without rotation)
    # Save original values
    orig_hmax = copy(parent(topo_info.hmax))
    orig_hmin = copy(parent(topo_info.hmin))
    orig_t11 = copy(parent(topo_info.t11))
    orig_t12 = copy(parent(topo_info.t12))
    orig_t21 = copy(parent(topo_info.t21))
    orig_t22 = copy(parent(topo_info.t22))

    @. topo_info.t11 = FT(-1.0)
    @. topo_info.t22 = FT(-1.0)
    @. topo_info.t12 = FT(0.0)
    @. topo_info.t21 = FT(0.0)

    # Set values_at_z_pbl directly for idealized test
    # We use surface values since we want the wave to start from the surface
    @. values_at_z_pbl.:1 = garner_fig7_density_profile(FT(0.0))   # ρ_pbl
    @. values_at_z_pbl.:2 = V_surf                                   # u_pbl
    @. values_at_z_pbl.:3 = FT(0.0)                                  # v_pbl
    @. values_at_z_pbl.:4 = N_surf                                   # N_pbl

    # OGW parameters for Figure 7
    ogw_params_fig7 = (;
        Fr_crit = Fr_crit_fig7,
        topo_γ = γ_fig7,
        topo_β = β_fig7,
        topo_ϵ = ϵ_fig7,
        topo_a0 = a0_fig7,
        topo_a1 = a1_fig7,
        topo_ρscale = FT(1.2),
        topo_L0 = FT(80000.0),
    )

    # Get additional cache fields needed for saturation profile
    (; topo_ᶜτ_sat, topo_ᶠτ_sat, topo_ᶠVτ) = p.orographic_gravity_wave
    (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

    # Use idealized pressure field (consistent with idealized density)
    ᶜp_idealized = similar(Y.c.ρ)
    @. ᶜp_idealized = garner_fig7_pressure_profile(ᶜz_field)
    ᶜp = ᶜp_idealized

    # Extract height array for plotting (convert to km)
    # Use VIJFH indexing: [:,1,1,1,1] extracts first column (all vertical levels)
    nlevels = Spaces.nlevels(axes(Y.c))
    z_km = parent(ᶜz_field)[:, 1, 1, 1, 1] ./ 1000.0  # Convert to km

    # Storage for profiles
    flux_hmin0 = zeros(FT, nlevels)
    flux_hmin_eq_hmax = zeros(FT, nlevels)
    forcing_hmin0 = zeros(FT, nlevels)
    forcing_hmin_eq_hmax = zeros(FT, nlevels)

    # Extract wind profile from first column
    wind_profile_vec = parent(u_phy)[:, 1, 1, 1, 1]

    println("  Number of vertical levels: $nlevels")
    println("  Heights (km): $(round.(z_km, digits=1))")
    println("  Height range: $(z_km[1]) to $(z_km[end]) km")
    println(
        "  Wind range: $(minimum(wind_profile_vec)) to $(maximum(wind_profile_vec)) m/s",
    )
    println("  Wind profile at each level (m/s):")
    for (k, (z, w)) in enumerate(zip(z_km, wind_profile_vec))
        println("    Level $k: z=$(round(z, digits=2)) km, V=$(round(w, digits=1)) m/s")
    end

    println("\n  Computing Case A: h_min = 0 (broadband)...")

    #--- Case A: h_min = 0 (broadband) ---
    @. topo_info.hmax = h_max_val
    @. topo_info.hmin = FT(0.0)
    # Set PBL height high enough to include the first model level for all columns
    # With terrain-following coordinates, z varies by column - use a high value
    @. topo_ᶜz_pbl = FT(2000.0)

    # Debug: check z range at first level across all columns
    z_first_level = parent(ᶜz_field)[1, :, :, :, :]
    println("    First model level z range: $(extrema(z_first_level)) m")
    println("    PBL height set to: 2000.0 m")

    # Compute base flux
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
        ogw_params_fig7,
        topo_info,
        Y.c.ρ,
        u_phy,
        v_phy,
        ᶜz_field,
        ᶜbuoyancy_frequency,
    )

    # Debug: check values_at_z_pbl after calc_base_flux!
    ρ_pbl_vals = parent(values_at_z_pbl.:1)
    println("    ρ_pbl range: $(extrema(ρ_pbl_vals))")

    τ_l_A = mean(parent(topo_τ_l))
    τ_p_A = mean(parent(topo_τ_p))
    τ_np_A = mean(parent(topo_τ_np))
    println("    τ_l = $(τ_l_A), τ_p = $(τ_p_A), τ_np = $(τ_np_A)")
    println("    τ_p/τ_l = $(τ_p_A/τ_l_A) (should be ~1 for unsaturated, <1 for saturated)")
    println(
        "    FrU_max = $(mean(parent(topo_FrU_max))), FrU_min = $(mean(parent(topo_FrU_min)))",
    )
    println(
        "    FrU_sat = $(mean(parent(topo_FrU_sat))), FrU_clp = $(mean(parent(topo_FrU_clp)))",
    )

    # Compute saturation profile
    CA.calc_saturation_profile!(
        topo_ᶠτ_sat,
        topo_ᶠVτ,
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜτ_sat,
        topo_τ_x,
        topo_τ_y,
        topo_τ_p,
        topo_ᶜz_pbl,
        ogw_params_fig7,
        Y.c.ρ,
        u_phy,
        v_phy,
        ᶜp,
        ᶜbuoyancy_frequency,
        ᶜz_field,
    )

    # Extract saturation profile using VIJFH indexing (first column)
    τ_sat_col_A = parent(topo_ᶜτ_sat)[:, 1, 1, 1, 1]
    println(
        "    τ_sat profile: surface=$(τ_sat_col_A[1]), mid=$(τ_sat_col_A[nlevels÷2]), top=$(τ_sat_col_A[end])",
    )

    # Compute forcing
    fill!(ᶜuforcing, FT(0.0))
    fill!(ᶜvforcing, FT(0.0))
    dτ_sat_dz = similar(Y.c.ρ)

    CA.calc_propagate_forcing!(
        ᶜuforcing,
        ᶜvforcing,
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_ᶠτ_sat,
        dτ_sat_dz,
        Y.c.ρ,
    )

    # Extract forcing using VIJFH indexing
    forcing_col_A = parent(ᶜuforcing)[:, 1, 1, 1, 1]

    # Normalize flux: D/D* = τ_sat(z) / τ_sat(surface)
    τ_sat_surface_A = τ_sat_col_A[1]
    if τ_sat_surface_A > eps(FT)
        flux_hmin0 .= τ_sat_col_A ./ τ_sat_surface_A
    end

    # Normalize forcing by H_scale
    forcing_hmin0 .= forcing_col_A .* H_scale

    println("    Flux range: $(minimum(flux_hmin0)) to $(maximum(flux_hmin0))")
    println("    Forcing range: $(minimum(forcing_hmin0)) to $(maximum(forcing_hmin0))")

    println("\n  Computing Case B: h_min = h_max (monochromatic)...")

    #--- Case B: h_min = h_max (monochromatic) ---
    @. topo_info.hmin = h_max_val - FT(1.0)  # Small offset to avoid division issues

    # Compute base flux
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
        ogw_params_fig7,
        topo_info,
        Y.c.ρ,
        u_phy,
        v_phy,
        ᶜz_field,
        ᶜbuoyancy_frequency,
    )

    τ_l_B = mean(parent(topo_τ_l))
    τ_p_B = mean(parent(topo_τ_p))
    τ_np_B = mean(parent(topo_τ_np))
    println("    τ_l = $(τ_l_B), τ_p = $(τ_p_B), τ_np = $(τ_np_B)")
    println("    τ_p/τ_l = $(τ_p_B/τ_l_B)")

    # Compute saturation profile
    CA.calc_saturation_profile!(
        topo_ᶠτ_sat,
        topo_ᶠVτ,
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜτ_sat,
        topo_τ_x,
        topo_τ_y,
        topo_τ_p,
        topo_ᶜz_pbl,
        ogw_params_fig7,
        Y.c.ρ,
        u_phy,
        v_phy,
        ᶜp,
        ᶜbuoyancy_frequency,
        ᶜz_field,
    )

    # Extract saturation profile using VIJFH indexing (first column)
    τ_sat_col_B = parent(topo_ᶜτ_sat)[:, 1, 1, 1, 1]
    println(
        "    τ_sat profile: surface=$(τ_sat_col_B[1]), mid=$(τ_sat_col_B[nlevels÷2]), top=$(τ_sat_col_B[end])",
    )

    # Compute forcing
    fill!(ᶜuforcing, FT(0.0))
    fill!(ᶜvforcing, FT(0.0))

    CA.calc_propagate_forcing!(
        ᶜuforcing,
        ᶜvforcing,
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_ᶠτ_sat,
        dτ_sat_dz,
        Y.c.ρ,
    )

    # Extract forcing using VIJFH indexing
    forcing_col_B = parent(ᶜuforcing)[:, 1, 1, 1, 1]

    # Normalize flux: D/D* = τ_sat(z) / τ_sat(surface)
    τ_sat_surface_B = τ_sat_col_B[1]
    if τ_sat_surface_B > eps(FT)
        flux_hmin_eq_hmax .= τ_sat_col_B ./ τ_sat_surface_B
    end

    # Normalize forcing by H_scale
    forcing_hmin_eq_hmax .= forcing_col_B .* H_scale

    println(
        "    Flux range: $(minimum(flux_hmin_eq_hmax)) to $(maximum(flux_hmin_eq_hmax))",
    )
    println(
        "    Forcing range: $(minimum(forcing_hmin_eq_hmax)) to $(maximum(forcing_hmin_eq_hmax))",
    )

    # Restore original topo_info values
    parent(topo_info.hmax) .= orig_hmax
    parent(topo_info.hmin) .= orig_hmin
    parent(topo_info.t11) .= orig_t11
    parent(topo_info.t12) .= orig_t12
    parent(topo_info.t21) .= orig_t21
    parent(topo_info.t22) .= orig_t22

    println("\n  Profile ranges:")
    println("    Flux (h_min=0): $(minimum(flux_hmin0)) to $(maximum(flux_hmin0))")
    println(
        "    Flux (h_min=h_max): $(minimum(flux_hmin_eq_hmax)) to $(maximum(flux_hmin_eq_hmax))",
    )
    println("    Forcing (h_min=0): $(minimum(forcing_hmin0)) to $(maximum(forcing_hmin0))")
    println(
        "    Forcing (h_min=h_max): $(minimum(forcing_hmin_eq_hmax)) to $(maximum(forcing_hmin_eq_hmax))",
    )

    # Create Figure 7 plot
    println("\n" * "="^70)
    println("Creating Figure 7 plot...")
    println("="^70)

    plot_garner_fig7(
        joinpath(output_dir, "garner_fig7_vertical_profile.png"),
        collect(z_km),
        collect(flux_hmin0),
        collect(flux_hmin_eq_hmax),
        collect(forcing_hmin0),
        collect(forcing_hmin_eq_hmax),
        collect(wind_profile_vec),
    )

    println("\n" * "="^70)
    println("Garner 2005 Figure 7 reproduction complete!")
    println("Output directory: $output_dir")
    println("="^70)
end
