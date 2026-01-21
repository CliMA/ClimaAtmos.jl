"""
Generic utilities for remapping ClimaCore fields to lat/lon grids and creating
multi-panel comparison plots.

Usage example:
```julia
# Define panels for a 2x2 comparison plot
panels = [
    PlotPanel("ogwd_u", "ClimaAtmos u-forcing", (1,1); scale_factor=86400),
    PlotPanel("gfdl_udt_topo", "GFDL u-forcing", (1,2); scale_factor=86400),
    PlotPanel("ogwd_v", "ClimaAtmos v-forcing", (2,1); scale_factor=86400),
    PlotPanel("gfdl_vdt_topo", "GFDL v-forcing", (2,2); scale_factor=86400),
]

# Create plot configuration
plot_config = PlotConfig(
    vertical_levels = [21, 31],
    contour_levels = range(-10, 10; length=20),
)

# Run the full pipeline
remap_and_plot_3d(
    output_dir,
    variables_to_remap,
    field_data,
    Y_cpu,
    ᶜspace,
    panels,
    plot_config
)
```
"""

import ClimaAtmos
using NCDatasets
using CairoMakie
using Dates

include(
    joinpath(pkgdir(ClimaAtmos), "post_processing/remap", "remap_helpers.jl"),
)
include(joinpath(@__DIR__, "gw_plotutils.jl"))

"""
    PlotPanel

Specification for a single panel in a multi-panel plot.

# Fields
- `variable_name::String`: Name of the variable in the NetCDF file
- `title_template::String`: Title template (can include {z} placeholder for vertical level)
- `position::Tuple{Int,Int}`: Grid position (row, col) for this panel
- `scale_factor::Float64`: Multiplicative scaling factor for the data (default: 1.0)
- `offset::Float64`: Additive offset for the data (default: 0.0)
- `custom_process::Union{Function,Nothing}`: Optional custom processing function
- `colormap::Symbol`: Colormap for this panel (default: :vik100)
- `colorrange::Union{Nothing,Tuple}`: Override color range for this panel (default: nothing)
- `extendhigh::Symbol`: Color for values above range (default: :magenta)
- `extendlow::Symbol`: Color for values below range (default: :cyan)
"""
struct PlotPanel
    variable_name::String
    title_template::String
    position::Tuple{Int, Int}
    scale_factor::Float64
    offset::Float64
    custom_process::Union{Function, Nothing}
    colormap::Symbol
    colorrange::Union{Nothing, Tuple}
    extendhigh::Symbol
    extendlow::Symbol
end

function PlotPanel(
    variable_name::String,
    title_template::String,
    position::Tuple{Int, Int};
    scale_factor::Real = 1.0,
    offset::Real = 0.0,
    custom_process::Union{Function, Nothing} = nothing,
    colormap::Symbol = :vik100,
    colorrange::Union{Nothing, Tuple} = nothing,
    extendhigh::Symbol = :magenta,
    extendlow::Symbol = :cyan,
)
    return PlotPanel(
        variable_name,
        title_template,
        position,
        Float64(scale_factor),
        Float64(offset),
        custom_process,
        colormap,
        colorrange,
        extendhigh,
        extendlow,
    )
end

"""
    PlotConfig

Configuration for the entire plotting pipeline.

# Fields
- `plot_mode::Symbol`: Either `:horizontal_slice` (2D fields) or `:vertical_slice` (3D fields with k-indices)
- `vertical_levels::Vector{Int}`: Which vertical levels (k-indices) to plot (only for :vertical_slice mode)
- `contour_levels::Union{Int,AbstractRange}`: Number or range of contour levels
- `nlat::Int`: Number of latitude points in remapped grid (default: 90)
- `nlon::Int`: Number of longitude points in remapped grid (default: 180)
- `figure_size::Tuple{Int,Int}`: Figure size in pixels (default: (2000, 2000))
- `figure_bgcolor::Tuple{Float64,Float64,Float64}`: Background color (default: (0.98, 0.98, 0.98))
- `fontsize::Int`: Font size (default: 40)
- `yreversed::Bool`: Reverse y-axis (default: false)
- `output_prefix::String`: Prefix for output files (default: "plot")
- `output_format::String`: Output format (default: "png")
- `remap_mono::Bool`: Use monotone remapping (default: true)
- `cleanup_remap_files::Bool`: Delete intermediate remap files (default: false)
"""
struct PlotConfig
    plot_mode::Symbol
    vertical_levels::Vector{Int}
    contour_levels::Union{Int, AbstractRange}
    nlat::Int
    nlon::Int
    figure_size::Tuple{Int, Int}
    figure_bgcolor::Tuple{Float64, Float64, Float64}
    fontsize::Int
    yreversed::Bool
    output_prefix::String
    output_format::String
    remap_mono::Bool
    cleanup_remap_files::Bool
end

function PlotConfig(;
    plot_mode::Symbol = :vertical_slice,
    vertical_levels::Vector{Int} = [1],
    contour_levels::Union{Int, AbstractRange} = 25,
    nlat::Int = 90,
    nlon::Int = 180,
    figure_size::Tuple{Int, Int} = (2000, 2000),
    figure_bgcolor::Tuple = (0.98, 0.98, 0.98),
    fontsize::Int = 40,
    yreversed::Bool = false,
    output_prefix::String = "plot",
    output_format::String = "png",
    remap_mono::Bool = true,
    cleanup_remap_files::Bool = false,
)
    # Validate plot_mode
    if !(plot_mode in [:horizontal_slice, :vertical_slice])
        throw(ArgumentError("plot_mode must be :horizontal_slice or :vertical_slice"))
    end

    return PlotConfig(
        plot_mode,
        vertical_levels,
        contour_levels,
        nlat,
        nlon,
        figure_size,
        (
            Float64(figure_bgcolor[1]),
            Float64(figure_bgcolor[2]),
            Float64(figure_bgcolor[3]),
        ),
        fontsize,
        yreversed,
        output_prefix,
        output_format,
        remap_mono,
        cleanup_remap_files,
    )
end

"""
    remap_to_latlon(
        remap_dir::String,
        variable_names::Vector{String},
        field_data::Dict{String,Any},
        Y_cpu,
        ᶜspace;
        config::PlotConfig
    )

Remap ClimaCore fields to a regular lat/lon grid.

# Arguments
- `remap_dir`: Directory to store intermediate remap files
- `variable_names`: Names of variables to remap
- `field_data`: Dictionary mapping variable names to ClimaCore Field objects
- `Y_cpu`: CPU version of the state vector (for getting axes)
- `ᶜspace`: Center space for defining coordinates
- `config`: Plot configuration

# Returns
- Path to the remapped NetCDF file
"""
function remap_to_latlon(
    remap_dir::String,
    variable_names::Vector{String},
    field_data::Dict{String, <:Any},
    Y_cpu,
    ᶜspace;
    config::PlotConfig,
    FT = Float64,
)
    # Create directory if needed
    if !isdir(remap_dir)
        mkpath(remap_dir)
    end

    # Create timestamped filenames
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    datafile_cg = joinpath(remap_dir, "data_cg_$(timestamp).nc")
    weightfile = joinpath(remap_dir, "remap_weights_$(timestamp).nc")
    datafile_rll = joinpath(remap_dir, "data_rll_$(timestamp).nc")

    # Write ClimaCore fields to NetCDF
    NCDataset(datafile_cg, "c") do nc
        # Determine the appropriate space for coordinate definition
        # Use the space from the first variable to determine dimensionality
        sample_field = field_data[variable_names[1]]
        field_space = axes(sample_field)

        # Check if this is a 2D horizontal space or 3D space
        is_horizontal_only = ndims(parent(sample_field)) == 1  # 1D parent array means 2D horizontal field

        if is_horizontal_only && config.plot_mode == :horizontal_slice
            # For 2D horizontal fields, use the horizontal space
            # Extract horizontal space from the field
            def_space_coord(nc, field_space, type = "cgll")
        else
            # For 3D fields, use the full center space
            def_space_coord(nc, ᶜspace, type = "cgll")
        end

        nc_time = def_time_coord(nc)
        nc_time[1] = 1

        # Define and write each variable
        for var_name in variable_names
            if haskey(field_data, var_name)
                var_field = field_data[var_name]
                var_space = axes(var_field)

                # Write variable with appropriate space
                nc_var = defVar(nc, var_name, FT, var_space, ("time",))
                nc_var[:, 1] = var_field
            else
                @warn "Variable $var_name not found in field_data, skipping"
            end
        end
    end

    # Create weight file for remapping
    create_weightfile(
        weightfile,
        axes(Y_cpu.c),
        axes(Y_cpu.f),
        config.nlat,
        config.nlon,
        mono = config.remap_mono,
    )

    # Apply remapping
    apply_remap(datafile_rll, datafile_cg, weightfile, variable_names)

    # Cleanup if requested
    if config.cleanup_remap_files
        rm(datafile_cg; force = true)
        rm(weightfile; force = true)
    end

    return datafile_rll
end

"""
    create_multipanel_figure(
        output_path::String,
        datafile_rll::String,
        panels::Vector{PlotPanel},
        config::PlotConfig,
        vertical_level::Union{Int,Nothing} = nothing
    )

Create a multi-panel comparison figure from remapped data.

# Arguments
- `output_path`: Full path for the output figure file
- `datafile_rll`: Path to the remapped NetCDF file
- `panels`: Vector of PlotPanel specifications
- `config`: Plot configuration
- `vertical_level`: The vertical level index to plot (only for :vertical_slice mode)
"""
function create_multipanel_figure(
    output_path::String,
    datafile_rll::String,
    panels::Vector{PlotPanel},
    config::PlotConfig,
    vertical_level::Union{Int, Nothing} = nothing,
)
    # Read data from remapped file
    data = NCDataset(datafile_rll) do ds
        result = Dict{String, Any}()
        result["lon"] = Array(ds["lon"])
        result["lat"] = Array(ds["lat"])
        if haskey(ds, "z")
            result["z"] = Array(ds["z"])
        end

        # Load all variables referenced by panels
        for panel in panels
            if haskey(ds, panel.variable_name)
                result[panel.variable_name] = Array(ds[panel.variable_name])
            else
                @warn "Variable $(panel.variable_name) not found in remapped file"
            end
        end
        result
    end

    # Create figure
    fig = generate_empty_figure(;
        size = config.figure_size,
        bgcolor = config.figure_bgcolor,
        fontsize = config.fontsize,
    )

    # Get z-coordinate for title (if applicable)
    z_value = nothing
    if config.plot_mode == :vertical_slice
        if vertical_level === nothing
            throw(
                ArgumentError("vertical_level must be specified for :vertical_slice mode"),
            )
        end
        if haskey(data, "z")
            z_value = data["z"][vertical_level]
        end
    end

    # Create each panel
    for panel in panels
        if !haskey(data, panel.variable_name)
            continue
        end

        # Extract and process data for this panel
        raw_data = data[panel.variable_name]

        # Handle 2D vs 3D data based on plot mode
        if config.plot_mode == :horizontal_slice
            # For 2D horizontal slices, data is already 2D: [lon, lat, time]
            panel_data = size(raw_data, 3) > 0 ? raw_data[:, :, 1] : raw_data[:, :]
        else  # :vertical_slice
            # For 3D data, extract the specified vertical level: [lon, lat, z, time]
            panel_data = raw_data[:, :, vertical_level, 1]
        end

        # Apply scaling and offset
        panel_data = panel_data .* panel.scale_factor .+ panel.offset

        # Apply custom processing if provided
        if panel.custom_process !== nothing
            panel_data = panel.custom_process(panel_data)
        end

        # Generate title (replace {z} placeholder if present)
        if z_value !== nothing
            title =
                replace(panel.title_template, "{z}" => string(round(z_value; digits = 1)))
        else
            title = panel.title_template  # No z-value replacement for horizontal slices
        end

        # Create the plot
        create_plot!(
            fig;
            X = data["lon"],
            Y = data["lat"],
            Z = panel_data,
            levels = config.contour_levels,
            colorrange = panel.colorrange,
            title = title,
            p_loc = panel.position,
            yreversed = config.yreversed,
            colormap = panel.colormap,
            extendhigh = panel.extendhigh,
            extendlow = panel.extendlow,
        )
    end

    # Save the figure
    CairoMakie.save(output_path, fig)
    @info "Saved figure to $output_path"
end

"""
    remap_and_plot_3d(
        output_dir::String,
        variable_names::Vector{String},
        field_data::Dict{String,Any},
        Y_cpu,
        ᶜspace,
        panels::Vector{PlotPanel},
        config::PlotConfig;
        remap_dir::Union{String,Nothing} = nothing,
        FT = Float64
    )

Complete pipeline: remap ClimaCore fields and create multi-panel plots at specified
vertical levels.

# Arguments
- `output_dir`: Directory for output plot files
- `variable_names`: Names of all variables to remap
- `field_data`: Dictionary mapping variable names to ClimaCore Field objects
- `Y_cpu`: CPU version of the state vector
- `ᶜspace`: Center space for defining coordinates
- `panels`: Vector of PlotPanel specifications describing the layout
- `config`: Plot configuration
- `remap_dir`: Directory for intermediate remap files (default: output_dir/remap_data)
- `FT`: Floating point type (default: Float64)

# Example
```julia
# Prepare field data dictionary
field_data = Dict(
    "ogwd_u" => uforcing_cpu,
    "ogwd_v" => vforcing_cpu,
    "gfdl_udt_topo" => gfdl_ca_udt_topo_cpu,
    "gfdl_vdt_topo" => gfdl_ca_vdt_topo_cpu,
    "z_3d" => ᶜz_cpu,
)

# Define 2x1 comparison panels
u_panels = [
    PlotPanel("ogwd_u", "ClimaAtmos at z = {z} m", (1,1); scale_factor=86400),
    PlotPanel("gfdl_udt_topo", "GFDL at z = {z} m", (2,1); scale_factor=86400),
]

v_panels = [
    PlotPanel("ogwd_v", "ClimaAtmos at z = {z} m", (1,1); scale_factor=86400),
    PlotPanel("gfdl_vdt_topo", "GFDL at z = {z} m", (2,1); scale_factor=86400),
]

# Plot configuration
config = PlotConfig(
    vertical_levels = [21, 31],
    contour_levels = range(-10, 10; length=20),
)

# Generate u-forcing plots
remap_and_plot_3d(
    output_dir,
    collect(keys(field_data)),
    field_data,
    Y_cpu,
    ᶜspace,
    u_panels,
    config;
    output_prefix = "uforcing"
)

# Generate v-forcing plots
remap_and_plot_3d(
    output_dir,
    collect(keys(field_data)),
    field_data,
    Y_cpu,
    ᶜspace,
    v_panels,
    config;
    output_prefix = "vforcing"
)
```
"""
function remap_and_plot_3d(
    output_dir::String,
    variable_names::Vector{String},
    field_data::Dict{String, <:Any},
    Y_cpu,
    ᶜspace,
    panels::Vector{PlotPanel},
    config::PlotConfig;
    remap_dir::Union{String, Nothing} = nothing,
    output_prefix::Union{String, Nothing} = nothing,
    FT = Float64,
)
    # Set up directories
    mkpath(output_dir)
    if remap_dir === nothing
        remap_dir = joinpath(output_dir, "remap_data")
    end

    # Use custom prefix if provided, otherwise use config
    prefix = output_prefix !== nothing ? output_prefix : config.output_prefix

    # Remap to lat/lon grid
    @info "Remapping $(length(variable_names)) variables to lat/lon grid..."
    datafile_rll = remap_to_latlon(
        remap_dir,
        variable_names,
        field_data,
        Y_cpu,
        ᶜspace;
        config = config,
        FT = FT,
    )

    # Create plots based on mode
    if config.plot_mode == :horizontal_slice
        # For horizontal slices, create a single plot (no vertical levels)
        output_filename = "$(prefix).$(config.output_format)"
        output_path = joinpath(output_dir, output_filename)

        @info "Creating horizontal slice plot..."
        create_multipanel_figure(output_path, datafile_rll, panels, config, nothing)
    else  # :vertical_slice
        # For vertical slices, create plots for each vertical level
        for k in config.vertical_levels
            output_filename = "$(prefix)_k$(k).$(config.output_format)"
            output_path = joinpath(output_dir, output_filename)

            @info "Creating multi-panel plot for vertical level $k..."
            create_multipanel_figure(output_path, datafile_rll, panels, config, k)
        end
    end

    # Cleanup remapped file if requested
    if config.cleanup_remap_files
        rm(datafile_rll; force = true)
    end

    @info "Plotting complete! Output saved to $output_dir"
end

"""
    create_figure_set(
        output_dir::String,
        variable_names::Vector{String},
        field_data::Dict{String,Any},
        Y_cpu,
        ᶜspace,
        figure_specs::Dict{String,Vector{PlotPanel}},
        config::PlotConfig;
        kwargs...
    )

Create multiple figures with different panel configurations from the same remapped data.
This is more efficient than calling `remap_and_plot_3d` multiple times because it only
remaps once.

# Arguments
- `output_dir`: Directory for output plot files
- `variable_names`: Names of all variables to remap
- `field_data`: Dictionary mapping variable names to ClimaCore Field objects
- `Y_cpu`: CPU version of the state vector
- `ᶜspace`: Center space for defining coordinates
- `figure_specs`: Dictionary mapping figure names to their panel specifications
- `config`: Plot configuration
- `kwargs...`: Additional keyword arguments passed to remap_to_latlon

# Example
```julia
figure_specs = Dict(
    "uforcing" => [
        PlotPanel("ogwd_u", "ClimaAtmos u at z = {z} m", (1,1); scale_factor=86400),
        PlotPanel("gfdl_udt_topo", "GFDL u at z = {z} m", (2,1); scale_factor=86400),
    ],
    "vforcing" => [
        PlotPanel("ogwd_v", "ClimaAtmos v at z = {z} m", (1,1); scale_factor=86400),
        PlotPanel("gfdl_vdt_topo", "GFDL v at z = {z} m", (2,1); scale_factor=86400),
    ],
)

create_figure_set(output_dir, variable_names, field_data, Y_cpu, ᶜspace,
                  figure_specs, config)
```
"""
function create_figure_set(
    output_dir::String,
    variable_names::Vector{String},
    field_data::Dict{String, <:Any},
    Y_cpu,
    ᶜspace,
    figure_specs::Dict{String, Vector{PlotPanel}},
    config::PlotConfig;
    remap_dir::Union{String, Nothing} = nothing,
    FT = Float64,
)
    # Set up directories
    mkpath(output_dir)
    if remap_dir === nothing
        remap_dir = joinpath(output_dir, "remap_data")
    end

    # Remap once for all figures
    @info "Remapping $(length(variable_names)) variables to lat/lon grid..."
    datafile_rll = remap_to_latlon(
        remap_dir,
        variable_names,
        field_data,
        Y_cpu,
        ᶜspace;
        config = config,
        FT = FT,
    )

    # Create each figure set
    for (figure_name, panels) in figure_specs
        if config.plot_mode == :horizontal_slice
            # For horizontal slices, create a single plot per figure
            output_filename = "$(figure_name).$(config.output_format)"
            output_path = joinpath(output_dir, output_filename)

            @info "Creating $figure_name horizontal slice plot..."
            create_multipanel_figure(output_path, datafile_rll, panels, config, nothing)
        else  # :vertical_slice
            # For vertical slices, create plots for each vertical level
            for k in config.vertical_levels
                output_filename = "$(figure_name)_k$(k).$(config.output_format)"
                output_path = joinpath(output_dir, output_filename)

                @info "Creating $figure_name plot for vertical level $k..."
                create_multipanel_figure(output_path, datafile_rll, panels, config, k)
            end
        end
    end

    # Cleanup if requested
    if config.cleanup_remap_files
        rm(datafile_rll; force = true)
    end

    @info "All figures complete! Output saved to $output_dir"
end
