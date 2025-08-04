# ci_plots.jl:
#
# Automatic produce a PDF report of a given job_id.
#
# FAQ: How to add a new job?
#
# To add a new job, you need to define a new method for the `make_plots` function. The
# `make_plots` has to take two arguments: a `Val(:job)`, and a list of output paths. In most
# cases, `make_plots` will work with just one output path, but it still has to accept a list
# so that the function can be used to compare different outputs with the same report.
# Support for comparison is for the most part automated as long as `map_comparison` and
# `make_plot_generic` are used.
#
# Consider for example
# ```julia
#     function make_plots(
#         ::Val{:box_hydrostatic_balance},
#         output_paths::Vector{<:AbstractString},
#     )
#         simdirs = SimDir.(output_paths)
#         short_names, reduction = ["wa", "ua"], "average"
#         vars = map_comparison(simdirs, short_names) do simdir, short_name
#             return get(simdir; short_name, reduction)
#         end
#         make_plots_generic(
#             output_paths,
#             vars,
#             y = 0.0,
#             time = LAST_SNAP,
#             more_kwargs = YLINEARSCALE,
#         )
#     end
# ```
#
# This function takes the required arguments. First, it converts the `output_paths` in
# `SimDir`s. Then, it extracts the variables we want to plot. `map_comparison` is a small
# helper function that "broadcasts" your expression to work on multiple simdirs at the same
# time (used to produce comparison reports). Finally, the function calls
# `make_plots_generic` with the default plotting function (`ClimaAnalysis.plot!`).

import CairoMakie
import CairoMakie.Makie
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz
import ClimaAnalysis: SimDir, slice, read_var, average_xy, window, average_time
import ClimaAnalysis.Utils: kwargs as ca_kwargs

import ClimaCoreSpectra: power_spectrum_2d

using Poppler_jll: pdfunite
import Base.Filesystem
import Statistics: mean

const days = 86400

# Return the last common directory across several files
function common_dirname(files::Vector{T}) where {T <: AbstractString}
    # Split the path of each file into a vector of strings
    # e.g. "/home/user/file1.txt" -> ["home", "user", "file1.txt"]
    split_files = split.(files, '/')
    # Find the index of the last common directory
    last_common_dir =
        findfirst(
            i -> any(j -> split_files[1][i] != j, split_files[2:end]),
            1:length(split_files[1]),
        ) - 1
    return joinpath(split_files[1][1:last_common_dir]...)
end

function make_plots(sim, _)
    @warn "No plot found for $sim"
end

"""
    make_plots(sim, simulation_path::AbstractString)
    make_plots(sim, simulation_paths::Iterable{AbstractString})

Plot the corresponding sets for the given `sim`.

When `simulation_path` is a string, use the data in the path.

When `simulation_paths` is a collection of strings, use all those paths and making
side-by-side comparisons.
"""
function make_plots(sim, simulation_path::AbstractString)
    paths = [simulation_path]
    make_plots(sim, paths)
end

# The contour plot functions in ClimaAnalysis work by finding the nearest slice available.
# If we want the extremes, we can just ask for the slice closest to a very large number.
const LARGE_NUM = typemax(Int)
const LAST_SNAP = LARGE_NUM
const FIRST_SNAP = -LARGE_NUM
const BOTTOM_LVL = -LARGE_NUM
const TOP_LVL = LARGE_NUM

function Makie.get_tickvalues(yticks::Int, ymin, ymax)
    return range(max(ymin, 0), ymax, yticks)
end

YLINEARSCALE = Dict(
    :axis =>
        ca_kwargs(dim_on_y = true, yticks = 10, ytickformat = "{:.3e}"),
)

long_name(var) = var.attributes["long_name"]
short_name(var) = var.attributes["short_name"]
z_dim_name(var) = haskey(var.dims, "z_reference") ? "z_reference" : "z"

"""
    parse_var_attributes(var)

Takes in an OutputVar and parses some of its attributes into a short, informative string.
Used to generate unique titles when the same var is being plotted for several times/locations.
This could be extended to parse more attributes.

For example, the sample attributes:
attributes = Dict(
    "units" => "%",
    "short_name" => "cl",
    "slice_y" => "0.0",
    "long_name" => "Cloud fraction, Instantaneous x = 0.0 m y = 0.0 m",
    "slice_y_units" => "m",
    "slice_x_units" => "m",
    "comments" => "",
    "slice_x" => "0.0",
)
will be parsed into "cl, x = 0.0, y = 0.0"
"""
function parse_var_attributes(var)
    MISSING_STR = "MISSING_ATTRIBUTE"
    attr = var.attributes
    name = replace(short_name(var), "up" => "")

    attributes = ["slice_lat", "slice_lon", "slice_x", "slice_y", "slice_time"]
    info = [
        replace(key, "slice_" => "") * " = " * get(attr, key, MISSING_STR)
        for key in attributes
    ]
    # Filter out missing entries
    info = filter(x -> !occursin(MISSING_STR, x), [name, info...])

    return join(info, ", ")
end

"""
    make_plots_generic(
        output_path::Union{<:AbstractString, Vector{<:AbstractString}},
        vars,
        args...;
        plot_fn = nothing,
        output_name = "summary",
        summary_files = String[],
        MAX_NUM_COLS = 1,
        MAX_NUM_ROWS = min(4, length(vars)),
        kwargs...,
    )

Use `plot_fn` to plot `vars` properly handling pagination.

Arguments
=========

`output_path` can be a `String` or a list of `String`s. When it is a list of `String`s, it
is assumed that `vars` are coming from different simulations and they have to be compared.
Hence, the summary plot is saved to the first `output_path` and the number of columns is
fixed to be the same as the number of `output_path`s. `summary_files` are also assumed to be
in the first `output_path`.

`output_name` is the name of the file produced.

`summary_files` is an optional list of paths to prepend to the PDF produced by this
function. This is useful when building larger and more complex reports that required
different `plot_fn` to be produced.

Extra Arguments
===============

`args` and `kwargs` are passed to the plotting function `plot_fn`.

`MAX_NUM_COLS` and `MAX_NUM_ROWS` define the grid layout.
"""
function make_plots_generic(
    output_path::Union{<:AbstractString, Vector{<:AbstractString}},
    vars,
    args...;
    plot_fn = nothing,
    output_name = "summary",
    summary_files = String[],
    MAX_NUM_COLS = 1,
    MAX_NUM_ROWS = min(4, length(vars)),
    kwargs...,
)
    # When output_path is a Vector with multiple elements, this means that this function is
    # being used to produce a comparison plot. In that case, we modify the output name, and
    # the number of columns (to match how many simulations we are comparing).
    is_comparison = output_path isa Vector
    #
    # However, we don't want to do this when the vector only contains one element.
    if is_comparison && length(output_path) == 1
        # Fallback to the "output_path isa String" case
        output_path = output_path[1]
        is_comparison = false
    end

    if is_comparison
        MAX_NUM_COLS = length(output_path)
        save_path = output_path[1]
        output_name *= "_comparison"
    else
        save_path = output_path
    end

    # Default plotting function needs access to kwargs
    if isnothing(plot_fn)
        plot_fn = viz.plot!
    end

    MAX_PLOTS_PER_PAGE = MAX_NUM_ROWS * MAX_NUM_COLS
    vars_left_to_plot = length(vars)

    # Define fig, grid, and grid_pos, used below. (Needed for scope)
    function makefig()
        fig = CairoMakie.Figure(; size = (900, 300 * MAX_NUM_ROWS))
        if is_comparison
            for (col, path) in enumerate(output_path)
                # CairoMakie seems to use this Label to determine the width of the figure.
                # Here we normalize the length so that all the columns have the same width.
                LABEL_LENGTH = 40
                normalized_path =
                    lpad(path, LABEL_LENGTH + 1, " ")[(end - LABEL_LENGTH):end]

                CairoMakie.Label(fig[0, col], path)
            end
        end
        return fig
    end
    gridlayout() =
        map(1:MAX_PLOTS_PER_PAGE) do i
            row = mod(div(i - 1, MAX_NUM_COLS), MAX_NUM_ROWS) + 1
            col = mod(i - 1, MAX_NUM_COLS) + 1
            return fig[row, col] = CairoMakie.GridLayout()
        end

    fig = makefig()
    grid = gridlayout()
    page = 1
    grid_pos = 1

    for var in vars
        if grid_pos > MAX_PLOTS_PER_PAGE
            fig = makefig()
            grid = gridlayout()
            grid_pos = 1
        end

        plot_fn(grid[grid_pos], var, args...; kwargs...)
        grid_pos += 1

        # Flush current page
        if grid_pos > min(MAX_PLOTS_PER_PAGE, vars_left_to_plot)
            file_path = joinpath(save_path, "$(output_name)_$page.pdf")
            CairoMakie.resize_to_layout!(fig)
            CairoMakie.save(file_path, fig)
            push!(summary_files, file_path)
            vars_left_to_plot -= MAX_PLOTS_PER_PAGE
            page += 1
        end
    end

    output_file = joinpath(save_path, "$(output_name).pdf")

    pdfunite() do unite
        run(Cmd([unite, summary_files..., output_file]))
    end

    # Cleanup
    Filesystem.rm.(summary_files, force = true)
    return output_file
end

"""
    horizontal_average(var)

A `ClimaAnalysis.OutputVar` with a horizontal RMS average of the data in `var`.
"""
function horizontal_average(var)
    rms(var; dims) = sqrt.(mean(var .^ 2; dims))
    reduced_var = ClimaAnalysis.Var._reduce_over(rms, "x", var)
    if haskey(var.dims, "y")
        reduced_var = ClimaAnalysis.Var._reduce_over(rms, "y", reduced_var)
    end
    if haskey(var.attributes, "long_name")
        long_name = reduced_var.attributes["long_name"]
        reduced_var.attributes["long_name"] = long_name * ", Horizontal Average"
    end
    return reduced_var
end

"""
    vertical_average(var)

A `ClimaAnalysis.OutputVar` with a vertical RMS average of the data in `var`.
"""
function vertical_average(var)
    rms(var; dims) = sqrt.(mean(var .^ 2; dims))
    reduced_var = ClimaAnalysis.Var._reduce_over(rms, z_dim_name(var), var)
    if haskey(var.attributes, "long_name")
        long_name = reduced_var.attributes["long_name"]
        reduced_var.attributes["long_name"] = long_name * ", Vertical Average"
    end
    return reduced_var
end

"""
    compute_spectrum(var::ClimaAnalysis.OutputVar; mass_weight = nothing)

Compute the spectrum associated to the given variable. Returns a ClimaAnalysis.OutputVar.

This function is slow because a bunch of work is done over and over (in ClimaCoreSpectra).

It is advisable to window the OutputVar to a narrow set of times before passing it to this
function.

With this function, you can also take time-averages of spectra.
"""
function compute_spectrum(var::ClimaAnalysis.OutputVar; mass_weight = nothing)
    # power_spectrum_2d seems to work only when the two dimensions have precisely one
    # twice as many points as the other
    if "time" in keys(var.dims)
        time, dim1, dim2, dim3 = var.index2dim[1:4]
        times = var.dims[time]
    else
        dim1, dim2, dim3 = var.index2dim[1:3]
        times = []
    end

    len1 = length(var.dims[dim1])
    len2 = length(var.dims[dim2])

    len1 == 2len2 || error("Cannot take this spectrum ($len1 != 2 $len2)")

    (dim1 == "lon" || dim1 == "long") ||
        error("First dimension has to be longitude (found $dim1)")
    dim2 == "lat" || error("Second dimension has to be latitude (found $dim2)")
    dim3 == "z" || error("Third dimension has to be altitude (found $dim3)")

    FT = eltype(var.data)

    mass_weight =
        isnothing(mass_weight) ? ones(FT, length(var.dims[dim3])) : mass_weight

    # Number of spherical wave numbers, excluding the first and the last
    # This number was reverse-engineered from ClimaCoreSpectra
    num_output = Int((floor((2 * len1 - 1) / 3) + 1) / 2 - 1)

    mesh_info = nothing

    if !isempty(times)
        output_spectrum =
            zeros((length(times), num_output, length(var.dims[dim3])))
        dims = Dict(time => times)
        dim_attributes = Dict(time => var.dim_attributes[time])
        for index in 1:length(times)
            spectrum_data, _wave_numbers, _spherical, mesh_info =
                power_spectrum_2d(FT, var.data[index, :, :, :], mass_weight)
            output_spectrum[index, :, :] .=
                dropdims(sum(spectrum_data, dims = 1), dims = 1)[
                    (begin + 1):(end - 1),
                    :,
                ]
        end
    else
        dims = Dict{String, Vector{FT}}()
        dim_attributes = Dict{String, Dict{String, String}}()
        output_spectrum = zeros((num_output, length(var.dims[dim3])))
        spectrum_data, _wave_numbers, _spherical, mesh_info =
            power_spectrum_2d(FT, var.data[:, :, :], mass_weight)
        output_spectrum[:, :] .=
            dropdims(sum(spectrum_data, dims = 1), dims = 1)[
                (begin + 1):(end - 1),
                :,
            ]
    end

    w_numbers = collect(1:1:(mesh_info.num_spherical - 1))

    dims["Log10 Spherical Wavenumber"] = log10.(w_numbers)
    dims[dim3] = var.dims[dim3]
    dim_attributes["Log10 Spherical Wavenumber"] = Dict("units" => "")
    dim_attributes[dim3] = var.dim_attributes[dim3]

    attributes = Dict(
        "short_name" => "log_spectrum_" * var.attributes["short_name"],
        "long_name" => "Spectrum of " * var.attributes["long_name"],
        "units" => "",
    )

    return ClimaAnalysis.OutputVar(
        attributes,
        dims,
        dim_attributes,
        log10.(output_spectrum),
    )
end

"""
    map_comparison(func, simdirs, args)

Helper function to make comparison plots for different simdirs.

`make_plots_generic` can plot any `ClimaAnalysis.OutputVar`, regardless of their origin.
We use this property to help us run the same plotting workflow for multiple simulations so that
we can make side-by-side comparisons.

The idea is simple: given a `func`tion that returns an `OutputVar` for the `arg` (typically
a short name), we run it over all the simdirs in order. This will interleave the
`OutputVar`s from the various N `simdirs`. Then, if we fix the number of columns to be
exactly N, this will automatically result in the same plot repeated for the N `simdirs`.

For the most part, this interface is transparent: developers only have to worry about
preparing a plot for one instance, and if `map_comparison` is used, the same plot can be
automatically extended to comparing N simulations.

The signature for `func` has to be `(simdir, arg)`. You can use closures to define
more complex behaviors. `func` has to return a `OutputVar`.

Example
===========

The simplest example is to directly `get` an `OutputVar`:
```julia
short_names = ["ta", "wa"]
vars = map_comparison(get, simdirs, short_names)
make_plots_generic(
    simulation_path,
    vars,
    time = LAST_SNAP,
    x = 0.0, # Our columns are still 3D objects...
    y = 0.0,
    more_kwargs = YLINEARSCALE,
)
```
If we want to be more daring, we can mix in some information about `reductions` and `periods`
```julia
short_names = ["ta", "wa"]
reduction, period = "average", "1d"
vars = map_comparison(simdirs, short_names) do simdir, short_name
     get(simdir; short_name, reduction, period)
end
make_plots_generic(
    simulation_path,
    vars,
    time = LAST_SNAP,
    x = 0.0, # Our columns are still 3D objects...
    y = 0.0,
    more_kwargs = YLINEARSCALE,
)
```
"""
function map_comparison(func, simdirs, args)
    return vcat([[func(simdir, arg) for simdir in simdirs] for arg in args]...)
end

"""
    plot_spectrum_with_line!(grid_loc, spectrum; exponent = -3.0)

Plots the given spectrum alongside a line that identifies a power law.

Assumes 1D spectrum.
"""
function plot_spectrum_with_line!(grid_loc, spectrum; exponent = -3.0)
    ndims(spectrum.data) == 1 || error("Can only work with 1D spectrum")
    viz.plot!(grid_loc, spectrum)

    dim_name = spectrum.index2dim[begin]

    ax = CairoMakie.current_axis()

    # Ignore below wavenumber of 10
    spectrum_10 = ClimaAnalysis.window(spectrum, dim_name; left = log10(10))

    # Add reference line
    wavenumbers = spectrum_10.dims[dim_name]
    max_spectrum_10 = maximum(spectrum_10.data)
    wavenumber_at_max = wavenumbers[argmax(spectrum_10.data)]

    # Increase the intercept by 20 percent so that it hovers over the spectrum
    intercept = 1.2 * (max_spectrum_10 - exponent * wavenumber_at_max)
    reference_line(k) = exponent * k + intercept

    color = :orange
    CairoMakie.lines!(ax, wavenumbers, reference_line.(wavenumbers); color)
    CairoMakie.text!(
        ax,
        wavenumber_at_max,
        reference_line(wavenumber_at_max),
        text = "k^$exponent";
        color,
    )

    return nothing
end

"""
    plot_contours!(place, var; [n_contours], [kwargs]...)

Generic alternative to the default plotting function provided in ClimaAnalysis,
which uses a semi-transparent color scheme with appropriately centered contours.
Data with a small but nonempty range is centered around 0 before being plotted.
For constant data, a heatmap is used instead of a contour plot.

The number of contours is 22 by default, but can also be specified manually. Any
additional keyword arguments are passed to the `CairoMakie` plotting function.
"""
function plot_contours!(place, var; n_contours = 22, kwargs...)
    length(var.dims) == 2 || error("Can only plot 2D variables")

    var_name = var.attributes["short_name"]
    var_units = var.attributes["units"]
    dim1_name, dim2_name = var.index2dim
    dim1_units = var.dim_attributes[dim1_name]["units"]
    dim2_units = var.dim_attributes[dim2_name]["units"]
    dim1 = var.dims[dim1_name]
    dim2 = var.dims[dim2_name]

    CairoMakie.Axis(
        place[1, 1];
        title = var.attributes["long_name"],
        xlabel = "$dim1_name [$dim1_units]",
        ylabel = "$dim2_name [$dim2_units]",
        limits = (extrema(dim1), extrema(dim2)),
    )

    # Interpolate between the 11 Spectral colors, with the middle color replaced
    # by transparent white.
    spectral_colors = CairoMakie.to_colormap(:Spectral)
    colormap = setindex!(spectral_colors, CairoMakie.RGBA(1, 1, 1, 0), 6)
    highclip = extendhigh = spectral_colors[11]
    lowclip = extendlow = spectral_colors[1]

    # Center the contour levels around either 0, the average of the data, or the
    # nearest integer that falls into the data range.
    data_avg = mean(var.data)
    data_avg_int = round(Int, data_avg)
    data_min, data_max = extrema(var.data)
    data_mid = if data_min < 0 < data_max
        0
    elseif data_min < data_avg_int < data_max
        data_avg_int
    else
        data_avg
    end
    data_delta = maximum(value -> abs(value - data_mid), var.data)

    if data_delta == 0
        # For constant data, use a heatmap to avoid Colorbar's LineAxis error.
        plot_kwargs = (; colormap, highclip, lowclip, kwargs...)
        label = "$var_name [$var_units]"
        plot = CairoMakie.heatmap!(dim1, dim2, var.data; plot_kwargs...)
    else
        plot_kwargs = (; colormap, extendhigh, extendlow, kwargs...)
        if data_delta > abs(data_mid) / 1e6
            # Center contours around data_mid when data_delta >> |data_mid|.
            data = var.data
            label = "$var_name [$var_units]"
            levels =
                range(data_mid - data_delta, data_mid + data_delta, n_contours)
        else
            # Recenter data and contours around 0 when data_delta << |data_mid|.
            data = var.data .- data_mid
            label = "$var_name - $data_mid [$var_units]"
            levels = range(-data_delta, data_delta, n_contours)
        end
        plot = CairoMakie.contourf!(dim1, dim2, data; levels, plot_kwargs...)
    end
    CairoMakie.Colorbar(place[1, 2], plot; label)
end

ColumnPlots = Union{
    Val{:single_column_hydrostatic_balance_ft64},
    Val{:single_column_radiative_equilibrium_gray},
    Val{:single_column_radiative_equilibrium_clearsky},
    Val{:single_column_radiative_equilibrium_clearsky_prognostic_surface_temp},
    Val{:single_column_radiative_equilibrium_allsky_idealized_clouds},
}

function make_plots(::ColumnPlots, output_paths::Vector{<:AbstractString})
    simdirs = SimDir.(output_paths)
    short_names = ["ta", "wa"]
    vars = map_comparison(get, simdirs, short_names)

    make_plots_generic(
        output_paths,
        vars,
        time = LAST_SNAP,
        x = 0.0, # Our columns are still 3D objects...
        y = 0.0,
        MAX_NUM_COLS = length(simdirs),
        more_kwargs = YLINEARSCALE,
    )
end

function make_plots(
    ::Val{:box_hydrostatic_balance},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names, reduction = ["wa", "ua"], "average"
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return get(simdir; short_name, reduction)
    end
    make_plots_generic(
        output_paths,
        vars,
        y = 0.0,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
end

function make_plots(
    sim_type::Union{
        Val{:single_column_precipitation_test},
        Val{:single_column_precipitation_2M_test},
    },
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    # TODO: Move this plotting code into the same framework as the other ones
    simdir = simdirs[1]

    if sim_type isa Val{:single_column_precipitation_test}
        short_names = ["hus", "clw", "cli", "husra", "hussn", "ta"]
        figsize = (1200, 600)
        pr_row = 3
    else
        short_names =
            ["hus", "clw", "cli", "husra", "hussn", "ta", "cdnc", "ncra"]
        figsize = (1200, 800)
        pr_row = 4
    end

    vars = [
        slice(get(simdir; short_name), x = 0.0, y = 0.0) for
        short_name in short_names
    ]

    # We first prepare the axes with all the nice labels with ClimaAnalysis, then we use
    # CairoMakie to add the additional lines.
    fig = CairoMakie.Figure(; size = figsize)

    p_loc = [1, 1]

    axes = map(vars) do var
        viz.plot!(
            fig,
            var;
            time = 0.0,
            p_loc,
            more_kwargs = Dict(
                :plot => ca_kwargs(color = :navy),
                :axis => ca_kwargs(dim_on_y = true, title = ""),
            ),
        )

        # Make a grid of plots
        p_loc[2] += 1
        p_loc[2] > 3 && (p_loc[1] += 1; p_loc[2] = 1)
        return CairoMakie.current_axis()
    end

    col = Dict(500 => :blue2, 1000 => :royalblue, 1500 => :skyblue1)

    for (time, color) in col
        for (i, var) in enumerate(vars)
            CairoMakie.lines!(
                axes[i],
                slice(var; time).data,
                var.dims["z"],
                color = color,
            )
        end
    end

    # surface_precipitation
    surface_precip = read_var(simdir.variable_paths["pr"]["inst"]["10s"])
    viz.line_plot1D!(
        fig,
        slice(surface_precip, x = 0.0, y = 0.0);
        p_loc = [pr_row, 1:3],
    )

    file_path = joinpath(output_paths[1], "summary.pdf")
    CairoMakie.save(file_path, fig)
end

function make_plots(
    ::Val{:box_density_current_test},
    ::Val{:box_rising_thermal_test},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names = ["thetaa"]
    period = "10s"
    reduction = "inst"
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return get(simdir; short_name, reduction, period)
    end
    make_plots_generic(output_paths, vars, y = 0.0, time = LAST_SNAP)
end

const PeriodicTopographyTest2D = Union{
    Val{:gpu_plane_no_topography_float64_test},
    Val{:gpu_plane_cosine_hills_float64_test},
}
const PeriodicTopographyTest3D = Union{
    Val{:gpu_extruded_plane_cosine_hills_float64_test},
    Val{:gpu_box_cosine_hills_float64_test},
}
const MountainTest2D = Union{
    Val{:gpu_plane_agnesi_mountain_float64_test},
    Val{:gpu_plane_schar_mountain_float64_test},
    Val{:gpu_plane_schar_mountain_float32_test},
}
const SteadyStateTest =
    Union{PeriodicTopographyTest2D, PeriodicTopographyTest3D, MountainTest2D}

function make_plots(
    val::SteadyStateTest,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    is_mountain_test = val isa MountainTest2D
    is_3d = val isa PeriodicTopographyTest3D
    zd_rayleigh = 13e3 # Values inside the Rayleigh sponge shouldn't be plotted.

    rms_error_vars =
        Iterators.flatmap((horizontal_average, vertical_average)) do average
            Iterators.flatmap(("uaerror", "waerror")) do short_name
                Iterators.map(simdirs) do simdir
                    var = get(simdir; short_name)
                    var = window(var, z_dim_name(var); right = zd_rayleigh)
                    var = slice(var; time = Inf)
                    average(var)
                end
            end
        end
    orog_vars = Iterators.map(simdirs) do simdir
        slice(get(simdir; short_name = "orog"); time = Inf)
    end
    make_plots_generic(
        output_paths,
        [rms_error_vars..., orog_vars...];
        output_name = "final_rms_errors",
    )

    make_contour_plots(get_vars, short_names, output_name) = make_plots_generic(
        output_paths,
        [Iterators.flatmap(get_vars, short_names)...];
        output_name,
        plot_fn = plot_contours!,
    )

    for velocity_component in ("ua", "wa")
        short_names = velocity_component .* ("error", "", "predicted")
        mountain_output_name = "final_mountain_closeup_" * velocity_component
        time_series_output_name = "slice_time_series_" * velocity_component
        is_mountain_test &&
            make_contour_plots(short_names, mountain_output_name) do short_name
                Iterators.flatmap(simdirs) do simdir
                    var = get(simdir; short_name)
                    var = window(var, "x"; left = 35e3, right = 65e3)
                    var = is_3d ? slice(var; y = 0) : var
                    var = slice(var; time = Inf)
                    z_max_values =
                        endswith(short_name, "error") ? (1e3, zd_rayleigh) :
                        (zd_rayleigh,) # Add closeup view of errors below 1 km.
                    Iterators.map(z_max_values) do z_max
                        window(var, z_dim_name(var); right = z_max)
                    end
                end
            end
        make_contour_plots(short_names, time_series_output_name) do short_name
            Iterators.flatmap(simdirs) do simdir
                var = get(simdir; short_name)
                var = window(var, z_dim_name(var); right = zd_rayleigh)
                var = is_3d ? slice(var; y = 0) : var
                time_values = if endswith(short_name, "predicted")
                    (Inf,) # Predicted values are constant and only need 1 plot.
                elseif var.dims["time"][end] > 24 * 3600
                    (1, 2, 24, Inf) .* 3600
                elseif var.dims["time"][end] > 3 * 3600
                    (1, 2, 4, Inf) .* 3600
                else
                    (5, 10, 20, Inf) .* 60
                end
                Iterators.map(time -> slice(var; time), time_values)
            end
        end
    end
end

function make_plots(
    ::Val{:plane_density_current_test},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names = ["thetaa"]
    vars = map_comparison(get, simdirs, short_names)
    make_plots_generic(
        output_paths,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
end

function make_plots(
    ::Val{:hydrostatic_balance_ft64},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names, reduction = ["ua", "wa"], "average"
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon
    end
    make_plots_generic(
        output_paths,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
end

DryBaroWavePlots = Union{
    Val{:baroclinic_wave},
    Val{:baroclinic_wave_deepatmos},
    Val{:longrun_dry_baroclinic_wave},
    Val{:longrun_dry_baroclinic_wave_he60},
}

function make_plots(::DryBaroWavePlots, output_paths::Vector{<:AbstractString})
    simdirs = SimDir.(output_paths)
    short_names, reduction = ["pfull", "va", "wa", "rv"], "inst"
    short_names_spectra = ["ke"]
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return slice(get(simdir; short_name, reduction), time = 10days)
    end
    vars_spectra =
        map_comparison(simdirs, short_names_spectra) do simdir, short_name
            slice(
                compute_spectrum(
                    slice(get(simdir; short_name, reduction), time = 10days),
                ),
                z = 1500,
            )
        end

    tmp_file =
        make_plots_generic(output_paths, vars, z = 1500, output_name = "tmp")
    make_plots_generic(
        output_paths,
        vars_spectra;
        summary_files = [tmp_file],
        plot_fn = plot_spectrum_with_line!,
    )
end

SphereOrographyPlots = Union{
    Val{:baroclinic_wave_topography_dcmip_rs},
    Val{:baroclinic_wave_hughes2023},
}

function make_plots(
    ::SphereOrographyPlots,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names, reduction = ["pfull", "va", "wa", "rv"], "inst"
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return get(simdir; short_name, reduction)
    end
    make_plots_generic(output_paths, vars, z_reference = 1500, time = LAST_SNAP)
end

MoistBaroWavePlots =
    Union{Val{:baroclinic_wave_equil}, Val{:baroclinic_wave_equil_deepatmos}}

function make_plots(
    ::MoistBaroWavePlots,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names, reduction = ["pfull", "va", "wa", "rv", "hus"], "inst"
    short_names_spectra = ["ke", "hus"]
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return slice(get(simdir; short_name, reduction), time = 10days)
    end
    vars_spectra =
        map_comparison(simdirs, short_names_spectra) do simdir, short_name
            windowed_var = ClimaAnalysis.window(
                get(simdir; short_name, reduction),
                "time",
                left = 9days,
                right = 10days,
            )
            return slice(
                ClimaAnalysis.average_time(compute_spectrum(windowed_var)),
                z = 1500,
            )
        end

    tmp_file =
        make_plots_generic(output_paths, vars, z = 1500, output_name = "tmp")
    make_plots_generic(
        output_paths,
        vars_spectra;
        summary_files = [tmp_file],
        plot_fn = plot_spectrum_with_line!,
    )
end

LongMoistBaroWavePlots = Union{
    Val{:longrun_moist_baroclinic_wave},
    Val{:longrun_moist_baroclinic_wave_he60},
}

function make_plots(
    ::LongMoistBaroWavePlots,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names, reduction = ["pfull", "va", "wa", "rv", "hus"], "inst"
    short_names_spectra = ["ke", "hus"]
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        return slice(get(simdir; short_name, reduction), time = 10days)
    end
    vars_spectra =
        map_comparison(simdirs, short_names_spectra) do simdir, short_name
            slice(
                compute_spectrum(
                    slice(get(simdir; short_name, reduction), time = 10days),
                ),
                z = 1500,
            )
        end

    tmp_file =
        make_plots_generic(output_paths, vars, z = 1500, output_name = "tmp")
    make_plots_generic(
        output_paths,
        vars_spectra;
        summary_files = [tmp_file],
        plot_fn = plot_spectrum_with_line!,
    )
end

DryHeldSuarezPlots = Union{
    Val{:held_suarez},
    Val{:longrun_hydrostatic_balance},
    Val{:longrun_dry_held_suarez},
}

function make_plots(
    ::DryHeldSuarezPlots,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    short_names, reduction = ["ua", "ta"], "average"
    vars = map_comparison(simdirs, short_names) do simdir, short_name
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon
    end
    make_plots_generic(
        output_paths,
        vars,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
end

MoistHeldSuarezPlots = Union{
    Val{:held_suarez_equil},
    Val{:longrun_moist_held_suarez},
    Val{:longrun_moist_held_suarez_deepatmos},
}

function make_plots(
    ::MoistHeldSuarezPlots,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    short_names_3D, reduction = ["ua", "ta", "hus"], "average"
    short_names_2D = ["hfes", "evspsbl", "pr"]
    vars_3D = map_comparison(simdirs, short_names_3D) do simdir, short_name
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon
    end
    vars_2D = map_comparison(simdirs, short_names_2D) do simdir, short_name
        get(simdir; short_name, reduction)
    end
    make_plots_generic(
        output_paths,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
    make_plots_generic(
        output_paths,
        vars_2D,
        time = LAST_SNAP,
        output_name = "summary_2D",
    )
end

function make_plots(
    ::Union{
        Val{:aquaplanet_equil_allsky_gw_raw_zonalasym},
        Val{:gpu_aquaplanet_dyamond_summer},
    },
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    reduction = "average"
    short_names_3D = ["ua", "ta", "hus"]
    short_names_2D = [
        "rsdt",
        "rsds",
        "rsut",
        "rsus",
        "rlds",
        "rlut",
        "rlus",
        "hfes",
        "evspsbl",
        "pr",
    ]
    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names_3D[1],
        reduction,
    )
    if "1M" in available_periods
        period = "1M"
    elseif "30d" in available_periods
        period = "30d"
    elseif "10d" in available_periods
        period = "10d"
    elseif "1d" in available_periods
        period = "1d"
    elseif "1h" in available_periods
        period = "1h"
    end
    vars_3D = map_comparison(simdirs, short_names_3D) do simdir, short_name
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon
    end
    vars_2D = map_comparison(simdirs, short_names_2D) do simdir, short_name
        get(simdir; short_name, reduction, period)
    end
    make_plots_generic(
        output_paths,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
    make_plots_generic(
        output_paths,
        vars_2D,
        time = LAST_SNAP,
        output_name = "summary_2D",
    )
end

function make_plots(
    ::Union{
        Val{:aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean},
        Val{:aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64},
        Val{
            :longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_tvinsol_0M_slabocean,
        },
    },
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    reduction = "average"
    short_names_3D =
        ["ta", "thetaa", "rhoa", "ua", "va", "wa", "hur", "hus", "clw", "cli"]
    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names_3D[1],
        reduction,
    )
    if "1M" in available_periods
        period = "1M"
    elseif "30d" in available_periods
        period = "30d"
    elseif "10d" in available_periods
        period = "10d"
    elseif "1d" in available_periods
        period = "1d"
    elseif "12h" in available_periods
        period = "12h"
    end
    short_names_2D = [
        "rsdt",
        "rsds",
        "rsut",
        "rsus",
        "rlds",
        "rlut",
        "rlus",
        "hfes",
        "evspsbl",
        "ts",
        "pr",
    ]
    vars_3D = map_comparison(simdirs, short_names_3D) do simdir, short_name
        get(simdir; short_name, reduction, period) |> ClimaAnalysis.average_lon
    end
    vars_2D = map_comparison(simdirs, short_names_2D) do simdir, short_name
        get(simdir; short_name, reduction, period)
    end
    make_plots_generic(
        output_paths,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
    make_plots_generic(
        output_paths,
        vars_2D,
        time = LAST_SNAP,
        output_name = "summary_2D",
    )
end

AquaplanetPlots = Union{
    Val{:edonly_edmfx_aquaplanet},
    Val{:mpi_sphere_aquaplanet_rhoe_equil_clearsky},
    Val{:aquaplanet_nonequil_allsky_gw_res},
    Val{:aquaplanet_nonequil_allsky_gw_res_2M},
    Val{:rcemipii_sphere_diagnostic_edmfx},
    Val{:longrun_aquaplanet_allsky_0M},
    Val{:longrun_aquaplanet_allsky_diagedmf_0M},
    Val{:longrun_aquaplanet_allsky_progedmf_0M},
    Val{:longrun_aquaplanet_allsky_0M_earth},
    Val{:longrun_aquaplanet_dyamond},
    Val{:longrun_aquaplanet_allsky_tvinsol_0M_slabocean},
    Val{:amip_target_diagedmf},
    Val{:amip_target_edonly},
}

function make_plots(
    sim_type::AquaplanetPlots,
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    reduction = "average"
    short_names_3D =
        sim_type isa Val{:aquaplanet_nonequil_allsky_gw_res_2M} ?
        ["ua", "ta", "hus", "clw", "cli", "husra", "hussn", "cdnc", "ncra"] :
        ["ua", "ta", "hus"]
    short_names_2D = [
        "rsdt",
        "rsds",
        "rsut",
        "rsus",
        "rlds",
        "rlut",
        "rlus",
        "hfes",
        "evspsbl",
        "pr",
    ]
    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names_3D[1],
        reduction,
    )
    if "1M" in available_periods
        period = "1M"
    elseif "30d" in available_periods
        period = "30d"
    elseif "10d" in available_periods
        period = "10d"
    elseif "1d" in available_periods
        period = "1d"
    elseif "12h" in available_periods
        period = "12h"
    end
    vars_3D = map_comparison(simdirs, short_names_3D) do simdir, short_name
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon
    end
    vars_2D = map_comparison(simdirs, short_names_2D) do simdir, short_name
        get(simdir; short_name, reduction)
    end
    make_plots_generic(
        output_paths,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
    make_plots_generic(
        output_paths,
        vars_2D,
        time = LAST_SNAP,
        output_name = "summary_2D",
    )
end

Aquaplanet1MPlots = Union{
    Val{:sphere_aquaplanet_rhoe_equil_allsky_gw_res},
    Val{:longrun_aquaplanet_allsky_1M},
}

function make_plots(::Aquaplanet1MPlots, output_paths::Vector{<:AbstractString})
    simdirs = SimDir.(output_paths)

    reduction = "average"
    short_names_3D =
        ["ua", "ta", "hus", "husra", "hussn", "hur", "cl", "cli", "clw"]
    short_names_2D = [
        "rsdt",
        "rsds",
        "rsut",
        "rsus",
        "rlds",
        "rlut",
        "rlus",
        "hfes",
        "evspsbl",
        "pr",
    ]
    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names_3D[1],
        reduction,
    )
    if "1M" in available_periods
        period = "1M"
    elseif "30d" in available_periods
        period = "30d"
    elseif "10d" in available_periods
        period = "10d"
    elseif "1d" in available_periods
        period = "1d"
    elseif "12h" in available_periods
        period = "12h"
    end
    vars_3D = map_comparison(simdirs, short_names_3D) do simdir, short_name
        get(simdir; short_name, reduction) |> ClimaAnalysis.average_lon
    end
    vars_2D = map_comparison(simdirs, short_names_2D) do simdir, short_name
        get(simdir; short_name, reduction)
    end
    make_plots_generic(
        output_paths,
        vars_3D,
        time = LAST_SNAP,
        more_kwargs = YLINEARSCALE,
    )
    make_plots_generic(
        output_paths,
        vars_2D,
        time = LAST_SNAP,
        output_name = "summary_2D",
    )
end

LESBoxPlots = Union{Val{:les_box}}

"""
    plot_les_vert_profile!(grid_loc, var_group)

Helper function for `make_plots_generic`. Takes a list of variables and plots
them on the same axis.
"""
function plot_les_vert_profile!(grid_loc, var_group)
    z = var_group[1].dims["z"]
    units = var_group[1].attributes["units"]
    ax = CairoMakie.Axis(
        grid_loc[1, 1],
        ylabel = "z [$(var_group[1].dim_attributes["z"]["units"])]",
        xlabel = "$(short_name(var_group[1])) [$units]",
        title = parse_var_attributes(var_group[1]),
    )

    for var in var_group
        CairoMakie.lines!(ax, var.data, z, label = short_name(var))
    end
    length(var_group) > 1 && Makie.axislegend(ax)
end

function make_plots(
    sim_type::Union{LESBoxPlots},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    reduction = "inst"
    short_names = [
        "wa",
        "ua",
        "va",
        "ta",
        "thetaa",
        "ha",
        "hus",
        "hur",
        "cl",
        "clw",
        "cli",
    ]
    short_names = short_names ∩ collect(keys(simdirs[1].vars))

    # Window average from instantaneous snapshots?
    function horizontal_average(var)
        return average_xy(var)
    end
    function windowed_reduction(var)
        hours = 3600.0
        window_end = last(var.dims["time"])
        window_start = window_end - 2hours
        var_window = ClimaAnalysis.window(
            var,
            "time";
            left = window_start,
            right = window_end,
        )
        var_reduced = horizontal_average(average_time(var_window))
        return var_reduced
    end

    var_groups_xyt_reduced =
        map_comparison(simdirs, short_names) do simdir, short_name
            return [get(simdir; short_name, reduction) |> windowed_reduction]
        end

    var_groups_xy_reduced =
        map_comparison(simdirs, short_names) do simdir, short_name
            return [get(simdir; short_name, reduction) |> horizontal_average]
        end

    tmp_file = make_plots_generic(
        output_paths,
        var_groups_xyt_reduced,
        output_name = "tmp";
        plot_fn = plot_les_vert_profile!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )

    make_plots_generic(
        output_paths,
        vcat(var_groups_xy_reduced...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end


EDMFBoxPlots = Union{
    Val{:diagnostic_edmfx_test_box},
    Val{:diagnostic_edmfx_gabls_box},
    Val{:diagnostic_edmfx_bomex_box},
    Val{:diagnostic_edmfx_bomex_stretched_box},
    Val{:diagnostic_edmfx_dycoms_rf01_box},
    Val{:diagnostic_edmfx_trmm_box_0M},
    Val{:diagnostic_edmfx_dycoms_rf01_explicit_box},
    Val{:prognostic_edmfx_adv_test_column},
    Val{:prognostic_edmfx_gabls_column},
    Val{:prognostic_edmfx_bomex_fixtke_column},
    Val{:prognostic_edmfx_bomex_column},
    Val{:prognostic_edmfx_bomex_column_implicit},
    Val{:prognostic_edmfx_bomex_stretched_column},
    Val{:prognostic_edmfx_bomex_pigroup_column},
    Val{:prognostic_edmfx_bomex_implicit_column},
    Val{:prognostic_edmfx_dycoms_rf01_column},
    Val{:prognostic_edmfx_trmm_column_0M},
    Val{:prognostic_edmfx_simpleplume_column},
    Val{:prognostic_edmfx_gcmdriven_column},
    Val{:prognostic_edmfx_tv_era5driven_column},
    Val{:prognostic_edmfx_bomex_box},
    Val{:rcemipii_box_diagnostic_edmfx},
    Val{:prognostic_edmfx_soares_column},
    Val{:diagnostic_edmfx_trmm_stretched_box},
}

EDMFBoxPlotsWithPrecip = Union{
    Val{:prognostic_edmfx_rico_column},
    Val{:prognostic_edmfx_trmm_column},
}

DiagEDMFBoxPlotsWithPrecip = Union{
    Val{:diagnostic_edmfx_dycoms_rf02_box},
    Val{:diagnostic_edmfx_rico_box},
    Val{:diagnostic_edmfx_trmm_box},
}
"""
    plot_edmf_vert_profile!(grid_loc, var_group)

Helper function for `make_plots_generic`. Takes a list of variables and plots
them on the same axis.
"""
function plot_edmf_vert_profile!(grid_loc, var_group)
    z = ClimaAnalysis.altitudes(var_group[1])
    units = ClimaAnalysis.units(var_group[1])
    z_units = ClimaAnalysis.dim_units(var_group[1], "z")
    ax = CairoMakie.Axis(
        grid_loc[1, 1],
        ylabel = "z [$z_units]",
        xlabel = "$(short_name(var_group[1])) [$units]",
        title = parse_var_attributes(var_group[1]),
    )

    for var in var_group
        CairoMakie.lines!(ax, var.data, z, label = short_name(var))
    end
    length(var_group) > 1 && Makie.axislegend(ax)
end


"""
    plot_parsed_attribute_title!(grid_loc, var)

Helper function for `make_plots_generic`. Plots an OutputVar `var`,
setting the axis title to `parse_var_attributes(var)`
"""
plot_parsed_attribute_title!(grid_loc, var) = viz.plot!(
    grid_loc,
    var;
    more_kwargs = Dict(:axis => ca_kwargs(title = parse_var_attributes(var))),
)

"""
    pair_edmf_names(vars)

Groups updraft and gridmean EDMF short names into tuples.
Matches on the same variable short name with the suffix "up".
This assumes that the updraft variable name is the same as the corresponding
gridmean variable with the suffix "up".
"""
function pair_edmf_names(short_names)
    grouped_vars = Any[]
    short_names_to_be_processed = Set(short_names)

    for name in short_names
        # If we have already visited this name, go to the next one
        name in short_names_to_be_processed || continue

        # First, check if we have the pair of variables
        # We normalize the name to the gridmean version (base_name)
        # So, if we are visiting "va" or "vaup", we end up with
        # base_name = "va" and up_name = "vaup"
        base_name = replace(name, "up" => "")
        up_name = base_name * "up"

        if base_name in short_names_to_be_processed &&
           up_name in short_names_to_be_processed
            # Gridmean and updraft are available
            tuple_to_be_added = (base_name, up_name)
        else
            # Only single var (updraft OR gridmean) is available
            tuple_to_be_added = (name,)
        end

        foreach(n -> delete!(short_names_to_be_processed, n), tuple_to_be_added)
        push!(grouped_vars, tuple_to_be_added)
    end
    return grouped_vars
end

function make_plots(
    sim_type::Union{
        EDMFBoxPlots,
        EDMFBoxPlotsWithPrecip,
        DiagEDMFBoxPlotsWithPrecip,
    },
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    if sim_type isa EDMFBoxPlotsWithPrecip
        precip_names =
            ("husra", "hussn", "husraup", "hussnup", "husraen", "hussnen")
    elseif sim_type isa DiagEDMFBoxPlotsWithPrecip
        precip_names = ("husra", "hussn", "husraup", "hussnup")
    else
        precip_names = ()
    end

    short_names = [
        "wa",
        "waup",
        "ta",
        "taup",
        "hus",
        "husup",
        "arup",
        "tke",
        "ua",
        "thetaa",
        "thetaaup",
        "ha",
        "haup",
        "hur",
        "hurup",
        "lmix",
        "cl",
        "clw",
        "clwup",
        "cli",
        "cliup",
        precip_names...,
    ]
    reduction = "inst"

    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names[1],
        reduction,
    )
    if "5m" in available_periods
        period = "5m"
    elseif "10m" in available_periods
        period = "10m"
    elseif "30m" in available_periods
        period = "30m"
    end

    short_name_tuples = pair_edmf_names(short_names)
    var_groups_zt =
        map_comparison(simdirs, short_name_tuples) do simdir, name_tuple
            return [
                slice(
                    get(simdir; short_name, reduction, period),
                    x = 0.0,
                    y = 0.0,
                ) for short_name in name_tuple
            ]
        end

    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,) for
        group in var_groups_zt
    ]

    tmp_file = make_plots_generic(
        output_paths,
        output_name = "tmp",
        var_groups_z;
        plot_fn = plot_edmf_vert_profile!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )

    make_plots_generic(
        output_paths,
        vcat(var_groups_zt...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end

EDMFSpherePlots =
    Union{Val{:diagnostic_edmfx_aquaplanet}, Val{:prognostic_edmfx_aquaplanet}}

function make_plots(::EDMFSpherePlots, output_paths::Vector{<:AbstractString})
    simdirs = SimDir.(output_paths)

    short_names =
        ["ua", "wa", "waup", "thetaa", "ta", "taup", "haup", "tke", "arup"]

    reduction = "average"
    period = "1h"
    latitudes = [0.0, 30.0, 60.0, 90.0]

    short_name_tuples = pair_edmf_names(short_names)

    # The hierarchy is:
    # - A vector looping over variables
    #     - Containing, a vector looping over latitudes
    #     - Containing, tuples with one or two variables
    #   - Repeated for each simdir
    # All of this is flattened out to be a vector of tuples (with the two gridmean/updraft
    # variables)
    var_groups_zt = vcat(
        map_comparison(simdirs, short_name_tuples) do simdir, name_tuple
            return [
                (
                    slice(
                        get(simdir; short_name, reduction, period),
                        lon = 0.0,
                        lat = lat,
                    ) for short_name in name_tuple
                ) for lat in latitudes
            ]
        end...,
    )

    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,) for
        group in var_groups_zt
    ]

    tmp_file = make_plots_generic(
        output_paths,
        output_name = "tmp",
        var_groups_z;
        plot_fn = plot_edmf_vert_profile!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
    make_plots_generic(
        output_paths,
        vcat((var_groups_zt...)...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end


function make_plots(
    ::Val{:gcm_driven_scm},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)
    short_names_2D = [
        "rlut",
        "rlutcs",
        "rsut",
        "rsutcs",
        "clwvi",
        "lwp",
        "clivi",
        "dsevi",
        "clvi",
        "prw",
        "hurvi",
    ]
    short_names_3D = ["husv", "thetaa", "ta", "hur", "hus", "clw", "cl"]
    reduction = "inst"
    vars_2D = map_comparison(simdirs, short_names_2D) do simdir, short_name
        average_xy(get(simdir; short_name, reduction))
    end
    vars_3D = map_comparison(simdirs, short_names_3D) do simdir, short_name
        data = window(
            get(simdir; short_name, reduction),
            "z",
            left = 0,
            right = 4000,
        )
        return average_xy(data)
    end
    make_plots_generic(
        output_paths,
        vars_2D;
        MAX_NUM_COLS = 2,
        output_name = "summary_2D",
    )
    make_plots_generic(
        output_paths,
        vars_3D;
        MAX_NUM_COLS = 2,
        output_name = "summary_3D",
    )
end
