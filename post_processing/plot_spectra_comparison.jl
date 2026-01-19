# plot_spectra_comparison.jl
#
# Script to generate spectra plots for variables and superimpose them
# for comparison across multiple output directories.
#
# Usage:
#   include("post_processing/plot_spectra_comparison.jl")
#   plot_spectra_comparison(
#       ["/path/to/output_0001", "/path/to/output_0003"],
#       ["ke", "hus"];
#       z_level = 1500,
#       time = 10days,
#       output_dir = "/path/to/save/plots"
#   )

import CairoMakie
import CairoMakie.Makie
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz
import ClimaAnalysis: SimDir, slice, window, average_time
import ClimaCoreSpectra: power_spectrum_2d

# Import compute_spectrum and other utilities from ci_plots.jl
# Note: This assumes ci_plots.jl is in the same directory
include("ci_plots.jl")

const days = 86400

"""
    plot_spectra_comparison!(
        ax,
        spectra_list,
        labels;
        colors = nothing,
        linestyles = nothing,
        exponent = -3.0,
    )

Plot multiple spectra on the same axis for comparison.

# Arguments
- `ax`: CairoMakie axis to plot on
- `spectra_list`: Vector of OutputVar spectra (1D, already sliced to desired z_level)
- `labels`: Vector of labels for each spectrum (one per output directory)
- `colors`: Optional vector of colors (default: uses Makie's default palette)
- `linestyles`: Optional vector of line styles (default: all solid)
- `exponent`: Exponent for reference power law line (default: -3.0)

# Returns
- Nothing (modifies ax in place)
"""
function plot_spectra_comparison!(
    ax,
    spectra_list,
    labels;
    colors = nothing,
    linestyles = nothing,
    exponent = -3.0,
)
    # Default colors from Makie
    if isnothing(colors)
        colors = [
            :blue,
            :red,
            :green,
            :orange,
            :purple,
            :brown,
            :pink,
            :gray,
            :olive,
            :cyan,
        ]
    end

    # Default linestyles
    if isnothing(linestyles)
        linestyles = fill(:solid, length(spectra_list))
    end

    # Ensure we have enough colors and linestyles
    while length(colors) < length(spectra_list)
        push!(colors, colors[mod(length(colors), length(colors)) + 1])
    end
    while length(linestyles) < length(spectra_list)
        push!(linestyles, linestyles[mod(length(linestyles), length(linestyles)) + 1])
    end

    # Plot each spectrum
    for (i, (spectrum, label)) in enumerate(zip(spectra_list, labels))
        dim_name = spectrum.index2dim[begin]
        wavenumbers = spectrum.dims[dim_name]
        spectrum_data = spectrum.data

        CairoMakie.lines!(
            ax,
            wavenumbers,
            spectrum_data;
            label = label,
            color = colors[i],
            linestyle = linestyles[i],
            linewidth = 2,
        )
    end

    # Add reference power law line (using the first spectrum for positioning)
    if !isempty(spectra_list)
        spectrum = spectra_list[1]
        dim_name = spectrum.index2dim[begin]

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
        CairoMakie.lines!(
            ax,
            wavenumbers,
            reference_line.(wavenumbers);
            color,
            linestyle = :dash,
            linewidth = 1.5,
            label = "k^$exponent",
        )
    end

    # Add legend
    Makie.axislegend(ax; position = :rt)

    return nothing
end

"""
    plot_spectra_comparison(
        output_dirs::Vector{String},
        short_names::Vector{String};
        z_level = 1500.0,
        time = 10days,
        reduction = "inst",
        period = nothing,
        output_dir = nothing,
        colors = nothing,
        linestyles = nothing,
        exponent = -3.0,
        figsize = (1200, 800),
        save_format = "pdf",
    )

Generate and save spectra comparison plots for multiple variables across multiple output directories.

# Arguments
- `output_dirs`: Vector of output directory paths to compare
- `short_names`: Vector of variable short names to plot (e.g., ["ke", "hus", "ta"])
- `z_level`: Vertical level (in meters) to slice spectra at (default: 1500.0)
- `time`: Time to use for spectra (default: 10days). Can be a number or LAST_SNAP
- `reduction`: Reduction type (default: "inst")
- `period`: Optional period string (e.g., "1h", "1d"). If nothing, uses default
- `output_dir`: Directory to save plots (default: first output_dir)
- `colors`: Optional vector of colors for each output directory
- `linestyles`: Optional vector of line styles for each output directory
- `exponent`: Exponent for reference power law line (default: -3.0)
- `figsize`: Figure size tuple (default: (1200, 800))
- `save_format`: Format to save plots ("pdf" or "png", default: "pdf")

# Returns
- Vector of saved file paths
"""
function plot_spectra_comparison(
    output_dirs::Vector{String},
    short_names::Vector{String};
    z_level = 1500.0,
    time = 10days,
    reduction = "inst",
    period = nothing,
    output_dir = nothing,
    colors = nothing,
    linestyles = nothing,
    exponent = -3.0,
    figsize = (1200, 800),
    save_format = "pdf",
)
    # Extract coefficient values and create labels, then sort by coefficient value
    dir_label_pairs = map(output_dirs) do dir
        # Try to extract coeff_xxxxx pattern from the full path
        # Look for patterns like "coeff_0.1234" or "coeff_0.12345"
        coeff_match = match(r"coeff_([\d.]+)", dir)
        if !isnothing(coeff_match)
            coeff_value_str = coeff_match.captures[1]
            coeff_value = tryparse(Float64, coeff_value_str)
            if !isnothing(coeff_value)
                # Format as a meaningful label with consistent precision
                label = "Coeff = $coeff_value_str"
                return (dir, label, coeff_value)
            else
                label = "Coeff = $coeff_value_str"
                return (dir, label, Inf)  # Put non-numeric at end
            end
        else
            # Fallback to directory name if no coeff pattern found
            return (dir, basename(dir), Inf)  # Put non-matching at end
        end
    end

    # Sort by coefficient value
    sort!(dir_label_pairs, by = x -> x[3])
    
    # Extract sorted directories and labels
    output_dirs = [pair[1] for pair in dir_label_pairs]
    labels = [pair[2] for pair in dir_label_pairs]

    # Set default output directory (use first sorted directory)
    if isnothing(output_dir)
        output_dir = output_dirs[1]
    end

    # Create SimDir objects
    simdirs = SimDir.(output_dirs)

    saved_files = String[]

    # Process each variable
    for short_name in short_names
        @info "Processing variable: $short_name"

        # Get variables from all directories
        spectra_list = []
        for (i, simdir) in enumerate(simdirs)
            try
                # Get the variable
                if isnothing(period)
                    var = ClimaAnalysis.get(simdir; short_name, reduction)
                else
                    var = ClimaAnalysis.get(simdir; short_name, reduction, period)
                end

                # Slice to desired time
                if time == LAST_SNAP
                    var = slice(var; time = Inf)
                else
                    var = slice(var; time = time)
                end

                # Compute spectrum
                @info "  Computing spectrum for $(labels[i])..."
                spectrum = compute_spectrum(var)

                # Slice to desired z level
                # Check which z dimension exists and slice accordingly
                if haskey(spectrum.dims, "z_reference")
                    spectrum_sliced = slice(spectrum; z_reference = z_level)
                elseif haskey(spectrum.dims, "z")
                    spectrum_sliced = slice(spectrum; z = z_level)
                else
                    error("No z dimension found in spectrum for $short_name")
                end

                push!(spectra_list, spectrum_sliced)
            catch e
                @warn "  Failed to process $(labels[i]) for $short_name: $e"
                continue
            end
        end

        if isempty(spectra_list)
            @warn "  No spectra computed for $short_name, skipping..."
            continue
        end

        # Create figure
        fig = CairoMakie.Figure(; size = figsize)
        ax = CairoMakie.Axis(
            fig[1, 1],
            xlabel = "Log₁₀ Spherical Wavenumber",
            ylabel = "Log₁₀ Power Spectrum",
            title = "Spectrum of $(short_name) at z = $(z_level) m",
        )

        # Plot all spectra on the same axis
        plot_spectra_comparison!(
            ax,
            spectra_list,
            labels;
            colors,
            linestyles,
            exponent,
        )

        # Save figure
        filename = "spectrum_$(short_name)_z$(Int(z_level))m"
        if save_format == "pdf"
            filepath = joinpath(output_dir, "$(filename).pdf")
            CairoMakie.save(filepath, fig)
        elseif save_format == "png"
            filepath = joinpath(output_dir, "$(filename).png")
            CairoMakie.save(filepath, fig)
        else
            error("Unsupported save_format: $save_format. Use 'pdf' or 'png'.")
        end

        push!(saved_files, filepath)
        @info "  Saved: $filepath"
    end

    return saved_files
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Example: Compare spectra from two output directories
    parent_dir = "/Users/akshaysridhar/Research/Data/SpectraHyperdiffusion"
    output_dirs = [
        joinpath(parent_dir, "longrun_aquaplanet_allsky_1M_coeff_0.09285", "output_0000"),
        joinpath(parent_dir, "longrun_aquaplanet_allsky_1M_coeff_0.1238", "output_0000"),
        joinpath(parent_dir, "longrun_aquaplanet_allsky_1M_coeff_0.15475", "output_0000"),
        joinpath(parent_dir, "longrun_aquaplanet_allsky_1M_coeff_0.1857", "output_0000"),
        joinpath(parent_dir, "longrun_aquaplanet_allsky_1M_coeff_0.3714", "output_0000"),
    ]

    # Variables to plot
    short_names = ["ke", "hus", "ta"]

    # Generate comparison plots
    saved_files = plot_spectra_comparison(
        output_dirs,
        short_names;
        z_level = 1500.0,
        time = 50days,
        reduction = "inst",
        save_format = "pdf",
    )

    @info "Generated $(length(saved_files)) spectra comparison plots"
    for file in saved_files
        println("  - $file")
    end
end
