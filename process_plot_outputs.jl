"""
    process_plot_output.jl

A script to organize all of the output paths by their initial conditions.
"""

import ClimaAnalysis
import CairoMakie
import ClimaAnalysis.Visualize as viz

"""
    EDMFParams

A structure for the output paths and their corresponding initial conditions.
"""
Base.@kwdef struct EDMFParams
    path::String
    qtot0::Float64
    theta0::Float64
    thetai::Float64
    zi::Float64
    N::Float64
end

"""
    parse_folder_name(path)

Use regex to get the initial conditions embedded in each output path and 
constructs a EDMFParams object.
"""
function parse_folder_name(path::String)
    # Regex expression.
    pattern = r"prognostic_edmfx_dycoms_rf02_column_qtot0_(\d+\.?\d*)_theta0_(\d+\.?\d*)_thetai_(\d+\.?\d*)_zi_(\d+\.?\d*)_prescribedN_([-\d\.eE]+)"
    m = match(pattern, path)

    # Construct EDMFParams object.
    return EDMFParams(
        path = path,
        qtot0 = parse(Float64, m.captures[1]),
        theta0 = parse(Float64, m.captures[2]),
        thetai = parse(Float64, m.captures[3]),
        zi = parse(Float64, m.captures[4]),
        N = parse(Float64, m.captures[5]),
    )
end

"""
    make_edmf_vec(paths)

Takes in a output paths from readdir and returns a vector of EDMFParam 
structures.
"""
# Returns a vector of EDMFParam structures.
function make_edmf_vec(paths::Vector{String})
    return [parse_folder_name(path) for path in paths]
end

"""
    plot_1M_edmf(output_dir, emdfparams)

# Plots all LWP over time for EDMF+1M outputs. Uses LWP over time since no CDNC.
"""
function plot_1M_edmf(output_dir::String, edmfparams::Vector{EDMFParams})
    # Base figure.
    fig = CairoMakie.Figure(size = (600, 450))
    ax = CairoMakie.Axis(
        fig[1, 1],
        xlabel = "t [s]",
        ylabel = "lwp [g m-2]",
        title = "LWP Over Time",
        yscale = log10,
        )
    CairoMakie.ylims!(ax, (1, 1e4))

    all_paths = [edmfparam.path for edmfparam in edmfparams]

    for out in all_paths
        simdir = ClimaAnalysis.SimDir(
            joinpath(output_dir, "$out", "output_0000")
        )

        lwp = get(simdir; short_name="lwp", reduction="inst", period="10m")
        lwp_slice = slice(lwp, x=0.0, y=0.0)
        rwp = get(simdir; short_name="rwp", reduction="inst", period="10m")
        rwp_slice = slice(rwp, x=0.0, y=0.0)
        t = ClimaAnalysis.times(lwp_slice)

        rwp_lwp_slice = rwp_slice.data ./ lwp_slice.data
        rwp_lwp_filtered = [isfinite(c) ? c : 0.0 for c in rwp_lwp_slice]

        CairoMakie.lines!(
            ax, 
            t, 
            lwp_slice.data.*1000, 
            color=rwp_lwp_filtered, 
            colormap = :viridis, 
            colorrange = (0, 1)
            )
    end

    CairoMakie.Colorbar(fig[1, 2], label = "RWP/LWP")

    return fig

end
