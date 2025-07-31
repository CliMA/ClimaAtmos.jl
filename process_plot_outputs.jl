"""
    process_plot_output.jl

A script to organize all of the output paths by their initial conditions.
"""

# Dependencies.
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
        qtot0 = parse(Float64, m.captures[1]), #[6.5, 8.5, 10.5]
        theta0 = parse(Float64, m.captures[2]), #[284, 287, 290, 294]
        thetai = parse(Float64, m.captures[3]), #[6.0, 8.0, 10.0]
        zi = parse(Float64, m.captures[4]), #[500, 800, 1000, 1300]
        N = parse(Float64, m.captures[5]), #[3e7, 1e8, 2.5e8, 5e8]
    )
end

"""
    make_edmf_vec(paths)

Takes in a output path and returns a vector of EDMFParam structures.
"""
# Returns a vector of EDMFParam structures.
function make_edmf_vec(output_dir::String)
    return [parse_folder_name(path) for path in readdir(output_dir, join=true)]
end

"""
    group_by_fields(emdfparams, field)

Takes a vector of EDMFParam objects and returns a dictionary where they're 
grouped by values of a specific field.
"""
function group_by_field(edmfparams::Vector{EDMFParams}, field::Symbol)
    grouped = Dict{Any, Vector{EDMFParams}}()

    for edmfparam in edmfparams
        key = getproperty(edmfparam, field)
        push!(get!(grouped, key, Vector{EDMFParams}()), edmfparam)
    end

    return grouped
end

"""
    plot_1M_edmf(output_dir, emdfparams)

# Base plotting function for LWP over time on an Axis object.
"""
function plot_1M_edmf(edmfparams::Vector{EDMFParams}, ax::CairoMakie.Axis)
    all_paths = [edmfparam.path for edmfparam in edmfparams]

    # For each path in the given vector, add it on the the Axis object.
    for out in all_paths
        simdir = ClimaAnalysis.SimDir(
            joinpath("$out", "output_0000")
        )

        # Liquid water path.
        lwp = ClimaAnalysis.get(
            simdir; 
            short_name="lwp", 
            reduction="inst", 
            period="10m"
            )
        lwp_slice = ClimaAnalysis.slice(lwp, x=0.0, y=0.0)

        # Rain water path.
        rwp = ClimaAnalysis.get(
            simdir; 
            short_name="rwp", 
            reduction="inst",
            period="10m"
            )
        rwp_slice = ClimaAnalysis.slice(rwp, x=0.0, y=0.0)

        # Time.
        t = ClimaAnalysis.times(lwp_slice)

        # RWP/LWP for the color bar.
        rwp_lwp_slice = rwp_slice.data ./ lwp_slice.data
        rwp_lwp_filtered = [isfinite(c) ? c : 0.0 for c in rwp_lwp_slice]

        CairoMakie.lines!(
            ax, 
            t, 
            lwp_slice.data.*1000, # Convert from kg m^-2 to g m^-2.
            color=rwp_lwp_filtered, # Ratio so don't need to convert.
            colormap = :viridis, 
            colorrange = (0, 1)
            )
    end
end

"""
"""
function plot_1M_basic(edmfparams::Vector{EDMFParams} ; title::String="LWP Over Time")
    # Base figure.
    fig = CairoMakie.Figure(size = (600, 450))
    ax = CairoMakie.Axis(
        fig[1, 1],
        xlabel = "t [s]",
        ylabel = "lwp [g m-2]",
        title = title,
        yscale = log10,
        )
    CairoMakie.ylims!(ax, (1, 1e4))

    plot_1M_edmf(edmfparams, ax)

    CairoMakie.Colorbar(fig[1, 2], label = "RWP/LWP")

    return fig
end 

"""
"""
function compare_1M_edmf(edmfparams::Vector{EDMFParams}, plotparams::Dict{Symbol, Vector{Float64}}; split_by::Symbol=:none)
    # Keep all params specified in plotparams dictionary.
    keep_params = edmfparams

    for (key, values) in plotparams
        edmfparam_add = Vector{EDMFParams}()
        emdfparam_dict = group_by_field(keep_params, key)

        for val in values
            edmfparam_add = vcat(edmfparam_add, emdfparam_dict[val])
        end
        keep_params = edmfparam_add
    end

    # Split the remaining strings by the attribute specified in split_by.
    if split_by != :none
        grouped = group_by_field(keep_params, split_by)
        verbose_title = join(["$key: $values" for (key, values) in plotparams if key != split_by], ", ")

        n = length(keys(grouped))
        max_col = 2
        max_row = ceil(Int, n / max_col)

        index = 1

        fig = CairoMakie.Figure(size = (500 * max_col, 450 * max_row))

        for (key, values) in grouped
            row = div(index - 1, 2) + 1
            col = mod(index - 1, 2) + 1
            ax = CairoMakie.Axis(
            fig[row, col],
            xlabel = "t [s]",
            ylabel = "lwp [g m-2]",
            title = "$split_by = $key | $verbose_title",
            yscale = log10,
            )
            CairoMakie.ylims!(ax, (1, 1e4))
            index += 1
            plot_1M_edmf(values, ax)
        end

        CairoMakie.Colorbar(fig[:, max_col+1], label = "RWP/LWP")

    else
        fig = plot_1M_basic(keep_params)
    end

    return fig

end
