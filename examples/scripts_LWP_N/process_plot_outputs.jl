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
    edmf::String
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
    pattern =
        r"(prognostic|diagnostic)_edmfx_dycoms_rf02_column_qtot0_(\d+\.?\d*)_theta0_(\d+\.?\d*)_thetai_(\d+\.?\d*)_zi_(\d+\.?\d*)_prescribedN_([-\d\.eE]+)"
    m = match(pattern, path)

    # Construct EDMFParams object.
    return EDMFParams(
        path = path,
        edmf = m.captures[1],
        qtot0 = parse(Float64, m.captures[2]), #[6.5, 8.5, 10.5]
        theta0 = parse(Float64, m.captures[3]), #[284, 287, 291, 294]
        thetai = parse(Float64, m.captures[4]), #[6.0, 8.0, 10.0]
        zi = parse(Float64, m.captures[5]), #[500, 800, 1000, 1300]
        N = parse(Float64, m.captures[6]), #[3e7, 1e8, 2.5e8, 5e8]
    )
end

"""
    make_edmf_vec(paths)

Takes in a output path and returns a vector of EDMFParam structures.
"""
# Returns a vector of EDMFParam structures.
function make_edmf_vec(output_dir::String)
    return [
        parse_folder_name(path) for path in readdir(output_dir, join = true)
    ]
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
    plot_1M_edmf_timeseries!(emdfparams, ax)

Base plotting function for 1 moment LWP over time on an Axis object.
"""
function plot_1M_edmf_timeseries!(
    edmfparams::Vector{EDMFParams},
    ax::CairoMakie.Axis,
)
    all_paths = [edmfparam.path for edmfparam in edmfparams]

    # For each path in the given vector, add it on the the Axis object.
    for out in all_paths
        simdir = ClimaAnalysis.SimDir(joinpath("$out", "output_0000"))

        # Liquid water path.
        lwp = ClimaAnalysis.get(
            simdir;
            short_name = "lwp",
            reduction = "inst",
            period = "10m",
        )
        lwp_slice = ClimaAnalysis.slice(lwp, x = 0.0, y = 0.0)
        lwp_skip = ClimaAnalysis.window(lwp_slice, "time", left = 7200)
        lwp_data = lwp_skip.data * 1e3

        # Rain water path.
        rwp = ClimaAnalysis.get(
            simdir;
            short_name = "rwp",
            reduction = "inst",
            period = "10m",
        )
        rwp_slice = ClimaAnalysis.slice(rwp, x = 0.0, y = 0.0)
        rwp_skip = ClimaAnalysis.window(rwp_slice, "time", left = 7200)
        rwp_data = rwp_skip.data * 1e3

        # Time.
        t = ClimaAnalysis.times(lwp_skip) / 3600

        # RWP/LWP for the color bar.
        rwp_lwp_data = rwp_data ./ lwp_data

        CairoMakie.lines!(
            ax,
            t,
            lwp_data,
            color = rwp_lwp_data, # Ratio so don't need to convert.
            colormap = :viridis,
            colorrange = (0, 1),
        )
    end
end

"""
    plot_1M_edmf_N!(emdfparams, ax)

Base plotting function for 1 moment LWP vs N on an Axis object.
"""
function plot_1M_edmf_N!(edmfparams::Vector{EDMFParams}, ax::CairoMakie.Axis)
    all_paths = [(edmfparam.path, edmfparam.N) for edmfparam in edmfparams]

    # For each path in the given vector, add it on the the Axis object.
    for (out, prescribed_N) in all_paths
        simdir = ClimaAnalysis.SimDir(joinpath("$out", "output_0000"))

        # Liquid water path.
        lwp = ClimaAnalysis.get(
            simdir;
            short_name = "lwp",
            reduction = "inst",
            period = "10m",
        )
        lwp_slice = ClimaAnalysis.slice(lwp, x = 0.0, y = 0.0)
        lwp_skip = ClimaAnalysis.window(lwp_slice, "time", left = 7200)
        lwp_data = lwp_skip.data * 1e3 # Convert from kg m^-2 to g m^-2.

        # Rain water path.
        rwp = ClimaAnalysis.get(
            simdir;
            short_name = "rwp",
            reduction = "inst",
            period = "10m",
        )
        rwp_slice = ClimaAnalysis.slice(rwp, x = 0.0, y = 0.0)
        rwp_skip = ClimaAnalysis.window(rwp_slice, "time", left = 7200)
        rwp_data = rwp_skip.data * 1e3 # Convert from kg m^-2 to g m^-2.

        # Number concentration.
        N = fill(prescribed_N, length(lwp_data)) * 1e-6

        # RWP/LWP for the color bar.
        rwp_lwp_data = rwp_data ./ lwp_data

        CairoMakie.scatter!(
            ax,
            lwp_data,
            N,
            color = rwp_lwp_data, # Ratio so don't need to convert.
            markersize = 5,
            colormap = :viridis,
            colorrange = (0, 1),
        )

        CairoMakie.scatter!(ax, lwp_data[1], N[1], color = :grey)
    end
end

"""
    plot_2M_edmf_N!(emdfparams, ax)

Base plotting function for 2 moment LWP over time on an Axis object.
"""
function plot_2M_edmf_timeseries!(
    edmfparams::Vector{EDMFParams},
    ax::CairoMakie.Axis,
)
    # TODO: Implement.
end

"""
    plot_2M_edmf_N!(emdfparams, ax)

Base plotting function for 2 moment LWP vs N on an Axis object.
"""
function plot_2M_edmf_N!(edmfparams::Vector{EDMFParams}, ax::CairoMakie.Axis)
    # TODO: Implement.
end

"""
    decide_plotting!(edmfparams, ax, is_1M, is_time)

Decides which Axis plotting function to call based on is_1M and is_time.
"""
function decide_plotting!(
    edmfparams::Vector{EDMFParams},
    ax::CairoMakie.Axis;
    is_1M::Bool = false,
    is_time::Bool = false,
)
    if is_1M # 1 Moment plotting.
        if is_time
            plot_1M_edmf_timeseries!(edmfparams, ax)
        else
            plot_1M_edmf_N!(edmfparams, ax)
        end
    else # 2 Moment plotting.
        if is_time
            # plot_2M_edmf_timeseries(edmfparams, ax) TODO: IMPLEMENT!
        else
            # plot_2M_edmf_N(edmfparams, ax) TODO: IMPLEMENT!
        end
    end
end

"""
    decide_title(is_1M, is_time)

Decides which labels to use based on is_1M and is_time.
"""
function decide_title(is_1M::Bool = false, is_time::Bool = false)
    # Title add on.
    if is_1M
        title_M = "EDMF+1M"
        save_M = "EDMF_1M"
    else
        title_M = "EDMF+2M"
        save_M = "EDMF_2M"
    end

    # Axis labels setup.
    if is_time
        title = "$title_M LWP Over Time"
        xlab = "t [hr]"
        ylab = "lwp [g m-2]"
        save_title = "$(save_M)_timeseries.png"
        xlims = nothing
        ylims = (1e0, 1e4)
    else
        title = "$title_M LWP vs N"
        xlab = "lwp [g m-2]"
        ylab = "N [cm-3]"
        save_title = "$(save_M)_LWP_N.png"
        xlims = (0.5e0, 0.5e3)
        ylims = (1, 1e3)
    end

    return save_title, title, xlab, ylab, xlims, ylims
end

"""
    plot_edmf(edmfparams, is_1M, is_time, save, replace_save, replace_title)

Generates a basic plot without separation by initial conditions.
"""
function plot_edmf(
    edmfparams::Vector{EDMFParams};
    is_1M::Bool = false,
    is_time::Bool = false,
    save::Bool = false,
    replace_save::String = "",
    replace_title::String = "",
)
    # Base figure.
    fig = CairoMakie.Figure(size = (600, 450))

    # Unpack results from decide_title.
    save_title, title, xlab, ylab, xlims, ylims = decide_title(is_1M, is_time)

    if replace_save != ""
        save_title = replace_save
    end

    if replace_title != ""
        title = replace_title
    end

    ax = CairoMakie.Axis(
        fig[1, 1],
        xlabel = xlab,
        ylabel = ylab,
        title = title,
        xscale = is_time ? identity : log10,
        yscale = log10,
    )
    if !is_time
        CairoMakie.xlims!(ax, xlims)
    end
    CairoMakie.ylims!(ax, ylims)

    decide_plotting!(edmfparams, ax, is_1M = is_1M, is_time = is_time)

    CairoMakie.Colorbar(fig[1, 2], label = "RWP/LWP")

    if save
        CairoMakie.save(save_title, fig)
    end

    return fig
end

"""
    compare_edmf(edmfparams, plotparams, split_by, is_1M, is_time, save, replace_save, replace_title)

Generates plots that are separated out based on their intial conditions for 
easier comparison.
"""
function compare_edmf(
    edmfparams::Vector{EDMFParams},
    plotparams::Dict{Symbol, Vector{Any}};
    split_by::Symbol = :none,
    is_1M::Bool = false,
    is_time::Bool = false,
    save::Bool = false,
    replace_save::String = "",
    replace_title::String = "",
)
    # Keep all params specified in plotparams dictionary.
    keep_params = edmfparams

    # Unpack results from decide_title.
    save_title, title, xlab, ylab, xlims, ylims = decide_title(is_1M, is_time)

    if replace_save != ""
        save_title = replace_save
    end

    if replace_title != ""
        title = replace_title
    end

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
        verbose_title = join(
            [
                replace("$key: $value", "Any" => "") for
                (key, value) in plotparams if key != split_by
            ],
            ", ",
        )

        n = length(keys(grouped))
        max_col = n > 1 ? 2 : 1
        max_row = ceil(Int, n / max_col)
        index = 1

        fig = CairoMakie.Figure(size = (500 * max_col, 450 * max_row))

        for (key, values) in grouped
            row = div(index - 1, 2) + 1
            col = mod(index - 1, 2) + 1
            ax = CairoMakie.Axis(
                fig[row, col],
                xlabel = xlab,
                ylabel = ylab,
                title = "$split_by = $key \n$verbose_title",
                xscale = is_time ? identity : log10,
                yscale = log10,
            )
            if !is_time
                CairoMakie.xlims!(ax, xlims)
            end
            CairoMakie.ylims!(ax, ylims)
            index += 1

            decide_plotting!(values, ax, is_1M = is_1M, is_time = is_time)

        end

        CairoMakie.Colorbar(fig[:, max_col + 1], label = "RWP/LWP")

    else
        fig = plot_edmf(
            keep_params,
            is_1M = is_1M,
            is_time = is_time,
            save = save,
            replace_save = replace_save,
            replace_title = replace_title,
        )
    end

    if save
        CairoMakie.save(save_title, fig)
    end

    return fig

end

"""
"""
function keep(edmfparam::EDMFParams)
    path = edmfparam.path

    simdir = ClimaAnalysis.SimDir(joinpath("$path", "output_0000"))

    # Liquid water path.
    lwp = ClimaAnalysis.get(
        simdir;
        short_name = "lwp",
        reduction = "inst",
        period = "10m",
    )
    lwp_slice = ClimaAnalysis.slice(lwp, x = 0.0, y = 0.0)
    lwp_skip = ClimaAnalysis.window(lwp_slice, "time", left = 7200)
    lwp_data = lwp_skip.data * 1e3 # Convert from kg m^-2 to g m^-2.

    above_cond = all(lwp_data .> 0.5)
    full_len = length(lwp_data) == 61

    return above_cond & full_len
end

"""
"""
function filter_runs(edmfparams::Vector{EDMFParams})
    return [edmfparam for edmfparam in edmfparams if keep(edmfparam)]
end
