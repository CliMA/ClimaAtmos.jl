using Plots

include(joinpath(@__DIR__, "plot_helpers.jl"))

import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--nc_dir"
        help = "NetCDF directory"
        arg_type = String
        "--fig_dir"
        help = "Figure directory"
        arg_type = String
        "--case_name"
        help = "Experiment Name"
        arg_type = String
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

function get_params()
    (s, parsed_args) = parse_commandline()
    nc_dir = parsed_args["nc_dir"]
    fig_dir = parsed_args["fig_dir"]
    case_name = parsed_args["case_name"]
    if isnothing(fig_dir)
        fig_dir = joinpath(nc_dir, "fig")
    end
    if isnothing(case_name)
        case_name = "aquaplanet"
    end
    mkpath(fig_dir)

    nc_files = filter(
        x -> startswith(basename(x), "day") && endswith(x, ".nc"),
        readdir(nc_dir, join = true),
    )

    return (; nc_files, fig_dir, case_name)
end

(; nc_files, fig_dir, case_name) = get_params()

if case_name == "dry_baroclinic_wave"
    generate_paperplots(case_name, args...) =
        generate_paperplots_dry_baro_wave(args...)
elseif case_name == "moist_baroclinic_wave"
    generate_paperplots(case_name, args...) =
        generate_paperplots_moist_baro_wave(args...)
elseif case_name == "dry_held_suarez"
    generate_paperplots(case_name, args...) =
        generate_paperplots_held_suarez(args...; moist = false)
elseif case_name == "moist_held_suarez" || case_name == "aquaplanet"
    generate_paperplots(case_name, args...) =
        generate_paperplots_held_suarez(args...; moist = true)
else
    error("Unknown `case_name`: $(case_name)")
end

generate_paperplots(case_name, fig_dir, nc_files)
