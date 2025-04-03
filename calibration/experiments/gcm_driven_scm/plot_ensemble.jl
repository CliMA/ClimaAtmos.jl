using ArgParse
import EnsembleKalmanProcesses: TOMLInterface
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaCalibrate: observation_map, ExperimentConfig
using ClimaAnalysis
import ClimaCalibrate
using Plots
using JLD2
using Statistics
using YAML

include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")


function parse_command_line_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--output_dir"
        help = "Directory for output files"
        arg_type = String
        default = "output/exp_1"

        "--figs_dir"
        help = "Directory for output figs"
        arg_type = String
        default = "figs/exp_default"

        "--config_i"
        help = "Config index to plot"
        arg_type = Int
        default = 1

        "--iterations"
        help = "Iterations to plot (as i.e., `0:5` or `3`) (default is all iterations)"
        arg_type = String
        default = nothing
    end
    return parse_args(s)
end

# Function to get all config indices from a given iteration and member directory
function get_config_indices(output_dir, iteration)
    iteration_dir = joinpath(output_dir, "iteration_$(iteration)")
    member_dirs = filter(isdir, readdir(iteration_dir, join = true))
    config_indices = Set{Int}()

    for member_dir in member_dirs
        config_dirs = filter(isdir, readdir(member_dir, join = true))
        for config_dir in config_dirs
            config_name = basename(config_dir)
            if startswith(config_name, "config_")
                config_index = parse(Int, split(config_name, "_")[2])
                push!(config_indices, config_index)
            end
        end
    end

    return sort(collect(config_indices))
end

# Get command-line arguments
parsed_args = parse_command_line_args()

# Assign variables based on the parsed arguments
output_dir = parsed_args["output_dir"]
figs_dir = parsed_args["figs_dir"]
config_i = parsed_args["config_i"]

# ylims = (0, 4000) # y limits for plotting (`z` coord)
ylims = nothing
# iterations = !isnothing(parsed_args["iterations"]) ? parse(Int, parsed_args["iterations"]) : nothing
if !isnothing(parsed_args["iterations"])
    if occursin(":", parsed_args["iterations"])
        start_iter, end_iter = split(parsed_args["iterations"], ":")
        iterations = parse(Int, start_iter):parse(Int, end_iter)
    else
        iterations = parse(Int, parsed_args["iterations"])
    end
else
    iterations = nothing
end

var_names = (
    "thetaa",
    "hus",
    "clw",
    "entr",
    "detr",
    "waup",
    "tke",
    "arup",
    "turbentr",
    "lmix",
    "nh_pressure",
) # "cli"
reduction = "inst"

config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
const z_max = config_dict["z_max"]
const z_cal_grid = config_dict["z_cal_grid"]
cal_var_names = config_dict["y_var_names"]
push!(cal_var_names, "wap")
const const_noise_by_var = config_dict["const_noise_by_var"]
const n_vert_levels = config_dict["dims_per_var"]
model_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))

ref_paths, _ = get_les_calibration_library()
atmos_config = CA.AtmosConfig(model_config_dict)

# Initialize config_indices
config_indices = []

# If config_i is not specified, infer from directory structure
if isnothing(config_i)
    iterations = get_iters_with_config(config_i, config_dict)
    config_indices = get_config_indices(output_dir, iterations[1])
else
    config_indices = [config_i]
end

xlims_dict = Dict(
    "arup" => (-0.1, 0.25),
    "entr" => (-1e-6, 0.005),
    "detr" => (-1e-6, 0.005),
    "clw" => "auto",
    "cli" => "auto",
    "lmix" => (-0.25, 750),
)


function compute_plot_limits(data; margin_ratio = 2.5, fixed_margin = 1.0)

    min_val = minimum(data)
    max_val = maximum(data)

    data_range = max_val - min_val
    margin = data_range == 0 ? fixed_margin : margin_ratio * data_range
    limits = (min_val - margin, max_val + margin)

    return limits
end


for iteration in iterations
    @show "Iter: $iteration"
    for config_i in config_indices
        @show "Config: $config_i"
        for var_name in var_names

            data, zc_model = ensemble_data(
                process_profile_variable,
                iteration,
                config_i,
                config_dict;
                var_name = var_name,
                reduction = reduction,
                output_dir = output_dir,
                z_max = z_max,
                n_vert_levels = n_vert_levels,
                return_z_interp = true,
            )

            # size(data) = [num vertical levels x ensemble size]
            eki_filepath = joinpath(
                ClimaCalibrate.path_to_iteration(output_dir, iteration),
                "eki_file.jld2",
            )
            eki = JLD2.load_object(eki_filepath)

            plot(legend = false)
            for i in 1:size(data, 2)

                y_var = data[:, i]
                Plots.plot!(y_var, zc_model)
            end
            if in(var_name, cal_var_names)
                ref_path = ref_paths[config_i]
                cfsite_number, _, _, _ = parse_les_path(ref_path)
                forcing_type = get_cfsite_type(cfsite_number)

                ti = config_dict["y_t_start_sec"]
                ti = isa(ti, AbstractFloat) ? ti : ti[forcing_type]
                tf = config_dict["y_t_end_sec"]
                tf = isa(tf, AbstractFloat) ? tf : tf[forcing_type]

                y_truth, Σ_obs, norm_vec_obs = get_obs(
                    ref_path,
                    [var_name],
                    zc_model;
                    ti = ti,
                    tf = tf,
                    Σ_const = const_noise_by_var,
                    z_score_norm = false,
                )
                Plots.plot!(y_truth, zc_model, color = :black, label = "LES")
            end


            xlims = get(xlims_dict, var_name, nothing)

            if xlims == "auto"
                xlims = compute_plot_limits(y_truth)
            end

            cfsite_info = get_cfsite_info_from_path(ref_paths[config_i])
            forcing_model = cfsite_info["forcing_model"]
            experiment = cfsite_info["experiment"]
            month = cfsite_info["month"]
            cfsite_number = cfsite_info["cfsite_number"]

            Plots.plot!(
                xlabel = var_name,
                ylabel = "Height (z)",
                title = "$var_name (cfsite: $cfsite_number, month: $month, model: $forcing_model)",
                xlims = xlims,
                ylims = ylims,
            )

            plot_rel_dir = joinpath(
                output_dir,
                "plots",
                "ensemble_plots",
                "config_$(config_i)",
            )
            if !isdir(plot_rel_dir)
                mkpath(plot_rel_dir)
            end

            savefig(
                joinpath(
                    plot_rel_dir,
                    "ensemble_plot_$(var_name)_i_$(iteration).png",
                ),
            )
        end
    end
end
