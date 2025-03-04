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


output_dir = "output/exp_1" # output directory
config_i = 1 # config to plot
ylims = (0, 4000) # y limits for plotting (`z` coord)
iterations = nothing # iterations to plot (i.e., 0:2). default is all iterations
var_names =
    ("thetaa", "hus", "clw", "arup", "entr", "detr", "waup", "tke", "turbentr")
reduction = "inst"

config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))
const z_max = config_dict["z_max"]
const z_cal_grid = config_dict["z_cal_grid"]
const cal_var_names = config_dict["y_var_names"]
const const_noise_by_var = config_dict["const_noise_by_var"]
const n_vert_levels = config_dict["dims_per_var"]
model_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))

ref_paths, _ = get_les_calibration_library()
atmos_config = CA.AtmosConfig(model_config_dict)


if isnothing(iterations)
    iterations = get_iters_with_config(config_i, config_dict)
end

xlims_dict = Dict("arup" => (-0.1, 0.4), "clw" => "auto")


function compute_plot_limits(data; margin_ratio = 0.5, fixed_margin = 1.0)

    min_val = minimum(data)
    max_val = maximum(data)

    data_range = max_val - min_val
    margin = data_range == 0 ? fixed_margin : margin_ratio * data_range
    limits = (min_val - margin, max_val + margin)

    return limits
end


for iteration in iterations
    @show "Iter: $iteration"
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

        Plots.plot!(
            xlabel = var_name,
            ylabel = "Height (z)",
            title = var_name,
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
