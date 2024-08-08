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


config_i = 1 # config to plot
N_VERT_LEVELS = 30 # variables to plot
var_names = ("thetaa", "hus", "clw", "arup", "entr", "detr", "waup", "tke")
reduction = "inst"


config_dict = YAML.load_file("experiment_config.yml")
const model_config = config_dict["model_config"]
const z_max = config_dict["z_max"]
const cal_var_names = config_dict["y_var_names"]
const output_dir = config_dict["output_dir"]
model_config_dict = YAML.load_file(model_config)

ref_paths, _ = get_les_calibration_library()
atmos_config = CA.AtmosConfig(model_config_dict)

# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config, z_max = z_max)

iterations = get_iters_with_config(config_i, config_dict)

xlims_dict = Dict("arup" => (-0.1, 1.2))

for iteration in iterations
    @show "Iter: $iteration"
    for var_name in var_names

        data = ensemble_data(
            iteration,
            config_i;
            var_name = var_name,
            reduction = reduction,
            output_dir = output_dir,
            z_max = z_max,
            n_vert_levels = N_VERT_LEVELS,
        )
        # size(data)= [num vertical levels x ensemble size]
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
            y_truth, Σ_obs, norm_vec_obs = get_obs(
                ref_paths[config_i],
                [var_name],
                zc_model;
                ti = config_dict["y_t_start_sec"],
                tf = config_dict["y_t_end_sec"],
                Σ_const = 0.05,
                z_score_norm = false,
            )
            Plots.plot!(y_truth, zc_model, color = :black, label = "LES")
        end


        xlims = get(xlims_dict, var_name, nothing)

        Plots.plot!(
            xlabel = var_name,
            ylabel = "Height (z)",
            title = "Matrix Columns vs. Height",
            xlims = xlims,
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
