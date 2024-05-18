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


config_dict = YAML.load_file("experiment_config.yml")
const model_config = config_dict["model_config"]
model_config_dict = YAML.load_file(model_config)

const output_dir = config_dict["output_dir"]


ref_paths = get_les_calibration_library()

N_VERT_LEVELS = 190


function get_iters_with_config(config_i::Int)
    config_i_dir = "config_$config_i"
    iters_with_config = []
    for iter in 0:config_dict["n_iterations"]
        for m in 1:config_dict["ensemble_size"]
            member_path =
                TOMLInterface.path_to_ensemble_member(output_dir, iter, m)

            if isdir(member_path)
                dirs = filter(
                    entry -> isdir(joinpath(member_path, entry)),
                    readdir(member_path),
                )

                if (config_i_dir in dirs) & ~(iter in iters_with_config)
                    push!(iters_with_config, iter)
                end

            else
                @show "Iteration not reached: $iter"
            end

        end
    end
    return iters_with_config
end


function ensemble_data(
    iteration,
    config_i::Int,
    ;
    var_name = "hus",
    reduction = "inst",
    output_dir = nothing,
)

    G_ensemble =
        Array{Float64}(undef, N_VERT_LEVELS, config_dict["ensemble_size"])

    for m in 1:config_dict["ensemble_size"]

        try
            member_path =
                TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
            simdir = SimDir(
                joinpath(member_path, "config_$config_i", "output_active"),
            )
            G_ensemble[:, m] .= get_var_data(
                simdir,
                var_name;
                t_start = config_dict["g_t_start_sec"],
                t_end = config_dict["g_t_end_sec"],
            )
        catch err
            @info "Error during observation map for ensemble member $m" err
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function get_var_data(simdir, var_name; t_start, t_end, reduction = "inst")

    var_i = get(simdir; short_name = var_name, reduction = reduction)
    sim_t_end = var_i.dims["time"][end]

    if sim_t_end < 0.95 * t_end
        throw(ErrorException("Simulation failed."))
    end
    # take time-mean of last N hours
    var_i_ave =
        average_time(window(var_i, "time", left = t_start, right = sim_t_end))

    y_var_i = slice(var_i_ave, x = 1, y = 1).data


    return y_var_i

end




cal_vars = ["thetaa", "hus", "clw"]

var_names = ("thetaa", "hus", "clw", "arup", "entr", "detr", "waup", "tke") #wa")
reduction = "inst"

config_i = 1

atmos_config = CA.AtmosConfig(model_config_dict)

# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config)

iterations = get_iters_with_config(config_i)

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
        )
        eki_filepath = joinpath(
            ClimaCalibrate.path_to_iteration(output_dir, iteration),
            "eki_file.jld2",
        )
        eki = JLD2.load_object(eki_filepath)
        prior_path = joinpath(config_dict["prior"])
        prior = ClimaCalibrate.get_prior(prior_path)

        plot(legend = false)  # Initialize an empty plot
        for i in 1:size(data, 2)

            y_var = data[:, i]

            plot!(y_var, zc_model)  # Append each column to the plot
        end
        if in(var_name, cal_vars)
            # y_truth = eki.obs_mean * sqrt(norm_vec_obs[1])
            y_truth, Σ_obs, norm_vec_obs = get_obs(
                ref_paths[config_i],
                [var_name],
                zc_model;
                ti = config_dict["y_t_start_sec"],
                tf = config_dict["y_t_end_sec"],
                Σ_const = 0.05,
                z_score_norm = false,
            )
            plot!(y_truth, zc_model, color = :black, label = "LES")
        end


        xlims = get(xlims_dict, var_name, nothing)

        plot!(
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
