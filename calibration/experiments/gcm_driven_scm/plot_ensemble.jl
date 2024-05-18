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


config_dict = YAML.load_file("experiment_config.yml")
model_config_dict = YAML.load_file("model_config.yml")


function ensemble_data(
    iteration;
    var_name = "hus",
    reduction = "inst",
    output_dir = nothing,
)

    # if !isnothing(output_dir)
    #     output_dir = output_dir
    #     (; ensemble_size,) = config
    # else
    #     (; ensemble_size, output_dir) = config
    # end
    # dims = 60 # num vertical levels x num variables 


    # G_ensemble = Array{Float64}(undef, dims..., ensemble_size)
    G_ensemble = Array{Float64}(
        undef,
        60,
        config_dict["ensemble_size"],
    )

    # f = JLD2.jldopen("norm_vec_obs.jld2", "r+")
    # norm_vec_obs = f["norm_vec_obs"][:]

    f_diagnostics = JLD2.jldopen(
        joinpath(config_dict["output_dir"], "norm_vec_obs.jld2"),
        "r+",
    )

    for m in 1:config_dict["ensemble_size"]
        member_path =
            TOMLInterface.path_to_ensemble_member(output_dir, iteration, m)
        simdir = SimDir(joinpath(member_path, "output_active"))
        try
            G_ensemble[:, m] .=
                get_var_data(simdir,
                var_name; 
                    # y_names = config_dict["y_var_names"],
                    t_start = config_dict["g_t_start_sec"],
                    t_end = config_dict["g_t_end_sec"],)
        catch err
            @info "Error during observation map for ensemble member $m" err
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function get_var_data(
    simdir,
    var_name;
    # y_names, 
    t_start,
    t_end,
    reduction = "inst",
)

    var_i = get(simdir; short_name = var_name, reduction = reduction)
    sim_t_end = var_i.dims["time"][end]

    if sim_t_end < 0.95 * t_end
        throw(ErrorException("Simulation failed."))
    end
    # take time-mean of last N hours
    var_i_ave = average_time(
        window(
            var_i,
            "time",
            left = t_start,
            right = sim_t_end,
        ),
    )

    y_var_i = slice(var_i_ave, x = 1, y = 1).data


    return y_var_i

end



# get eki object 
output_dir = joinpath("output", "gcm_driven_scm")


cal_vars = ["thetaa", "hus", "clw"]

var_names = ("thetaa", "hus", "clw", "arup", "entr", "detr", "waup") #wa")
# var_name = "arup"
reduction = "inst"

iteration = 0


atmos_config = CA.AtmosConfig(model_config_dict)
experiment_config_dict = YAML.load_file("experiment_config.yml")

# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config)

# zc_model = collect(33.333333:66.666666:4000.0)

f = JLD2.jldopen("norm_vec_obs.jld2", "r+")
norm_vec_obs = f["norm_vec_obs"][:]

for var_name in var_names

    data = ensemble_data(
        iteration;
        var_name = var_name,
        reduction = reduction,
        output_dir = output_dir,
    )
    eki_filepath = joinpath(
        ClimaCalibrate.path_to_iteration(output_dir, iteration),
        "eki_file.jld2",
    )
    eki = JLD2.load_object(eki_filepath)
    prior_path = joinpath("./prior.toml")
    prior = ClimaCalibrate.get_prior(prior_path)

    plot(legend = false)  # Initialize an empty plot
    for i in 1:size(data, 2)

        y_var = data[:, i]

        plot!(y_var, zc_model)  # Append each column to the plot
    end
    if in(var_name, cal_vars)
        # y_truth = eki.obs_mean * sqrt(norm_vec_obs[1])
        y_truth, Σ_obs, norm_vec_obs = get_obs(
            model_config_dict["external_forcing_file"],
            [var_name],
            zc_model;
            ti = experiment_config_dict["y_t_start_sec"],
            tf = experiment_config_dict["y_t_end_sec"],
            Σ_const = 0.05,
            z_score_norm = false, 
        )
        plot!(y_truth, zc_model, color = :black, label = "LES")
    end

    plot!(
        xlabel = var_name,
        ylabel = "Height (z)",
        title = "Matrix Columns vs. Height",
    )

    plot_rel_dir = joinpath(output_dir, "plots", "ensemble_plots")
    if !isdir(plot_rel_dir)
        mkpath(plot_rel_dir)
    end

    savefig(
        joinpath(plot_rel_dir, "ensemble_plot_$(var_name)_i_$(iteration).png"),
    )

end
