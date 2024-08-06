import EnsembleKalmanProcesses: TOMLInterface
import ClimaCalibrate: observation_map, ExperimentConfig
using ClimaAnalysis
using JLD2
using YAML

config_dict = YAML.load_file("experiment_config.yml")

function observation_map(iteration)

    G_ensemble = Array{Float64}(
        undef,
        config_dict["dims"]...,
        config_dict["ensemble_size"],
    )

    # f_diagnostics = JLD2.jldopen(
    #     joinpath(config_dict["output_dir"], "norm_vec_obs.jld2"),
    #     "r+",
    # )

    for m in 1:config_dict["ensemble_size"]
        member_path = TOMLInterface.path_to_ensemble_member(
            config_dict["output_dir"],
            iteration,
            m,
        )
        simdir = SimDir(joinpath(member_path, "output_active"))
        try
            G_ensemble[:, m] .= process_member_data(
                simdir;
                y_names = config_dict["y_var_names"],
                t_start = config_dict["g_t_start_sec"],
                t_end = config_dict["g_t_end_sec"],
                # norm_vec_obs = f_diagnostics["norm_vec_obs"],
            )
        catch err
            @info "Error during observation map for ensemble member $m" err
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(
    simdir;
    y_names,
    reduction = "inst",
    t_start,
    t_end,
    # norm_vec_obs = [0.0, 1.0],
    # normalize = true,
)

    g = Float64[]

    for (i, y_name) in enumerate(y_names)
        var_i = get(simdir; short_name = y_name, reduction = reduction)
        sim_t_end = var_i.dims["time"][end]

        if sim_t_end < 0.95 * t_end
            throw(ErrorException("Simulation failed."))
        end
        # take time-mean
        var_i_ave = average_time(
            window(var_i, "time", left = t_start, right = sim_t_end),
        )

        y_var_i = slice(var_i_ave, x = 1, y = 1).data
        # if normalize
        #     y_μ, y_σ = norm_vec_obs[i, 1], norm_vec_obs[i, 2]
        #     y_var_i = (y_var_i .- y_μ) ./ y_σ
        # end

        append!(g, y_var_i)
    end

    return g
end