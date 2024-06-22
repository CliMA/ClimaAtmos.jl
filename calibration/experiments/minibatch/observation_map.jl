import EnsembleKalmanProcesses as EKP
import ClimaCalibrate: observation_map, ExperimentConfig, path_to_ensemble_member
using ClimaAnalysis
using JLD2

const config_dict = YAML.load_file(joinpath(@__DIR__, "experiment_config.yml"))
const output_dir = config_dict["output_dir"]

function observation_map(iteration)

    G_ensemble = Array{Float64}(
        undef,
        config_dict["dims"]...,
        config_dict["ensemble_size"],
    )

    f_diagnostics = JLD2.jldopen(
        joinpath(config_dict["output_dir"], "norm_factors.jld2"),
        "r+",
    )

    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    for m in 1:config_dict["ensemble_size"]
        member_path = path_to_ensemble_member(
            config_dict["output_dir"],
            iteration,
            m,
        )
        try
            G_ensemble[:, m] .= process_member_data(
                member_path, eki;
                y_names = config_dict["y_var_names"],
                t_start = config_dict["g_t_start_sec"],
                t_end = config_dict["g_t_end_sec"],
                norm_factors_dict = f_diagnostics["norm_factors_dict"],
            )
        catch err
            @info "Error during observation map for ensemble member $m" err
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(
    member_path,
    eki;
    y_names,
    reduction = "inst",
    t_start,
    t_end,
    norm_factors_dict = nothing,
)
    forcing_file_indices = EKP.get_current_minibatch(eki)
    g = Float64[]
    for i in forcing_file_indices
        # simdir = SimDir(joinpath(member_path * "_config_$i", "output_active"))
        simdir = SimDir(joinpath(member_path, "config_$i", "output_active"))
        for (i, y_name) in enumerate(y_names)
            var_i = get(simdir; short_name = y_name, reduction)
            sim_t_end = var_i.dims["time"][end]

            if sim_t_end < 0.95 * t_end
                throw(ErrorException("Simulation failed."))
            end
            # take time-mean
            var_i_ave = average_time(
                window(var_i, "time", left = t_start, right = sim_t_end),
            )

            y_var_i = slice(var_i_ave, x = 1, y = 1).data
            if !isnothing(norm_factors_dict)
                y_μ, y_σ = norm_factors_dict[y_name]
                y_var_i = (y_var_i .- y_μ) ./ y_σ
            end

            append!(g, y_var_i)
        end
    end

    return g
end
