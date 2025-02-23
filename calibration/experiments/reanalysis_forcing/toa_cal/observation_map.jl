import EnsembleKalmanProcesses as EKP
import ClimaCalibrate:
    observation_map, ExperimentConfig, path_to_ensemble_member, path_to_iteration
using ClimaAnalysis
using JLD2
using Statistics

function observation_map(iteration; config_dict::Dict)

    full_dim =
        config_dict["dims_per_var"] *
        length(config_dict["y_var_names"]) *
        config_dict["batch_size"]
    
    G_ensemble =
        Array{Float64}(undef, full_dim..., config_dict["ensemble_size"])

    iter_path = path_to_iteration(config_dict["output_dir"], iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    for m in 1:config_dict["ensemble_size"]
        member_path =
            path_to_ensemble_member(config_dict["output_dir"], iteration, m)
        try
            G_ensemble[:, m] .= process_member_data(
                member_path,
                eki;
                y_names = config_dict["y_var_names"],
                t_start = config_dict["g_t_start_sec"],
                t_end = config_dict["g_t_end_sec"],
                z_max = config_dict["z_max"],
                norm_factors_dict = config_dict["norm_factors_by_var"],
                log_vars = config_dict["log_vars"],
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
    z_max = "nothing",
    norm_factors_dict = nothing,
    log_vars = [],
)
    forcing_file_indices = EKP.get_current_minibatch(eki)
    g = Float64[]
    for i in forcing_file_indices
        simdir = SimDir(joinpath(member_path, "config_$i", "output_0000"))
        for (i, y_name) in enumerate(y_names)
            y_var_i = process_profile_variable(
                simdir,
                y_name;
                reduction,
                t_start,
                t_end,
                z_max,
                norm_factors_dict,
                log_vars,
            )
            append!(g, y_var_i)
        end
    end
    return g
end


function process_profile_variable(
    simdir,
    y_name;
    reduction,
    t_start,
    t_end,
    z_max = "nothing",
    norm_factors_dict = nothing,
    log_vars = [],
)
    #println("Processing $y_name")
    if y_name == "rsut"
        return process_rsut(simdir, reduction, t_start, t_end)
    end

    var_i = get(simdir; short_name = y_name, reduction = reduction)

    # subset vertical coordinate
    if z_max ∉ ("nothing", nothing) # depends on how its read in
        z_window = filter(x -> x <= z_max, var_i.dims["z"])
        var_i = window(var_i, "z", right = maximum(z_window))
    end
    sim_t_end = var_i.dims["time"][end]
    #println("sim time: $(sim_t_end/86400.) days")

    if sim_t_end < 0.95 * t_end
        throw(ErrorException("Simulation failed."))
    end

    # take time-mean
    var_i_ave =
        average_time(window(var_i, "time", left = t_start, right = sim_t_end))
    y_var_i = slice(var_i_ave, x = 1, y = 1).data

    if y_name in log_vars
        y_var_i = log10.(y_var_i .+ 1e-12)
    end

    # normalize
    # if !isnothing(norm_factors_dict)
    #     y_μ, y_σ = norm_factors_dict[y_name]
    #     y_var_i = (y_var_i .- y_μ) ./ y_σ
    # end

    return y_var_i

end

function process_rsut(simdir, reduction, t_start, t_end)
    # get rsdt (TOA incoming solar radiation) and rsut (TOA outgoing solar radiation)
    # net TOA = rsdt - rsut
    @info "Computing net shortwave radiation, NOT rsut!"
    rsdt = get(simdir; short_name = "rsdt", reduction = reduction)
    rsut = get(simdir; short_name = "rsut", reduction = reduction)

    sim_t_end = rsut.dims["time"][end]
    if sim_t_end < 0.95 * t_end
        throw(ErrorException("Simulation failed."))
    end

    rsdt_ave = average_time(window(rsdt, "time", left = t_start, right = sim_t_end))
    rsut_ave = average_time(window(rsut, "time", left = t_start, right = sim_t_end))

    rsdt_data = slice(rsdt_ave, x = 1, y = 1).data
    rsut_data = slice(rsut_ave, x = 1, y = 1).data

    # net shortwave radiation is incoming minus outgoing shortwave radiation
    net_shortwave_radiation = rsdt_data .- rsut_data

    return net_shortwave_radiation
end
