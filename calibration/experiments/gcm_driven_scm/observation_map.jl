import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL
import ClimaCalibrate: path_to_ensemble_member
using ClimaAnalysis
using JLD2
using Statistics
using Logging

"""Suppress Info and Warnings for any function"""
function suppress_logs(f, args...; kwargs...)
    Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        f(args...; kwargs...)
    end
end

function CAL.observation_map(
    iteration;
    config_dict = YAML.load_file(
        joinpath(dirname(Base.active_project()), "experiment_config.yml"),
    ),
)

    full_dim =
        config_dict["dims_per_var"] *
        length(config_dict["y_var_names"]) *
        config_dict["batch_size"]

    G_ensemble =
        Array{Float64}(undef, full_dim..., config_dict["ensemble_size"])

    iter_path = CAL.path_to_iteration(config_dict["output_dir"], iteration)
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
                z_cal_grid = config_dict["z_cal_grid"],
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
    z_max = nothing,
    z_cal_grid = nothing,
    norm_factors_dict = nothing,
    log_vars = [],
)
    forcing_file_indices = EKP.get_current_minibatch(eki)
    g = Float64[]
    for i in forcing_file_indices
        simulation_dir = joinpath(member_path, "config_$i", "output_0000")
        model_config_dict = YAML.load_file(joinpath(simulation_dir, ".yml"))
        # suppress logs when creating model config, z grids to avoid cluttering output
        model_config = suppress_logs(CA.AtmosConfig, model_config_dict)

        cfsite_number = model_config_dict["cfsite_number"]
        site_num = parse(Int, replace(cfsite_number, r"[^0-9]" => ""))
        forcing_type = get_cfsite_type(site_num)

        z_interp = get_cal_z_grid(model_config, z_cal_grid, forcing_type)

        simdir = SimDir(simulation_dir)
        for (i, y_name) in enumerate(y_names)
            y_var_i = process_profile_variable(
                simdir,
                y_name;
                reduction,
                t_start,
                t_end,
                z_max,
                z_interp,
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
    z_max = nothing,
    z_interp = nothing,
    norm_factors_dict = nothing,
    log_vars = [],
)

    var_i = get(simdir; short_name = y_name, reduction)

    # subset vertical coordinate
    if !isnothing(z_max)
        z_window = filter(x -> x <= z_max, var_i.dims["z"])
        var_i = window(var_i, "z", right = maximum(z_window))
    end

    sim_t_end = nothing
    try
        sim_t_end = var_i.dims["time"][end]
    catch e
        if isa(e, BoundsError)
            throw(ErrorException("Simulation failed at: t=0"))
        end
    end

    if sim_t_end < 0.95 * t_end
        throw(ErrorException("Simulation failed at: $sim_t_end"))
    end

    # take time-mean
    var_i_ave =
        average_time(window(var_i, "time", left = t_start, right = sim_t_end))
    y_var_i = slice(var_i_ave, x = 1, y = 1).data

    # interpolate to calibration grid
    if !isnothing(z_interp)
        y_var_i = interp_prof_1D(y_var_i, var_i.dims["z"], z_interp)
    end

    if y_name in log_vars
        y_var_i = log10.(y_var_i .+ 1e-12)
    end

    # normalize
    if !isnothing(norm_factors_dict)
        y_μ, y_σ = norm_factors_dict[y_name]
        y_var_i = (y_var_i .- y_μ) ./ y_σ
    end

    return y_var_i

end
