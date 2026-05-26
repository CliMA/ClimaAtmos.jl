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

"""
    precompute_z_grids(config_dict)

Compute z-grids once for each forcing type (e.g. "shallow", "deep") to avoid
repeatedly creating expensive CA.AtmosConfig objects inside the observation map loop.
"""
function precompute_z_grids(config_dict)
    z_cal_grid = config_dict["z_cal_grid"]
    model_config_dict =
        YAML.load_file(joinpath(dirname(Base.active_project()), config_dict["model_config"]))
    atmos_config = suppress_logs(CA.AtmosConfig, model_config_dict)

    z_grids = Dict{String, Vector{Float64}}()
    for forcing_type in keys(z_cal_grid)
        z_grids[forcing_type] =
            get_cal_z_grid(atmos_config, z_cal_grid, forcing_type)
    end
    return z_grids
end

function CAL.observation_map(
    ::GCMDrivenSCMInterface,
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

    z_grids_by_type = precompute_z_grids(config_dict)

    n_failures = 0
    first_error = nothing
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
                z_grids_by_type = z_grids_by_type,
                norm_factors_dict = config_dict["norm_factors_by_var"],
                log_vars = config_dict["log_vars"],
            )
        catch err
            n_failures += 1
            if isnothing(first_error)
                first_error = (m, err, catch_backtrace())
            end
            G_ensemble[:, m] .= NaN
        end
    end

    N = config_dict["ensemble_size"]
    if n_failures > 0
        m, err, bt = first_error
        @warn "Observation map: $n_failures / $N members failed. First failure (member $m):"
        showerror(stderr, err, bt)
        println(stderr)
        flush(stderr)
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
    z_grids_by_type = nothing,
    norm_factors_dict = nothing,
    log_vars = [],
)
    forcing_file_indices = EKP.get_current_minibatch(eki)
    g = Float64[]
    for i in forcing_file_indices
        simulation_dir = joinpath(member_path, "config_$i", "output_active")
        model_config_dict = YAML.load_file(joinpath(simulation_dir, ".yml"))

        cfsite_number = model_config_dict["cfsite_number"]
        site_num = parse(Int, replace(cfsite_number, r"[^0-9]" => ""))
        forcing_type = get_cfsite_type(site_num)

        z_interp = z_grids_by_type[forcing_type]

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
        throw(ErrorException(
            "Simulation too short: sim_t_end=$(sim_t_end)s but need " *
            "0.95 * g_t_end=$(0.95 * t_end)s. Check that model config " *
            "t_end >= experiment config g_t_end_sec.",
        ))
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
