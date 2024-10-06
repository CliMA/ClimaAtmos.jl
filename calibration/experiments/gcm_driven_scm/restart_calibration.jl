using ArgParse
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import JLD2
using LinearAlgebra

include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "output_dir"
        help = "The output directory for calibration"
        "--restart_iteration"
        help = "Optional: specify the iteration to restart from"
        arg_type = Int
        default = -1
    end

    parsed_args = parse_args(s)
    output_dir = parsed_args["output_dir"]
    restart_iteration = parsed_args["restart_iteration"]

    experiment_dir = dirname(Base.active_project())
    const model_interface = joinpath(experiment_dir, "model_interface.jl")

    const experiment_config =
        YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))

    for (key, value) in experiment_config
        if key != "output_dir"
            @eval const $(Symbol(key)) = $value
        end
    end

    const prior = CAL.get_prior(joinpath(output_dir, "configs", "prior.toml"))

    model_config_dict = YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))
    atmos_config = CA.AtmosConfig(model_config_dict)

    ref_paths, _ = get_les_calibration_library()
    obs_vec = []

    for ref_path in ref_paths
        cfsite_number, _, _, _ = parse_les_path(ref_path)
        forcing_type = get_cfsite_type(cfsite_number)
        zc_model = get_cal_z_grid(atmos_config, z_cal_grid, forcing_type)

        ti = experiment_config["y_t_start_sec"]
        ti = isa(ti, AbstractFloat) ? ti : ti[forcing_type]
        tf = experiment_config["y_t_end_sec"]
        tf = isa(tf, AbstractFloat) ? tf : tf[forcing_type]

        y_obs, Σ_obs, norm_vec_obs = get_obs(
            ref_path,
            experiment_config["y_var_names"],
            zc_model;
            ti = ti,
            tf = tf,
            norm_factors_dict = norm_factors_by_var,
            z_score_norm = true,
            log_vars = log_vars,
            Σ_const = const_noise_by_var,
            Σ_scaling = "const",
        )

        push!(
            obs_vec,
            EKP.Observation(
                Dict(
                    "samples" => y_obs,
                    "covariances" => Σ_obs,
                    "names" => split(ref_path, "/")[end],
                ),
            ),
        )
    end

    series_names = [ref_paths[i] for i in 1:length(ref_paths)]

    rfs_minibatcher =
        EKP.FixedMinibatcher(collect(1:experiment_config["batch_size"]))
    observations = EKP.ObservationSeries(obs_vec, rfs_minibatcher, series_names)

    @info "Initializing calibration" n_iterations ensemble_size output_dir

    current_iterations = sort([parse(Int, split(x, "_")[2]) for x in readdir(output_dir) if isdir(joinpath(output_dir, x)) && occursin("iteration_", x)])

    if restart_iteration == -1
        latest_iteration = current_iterations[end]
    else
        latest_iteration = restart_iteration
    end

    last_iteration_dir = joinpath(output_dir, "iteration_$(lpad(latest_iteration, 3, '0'))")

    if isdir(last_iteration_dir)
        member_dirs = filter(x -> startswith(x, "member_"), readdir(last_iteration_dir))
        for member_dir in member_dirs
            rm(joinpath(last_iteration_dir, member_dir); force = true, recursive = true)
        end
    end

    @info "Restarting from iteration $latest_iteration"

    eki = nothing
    hpc_kwargs = CAL.kwargs(
        time = 90,
        mem_per_cpu = "12G",
        cpus_per_task = batch_size + 1,
        ntasks = 1,
        nodes = 1,
        reservation = "clima",
    )
    module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
    for iter in latest_iteration:(n_iterations - 1)
        @info "Iteration $iter"
        jobids = map(1:ensemble_size) do member
            @info "Running ensemble member $member"
            member_dir = joinpath(output_dir, "iteration_$(lpad(iter, 3, '0'))", "member_$(lpad(member, 3, '0'))")
            CAL.slurm_model_run(
                iter,
                member,
                output_dir,
                experiment_dir,
                model_interface,
                module_load_str;
                hpc_kwargs,
            )
        end

        statuses = CAL.wait_for_jobs(
            jobids,
            output_dir,
            iter,
            experiment_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            verbose = false,
            reruns = 0,
        )
        CAL.report_iteration_status(statuses, output_dir, iter)
        @info "Completed iteration $iter, updating ensemble"
        G_ensemble = CAL.observation_map(iter; config_dict = experiment_config)
        CAL.save_G_ensemble(output_dir, iter, G_ensemble)
        eki = CAL.update_ensemble(output_dir, iter, prior)
    end
end

main()
