using ArgParse
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import JLD2
using LinearAlgebra
using Random
using Flux
using TOML
using Distributions


include("helper_funcs.jl")
include("observation_map.jl")
include("get_les_metadata.jl")
include("nn_helpers.jl")

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--output_dir"
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
    model_interface = joinpath(experiment_dir, "model_interface.jl")

    experiment_config =
        YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))

    for (key, value) in experiment_config
        if key != "output_dir"
            @eval $(Symbol(key)) = $value
        end
    end

    if model_config_dict["mixing_length_model"] == "nn"
        prior = create_prior_with_nn(
            prior_path,
            pretrained_nn_path;
            arc = [8, 20, 15, 10, 1],
        )
    else
        const prior = CAL.get_prior(joinpath(experiment_dir, prior_path))
    end

    model_config_dict =
        YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))
    atmos_config = CA.AtmosConfig(model_config_dict)


    @info "Initializing calibration" n_iterations ensemble_size output_dir

    current_iterations = sort([
        parse(Int, split(x, "_")[2]) for x in readdir(output_dir) if
        isdir(joinpath(output_dir, x)) && occursin("iteration_", x)
    ])

    if restart_iteration == -1
        latest_iteration = current_iterations[end]
    else
        latest_iteration = restart_iteration
    end

    last_iteration_dir =
        joinpath(output_dir, "iteration_$(lpad(latest_iteration, 3, '0'))")
    @info "Restarting from iteration $latest_iteration"

    if isdir(last_iteration_dir)
        member_dirs =
            filter(x -> startswith(x, "member_"), readdir(last_iteration_dir))
        for member_dir in member_dirs
            config_dirs = filter(
                x -> startswith(x, "config_"),
                readdir(joinpath(last_iteration_dir, member_dir)),
            )
            for config_dir in config_dirs
                rm(
                    joinpath(last_iteration_dir, member_dir, config_dir);
                    force = true,
                    recursive = true,
                )
            end
        end
    end


    eki_path = joinpath(
        CAL.path_to_iteration(output_dir, latest_iteration),
        "eki_file.jld2",
    )
    eki = JLD2.load_object(eki_path)

    hpc_kwargs = CAL.kwargs(
        time = 180,
        mem_per_cpu = "12G",
        cpus_per_task = min(batch_size + 1, 5),
        ntasks = 1,
        nodes = 1,
        # reservation = "clima",
    )
    module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
    for iter in latest_iteration:(n_iterations - 1)
        @info "Iteration $iter"
        jobids = map(1:ensemble_size) do member
            @info "Running ensemble member $member"

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
            verbose = true,
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
