import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
import YAML

experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
const n_iterations = experiment_config["n_iterations"]
const ensemble_size = experiment_config["ensemble_size"]
const prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))
const observations = EKP.ObservationSeries(

)

#= TODO:
1. Fix dimensions for obs map
2. 
=#

# ExperimentConfig is created from a YAML file within the experiment_dir
@info "Initializing calibration" n_iterations ensemble_size output_dir
CAL.initialize(ensemble_size, observations, prior; scheduler = EKP.DefaultScheduler(0.5))

eki = nothing
slurm_kwargs = CAL.kwargs(time = 30, mem = "16G")
module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
for iter in 0:(n_iterations - 1)
    @info "Iteration $iter"
    jobids = map(1:ensemble_size) do member
        @info "Running ensemble member $member"
        CAL.sbatch_model_run(
            iter,
            member,
            output_dir,
            experiment_dir,
            model_interface,
            module_load_str;
            slurm_kwargs,
        )
    end

    statuses = CAL.wait_for_jobs(
        jobids,
        output_dir,
        iter,
        experiment_dir,
        model_interface,
        module_load_str;
        slurm_kwargs,
        verbose,
    )
    CAL.report_iteration_status(statuses, output_dir, iter)
    @info "Completed iteration $iter, updating ensemble"
    G_ensemble = CAL.observation_map(iter)
    CAL.save_G_ensemble(config, iter, G_ensemble)
    eki = CAL.update_ensemble(config, iter)
end