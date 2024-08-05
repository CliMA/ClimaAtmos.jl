#= End-to-end test
Runs a perfect model calibration, calibrating on the parameter `astronomical_unit`
with top-of-atmosphere radiative shortwave flux in the loss function.

The calibration is run twice, once on the backend obtained via `get_backend()`
and once on the `JuliaBackend`. The output of each calibration is tested individually
and compared to ensure reproducibility.
=#
import ClimaAtmos as CA
import ClimaAnalysis: SimDir, get, slice, average_xy
import CairoMakie
import JLD2
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test
using Revise

# load observation map
include("observation_map.jl")


# add processes
using Distributed
addprocs(9)

@everywhere begin
    using Revise
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    const experiment_dir = joinpath(pkgdir(CA), "calibration", "experiments", "gcm_driven_scm")
    const model_interface =
        joinpath(pkgdir(CA), "calibration", "model_interface.jl")
    const output_dir = joinpath("output", "gcm_driven_scm")
    # include model interface
    includet(model_interface)
end


# Generate observations
obs_path = joinpath(experiment_dir, "observations.jld2")
# if !isfile(obs_path)
#     @info "Generating observations"
#     config = CA.AtmosConfig(joinpath(experiment_dir, "model_config.yml"))
#     simulation = CA.get_simulation(config)
#     CA.solve_atmos!(simulation)
#     observations = Vector{Float64}(undef, 1)
#     observations .= process_member_data(SimDir(simulation.output_dir))
#     JLD2.save_object(obs_path, observations)
# end

# Initialize experiment data
observations = JLD2.load_object(obs_path)
noise = 0.1 * I
n_iterations = 3
ensemble_size = 9
prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))
experiment_config = CAL.ExperimentConfig(;
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
    prior,
)
# Run calibration
julia_eki = CAL.calibrate(CAL.JuliaBackend, experiment_config)



#############################
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML

# import ClimaCalibrate:
#     calibrate, ExperimentConfig, CaltechHPC, get_prior, kwargs
import YAML
import JLD2
using LinearAlgebra

include("observation_map.jl")

experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
const n_iterations = experiment_config["n_iterations"]
const ensemble_size = experiment_config["ensemble_size"]
const output_dir = experiment_config["output_dir"]
const model_config = experiment_config["model_config"]
const prior =
    CAL.get_prior(joinpath(experiment_dir, experiment_config["prior"]))

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

if !isdir(output_dir)
    mkpath(output_dir)
end

# load observations
obs_path = joinpath(experiment_dir, "observations.jld2")
observations = JLD2.load_object(obs_path)

# rfs_minibatcher =
#     EKP.RandomFixedSizeMinibatcher(1)
# observations = EKP.ObservationSeries(observations, rfs_minibatcher)

noise = 0.1 * I
cal_ex_config = CAL.ExperimentConfig(;
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
    prior,
)
# ExperimentConfig is created from a YAML file within the experiment_dir
@info "Initializing calibration" n_iterations ensemble_size output_dir
CAL.initialize(
    cal_ex_config;
    scheduler = EKP.DefaultScheduler(1),
)

eki = nothing
hpc_kwargs = CAL.kwargs(time = 60, mem = "16G")
module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
for iter in 0:(n_iterations - 1)
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
    G_ensemble = CAL.observation_map(iter)
    CAL.save_G_ensemble(output_dir, iter, G_ensemble)
    eki = CAL.update_ensemble(output_dir, iter, prior)
end







hpc_kwargs = CAL.kwargs(time = 60, mem = "16G")
julia_eki = CAL.calibrate(CAL.CaltechHPCBackend, experiment_dir; model_interface, hpc_kwargs)

# Run calibration on CaltechHPC backend
experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
const experiment_config =
    YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
const n_iterations = experiment_config["n_iterations"]
const ensemble_size = experiment_config["ensemble_size"]
const output_dir = experiment_config["output_dir"]
const model_config = experiment_config["model_config"]
const prior =
    CAL.get_prior(joinpath(experiment_dir, experiment_config["prior"]))

# ExperimentConfig is created from a YAML file within the experiment_dir
@info "Initializing calibration" n_iterations ensemble_size output_dir
CAL.initialize(
    ensemble_size,
    observations,
    noise,
    prior,
    output_dir;
    scheduler = EKP.DefaultScheduler(1),
)

eki = nothing
hpc_kwargs = CAL.kwargs(time = 60, mem = "16G")
module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
for iter in 1:(n_iterations - 1)
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
    G_ensemble = CAL.observation_map(iter)
    CAL.save_G_ensemble(output_dir, iter, G_ensemble)
    eki = CAL.update_ensemble(output_dir, iter, prior)
end