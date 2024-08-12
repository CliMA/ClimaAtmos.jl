# ENV["CLIMACOMMS_DEVICE"] = "CUDA"
# Run SCM calibration of toa radiative fluxes
import ClimaAtmos as CA
import ClimaAnalysis: SimDir, get, slice, average_xy
import CairoMakie
import JLD2
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test
import ClimaCalibrate as CAL
import YAML

# load observation map
include("observation_map.jl")
# include model interface

experiment_dir = dirname(Base.active_project())
const model_interface = joinpath(experiment_dir, "model_interface.jl")
include(model_interface)


const experiment_config = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
const n_iterations = experiment_config["n_iterations"]
const ensemble_size = experiment_config["ensemble_size"]
const output_dir = experiment_config["output_dir"]
const model_config = experiment_config["model_config"]
const prior =
    CAL.get_prior(joinpath(experiment_dir, experiment_config["prior"]))


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
observations = JLD2.load_object(obs_path)

# Initialize experiment data
noise = JLD2.load_object(joinpath(experiment_dir, experiment_config["noise"]))
#prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))
cal_ex_config = CAL.ExperimentConfig(;
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
    prior,
)

# load configs and directories 
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

if !isdir(output_dir)
    mkpath(output_dir)
end


# ExperimentConfig is created from a YAML file within the experiment_dir
@info "Initializing calibration" n_iterations ensemble_size output_dir
CAL.initialize(
    cal_ex_config;
    scheduler = EKP.DataMisfitController(terminate_at = 1),
    localization_method = EKP.NoLocalization(),
    failure_handler_method = EKP.SampleSuccGauss(),
    accelerator = EKP.DefaultAccelerator(),
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
            # hpc_kwargs,
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



CAL.calibrate(experiment_dir)



# hpc_kwargs = CAL.kwargs(time = 60, mem = "16G")
# julia_eki = CAL.calibrate(CAL.CaltechHPCBackend, experiment_dir; model_interface, hpc_kwargs)

# # Run calibration on CaltechHPC backend
# experiment_dir = dirname(Base.active_project())
# const model_interface = joinpath(experiment_dir, "model_interface.jl")
# const experiment_config =
#     YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
# const n_iterations = experiment_config["n_iterations"]
# const ensemble_size = experiment_config["ensemble_size"]
# const output_dir = experiment_config["output_dir"]
# const model_config = experiment_config["model_config"]
# const prior =
#     CAL.get_prior(joinpath(experiment_dir, experiment_config["prior"]))

# # ExperimentConfig is created from a YAML file within the experiment_dir
# @info "Initializing calibration" n_iterations ensemble_size output_dir
# CAL.initialize(
#     ensemble_size,
#     observations,
#     noise,
#     prior,
#     output_dir;
#     scheduler = EKP.DefaultScheduler(1),
# )

# eki = nothing
# hpc_kwargs = CAL.kwargs(time = 60, mem = "16G")
# module_load_str = CAL.module_load_string(CAL.CaltechHPCBackend)
# for iter in 1:(n_iterations - 1)
#     @info "Iteration $iter"
#     jobids = map(1:ensemble_size) do member
#         @info "Running ensemble member $member"
#         CAL.slurm_model_run(
#             iter,
#             member,
#             output_dir,
#             experiment_dir,
#             model_interface,
#             module_load_str;
#             hpc_kwargs,
#         )
#     end

#     statuses = CAL.wait_for_jobs(
#         jobids,
#         output_dir,
#         iter,
#         experiment_dir,
#         model_interface,
#         module_load_str;
#         hpc_kwargs,
#         verbose = true,
#         reruns = 0,
#     )
#     CAL.report_iteration_status(statuses, output_dir, iter)
#     @info "Completed iteration $iter, updating ensemble"
#     G_ensemble = CAL.observation_map(iter)
#     CAL.save_G_ensemble(output_dir, iter, G_ensemble)
#     eki = CAL.update_ensemble(output_dir, iter, prior)
# end