import ClimaCalibrate
import EnsembleKalmanProcesses as EKP
import EnsembleKalmanProcesses.ParameterDistributions as PD

import Distributed
import JLD2
import CairoMakie

model_interface_filepath = joinpath("model_interface", "model_interface.jl")
include(model_interface_filepath)

# Set up model interface
config = joinpath("config", "model_configs", "prognostic_edmfx_trmm_column.yml")
output_dir = "perfect_scm_calibration"
diagnostic_dict =
    Dict(
        "short_name" => ["lwp", "rwp", "iwp", "swp"],
        "period" => "20mins",
        "reduction_time" => "average",
    )
interface = PerfectAtmosModelInterface(config, output_dir, diagnostic_dict)

# Generate data from model simulation with observational data specified by
# diagnostic_dict
obs_vec_filename = "observation_vec.jld2"
obs_vec_filepath = joinpath(interface.output_dir, obs_vec_filename)
diagnostics_filepath = generate_perfect_model_data(
    interface;
    tomls = [joinpath(@__DIR__, "perfect_model_parameters.toml")],
)
make_observation_vec(
    interface,
    diagnostics_filepath,
    obs_vec_filepath;
    overwrite = false, # Set to true to always recreate the observations
)

# Create the observation series
obs_vec = JLD2.load_object(obs_vec_filepath)
minibatch_size = 1
obs_series = EKP.ObservationSeries(
    Dict(
        "observations" => obs_vec,
        "minibatcher" => ClimaCalibrate.minibatcher_over_samples(
            length(obs_vec),
            minibatch_size,
        ),
    ),
)

# Set seed for ensure reproducibility
rng_seed = 1234
rng = Random.MersenneTwister(rng_seed)

# Set priors for parameters of interest
means = [5.0, 1.0, 0.01, 0.01]
mean_std_dev_lub(m) = (m + 0.05 * m, m * 0.2, 0.0, Inf)
calibration_priors = [
    PD.constrained_gaussian(
        "fixed_rain_terminal_velocity",
        mean_std_dev_lub(means[1])...,
    ),
    PD.constrained_gaussian(
        "fixed_snow_terminal_velocity",
        mean_std_dev_lub(means[2])...,
    ),
    PD.constrained_gaussian(
        "fixed_cloud_liquid_terminal_velocity",
        mean_std_dev_lub(means[3])...,
    ),
]
prior = EKP.combine_distributions(calibration_priors)

# Plot parameter distribution
fig_priors = CairoMakie.Figure(size = (700, 400))
EKP.Visualize.plot_parameter_distribution(fig_priors[1, 1], prior)
plots_dir = joinpath(output_dir, "plots")
mkpath(plots_dir)
CairoMakie.save(joinpath(plots_dir, "parameter_distribution.png"), fig_priors)

ekp = EKP.EnsembleKalmanProcess(
    obs_series,
    EKP.TransformUnscented(prior, impose_prior = true);
    verbose = true,
    rng,
    scheduler = EKP.DataMisfitController(terminate_at = 1000000),
)

backend = ClimaCalibrate.JuliaBackend()
# If we are using Slurm, then we will use the Distributed backend to parallelize
# the forward models.
# The number of tasks in Slurm script should be equal to the number of ensemble
# members
slurm_ntasks = parse(Int, get(ENV, "SLURM_NTASKS", "0"))
# When using the WorkerBackend, you need to recompile everything again on the
# workers. Since running the simulation is faster than compilation, it is better
# to use the JuliaBackend even though the forward runs are not ran in parallel.
is_buildkite = parse(Bool, get(ENV, "BUILDKITE", "false"))
if slurm_ntasks > 1 && !is_buildkite
    # This could be more generic (e.g. support PBS and local machines) at the
    # cost of more complexity
    ensemble_size = EKP.get_N_ens(ekp)
    if ensemble_size != slurm_ntasks
        @warn "The ensemble size ($ensemble_size) is not the same as the number of Slurm tasks ($slurm_ntasks)"
    end
    Distributed.addprocs(ClimaCalibrate.SlurmManager())
    Distributed.@everywhere include($model_interface_filepath)
    backend = ClimaCalibrate.WorkerBackend()
end

# Run calibration
n_iterations = 5
eki = ClimaCalibrate.calibrate(
    backend,
    ekp,
    interface,
    n_iterations,
    prior,
    interface.output_dir,
)

@info "Final constrained parameters:" EKP.get_ϕ_mean_final(prior, ekp)
@info "Final unconstrained parameters:" EKP.get_u_mean_final(eki)

# You can inspect the final mean forward model evaluation as `OutputVar`s with
# `ClimaCalibrate.ObservationRecipe.reconstruct_g_mean_final` and you can
# inspect the `EKP.Observation` with
# `ClimaCalibrate.ObservationRecipe.reconstruct_vars`.
