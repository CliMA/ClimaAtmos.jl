using Revise
using LinearAlgebra
using Statistics
using YAML
using JLD2
using CairoMakie
using ClimaAnalysis
@info "Packages Loaded..."
simdir = ""
iteration = parse(Int, ARGS[1])
calibration_name = ARGS[2]

includet(joinpath(simdir, "utils.jl"))
config_dict = YAML.load_file(joinpath(simdir, "experiment_config.yml"))

# load eki object
eki = JLD2.load_object(joinpath(simdir, "output/$calibration_name/iteration_$(lpad(iteration, 3, "0"))/eki_file.jld2"))
prior = CAL.get_prior(config_dict["prior"])

if !isdir("post_processing/$calibration_name/")
    mkdir("post_processing/$calibration_name/")
end
@info "Starting to Plot..."
plot = plot_parameters(eki, prior, params_true)
save("post_processing/$calibration_name/param_plot.png", plot)

# load variances and observations
variances = diag(JLD2.load_object(joinpath(simdir, "obs_noise_cov.jld2")))
observations = JLD2.load_object(joinpath(simdir, "observations.jld2"))
# generate true observations again
true_obs = gen_obs(joinpath(simdir,"../perf_gcm_driven_scm/output/gcm_driven_scm/output_0003"))

dist_plot = plot_start_end_distributions(eki, config_dict, observations, variances, true_obs)

save("post_processing/$calibration_name/dist_plot.png", dist_plot)

obs_loss_plot = loss_plot(eki, observations, n_iters = iteration, n_metrics = config_dict["dims"])
save("post_processing/$calibration_name/obs_loss_plot.png", obs_loss_plot)