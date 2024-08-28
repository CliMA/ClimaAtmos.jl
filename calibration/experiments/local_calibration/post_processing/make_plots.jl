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
param_plot = plot_parameters(eki, prior, params_true)
save("post_processing/$calibration_name/param_plot.png", param_plot)

# load variances and observations
# variances = diag(JLD2.load_object(joinpath(simdir, "obs_noise_cov.jld2")))
# observations = JLD2.load_object(joinpath(simdir, "observations.jld2"))

# observations = [1.197289105076002, 1092.9633982962446, 32.31454740246987, 0.45136938979504854, 279.3650961190702, 289.94620843209833, 283.0394062847844, 122.99602976632545]
# variances = [0.005347825979964824, 411.3305870069564, 0.009044231731095352, 6.683379599516927e-6, 0.13426024922169746, 0.03037344351410866, 0.1506983491126448, 0.001019656958524138]
# generate true observations again
# true_obs = gen_obs(joinpath(simdir,"../perf_gcm_driven_scm/output/gcm_driven_scm/output_0003"))

dist_plot = plot_start_end_distributions(eki, config_dict)

save("post_processing/$calibration_name/dist_plot.png", dist_plot)

obs_loss_plot = loss_plot(eki, config_dict)
save("post_processing/$calibration_name/obs_loss_plot.png", obs_loss_plot)