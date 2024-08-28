using Revise
using LinearAlgebra
using Statistics
using YAML
import ClimaAtmos as CA
includet("observation_map.jl")
include(model_interface)

config_dict = YAML.load_file("experiment_config.yml")
@info config_dict["y_var_names"]
#.1 kg/m^2, 100 m, 10^7
#observational_variances = [.1, 100, 1e7, 3, .03, 1, 1, 1, 1]
observational_variances = ones(config_dict["dims"])

# generate observations
output_dir  = config_dict["output_dir"]
obs_path = joinpath(output_dir, "observations.jld2")

if !isdir(output_dir)
    mkdir(output_dir)
end
vi_wght = 10 # we'll upweight the importance of radiative vi's to compete with the profiles
norm_factors_dict = Dict(
    "thetaa" => [306.172, 8.07383, 1],
    "hus" => [0.0063752, 0.00471147, 1],
    "husv" => [0.0063752, 0.00471147, 1],
    "clw" => [2.67537e-6, 4.44155e-6, 1],
    "lwp" => [1, .1^2, 1],
    "prw" => [30, 3^2, 1],
    "clwvi" => [1.25, .1^2, 1],
    "clvi" => [1100, 100^2, 1],
    "husvi" => [32, 3^2, 1],
    "hurvi" => [.45, .03^2, 1],
    "rlut" => [279.5, 1^2, vi_wght],
    "rlutcs" => [290, 1^2, vi_wght],
    "rsut" => [283.5, 1^2, vi_wght],
    "rsutcs" => [123, 1^2, vi_wght],
)
try
    JLD2.jldsave(
        joinpath(output_dir, "norm_factors.jld2");
        norm_factors_dict = norm_factors_dict,
    )
catch
    println("Norm Factors already saved: $(joinpath(output_dir, "norm_factors.jld2"))")
end

if !isfile(obs_path)
    @info "Generating observations"
    config = CA.AtmosConfig("../perf_gcm_driven_scm/prognostic_edmfx_gcmdriven_column.yml"; job_id = "gcm_driven_scm")
    simulation = CA.get_simulation(config)
    CA.solve_atmos!(simulation)
end

f_diagnostics = JLD2.jldopen(
    joinpath(config_dict["output_dir"], "norm_factors.jld2"),
    "r+",
)

observations = process_member_data(SimDir("output/gcm_driven_scm/output_active"), 
                    y_names = config_dict["y_var_names"],
                    t_start = config_dict["g_t_start_sec"],
                    t_end = config_dict["g_t_end_sec"],
                    norm_vec_obs = f_diagnostics["norm_factors_dict"],
)

@info "Observations generated"
@info "Observations: $(round.(observations, digits = 2))"
@info "Observational variances: $(round.(observational_variances, digits= 2))"

@info "Adding Noise to Observations - should we be doing sqrt?"
observations_noisy = observations .+ randn(length(observations)) .* observational_variances .* .1
@info "Observations with noise: $(round.(observations, digits = 2))"

# save observations and observational variances
JLD2.save_object(joinpath(config_dict["output_dir"], "observations.jld2"), observations)
JLD2.save_object(joinpath(config_dict["output_dir"], "obs_noise_cov.jld2"), Diagonal(observational_variances))
