using Revise
using LinearAlgebra
using Statistics
using YAML
import ClimaAtmos as CA
includet("observation_map.jl")

config_dict = YAML.load_file("experiment_config.yml")
@info config_dict["y_var_names"]
#.1 kg/m^2, 100 m, 10^7
#observational_variances = [.1, 100, 1e7, 3, .03, 1, 1, 1, 1]
observational_variances = ones(config_dict["dims"])

# generate observations
obs_path = joinpath(config_dict["output_dir"], "observations.jld2")
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
