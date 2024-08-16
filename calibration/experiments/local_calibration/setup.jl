using Revise
using LinearAlgebra
using Statistics
using YAML
includet("observation_map.jl")

exp_config = YAML.load_file("experiment_config.yml")
@info config_dict["y_var_names"]
#.1 kg/m^2, 100 m, 10^7
observational_variances = [.1, 100, 1e7, 3, .03, 1, 1, 1, 1]

# generate observations
# if !isfile(obs_path)
#     @info "Generating observations"
#     config = CA.AtmosConfig(joinpath(experiment_dir, "model_config.yml"))
#     simulation = CA.get_simulation(config)
#     CA.solve_atmos!(simulation)
#     observations = Vector{Float64}(undef, 1)
#     observations .= process_member_data(SimDir(simulation.output_dir))
#     JLD2.save_object(obs_path, observations)
# end

observations = process_member_data(SimDir("../perf_gcm_driven_scm/output/gcm_driven_scm/output_active"), 
                    y_names = exp_config["y_var_names"],
                    t_start = exp_config["g_t_start_sec"],
                    t_end = exp_config["g_t_end_sec"],
)
@info "Observations generated"
@info "Observations: $(round.(observations, digits = 2))"
@info "Observational variances: $(round.(observational_variances, digits= 2))"

@info "Adding Noise to Observations - should we be doing sqrt?"
observations += randn(length(observations)) .* observational_variances
@info "Observations with noise: $(round.(observations, digits = 2))"

# save observations and observational variances
JLD2.save_object("observations.jld2", observations)
JLD2.save_object("obs_noise_cov.jld2", Diagonal(observational_variances))
