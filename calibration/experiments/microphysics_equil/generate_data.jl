### to go in here:
# - generate synthetic observation


"""
Generate perfect model simulation. In initial test case, run baroclinic wave w/equil microphysics parameterization
"""
function generate()
    # to do:
    #   - change toml to the updated parameter value (???)
    #   - otherwise looks mostly correct?

    # generate wanted config
    experiment_config_dict = YAML.load_file(joinpath(experiment_dir, model_config))
    config_dict = YAML.load_file(joinpath(experiment_dir, model_config))
    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    member_path = path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end
    config_dict["output_default_diagnostics"] = false

    # run it and save it somewhere?

end


"""
Add noise to model truth
"""
function synthetic_observed_y(x_inputs; data_path = "data", apply_noise = false)
    
end