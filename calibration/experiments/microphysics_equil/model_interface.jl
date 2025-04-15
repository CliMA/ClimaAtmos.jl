### to go in here:
# - forward_model function

import ClimaAtmos as CA
import YAML
import ClimaCalibrate:
    set_up_forward_model, run_forward_model, path_to_ensemble_member

# to do -- change this to the equil baroclinic wave config
const experiment_config_dict =
    YAML.load_file(joinpath(@__DIR__, "baroclinic_wave_equil.yml"))

const output_dir = experiment_config_dict["output_dir"]
const model_config = experiment_config_dict["model_config"]

function CAL.forward_model(iteration, member)
    base_config_dict = YAML.load_file(joinpath(@__DIR__, model_config))

    iter_path = CAL.path_to_iteration(output_dir, iteration)
    eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    member_path = path_to_ensemble_member(output_dir, iteration, member)

    config_dict = deepcopy(base_config_dict)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        config_dict["toml"] = abspath.(config_dict["toml"])
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end
    config_dict["output_default_diagnostics"] = false

    # ADD JULIANS VERSION ITS MORE UP TO DATE

end







@everywhere function run_atmos_simulation(atmos_config)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        !isnothing(sol_res.sol) && sol_res.sol .= eltype(sol_res.sol)(NaN)
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end


function run_forward_model(atmos_configs)
    @info "Preparing to run $(length(atmos_configs)) model simulations in parallel."
    println("Number of workers: ", nprocs())

    start_time = time()

    pmap(run_atmos_simulation, atmos_configs)

    end_time = time()
    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished all model simulations. Total time taken: $(elapsed_time) minutes."
end