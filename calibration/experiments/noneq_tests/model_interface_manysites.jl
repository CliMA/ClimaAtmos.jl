redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

import ClimaAtmos as CA
import YAML
import ClimaComms
ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: forward_model, path_to_ensemble_member
import ClimaCalibrate as CAL

# FUNCTIONS
CAL.default_worker_pool() = WorkerPool(workers())

function run_atmos_simulation(atmos_config)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        if !isnothing(sol_res.sol)
            T = eltype(sol_res.sol)
            if T !== Any && isconcretetype(T)
                sol_res.sol .= T(NaN)
            else
                fill!(sol_res.sol, NaN)
            end
        end
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end

function CAL.forward_model(parameter_path, lat, lon, start_date)
    base_config_dict = YAML.load_file(joinpath(@__DIR__, "diagnostic_edmfx_diurnal_scm_imp.yml"))
    config_dict = deepcopy(base_config_dict)

    # update the config_dict with site latitude / longitude
    config_dict["site_latitude"] = lat
    config_dict["site_longitude"] = lon
    config_dict["start_date"] = start_date

    # set the data output directory
    member_path = dirname(parameter_path)
    member_path = joinpath(member_path, "location_$(lat)_$(lon)_$(start_date)")
    config_dict["output_dir"] = member_path

    # add the perturbation toml to the config_dict
    # if haskey(config_dict, "toml")
    #     config_dict["toml"] = abspath.(config_dict["toml"])
    #     push!(config_dict["toml"], parameter_path)
    # else
    #     config_dict["toml"] = [parameter_path]
    # end

    push!(config_dict["toml"], truth_toml)
    
    comms_ctx = ClimaComms.SingletonCommsContext()
    config = CA.AtmosConfig(config_dict; comms_ctx)

    start_time = time()
    try
        run_atmos_simulation(config)
    catch e
        @warn "Simulation crashed for parameter file $(parameter_path): $(e)"
        return
    end
    end_time = time()

    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished simulation. Total time taken: $(elapsed_time) minutes."

end