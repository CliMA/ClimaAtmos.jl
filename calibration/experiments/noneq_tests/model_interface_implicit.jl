# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

import ClimaAtmos as CA
import YAML
import ClimaComms
ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: forward_model, path_to_ensemble_member
import ClimaCalibrate as CAL

function CAL.forward_model(iteration, member, config_dict = nothing)
    experiment_dir = dirname(Base.active_project())
    if isnothing(config_dict)
        config_dict =
            YAML.load_file(joinpath(experiment_dir, "diagnostic_edmfx_diurnal_scm_imp_noneq_1M_implicit_mixed_phase_site.yml"))
    end
    output_dir = "/home/oalcabes/EKI_output/test_9" #config_dict["output_dir"] # need to input
    member_path = path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path
    parameter_path = joinpath(member_path, "parameters.toml")
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    # Turn off default diagnostics
    config_dict["output_default_diagnostics"] = false
    atmos_config = CA.AtmosConfig(
        config_dict;
        comms_ctx = ClimaComms.SingletonCommsContext(),
    )
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
    return simulation
end