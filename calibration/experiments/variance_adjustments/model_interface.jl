# Local forward model for ClimaCalibrate (extends / replaces methods when this file is included after importing CAL).

import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import ClimaCalibrate as CAL

const _EXP_CONFIG = Ref{Union{Nothing, Dict{Any, Any}}}(nothing)

function load_experiment_config!(experiment_dir::AbstractString)
    path = joinpath(experiment_dir, "experiment_config.yml")
    _EXP_CONFIG[] = YAML.load_file(path)
    return _EXP_CONFIG[]
end

function experiment_config()
    c = _EXP_CONFIG[]
    isnothing(c) && error("Call load_experiment_config! first.")
    return c
end

"""
    build_atmos_config_dict(member, iteration, experiment_dir) -> Dict

Merge `experiment_config.yml` knobs (`quadrature_order`, subcell flag, case TOML)
with the case YAML and ensemble `parameters.toml`.
"""
function build_atmos_config_dict(
    member::Integer,
    iteration::Integer,
    experiment_dir::AbstractString,
)
    expc = experiment_config()
    model_rel = expc["model_config_path"]
    config_dict = YAML.load_file(joinpath(experiment_dir, model_rel))

    out_root = expc["output_dir"]
    member_path = CAL.path_to_ensemble_member(out_root, iteration, member)
    config_dict["output_dir"] = member_path

    param_path = joinpath(member_path, "parameters.toml")
    atmos_root = pkgdir(CA)
    scm_toml = joinpath(atmos_root, "toml", expc["scm_toml"])
    config_dict["toml"] = [scm_toml, param_path]

    config_dict["quadrature_order"] = expc["quadrature_order"]
    config_dict["sgs_quadrature_subcell_geometric_variance"] =
        expc["sgs_quadrature_subcell_geometric_variance"]

    config_dict["output_default_diagnostics"] = get(config_dict, "output_default_diagnostics", false)
    return config_dict
end

function CAL.forward_model(iteration, member, config_dict = nothing)
    experiment_dir = dirname(Base.active_project())
    isnothing(_EXP_CONFIG[]) && load_experiment_config!(experiment_dir)
    cfg = isnothing(config_dict) ? build_atmos_config_dict(member, iteration, experiment_dir) : config_dict
    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = ClimaComms.SingletonCommsContext(),
    )
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        error("ClimaAtmos simulation crashed; see stack trace.")
    end
    return simulation
end
