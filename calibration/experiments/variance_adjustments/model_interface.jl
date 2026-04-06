# Local forward model for ClimaCalibrate (extends / replaces methods when this file is included after importing CAL).

include(joinpath(@__DIR__, "experiment_common.jl"))

import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import ClimaCalibrate as CAL

const _EXP_CONFIG = Ref{Union{Nothing, Dict{Any, Any}}}(nothing)

function load_experiment_config!(experiment_dir::AbstractString)
    path = va_experiment_config_path(experiment_dir)
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

Build the `AtmosConfig` dict for one ensemble member: load case YAML (`model_config_path`), set
`output_dir` to the member directory, merge SCM baseline + EKI `parameters.toml` into a single TOML
(see [`va_write_combined_member_atmos_parameters_toml`](@ref) in `experiment_common.jl` — **why** this
is required), and apply `experiment_config.yml` knobs (`quadrature_order`, subcell variance flag).

Case YAML uses `toml: []`; this function sets `config_dict["toml"]` to
`[joinpath(member_path, VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME)]` only.
"""
function build_atmos_config_dict(
    member::Integer,
    iteration::Integer,
    experiment_dir::AbstractString,
)
    expc = experiment_config()
    model_rel = expc["model_config_path"]
    config_dict = YAML.load_file(joinpath(experiment_dir, model_rel))

    out_rel = expc["output_dir"]
    out_root = isabspath(out_rel) ? String(out_rel) : joinpath(experiment_dir, out_rel)
    member_path = CAL.path_to_ensemble_member(out_root, iteration, member)
    config_dict["output_dir"] = member_path

    param_path = joinpath(member_path, "parameters.toml")
    scm_toml = va_scm_toml_path(experiment_dir, expc["scm_toml"])
    mkpath(member_path)
    combined_path = joinpath(member_path, VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME)
    va_write_combined_member_atmos_parameters_toml(scm_toml, param_path, combined_path)
    config_dict["toml"] = [combined_path]

    config_dict["quadrature_order"] = expc["quadrature_order"]
    config_dict["sgs_quadrature_subcell_geometric_variance"] =
        expc["sgs_quadrature_subcell_geometric_variance"]

    config_dict["output_default_diagnostics"] = get(config_dict, "output_default_diagnostics", false)
    return config_dict
end

function CAL.forward_model(iteration, member, config_dict = nothing)
    experiment_dir = dirname(Base.active_project())
    isnothing(_EXP_CONFIG[]) && load_experiment_config!(experiment_dir)
    expc = experiment_config()
    cfg = isnothing(config_dict) ? build_atmos_config_dict(member, iteration, experiment_dir) : config_dict
    job_id = something(get(cfg, "job_id", nothing), va_job_id_member(expc, iteration, member))
    @debug "forward_model" calibration_mode = get(expc, "calibration_mode", nothing) iteration member job_id
    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = va_comms_ctx(),
        job_id,
    )
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        error("ClimaAtmos simulation crashed; see stack trace.")
    end
    return simulation
end
