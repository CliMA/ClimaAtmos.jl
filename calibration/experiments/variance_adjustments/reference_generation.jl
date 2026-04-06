# Reference truth run + write `observations_path` from the active experiment YAML (definitions only; no `Pkg.activate`, no auto-run).
#
# Call from a script after activating this directory's project:
#   include("reference_generation.jl"); generate_observations_reference!()

include(joinpath(@__DIR__, "experiment_common.jl"))

import ClimaAtmos as CA
import ClimaCalibrate as CAL # before observation_map.jl extends CAL.observation_map
import ClimaComms
import YAML
import JLD2

# Load here, not inside `generate_observations_reference!()`: runtime `include` defines
# `process_member_column` in a newer world age than the caller (Julia 1.12+).
include(joinpath(@__DIR__, "observation_map.jl"))

const _REFERENCE_EXPERIMENT_DIR = dirname(@__FILE__) |> abspath

function reference_config_dict(experiment_dir::AbstractString)
    expc = va_load_experiment_config(experiment_dir)
    cfg = YAML.load_file(joinpath(experiment_dir, expc["model_config_path"]))
    ref_out = va_reference_output_dir(experiment_dir, expc)
    mkpath(ref_out)
    cfg["output_dir"] = ref_out
    cfg["quadrature_order"] = expc["quadrature_order"]
    cfg["sgs_quadrature_subcell_geometric_variance"] =
        expc["sgs_quadrature_subcell_geometric_variance"]
    cfg["toml"] = [va_scm_toml_path(experiment_dir, expc["scm_toml"])]
    cfg["output_default_diagnostics"] = get(cfg, "output_default_diagnostics", false)
    return cfg, expc
end

function generate_observations_reference!()
    experiment_dir = _REFERENCE_EXPERIMENT_DIR
    cfg, expc = reference_config_dict(experiment_dir)
    job_id = va_job_id_reference(expc)
    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = va_comms_ctx(),
        job_id,
    )
    sim = CA.get_simulation(atmos_config)
    CA.solve_atmos!(sim)
    n_y = va_expected_obs_length(experiment_dir, expc)
    y_template = ones(Float64, n_y)
    active = joinpath(cfg["output_dir"], "output_active")
    y = process_member_column(active, y_template)
    any(isnan, y) &&
        @warn "Observation vector contains NaNs; check diagnostics vs observation_fields in experiment_config.yml"
    out_jld = va_observations_abs_path(experiment_dir, expc)
    mkpath(dirname(out_jld))
    JLD2.save_object(out_jld, y)
    @info "Wrote observations" out_jld length(y) job_id
    return y
end
