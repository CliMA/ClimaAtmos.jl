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

const _REFERENCE_MERGED_TRUTH_TOML_BASENAME = "_reference_merged_truth.toml"

function _eki_root_has_iteration_dir(eki_root::AbstractString)::Bool
    isdir(eki_root) || return false
    for name in readdir(eki_root)
        m = match(r"^iteration_(\d+)$", name)
        m === nothing && continue
        isdir(joinpath(eki_root, name)) && return true
    end
    return false
end

"""
    _reference_truth_atmos_toml!(experiment_dir, expc, ref_out) -> path

ClimaParams TOML for the **reference** ClimaAtmos run that defines observation vector **`y`**.

**Only** **`reference_truth_from_eki`** is supported: merge **`scm_toml`** layers with **`parameters.toml`** from an
existing EKI ensemble member (calibrated θ). Keys: optional **`eki_config`**, optional **`iteration`** (default:
latest completed iteration), **`member`** (`best` or integer; **`best`** uses
[`va_eki_best_member_by_obs_loss`](@ref) and needs an existing **`observations.jld2`** — use a fixed integer if
you do not have **`y`** yet).

There is **no** SCM-baseline-only path for **`y`**. If EKI output is missing, run calibration first or copy an
existing **`iteration_*`** tree into this slice’s **`output_dir`**.
"""
function _reference_truth_atmos_toml!(
    experiment_dir::AbstractString,
    expc,
    ref_out::AbstractString,
)::String
    rtf = get(expc, "reference_truth_from_eki", nothing)

    if rtf !== nothing && rtf !== false
        spec = rtf isa AbstractDict ? rtf :
            error("reference_truth_from_eki must be a mapping (e.g. iteration / member / eki_config)")
        config_relp = get(spec, "eki_config", nothing)
        expc_eki = config_relp === nothing ? expc :
            va_load_experiment_config(experiment_dir, string(config_relp))
        eki_root = va_eki_output_root(experiment_dir, expc_eki)
        if !_eki_root_has_iteration_dir(eki_root)
            error(
                "reference_truth_from_eki requires EKI output under $eki_root (at least one iteration_* directory). " *
                    "Run calibration for this slice first, or copy an existing EKI output tree there.",
            )
        end
        iter_raw = get(spec, "iteration", nothing)
        iteration = if iter_raw === nothing
            va_latest_eki_iteration_number(
                experiment_dir,
                config_relp === nothing ? nothing : string(config_relp),
            )
        else
            Int(iter_raw)
        end
        mem = get(spec, "member", "best")
        rel_for_best = config_relp === nothing ? basename(va_experiment_config_path(experiment_dir)) :
            string(config_relp)
        member = if mem == "best" || mem === nothing
            va_eki_best_member_by_obs_loss(experiment_dir, rel_for_best, iteration)
        else
            Int(mem)
        end
        member_path = CAL.path_to_ensemble_member(eki_root, iteration, member)
        param_path = joinpath(member_path, "parameters.toml")
        isfile(param_path) || error(
            "Missing EKI parameters at $param_path — run calibration for this slice first, or fix reference_truth_from_eki.",
        )
        baseline = va_merge_scm_baseline_dict(experiment_dir, expc["scm_toml"])
        out_toml = joinpath(ref_out, _REFERENCE_MERGED_TRUTH_TOML_BASENAME)
        va_write_combined_member_atmos_parameters_toml(baseline, param_path, out_toml)
        @info "Reference truth ClimaAtmos TOML: SCM baseline + EKI member" out_toml iteration member eki_root
        return out_toml
    else
        error(
            "Set reference_truth_from_eki in the experiment YAML (SCM + EKI member θ). " *
                "Observation vector y is not built from uncalibrated SCM baseline alone.",
        )
    end
end

function reference_config_dict(experiment_dir::AbstractString)
    expc = va_load_experiment_config(experiment_dir)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    ref_out = va_reference_output_dir(experiment_dir, expc)
    mkpath(ref_out)
    cfg["output_dir"] = ref_out
    cfg["quadrature_order"] = expc["quadrature_order"]
    cfg["sgs_distribution"] = va_resolve_sgs_distribution_for_atmos_config(cfg, expc)
    merged_scm = _reference_truth_atmos_toml!(experiment_dir, expc, ref_out)
    cfg["toml"] = [merged_scm]
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
