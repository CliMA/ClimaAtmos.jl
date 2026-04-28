# YAML paths (relative to the experiment directory) for multi-case EKI + post-analysis.
# Named configs live under `experiment_configs/`; default when unset is `config/experiment_config.yml` (TRMM).
# Edit this list to match the experiment YAMLs you want in `scripts/run_calibration_sweep.jl` and `scripts/run_full_study.jl` / `lib/run_full_study.jl`.
# **`va_calibration_sweep_configs`** defaults to the **idealized** PyCLES-backed columns only (same list as
# [`va_idealized_calibration_sweep_configs`](@ref)). GoogleLES CloudBench YAMLs exist under `experiment_configs/` but are
# **not** included until that track is verified — add them here explicitly when ready.
# Must match `VA_EXPERIMENT_CONFIGS_DIR` in `experiment_common.jl`.
const _VA_SL = "experiment_configs"

"""EKI calibration YAML list for `run_calibration_sweep` / full study (idealized SCM cases)."""
function va_calibration_sweep_configs()
    return va_idealized_calibration_sweep_configs()
end

"""Varfix-off YAMLs used for the **naive** track: after EKI, run varfix-**on** forwards with the same member TOML (`scripts/run_full_study.jl`, `scripts/run_naive_varfix_on_forwards.jl`). Matches [`va_calibration_sweep_configs`](@ref)."""
function va_naive_varfix_off_source_configs()
    return va_calibration_sweep_configs()
end

"""Idealized SCM + PyCLES `Stats*.nc` (`les_truth.stats_file` / `LES_STATS_FILE`) — not GoogleLES / GCM-cluster truth."""
function va_idealized_calibration_sweep_configs()
    return String[
        "$(_VA_SL)/experiment_config_trmm_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_dycoms_rf01_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_dycoms_rf02_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_bomex_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_gabls_N3_varfix_off.yml",
    ]
end

"""
    va_naive_vs_calibrated_varfix_on_yaml_pairs(experiment_dir) -> Vector{Tuple{String,String}}

Each `(varfix_off_yaml, varfix_on_yaml)` names the **same** case and `quadrature_order` so we can plot
**naive varfix-on** (varfix-off params) vs **separately calibrated varfix-on** on one axis. Only pairs whose
**both** files exist under `experiment_dir` are returned.
"""
function va_naive_vs_calibrated_varfix_on_yaml_pairs(experiment_dir::AbstractString)
    return Tuple{String, String}[]
end
