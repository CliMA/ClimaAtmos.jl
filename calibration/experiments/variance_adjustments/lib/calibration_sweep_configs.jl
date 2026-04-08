# YAML paths (relative to the experiment directory) for multi-slice EKI + post-analysis.
# Named slices live under `experiment_configs/`; default slice is `config/experiment_config.yml`.
# Edit this list to match the slices you want in `scripts/run_calibration_sweep.jl` and `scripts/run_full_study.jl` / `lib/run_full_study.jl`.
# Idealized PyCLES slices: `va_idealized_calibration_sweep_configs()` (TRMM, DYCOMS, BOMEX, GABLS).
# Must match `VA_EXPERIMENT_CONFIGS_DIR` in `experiment_common.jl`.
const _VA_SL = "experiment_configs"

function va_calibration_sweep_configs()
    return String[
        "$(_VA_SL)/experiment_config_googleles_01_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_02_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_03_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_04_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_05_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_06_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_07_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_08_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_09_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_googleles_10_N3_varfix_off.yml",
    ]
end

"""Varfix-off YAMLs used for the **naive** track: after EKI, run varfix-**on** forwards with the same member TOML (`scripts/run_full_study.jl`, `scripts/run_naive_varfix_on_forwards.jl`)."""
function va_naive_varfix_off_source_configs()
    return String[
        "$(_VA_SL)/experiment_config_gcm_cfsite04_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_gcm_cfsite08_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_gcm_cfsite11_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_gcm_cfsite19_N3_varfix_off.yml",
        "$(_VA_SL)/experiment_config_gcm_cfsite23_N3_varfix_off.yml",
    ]
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
