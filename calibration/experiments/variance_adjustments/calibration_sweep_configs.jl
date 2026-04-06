# YAML filenames (relative to the experiment directory) for multi-slice EKI + post-analysis.
# Edit this list to match the slices you want in `run_calibration_sweep.jl` and `run_full_study.jl`.
function va_calibration_sweep_configs()
    return String[
        "experiment_config.yml",
        "experiment_config_trmm_N3_varfix_on.yml",
        "experiment_config_dycoms_N3_varfix_off.yml",
    ]
end

"""Varfix-off YAMLs used for the **naive** track: after EKI, run varfix-**on** forwards with the same member TOML (`run_full_study.jl`, `scripts/run_naive_varfix_on_forwards.jl`)."""
function va_naive_varfix_off_source_configs()
    return String["experiment_config.yml", "experiment_config_dycoms_N3_varfix_off.yml"]
end
