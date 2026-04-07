# Named EKI slice YAMLs

Per-case / per-parameter experiment configs (`experiment_config_*_*.yml`) live here so the experiment root stays readable.

The default **`../experiment_config.yml`** (single active slice when `VA_EXPERIMENT_CONFIG` is unset) remains next to `run_full_study.jl` and the drivers.

Paths in code and registries use the prefix **`experiment_configs/`** (relative to the `variance_adjustments/` directory).
