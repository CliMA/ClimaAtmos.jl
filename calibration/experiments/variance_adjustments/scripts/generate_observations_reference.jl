# CLI: reference truth run + write **`observations_path`** from the active experiment YAML (activates project).
#
#   julia --project=. scripts/generate_observations_reference.jl
#   VA_EXPERIMENT_CONFIG=experiment_configs/experiment_config_gcm_cfsite23_N3_varfix_off.yml julia --project=. scripts/generate_observations_reference.jl
#
import Pkg
const _ROOT = dirname(@__DIR__) |> abspath
Pkg.activate(_ROOT)
include(joinpath(_ROOT, "lib", "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_ROOT, "lib", "reference_generation.jl"))
generate_observations_reference!()
