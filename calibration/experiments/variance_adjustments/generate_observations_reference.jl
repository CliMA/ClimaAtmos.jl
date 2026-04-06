# CLI: reference truth run + write **`observations_path`** from the active experiment YAML (activates project).
#
#   julia --project=. generate_observations_reference.jl
#   VA_EXPERIMENT_CONFIG=experiment_config_trmm_N3_varfix_on.yml julia --project=. generate_observations_reference.jl
#
import Pkg
const _ROOT = dirname(@__FILE__) |> abspath
Pkg.activate(_ROOT)
include(joinpath(_ROOT, "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_ROOT, "reference_generation.jl"))
generate_observations_reference!()
