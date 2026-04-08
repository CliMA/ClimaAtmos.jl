# CLI: EKI only (activates project). Needs `observations_path` from the active experiment YAML (see `VA_EXPERIMENT_CONFIG`).
#
#   julia --project=. scripts/run_calibration.jl
#
import Pkg
const _ROOT = dirname(@__DIR__) |> abspath
Pkg.activate(_ROOT)
include(joinpath(_ROOT, "lib", "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_ROOT, "lib", "eki_calibration.jl"))
run_variance_calibration!(va_eki_calibration_options_from_env())
