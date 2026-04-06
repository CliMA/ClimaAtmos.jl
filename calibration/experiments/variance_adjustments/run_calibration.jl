# CLI: EKI only (activates project). Needs `observations_path` from the active experiment YAML (see `VA_EXPERIMENT_CONFIG`).
#
#   julia --project=. run_calibration.jl
#
import Pkg
const _ROOT = dirname(@__FILE__) |> abspath
Pkg.activate(_ROOT)
include(joinpath(_ROOT, "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_ROOT, "eki_calibration.jl"))
run_variance_calibration!()
