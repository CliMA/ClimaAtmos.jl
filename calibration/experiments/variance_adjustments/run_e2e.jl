# End-to-end pipeline in one Julia process (one precompile).
#
# What this does (single slice only):
#   1) Optional `Pkg.instantiate`
#   2) Reference truth SCM + build observation vector **y** at `observations_path` (from active YAML)
#   3) EKI calibration into `output_dir` (from active YAML)
#
# What this does **not** do:
#   - No figures — use `include("analysis/plotting/run_post_analysis.jl"); va_run_post_analysis!()` after a run.
#   - No grid over `quadrature_order` or varfix — those are fixed in `experiment_config.yml` (or
#     `VA_EXPERIMENT_CONFIG`). Forward-only grids: `scripts/sweep_forward_runs.jl`. Several EKI YAMLs:
#     `scripts/run_calibration_sweep.jl` or rerun with different `VA_EXPERIMENT_CONFIG`.
# For the multi-YAML README workflow (EKI sweep + figures, optional forward grid), use `run_full_study.jl`.
# See README "One command: full README workflow".
#
#   julia --project=. run_e2e.jl
#
# REPL (activate this project first):
#   include("run_e2e.jl")
#
# Skip steps: VA_SKIP_INSTANTIATE=1, VA_SKIP_REFERENCE=1, VA_SKIP_CALIBRATION=1
# Alternate YAML: VA_EXPERIMENT_CONFIG=other.yml julia --project=. run_e2e.jl
#
import Pkg

const _VA_ROOT = dirname(@__FILE__) |> abspath
Pkg.activate(_VA_ROOT)
include(joinpath(_VA_ROOT, "stdio_flush.jl"))
va_setup_stdio_flushing!()
if get(ENV, "VA_SKIP_INSTANTIATE", "") != "1"
    Pkg.instantiate()
end

include(joinpath(_VA_ROOT, "reference_generation.jl"))
if get(ENV, "VA_SKIP_REFERENCE", "") != "1"
    generate_observations_reference!()
else
    @info "Skipping reference run (VA_SKIP_REFERENCE=1)"
end
va_flush_stdio()

include(joinpath(_VA_ROOT, "eki_calibration.jl"))
if get(ENV, "VA_SKIP_CALIBRATION", "") != "1"
    run_variance_calibration!()
else
    @info "Skipping calibration (VA_SKIP_CALIBRATION=1)"
end
va_flush_stdio()

@info "Pipeline finished" root = _VA_ROOT
va_flush_stdio()
