# Build `observations.jld2` at `observations_path` from GCM-forced LES Stats.nc (same path stack as
# `calibration/experiments/gcm_driven_scm`). Activate this directory's project first.
#
#   julia --project=. scripts/build_observations_from_les.jl
#   LES_STATS_FILE=/path/to/Stats.nc julia --project=. scripts/build_observations_from_les.jl
#   VA_EXPERIMENT_CONFIG=experiment_configs/experiment_config_gcm_cfsite23_N3_varfix_off.yml julia --project=. scripts/build_observations_from_les.jl

import Pkg
const _ROOT = dirname(@__DIR__) |> abspath
Pkg.activate(_ROOT)

include(joinpath(_ROOT, "lib", "les_truth_build.jl"))
va_write_les_observations_jld2!(_ROOT)
