# Loaded on each worker via `@everywhere include(...)` from `run_variance_calibration!()`.
# Imports and includes must live here, not inside a function: Julia 1.12+ forbids
# `import`/`using` inside function bodies (including `@everywhere begin ... end` blocks).

ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"
import Pkg
Pkg.activate(_va_exp_dir)
import ClimaCalibrate as CAL
import ClimaAtmos as CA
include(joinpath(_va_exp_dir, "lib", "model_interface.jl"))
include(joinpath(_va_exp_dir, "lib", "observation_map.jl"))
load_experiment_config!(_va_exp_dir)
