#=
To run under NSightCompute:
````
ncu \
  -o output.ncu-rep \
  --import-source 1 \
  --profile-from-start=off \
  --set=full \
  julia --project ./prof_harness.jl <path-to-case-file>
```

To run with NSightSystems:
```
nsys profile \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,cuda,osrt \
    --gpu-metrics-device=all \
    --cuda-memory-usage=true \
    --output=output.nsys-rep \
    julia --project ./prof_harness.jl <path-to-case-file>
```

We also need to make sure that nsys/ncu is useing the same CUDA version as
Julia.

To change the version used by julia run, e.g.:
```
julia --project -e 'using CUDA; CUDA.set_runtime_version!(v"12.2")'
```

=#

# Activate ClimaCore's kernel renaming feature, so kernels show up in
# profiler output (nsys/ncu) with human-readable names constructed from
# the stack trace (function name, file, and line).
# Must be set before ClimaCore's CUDA extension (ClimaCoreCUDAExt) is loaded,
# since the setting is read in its `__init__`.
ENV["CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE"] = get(
    ENV, "CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE", "true",
)

using CUDA


if length(ARGS) != 1
    error(
        "You need to provide path to the 'case' file that defines 'case_setup' and 'case_run' functions.",
    )
end

# Load the simulation file
case_path = abspath(ARGS[1])

#=
The file getting include here needs to specify the 2 methods:
    - `case_setup`: returns a 'state'
    - `case_run(state)`: runs the functions we wish to measure
=#
include(case_path)

isdefined(Main, :case_setup) ||
    error("The required 'case_setup' function was not defined in the case file: $case_path")
isdefined(Main, :case_run) ||
    error("The required 'case_run' function was not defined in the case file: $case_path")



setup_time = @elapsed s = case_setup()

compile_time = @elapsed case_run(s)

@info """Basic run statistics:
    • Setup time: $setup_time [s]
    • Warmup (compilation) time: $compile_time [s]
"""

# Profile the section with the CUDA's build-in profiler
CUDA.@profile external=true begin
    case_run(s)
end
