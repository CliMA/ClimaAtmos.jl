#=
Run this script with, for example:
```
nsys profile --start-later=true --capture-range=cudaProfilerApi --kill=none --trace=nvtx,mpi,cuda,osrt --output=benchmark_step_gpu/report julia --project=.buildkite perf/benchmark_step_gpu.jl --config config/default_configs/default_config.yml --config config/common_configs/numerics_sphere_he30ze43.yml --config config/perf_configs/bm_baroclinic_wave_moist.yml 
```
Or
...
ncu --nvtx --call-stack --profile-from-start no --export benchmark_step_gpu/report --print-nvtx-rename kernel julia --project=.buildkite perf/benchmark_step_gpu.jl --config config/default_configs/default_config.yml --config config/common_configs/numerics_sphere_he30ze43.yml --config config/perf_configs/bm_baroclinic_wave_moist.yml 
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import Random
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaComms
import ClimaCore.DebugOnly: profile_rename_kernel_names

include("common.jl")
(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)

simulation = CA.get_simulation(config)
(; integrator) = simulation;
Y₀ = deepcopy(integrator.u);
@info "Compiling benchmark_step!..."
# turn on renaming of CUDA kernels based on stack trace
profile_rename_kernel_names() = true
CA.benchmark_step!(integrator, Y₀); # compile first

@info "Running benchmark_step_gpu!..."
n_steps = 5
comms_ctx = ClimaComms.context(integrator.u.c)
device = ClimaComms.device(comms_ctx)
CUDA.@profile external = true begin
    CA.benchmark_step!(integrator, Y₀, n_steps) # run
end
@info "Done running benchmark_step_gpu!"