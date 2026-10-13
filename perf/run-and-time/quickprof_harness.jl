#=
run with: julia  --project ./quickprof_harness.jl <path-to-case-file>

from the script directory.
=#

# Always try using Revise if available
# It will allow to skip setup+compilation overhead when iterating on a function
# using the harness to re-do measurements
try
    using Revise
catch e
    @warn "'Revise' module not available received: $e on import"
end

# Activate ClimaCore's kernel renaming feature, so kernels show up in
# profiler output (nsys/ncu) with human-readable names constructed from
# the stack trace (function name, file, and line).
# Must be set before ClimaCore's CUDA extension (ClimaCoreCUDAExt) is loaded,
# since the setting is read in its `__init__`.
ENV["CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE"] = get(
    ENV, "CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE", "true",
)

using CUDA
using DataFrames


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



function quick_profile()

    setup_time = @elapsed s = case_setup()

    compile_time = @elapsed case_run(s)

    @info """Basic run statistics:
        • Setup time: $setup_time [s]
        • Warmup (compilation) time: $compile_time [s]
    """

    # Profile the section with the CUDA's build-in profiler
    profile_data = CUDA.@profile begin
        case_run(s)
    end

    # We want to avoid truncating the data both in interactive and script use
    # Hence we call shows below with explicit IOContext
    io = IOContext(
        stdout,
        :limit => false,
        :displaysize => displaysize(stdout),
        :color => get(stdout, :color, false),
    )

    show(io, profile_data)
    summary = get_kernel_data(profile_data)

    show(io, summary;
        truncate = 0,
        allcols = true,
        allrows = true,
        summary = false,
        eltypes = false,
    )

    return (profile_data, summary)
end

cudim_to_tuple(grid::CUDA.CuDim3) = map(Int64, (grid.x, grid.y, grid.z))
cudim_to_tuple(::Missing) = missing


"""
    get_kernel_data(profile_data)

Identify unique kernels and generate DataFrame summary with launch statistics
and compile parameters (e.g. register usage).

Here we are relying on the internal representation of the profile data
This is not Public CUDA API and is unstable. Developed on CUDA.jl 5.11.3
It is possible for it to break on any major, minor (or even patch) version
"""
function get_kernel_data(profile_data)
    device_data = DataFrame(profile_data.device)

    transform!(device_data, [:stop, :start] => ByRow(-) => :duration)
    transform!(device_data, :grid => ByRow(cudim_to_tuple) => :grid)
    transform!(device_data, :block => ByRow(cudim_to_tuple) => :block)

    transform!(
        device_data,
        :shared_mem => ByRow(x -> x === missing ? missing : x.dynamic) => :dynamic_shmem,
    )
    transform!(
        device_data,
        :shared_mem => ByRow(x -> x === missing ? missing : x.static) => :static_shmem,
    )
    transform!(
        device_data,
        :local_mem => ByRow(x -> x === missing ? missing : x.thread) => :local_mem,
    )

    # Group by duplicates
    # We assume the kernel is the same if it has all the same:
    kernel_traits =
        [:name, :grid, :block, :dynamic_shmem, :static_shmem, :local_mem, :registers]
    groups = groupby(
        device_data,
        kernel_traits,
    )

    # Reduce each group to one row of summary statistics.
    # duration is in seconds (CUDA.jl stores start/stop as Float64 seconds).
    summary = combine(groups) do subdf
        durs = subdf.duration
        n = length(durs)
        (
            n_launches = n,
        )
    end

    return summary
end


(data, kernel_summary) = quick_profile()
