#=
run with: julia  --project ./timer_harness.jl <path-to-case-file>

from the script directory.
=#

# Always try using Revise if available
# It will allow to skip setup+compilation overhead when iterating on a function
# using the harness to re-do measurements
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(true)))
try
    using Revise
catch e
    @warn "'Revise' module not available received: $e on import"
end

using CUDA
using BenchmarkTools
using ClimaComms


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


function basic_timers()
    setup_time = @elapsed s = case_setup()

    compile_time = @elapsed case_run(s)

    benchmark = BenchmarkTools.@benchmark CUDA.@sync begin
        case_run($s)
    end

    display(benchmark)
    @info """Basic run statistics:
        • Setup time: $setup_time [s]
        • Warmup (compilation) time: $compile_time [s]
        • Mean GPU time: $(mean(benchmark.times * 1e-3)) [ms]
    """
    return benchmark
end

# Run the timers by default
basic_timers()
