redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
Random.seed!(1234)
import ClimaAtmos as CA

include("common.jl")

using CUDA, BenchmarkTools, OrderedCollections, StatsBase, PrettyTables # needed for CTS.benchmark_step
using Test
using ClimaComms
import SciMLBase
import ClimaTimeSteppers as CTS

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)

simulation = CA.get_simulation(config);
(; integrator) = simulation;
(; parsed_args) = config;

device = ClimaComms.device(config.comms_ctx)
(; table_summary, trials) = CTS.benchmark_step(
    integrator,
    device;
    crop = true,
    only = [
        "Wfact",
        "ldiv!",
        "T_imp!",
        "T_exp_T_lim!",
        # "lim!",
        "dss!",
        "post_explicit!",
        "post_implicit!",
        "step!",
    ],
)

SciMLBase.step!(integrator) # compile first

are_boundschecks_forced = Base.JLOptions().check_bounds == 1
# Benchmark allocation tests
@testset "Benchmark allocation tests" begin
    if device isa ClimaComms.CPUSingleThreaded && !are_boundschecks_forced
        function compare_mem(trials, name, mem)
            if haskey(trials, name)
                return trials[name].memory ≤ mem
            else
                @warn "key $name not found in `trials` dict."
                true
            end
        end
        @test compare_mem(trials, "Wfact", 0)
        @test compare_mem(trials, "ldiv!", 0)
        @test compare_mem(trials, "T_imp!", 0)
        @test compare_mem(trials, "T_exp_T_lim!", 9920)
        @test compare_mem(trials, "lim!", 0)
        @test compare_mem(trials, "dss!", 0)
        @test compare_mem(trials, "post_explicit!", 120)
        @test compare_mem(trials, "post_implicit!", 160)

        # It's difficult to guarantee zero allocations,
        # so let's just leave this as broken for now.
        @test_broken compare_mem(trials, "step!", 0)
    end
end

if get(ENV, "BUILDKITE", "") == "true"
    # Export table_summary
    import JSON
    path = pkgdir(CA)
    open(joinpath(path, "perf_benchmark_$job_id.json"), "w") do io
        JSON.print(io, table_summary)
    end
end

if config.comms_ctx isa ClimaComms.SingletonCommsContext && !isinteractive()
    include(joinpath(pkgdir(CA), "perf", "jet_report_nfailures.jl"))
end
