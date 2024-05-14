redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
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
(; table_summary, trials) = CTS.benchmark_step(integrator, device)

SciMLBase.step!(integrator) # compile first

are_boundschecks_forced = Base.JLOptions().check_bounds == 1
# Benchmark allocation tests
@testset "Benchmark allocation tests" begin
    if device isa ClimaComms.CPUSingleThreaded && !are_boundschecks_forced
        @test trials["Wfact"].memory == 0
        @test trials["ldiv!"].memory == 0
        @test trials["T_imp!"].memory == 0
        @test trials["T_exp_T_lim!"].memory ≤ 9920
        @test trials["lim!"].memory == 0
        @test trials["dss!"].memory == 0
        @test trials["post_explicit!"].memory ≤ 120
        @test trials["post_implicit!"].memory ≤ 160

        # It's difficult to guarantee zero allocations,
        # so let's just leave this as broken for now.
        @test_broken trials["step!"].memory == 0
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
