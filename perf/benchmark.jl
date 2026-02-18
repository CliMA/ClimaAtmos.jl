redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
import ClimaComms
ClimaComms.@import_required_backends
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
    crop = false,
    hcrop = 168,
    only = [
        "Wfact",
        "ldiv!",
        "T_imp!",
        "T_exp_T_lim!",
        # "lim!",
        "dss!",  # TODO: Rename to constrain_state! once ClimaTimeSteppers.jl updates its API
        "cache!",
        "cache_imp!",
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
                if trials[name].memory ≤ mem
                    @warn "Allocation limits for $name can be reduced to $(trials[name].memory)."
                else
                    @info "trials[$name].memory: $(trials[name].memory)"
                end
                return trials[name].memory ≤ mem
            else
                @warn "key $name not found in `trials` dict."
                true
            end
        end
        @test compare_mem(trials, "Wfact", 0)
        @test compare_mem(trials, "ldiv!", 0)
        @test compare_mem(trials, "T_imp!", 0)
        @test compare_mem(trials, "T_exp_T_lim!", 190420)
        @test compare_mem(trials, "lim!", 0)
        @test compare_mem(trials, "dss!", 0)
        @test compare_mem(trials, "cache!", 168)
        @test compare_mem(trials, "cache_imp!", 160)

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
