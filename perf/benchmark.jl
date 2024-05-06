redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
Random.seed!(1234)
import ClimaAtmos as CA
using Test
using ClimaComms

include("common.jl")
(s, parsed_args) = parse_commandline()
config = TargetConfig(parsed_args["target_config"])
device = ClimaComms.device(config.comms_ctx)

simulation = CA.get_simulation(config)
(; integrator, job_id, config_id) = simulation

(; parsed_args) = config

import SciMLBase
import ClimaTimeSteppers as CTS
SciMLBase.step!(integrator) # compile first

(; sol, u, p, dt, t) = integrator

W = get_W(integrator)
X = similar(u)

include("benchmark_utils.jl")

import OrderedCollections
import LinearAlgebra as LA
trials = OrderedCollections.OrderedDict()
#! format: off
trials["Wfact"] = get_trial(wfact_fun(integrator), (W, u, p, dt, t), "Wfact");
trials["linsolve"] = get_trial(LA.ldiv!, (X, W, u), "linsolve");
trials["implicit_tendency!"] = get_trial(implicit_fun(integrator), implicit_args(integrator), "implicit_tendency!");
trials["remaining_tendency!"] = get_trial(remaining_fun(integrator), remaining_args(integrator), "remaining_tendency!");
trials["additional_tendency!"] = get_trial(CA.additional_tendency!, (X, u, p, t), "additional_tendency!");
trials["hyperdiffusion_tendency!"] = get_trial(CA.hyperdiffusion_tendency!, remaining_args(integrator), "hyperdiffusion_tendency!");
trials["dss!"] = get_trial(CA.dss!, (u, p, t), "dss!");
trials["set_precomputed_quantities!"] = get_trial(CA.set_precomputed_quantities!, (u, p, t), "set_precomputed_quantities!");
trials["step!"] = get_trial(SciMLBase.step!, (integrator, ), "step!");
#! format: on

table_summary = OrderedCollections.OrderedDict()
for k in keys(trials)
    table_summary[k] = get_summary(trials[k])
end
tabulate_summary(table_summary)

are_boundschecks_forced = Base.JLOptions().check_bounds == 1
# Benchmark allocation tests
@testset "Benchmark allocation tests" begin
    if device isa ClimaComms.CPUSingleThreaded && !are_boundschecks_forced
        @test trials["Wfact"].memory == 0
        @test trials["linsolve"].memory == 0
        @test trials["implicit_tendency!"].memory == 0
        @test trials["remaining_tendency!"].memory ≤ 2480
        @test trials["additional_tendency!"].memory == 0
        @test trials["hyperdiffusion_tendency!"].memory ≤ 2480
        @test trials["dss!"].memory == 0
        @test trials["set_precomputed_quantities!"].memory ≤ 40
        @test_broken trials["set_precomputed_quantities!"].memory < 40

        # It's difficult to guarantee zero allocations,
        # so let's just leave this as broken for now.
        @test_broken trials["step!"].memory == 0
    end
end

if get(ENV, "BUILDKITE", "") == "true"
    # Export table_summary
    import JSON
    config_id = CA.get_config_id(config.config_file, config.parsed_args)
    path = pkgdir(CA)
    open(joinpath(path, "perf_benchmark_$config_id.json"), "w") do io
        JSON.print(io, table_summary)
    end
end

import ClimaComms
if config.comms_ctx isa ClimaComms.SingletonCommsContext && !isinteractive()
    include(joinpath(pkgdir(CA), "perf", "jet_report_nfailures.jl"))
end
