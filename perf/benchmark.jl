example_dir = joinpath(dirname(@__DIR__), "examples")

include("../examples/hybrid/cli_options.jl");

(s, parsed_args) = parse_commandline();
parsed_args["z_elem"] = 25;
parsed_args["h_elem"] = 12;
parsed_args["enable_threading"] = false


ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(example_dir, "hybrid", "driver.jl")

try
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

OrdinaryDiffEq.step!(integrator) # compile first
X = similar(integrator.u)

println("Wfact")
@time integrator.f.f1.Wfact(
    integrator.cache.W,
    integrator.u,
    integrator.p,
    integrator.dt,
    integrator.t,
);
@time integrator.f.f1.Wfact(
    integrator.cache.W,
    integrator.u,
    integrator.p,
    integrator.dt,
    integrator.t,
);
println("linsolve")
@time integrator.cache.linsolve(X, integrator.cache.W, integrator.u);
@time integrator.cache.linsolve(X, integrator.cache.W, integrator.u);
println("implicit_tendency!")
@time implicit_tendency!(X, integrator.u, integrator.p, integrator.t);
@time implicit_tendency!(X, integrator.u, integrator.p, integrator.t);
println("remaining_tendency!")
@time remaining_tendency!(X, integrator.u, integrator.p, integrator.t);
@time remaining_tendency!(X, integrator.u, integrator.p, integrator.t);



#import BenchmarkTools
#trial = BenchmarkTools.@benchmark OrdinaryDiffEq.step!($integrator)
#show(stdout, MIME("text/plain"), trial)
#println()
