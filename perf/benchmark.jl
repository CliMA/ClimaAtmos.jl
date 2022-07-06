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
X = similar(Y)

println("Wfact")
@time Wfact!(W, Y, p, dt, zero(dt));
@time Wfact!(W, Y, p, dt, zero(dt));
println("linsolve")
@time integrator.cache.linsolve(X, W, Y);
@time integrator.cache.linsolve(X, W, Y);
println("implicit_tendency!")
@time implicit_tendency!(X, Y, p, zero(dt));
@time implicit_tendency!(X, Y, p, zero(dt));
println("remaining_tendency!")
@time remaining_tendency!(X, Y, p, zero(dt));
@time remaining_tendency!(X, Y, p, zero(dt));



#import BenchmarkTools
#trial = BenchmarkTools.@benchmark OrdinaryDiffEq.step!($integrator)
#show(stdout, MIME("text/plain"), trial)
#println()
