ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"));

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")
dict = parsed_args_per_job_id(; trigger = "benchmark.jl")
parsed_args_prescribed = parsed_args_from_ARGS(ARGS)

# Start with performance target, but override anything provided in ARGS
parsed_args_target = dict["perf_target_unthreaded"];
parsed_args = merge(parsed_args_target, parsed_args_prescribed);


try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

function do_work!(integrator)
    for _ in 1:20
        OrdinaryDiffEq.step!(integrator)
    end
end

do_work!(integrator) # compile first
import Profile
Profile.clear_malloc_data()
Profile.clear()
prof = Profile.@profile begin
    do_work!(integrator)
end

(; output_dir, job_id) = simulation

import ProfileCanvas
include("profile_canvas_patch.jl")

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    output_dir = job_id
    mkpath(output_dir)
    html_file(joinpath(output_dir, "flame.html"))
else
    ProfileCanvas.view(Profile.fetch())
end
