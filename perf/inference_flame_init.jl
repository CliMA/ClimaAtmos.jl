ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"));

ENV["CI_PERF_SKIP_INIT"] = true # we only need haskey(ENV, "CI_PERF_SKIP_INIT") == true

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")
dict = parsed_args_per_job_id(; trigger = "driver.jl")
parsed_args_prescribed = parsed_args_from_ARGS(ARGS)

# Start with performance target, but override anything provided in ARGS
parsed_args_target = dict["compressible_edmf_bomex"];
parsed_args = merge(parsed_args_target, parsed_args_prescribed);
parsed_args["enable_threading"] = false;

# The callbacks flame graph is very expensive, so only do 2 steps.
const n_samples = occursin("callbacks", parsed_args["job_id"]) ? 2 : 20

try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile_init
        rethrow(err.error)
    end
end

function do_work!(integrator_args, integrator_kwargs)
    integrator = get_integrator(integrator_args, integrator_kwargs)
end

# This file needs some additional packages,
# let's add them to the BINDIR rather than
# the perf env.
try
    using SnoopCompile
    using FlameGraphs
    using ProfileView
catch
    run(`$(Base.julia_cmd()) --project=$(Sys.BINDIR) -e """
    import Pkg
    Pkg.add("SnoopCompile")
    Pkg.add("FlameGraphs")
    Pkg.add("ProfileView")
    """`)
end

using SnoopCompile
using FlameGraphs
using ProfileView
import Profile
Profile.clear_malloc_data()
Profile.clear()
tinf = @snoopi_deep begin
    do_work!(integrator_args, integrator_kwargs)
end

fg = flamegraph(tinf)
ProfileView.view(fg)
