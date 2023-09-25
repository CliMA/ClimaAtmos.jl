import Random
Random.seed!(1234)
import ClimaAtmos as CA
include("common.jl")
config = AtmosCoveragePerfConfig()
# To revive, define `args_integrator(::AtmosConfig)` and use that here
integrator = CA.get_integrator(config)

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
