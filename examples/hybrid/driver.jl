# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

import ClimaCore
ClimaCore.Fields.local_geometry_field(bc::Base.AbstractBroadcasted) =
    ClimaCore.Fields.local_geometry_field(axes(bc))
# Todo: move this to NullBroadcasts, or is there a better way?
import NullBroadcasts
# This makes the following pattern easier:
# ∑tendencies = lazy.(∑tendencies .+ viscous_sponge_tendency_uₕ(ᶜuₕ, viscous_sponge))
Base.broadcasted(::typeof(+), ::NullBroadcasts.NullBroadcasted, x) = x
Base.broadcasted(::typeof(+), x, ::NullBroadcasts.NullBroadcasted) = x
# Base.broadcasted(::typeof(-), ::NullBroadcasts.NullBroadcasted, x) = x
Base.broadcasted(::typeof(-), x, ::NullBroadcasts.NullBroadcasted) = x

ClimaCore.Operators.fd_shmem_is_supported(bc::Base.Broadcast.Broadcasted) = false
ClimaCore.Operators.use_fd_shmem() = false
# The existing implementation limits our ability to apply
# the same expressions from within kernels
ClimaComms.device(topology::ClimaCore.Topologies.DeviceIntervalTopology) =
    ClimaComms.CUDADevice()
ClimaCore.Fields.error_mismatched_spaces(::Type, ::Type) = nothing # causes unsupported dynamic function invocation


# if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
# end
simulation = CA.AtmosSimulation(config)
sol_res = CA.solve_atmos!(simulation)

# include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))
# if ClimaComms.iamroot(config.comms_ctx)
#     make_plots(Val(Symbol(simulation.job_id)), simulation.output_dir)
# end
