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

if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
end
#config.parsed_args["insolation"] = larcform1

simulation = CA.get_simulation(config)
#---------- my code -------------------
Y = simulation.integrator.u;
p = simulation.integrator.p;
t = simulation.integrator.t;

# Does not converge. Trying to implement in src
#include(joinpath(pkgdir(CA), "src", "solver", "types.jl"))
#=
struct Larcform1Insolation <: AbstractInsolation end
#Uniform insolation, magnitudes from Wing et al. (2018)
#Note that the TOA downward shortwave fluxes won't be the same as the values in the paper if add_isothermal_boundary_layer is true
function set_insolation_variables!(Y, p, t, ::Larcform1Insolation)
    FT = Spaces.undertype(axes(Y.c))
    (; rrtmgp_model) = p.radiation
    rrtmgp_model.cos_zenith .= cosd(FT(π)./3)
    rrtmgp_model.weighted_irradiance .= FT(0)
end
ins = Larcform1Insolation() # ins is insolation instantiated as type ::Larcform1Insolation
set_insolation_variables!(Y, p, t, ins)  # mutates p, maybe also Y?
#--------------------------------------------
=#


sol_res = CA.solve_atmos!(simulation)

include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))
if ClimaComms.iamroot(config.comms_ctx)
    make_plots(Val(Symbol(simulation.job_id)), simulation.output_dir)
end