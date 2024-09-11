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

# This runtime dispatch is only due to the
# possibility of printing error messages
import Thermodynamics as TD
TD.print_warning() = false

Random.seed!(1234)

if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
end
simulation = CA.get_simulation(config)
(; integrator) = simulation
import SciMLBase
SciMLBase.step!(integrator) # compile
include(joinpath(@__DIR__, "..", "..", "perf", "jet_report_nfailures.jl"))
