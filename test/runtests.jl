# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
using SafeTestsets
using Test

# Download test artifacts
include("download_artifacts.jl")

#! format: off
@safetestset "Aqua" begin @time include("aqua.jl") end
@safetestset "Dependencies" begin @time include("dependencies.jl") end
@safetestset "Callbacks" begin @time include("callbacks.jl") end
@safetestset "Utilities" begin @time include("utilities.jl") end
@safetestset "Parameter tests" begin @time include("parameters/parameter_tests.jl") end
@safetestset "Coupler Compatibility" begin @time include("coupler_compatibility.jl") end
@safetestset "Configuration tests" begin @time include("config.jl") end
@safetestset "surface albedo tests" begin @time include("surface_albedo.jl") end
@safetestset "Radiation interface tests" begin @time include("rrtmgp_interface.jl") end
@safetestset "Sponge interface tests" begin @time include("parameterized_tendencies/sponge/rayleigh_sponge.jl") end
@safetestset "Precipitation interface tests" begin @time include("parameterized_tendencies/microphysics/precipitation.jl") end
@safetestset "Model getters" begin @time include("solver/model_getters.jl") end
@safetestset "Topography tests" begin @time include("topography.jl") end
@safetestset "Restarts" begin @time include("restart.jl") end

#! format: on

nothing
