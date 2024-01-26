# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively. 
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
using SafeTestsets
using Test

#! format: off
@safetestset "Aqua" begin @time include("aqua.jl") end
@safetestset "Utilities" begin @time include("utilities.jl") end
@safetestset "Parameter tests" begin @time include("parameter_tests.jl") end
@safetestset "Coupler Compatibility" begin @time include("coupler_compatibility.jl") end
@safetestset "Configuration tests" begin @time include("config.jl") end
#! format: on

nothing
