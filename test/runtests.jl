using SafeTestsets
using Test

@safetestset "Aqua" begin
    @time include("utilities.jl")
    @time include("parameter_tests.jl")
    @time include("aqua.jl")
end

nothing
