using SafeTestsets
using Test

@safetestset "Aqua" begin
    @time include("utilities.jl")
    @time include("aqua.jl")
end

nothing
