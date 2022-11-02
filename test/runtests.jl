using SafeTestsets
using Test

@safetestset "Aqua" begin
    @time include("aqua.jl")
end

nothing
