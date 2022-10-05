using Test
using ClimaAtmos

#####
##### Run tests!
#####

group = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "ClimaAtmos" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_domains.jl")
            include("test_models.jl")
            include("test_thomas.jl")
            include("test_parameters.jl")
        end
    end
end

using Test
import ClimaAtmos
using Aqua

@testset "Aqua tests - unbound args" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    Aqua.test_unbound_args(ClimaAtmos)
end

nothing
