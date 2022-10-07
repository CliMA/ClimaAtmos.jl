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

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ClimaAtmos; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ClimaAtmos", pkgdir(last(x).module)), ambs)
    for method_ambiguity in ambs
        @show method_ambiguity
    end
    # If the number of ambiguities is less than the limit below,
    # then please lower the limit based on the new number of ambiguities.
    # We're trying to drive this number down to zero to reduce latency.
    @info "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) ≤ 118

    # returns a vector of all unbound args
    # ua = Aqua.detect_unbound_args_recursively(ClimaCore)
end

nothing
