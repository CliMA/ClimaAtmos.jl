if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(@__DIR__)))
end
using Test

using Base.CoreLogging
using Documenter: doctest
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
        end

        disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
        doctest(ClimaAtmos)
        disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging
    end
end

nothing
