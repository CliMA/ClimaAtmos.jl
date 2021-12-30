using Test

using Base.CoreLogging
using Documenter: doctest
using JLD2
using OrdinaryDiffEq: SSPRK33, CallbackSet, DiscreteCallback
using Plots
using UnPack

using ClimaCore: Geometry, Spaces, Fields
using ClimaAtmos
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models.SingleColumnModels
using ClimaAtmos.Models.Nonhydrostatic2DModels
using ClimaAtmos.Models.Nonhydrostatic3DModels
using ClimaAtmos.Callbacks
using ClimaAtmos.Simulations
using CLIMAParameters

#float_types = (Float32, Float64)
float_types = (Float64,)

#####
##### Run tests!
#####

group = get(ENV, "TEST_GROUP", :all) |> Symbol

include("test_cases.jl")

@testset "ClimaAtmos" begin
    if group == :integration || group == :all
        @testset "Integration tests" begin
            include("test_domains.jl")
            test_cases(:integration)
        end

        disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
        doctest(ClimaAtmos)
        disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging
    end

    if group == :regression || group == :all
        @testset "Regression Tests" begin
            test_cases(:regression)
        end
    end

    if group == :validation
        @testset "Validation Tests" begin
            test_cases(:validation)
        end
    end
end

nothing
