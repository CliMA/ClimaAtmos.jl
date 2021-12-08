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
using ClimaAtmos.Models.ShallowWaterModels
using ClimaAtmos.Models.SingleColumnModels
using ClimaAtmos.Models.Nonhydrostatic2DModels
using ClimaAtmos.Callbacks
using ClimaAtmos.Simulations

float_types = (Float32, Float64)

#####
##### Run tests!
#####

group = get(ENV, "TEST_GROUP", :all) |> Symbol

include("test_simulations.jl")

@testset "ClimaAtmos" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_domains.jl")
            include("test_callbacks.jl")
            test_simulations(:unit)
        end

        disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
        doctest(ClimaAtmos)
        disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging
    end

    if group == :regression || group == :all
        @testset "Regression Tests" begin
            test_simulations(:regression)
        end
    end

    if group == :validation
        @testset "Validation Tests" begin
            test_simulations(:validation)
        end
    end
end

nothing
