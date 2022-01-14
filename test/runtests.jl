using Test

using Base.CoreLogging
using Documenter: doctest
using JLD2
using OrdinaryDiffEq: SSPRK33, CallbackSet, DiscreteCallback
using Plots
using UnPack

using ClimaAtmos

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
            include("test_models.jl")
        end

        disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
        doctest(ClimaAtmos)
        disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging
    end

    if group == :regression || group == :all
        @testset "Regression Tests" begin
            test_names = (
                :test_1d_ekman_column,
                :test_2d_rising_bubble,
                :test_3d_rising_bubble,
                :test_3d_solid_body_rotation,
                :test_3d_balanced_flow,
                :test_3d_baroclinic_wave,
            )
            test_cases(test_names, :regression)
        end
    end

    if group == :validation
        @testset "Validation Tests" begin
            test_names = (
                :test_1d_ekman_column,
                :test_2d_rising_bubble,
                :test_3d_rising_bubble,
                :test_3d_solid_body_rotation,
                :test_3d_balanced_flow,
                :test_3d_baroclinic_wave,
            )
            test_cases(:validation, :validation)
        end
    end
end

nothing
