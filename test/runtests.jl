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

    if group == :regression || group == :all
        @testset "Regression Tests" begin
            include("test_cases.jl")
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
            include("test_cases.jl")
            test_names = (
                :test_1d_ekman_column,
                :test_2d_rising_bubble,
                :test_3d_rising_bubble,
                :test_3d_solid_body_rotation,
                :test_3d_balanced_flow,
                :test_3d_baroclinic_wave,
            )
            test_cases(test_names, :validation)
        end
    end
end

nothing
