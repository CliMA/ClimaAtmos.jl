using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Chemistry Tendencies" begin

    # ========================================================================
    # Default: no chemistry model → no-op tendency
    # ========================================================================
    @testset "Default chemistry model is nothing" begin
        @test isnothing(CA.AtmosChem().chemistry_model)
    end

    @testset "chemistry_tendency! with nothing is a no-op" begin
        result = CA.chemistry_tendency!(nothing, nothing, nothing, 0.0, nothing)
        @test isnothing(result)
    end
end
