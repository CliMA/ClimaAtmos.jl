using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Chemistry Tendencies" begin

    # ========================================================================
    # Default: no chemistry model → no-op tendency
    # ========================================================================
    @testset "Default chemistry model is nothing" begin
        model = CA.AtmosModel()
        @test model.chemistry_model === nothing
    end

    @testset "chemistry_tendency! with nothing is a no-op" begin
        result = CA.chemistry_tendency!(nothing, nothing, nothing, 0.0, nothing)
        @test result === nothing
    end

    # ========================================================================
    # With Musica loaded: GasPhaseChem prints the version string
    # ========================================================================
    @testset "GasPhaseChem prints MUSICA version" begin
        import Musica
        @test_logs (:info, r"MUSICA version: .+") CA.chemistry_tendency!(
            nothing,
            nothing,
            nothing,
            0.0,
            CA.GasPhaseChem(),
        )
    end
end
