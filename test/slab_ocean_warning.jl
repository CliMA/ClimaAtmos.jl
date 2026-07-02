# Test that a warning is emitted when SlabOceanSST is used with a setup that provides
# a surface_condition temperature.
using Test
using Logging
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "SlabOceanSST Warning" begin
    FT = Float32
    params = CA.ClimaAtmosParameters(FT)

    # Shared config for the two SlabOceanSST testsets below
    config_slab = CA.AtmosConfig(
        Dict(
            "initial_condition" => "PrescribedColumn",
            "microphysics_model" => "dry",
            "config" => "column",
            "prognostic_surface" => "SlabOceanSST",
            "surface_setup" => "DefaultMoninObukhov",
            "albedo_model" => "ConstantAlbedo",
            "FLOAT_TYPE" => string(FT),
        ),
    )

    # TRMM_LBA is chosen because its surface_condition provides a temperature value
    # (see src/setups/TRMM_LBA.jl). If this changes in the future, this test
    # may need to be updated to use a different setup that provides a surface temperature.
    setup_type = CA.Setups.TRMM_LBA(; prognostic_tke = false, thermo_params = params.thermodynamics_params)

    # Sanity-check the fixture: TRMM_LBA must provide a surface temperature
    setup_pieces = CA.Setups.surface_condition(setup_type, params)
    @test !isnothing(setup_pieces.temperature)

    @testset "Warning emitted when setup provides surface temperature" begin
        log_buffer = IOBuffer()
        logger = SimpleLogger(log_buffer, Logging.Warn)

        with_logger(logger) do
            surface = CA.AtmosSurface(config_slab, params, FT; setup_type)
            @test surface.temperature isa CA.SurfaceConditions.SlabOceanTemperature
        end

        log_output = String(take!(log_buffer))
        @test contains(log_output, "`SlabOceanSST` is active")
        @test contains(log_output, "surface_condition")
        @test contains(log_output, "overwritten")
        @test contains(log_output, "prognostic initialization")
    end

    @testset "No warning when setup_type is nothing" begin
        log_buffer = IOBuffer()
        logger = SimpleLogger(log_buffer, Logging.Warn)

        with_logger(logger) do
            surface = CA.AtmosSurface(config_slab, params, FT; setup_type = nothing)
            @test surface.temperature isa CA.SurfaceConditions.SlabOceanTemperature
        end

        log_output = String(take!(log_buffer))
        @test !contains(log_output, "`SlabOceanSST` is active")
    end

    @testset "No warning with PrescribedSST" begin
        config_prescribed = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrescribedColumn",
                "microphysics_model" => "dry",
                "config" => "column",
                "prognostic_surface" => "PrescribedSST",
                "surface_setup" => "DefaultMoninObukhov",
                "albedo_model" => "ConstantAlbedo",
                "FLOAT_TYPE" => string(FT),
            ),
        )

        log_buffer = IOBuffer()
        logger = SimpleLogger(log_buffer, Logging.Warn)

        with_logger(logger) do
            surface = CA.AtmosSurface(config_prescribed, params, FT; setup_type)
            @test surface.temperature isa CA.SurfaceConditions.AnalyticTemperature
        end

        log_output = String(take!(log_buffer))
        @test !contains(log_output, "`SlabOceanSST` is active")
    end
end
