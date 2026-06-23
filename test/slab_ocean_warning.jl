"""
    Test that a warning is emitted when SlabOceanSST is used with a setup that provides
    a surface_condition temperature.
"""
using Test
using Logging
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "SlabOceanSST Warning" begin
    FT = Float32
    
    @testset "Warning emitted when setup provides surface temperature" begin
        # Create a minimal config with SlabOceanSST
        config = CA.AtmosConfig(Dict(
            "initial_condition" => "PrescribedColumn",
            "microphysics_model" => "dry",
            "config" => "column",
            "prognostic_surface" => "SlabOceanSST",
            "surface_setup" => "DefaultMoninObukhov",
            "albedo_model" => "ConstantAlbedo",
            "FLOAT_TYPE" => string(FT),
        ))
        
        params = CA.Parameters.ClimaAtmosParameters(FT)
        
        # Create a setup that has a surface_condition with temperature.
        # TRMM_LBA is chosen because its surface_condition provides a temperature value
        # (see src/setups/TRMM_LBA.jl). If this changes in the future, this test
        # may need to be updated to use a different setup that provides a surface temperature.
        setup_type = CA.Setups.TRMM_LBA()
        
        # Verify that the setup indeed provides a surface temperature
        setup_pieces = CA.Setups.surface_condition(setup_type, params)
        @test !isnothing(setup_pieces.temperature) "TRMM_LBA setup must provide a surface_condition temperature"
        
        # Capture warnings
        log_buffer = IOBuffer()
        logger = SimpleLogger(log_buffer, Logging.Warn)
        
        with_logger(logger) do
            # This should emit a warning
            surface = CA.AtmosSurface(config, params, FT; setup_type = setup_type)
            
            # Verify that the surface was created with SlabOceanTemperature
            @test surface.temperature isa CA.SurfaceConditions.SlabOceanTemperature
        end
        
        # Check that a warning was logged
        log_output = String(take!(log_buffer))
        @test contains(log_output, "`SlabOceanSST` is active")
        @test contains(log_output, "surface_condition")
        @test contains(log_output, "overwritten")
        @test contains(log_output, "prognostic initialization")
    end
    
    @testset "No warning when setup_type is nothing" begin
        # Create a config with SlabOceanSST but no setup_type
        config = CA.AtmosConfig(Dict(
            "initial_condition" => "PrescribedColumn",
            "microphysics_model" => "dry",
            "config" => "column",
            "prognostic_surface" => "SlabOceanSST",
            "surface_setup" => "DefaultMoninObukhov",
            "albedo_model" => "ConstantAlbedo",
            "FLOAT_TYPE" => string(FT),
        ))
        
        params = CA.Parameters.ClimaAtmosParameters(FT)
        
        # Capture warnings
        log_buffer = IOBuffer()
        logger = SimpleLogger(log_buffer, Logging.Warn)
        
        with_logger(logger) do
            # This should NOT emit a warning since setup_type is nothing
            surface = CA.AtmosSurface(config, params, FT; setup_type = nothing)
            
            # Verify that the surface was created with SlabOceanTemperature
            @test surface.temperature isa CA.SurfaceConditions.SlabOceanTemperature
        end
        
        # Check that no warning was logged
        log_output = String(take!(log_buffer))
        @test !contains(log_output, "`SlabOceanSST` is active")
    end
    
    @testset "No warning with PrescribedSST" begin
        # Create a config with PrescribedSST
        config = CA.AtmosConfig(Dict(
            "initial_condition" => "PrescribedColumn",
            "microphysics_model" => "dry",
            "config" => "column",
            "prognostic_surface" => "PrescribedSST",
            "surface_setup" => "DefaultMoninObukhov",
            "albedo_model" => "ConstantAlbedo",
            "FLOAT_TYPE" => string(FT),
        ))
        
        params = CA.Parameters.ClimaAtmosParameters(FT)
        
        # Create a setup that has a surface_condition with temperature
        setup_type = CA.Setups.TRMM_LBA()
        
        # Capture warnings
        log_buffer = IOBuffer()
        logger = SimpleLogger(log_buffer, Logging.Warn)
        
        with_logger(logger) do
            # This should NOT emit a warning since it's PrescribedSST, not SlabOceanSST
            surface = CA.AtmosSurface(config, params, FT; setup_type = setup_type)
            
            # Verify that the surface was created with AnalyticTemperature
            @test surface.temperature isa CA.SurfaceConditions.AnalyticTemperature
        end
        
        # Check that no warning was logged
        log_output = String(take!(log_buffer))
        @test !contains(log_output, "`SlabOceanSST` is active")
    end
end
