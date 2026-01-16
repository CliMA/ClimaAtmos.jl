using Test
import ClimaAtmos as CA
import ClimaParams as CP
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

@testset "ClimaAtmosParameters Construction" begin
    for FT in (Float32, Float64)
        params = CA.ClimaAtmosParameters(FT)
        @test params isa CAP.ClimaAtmosParameters
        @test CAP.eltype(params) == FT

        # Verify sub-components types
        @test params.thermodynamics_params isa CAP.TD.Parameters.ThermodynamicsParameters
        @test params.turbconv_params isa CAP.TurbulenceConvectionParameters
        @test params.microphysics_cloud_params isa NamedTuple
        @test params.microphysics_cloud_params.liquid isa CAP.CM.Parameters.CloudLiquid
        
        # Verify physical constants (sanity check for Earth, would need to be changed for other planets)
        # R_d for dry air ~ 287 J/kg/K
        @test CAP.R_d(params) ≈ FT(287.0) rtol=0.01
        # Planet radius ~ 6371 km
        @test CAP.planet_radius(params) ≈ FT(6.371e6) rtol=0.01
        # Gravity ~ 9.8 m/s^2
        @test CAP.grav(params) ≈ FT(9.81) rtol=0.01
    end
end

@testset "AtmosConfig Parameter Overrides" begin
    # Test overriding a parameter via configuration logic
    # CA.AtmosConfig merges dicts into the parameters
    
    # Create a temporary TOML file for overrides
    mktemp() do path, io
        write(io, """
        [planet_radius]
        value = 1000.0
        type = "float"
        """)
        flush(io)
        
        # Pass the TOML file to AtmosConfig
        # Note: We need to pass the path as a list in the "toml" key
        config_dict = Dict("toml" => [path])
        config = CA.AtmosConfig(config_dict, job_id = "parameter_test_override")
        
        params = CA.ClimaAtmosParameters(config)
        
        # Check if override worked
        @test CAP.planet_radius(params) == 1000.0
        
        # Check that other parameters remained default-like (sanity check)
        @test CAP.R_d(params) ≈ 287.0 rtol=0.01
    end
end

@testset "TOML Integration" begin
    # Iterate over all TOML files in the package to ensure they load without error
    # This preserves the original test intent but cleanly
    toml_path = joinpath(pkgdir(CA), "toml")
    for (index, toml_file) in enumerate(readdir(toml_path))
        # Skip if not a .toml file
        endswith(toml_file, ".toml") || continue
        
        config_dict = Dict("toml" => [joinpath(toml_path, toml_file)])
        config = CA.AtmosConfig(config_dict, job_id = "parameter_test_toml_$(index)")
        
        @test CA.ClimaAtmosParameters(config) isa CAP.ClimaAtmosParameters
    end
end
