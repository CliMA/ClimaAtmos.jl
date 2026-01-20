using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import Thermodynamics as TD
import CloudMicrophysics as CM

include("../../test_helpers.jl")

# Helper to run standard precipitation tests for any configuration
function test_precipitation_setup!(Y, p, expected_vars)
    FT = eltype(Y)
    (; turbconv_model, moisture_model, microphysics_model) = p.atmos

    CA.set_precipitation_velocities!(
        Y,
        p,
        moisture_model,
        microphysics_model,
        turbconv_model,
    )
    CA.set_precipitation_cache!(Y, p, microphysics_model, turbconv_model)
    CA.set_precipitation_surface_fluxes!(Y, p, microphysics_model)

    # Verify expected cache variables exist
    for var_name in expected_vars
        @test var_name ∈ propertynames(p.precomputed)
    end
end

function test_precipitation_tendency!(ᶜYₜ, Y, p)
    FT = eltype(Y)
    (; turbconv_model, moisture_model, microphysics_model) = p.atmos

    CA.precipitation_tendency!(
        ᶜYₜ,
        Y,
        p,
        FT(0),
        moisture_model,
        microphysics_model,
        turbconv_model,
    )

    # Test water budget: total mass tendency = total water tendency
    @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot

    # No NaNs in tendencies
    @test !any(isnan, ᶜYₜ.c.ρ)
    @test !any(isnan, ᶜYₜ.c.ρq_tot)
    @test !any(isnan, ᶜYₜ.c.ρe_tot)
end

function test_terminal_velocities_nonnegative(p, velocity_vars, FT)
    for var in velocity_vars
        @test minimum(getproperty(p.precomputed, var)) >= FT(0)
    end
end

function test_cloud_fraction_bounds(p, FT)
    @test !any(isnan, p.precomputed.cloud_diagnostics_tuple.cf)
    @test minimum(p.precomputed.cloud_diagnostics_tuple.cf) >= FT(0)
    @test maximum(p.precomputed.cloud_diagnostics_tuple.cf) <= FT(1)
end

@testset "Precipitation" begin

    @testset "Equilibrium moisture + 0-moment" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "DYCOMS_RF02",
                "moist" => "equil",
                "precip_model" => "0M",
                "config" => "column",
                "output_default_diagnostics" => false,
            ),
            job_id = "precip_0M",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, moisture_model, microphysics_model) = p.atmos

        # Expected cache variables for 0-moment
        expected_vars =
            (:ᶜS_ρq_tot, :ᶜS_ρe_tot, :surface_rain_flux, :surface_snow_flux, :ᶜwₜqₜ, :ᶜwₕhₜ)
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency
        test_precipitation_tendency!(ᶜYₜ, Y, p)

        # Verify source equals tendency
        @test ᶜYₜ.c.ρ == p.precomputed.ᶜS_ρq_tot

        # No cloud condensate tendency for equilibrium model
        @test CA.cloud_condensate_tendency!(
            ᶜYₜ,
            Y,
            p,
            moisture_model,
            microphysics_model,
            turbconv_model,
        ) isa Nothing
    end

    @testset "NonEquilibrium moisture + 1-moment" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrecipitatingColumn",
                "moist" => "nonequil",
                "precip_model" => "1M",
                "config" => "column",
                "output_default_diagnostics" => false,
            ),
            job_id = "precip_1M",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, moisture_model, microphysics_model) = p.atmos

        # Expected cache variables for 1-moment
        expected_vars = (
            :ᶜSqₗᵖ, :ᶜSqᵢᵖ, :ᶜSqᵣᵖ, :ᶜSqₛᵖ,
            :surface_rain_flux, :surface_snow_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test limit helper function
        @test CA.limit(FT(10), FT(2), 5) == FT(1)

        # Test tendency
        test_precipitation_tendency!(ᶜYₜ, Y, p)

        # Additional 1M-specific NaN checks
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)
        @test !any(isnan, ᶜYₜ.c.ρq_rai)
        @test !any(isnan, ᶜYₜ.c.ρq_sno)

        # Terminal velocities must be non-negative
        test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ), FT)

        # Cloud condensate tendency should run without NaNs
        CA.cloud_condensate_tendency!(
            ᶜYₜ,
            Y,
            p,
            moisture_model,
            microphysics_model,
            turbconv_model,
        )
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)

        @test all(iszero, ᶜYₜ.c.ρq_tot)

        # Cloud fraction bounds
        test_cloud_fraction_bounds(p, FT)
    end

    @testset "NonEquilibrium moisture + 2-moment" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrecipitatingColumn",
                "moist" => "nonequil",
                "precip_model" => "2M",
                "config" => "column",
                "output_default_diagnostics" => false,
                "prescribed_aerosols" => ["SSLT01"],
            ),
            job_id = "precip_2M",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, moisture_model, microphysics_model) = p.atmos

        # Expected cache variables for 2-moment (includes number densities)
        expected_vars = (
            :ᶜSqₗᵖ, :ᶜSqᵢᵖ, :ᶜSqᵣᵖ, :ᶜSqₛᵖ, :ᶜSnₗᵖ, :ᶜSnᵣᵖ,
            :surface_rain_flux, :surface_snow_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₙₗ, :ᶜwₙᵣ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency
        test_precipitation_tendency!(ᶜYₜ, Y, p)

        # Additional 2M-specific NaN checks (includes number densities)
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)
        @test !any(isnan, ᶜYₜ.c.ρq_rai)
        @test !any(isnan, ᶜYₜ.c.ρq_sno)
        @test !any(isnan, ᶜYₜ.c.ρn_liq)
        @test !any(isnan, ᶜYₜ.c.ρn_rai)

        # Terminal velocities must be non-negative (includes number-weighted)
        test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₙₗ, :ᶜwₙᵣ), FT)

        # Cloud condensate tendency
        CA.cloud_condensate_tendency!(
            ᶜYₜ,
            Y,
            p,
            moisture_model,
            microphysics_model,
            turbconv_model,
        )
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)
        @test !any(isnan, ᶜYₜ.c.ρn_liq)

        @test all(iszero, ᶜYₜ.c.ρq_tot)

        # Cloud fraction bounds
        test_cloud_fraction_bounds(p, FT)
    end
end
