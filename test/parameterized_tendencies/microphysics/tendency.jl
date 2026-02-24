#=
Unit tests for tendency.jl

Tests cover:
1. microphysics_tendency! dispatch (DryModel returns nothing)
2. Integration tests (merged from precipitation.jl)
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import Thermodynamics as TD
import CloudMicrophysics as CM

include("../../test_helpers.jl")

# ============================================================================
# Dispatch Tests
# ============================================================================

@testset "Microphysics Tendency Dispatch" begin

    @testset "DryModel returns nothing" begin
        result = CA.microphysics_tendency!(
            nothing, nothing, nothing, nothing,
            CA.DryModel(), nothing,
        )
        @test isnothing(result)
    end
end

# ============================================================================
# Integration Test Helpers
# ============================================================================

function test_precipitation_setup!(Y, p, expected_vars)
    FT = eltype(Y)
    (; turbconv_model, microphysics_model) = p.atmos

    CA.set_precipitation_velocities!(
        Y, p, microphysics_model, turbconv_model,
    )
    CA.set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)
    CA.set_precipitation_surface_fluxes!(Y, p, microphysics_model)

    # Verify expected cache variables exist
    for var_name in expected_vars
        @test var_name ∈ propertynames(p.precomputed)
    end
end

function test_microphysics_tendency!(ᶜYₜ, Y, p)
    FT = eltype(Y)
    (; turbconv_model, microphysics_model) = p.atmos

    CA.microphysics_tendency!(
        ᶜYₜ, Y, p, FT(0),
        microphysics_model, turbconv_model,
    )

    # No NaNs in tendencies
    @test !any(isnan, ᶜYₜ.c.ρ)
    @test !any(isnan, ᶜYₜ.c.ρq_tot)
    @test !any(isnan, ᶜYₜ.c.ρe_tot)
end

function test_terminal_velocities_nonnegative(p, velocity_vars, FT)
    for var in velocity_vars
        @test minimum(getproperty(p.precomputed, var)) >= -eps(FT)
    end
end

function test_cloud_fraction_bounds(p, FT)
    @test !any(isnan, p.precomputed.ᶜcloud_fraction)
    @test minimum(p.precomputed.ᶜcloud_fraction) >= -eps(FT)
    @test maximum(p.precomputed.ᶜcloud_fraction) <= FT(1) + eps(FT)
end


# =================
# Integration Tests
# =================
@testset "Microphysics Tendency Integration" begin

    @testset "Equilibrium moisture + 0-moment" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "DYCOMS_RF02",
                "microphysics_model" => "0M",
                "config" => "column",
                "output_default_diagnostics" => false,
            ),
            job_id = "tendency_0M",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, microphysics_model) = p.atmos

        # Expected cache variables for 0-moment
        expected_vars =
            (:ᶜS_ρq_tot, :ᶜS_ρe_tot, :surface_rain_flux, :surface_snow_flux, :ᶜwₜqₜ, :ᶜwₕhₜ)
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # Test water budget: total mass tendency = total water tendency
        @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot

        # Verify source equals tendency
        @test ᶜYₜ.c.ρ == p.precomputed.ᶜS_ρq_tot


    end

    @testset "NonEquilibrium moisture + 1-moment" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrecipitatingColumn",
                "microphysics_model" => "1M",
                "config" => "column",
                "output_default_diagnostics" => false,
            ),
            job_id = "tendency_1M",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, microphysics_model) = p.atmos

        # Expected cache variables for 1-moment
        expected_vars = (
            :ᶜSqₗᵐ, :ᶜSqᵢᵐ, :ᶜSqᵣᵐ, :ᶜSqₛᵐ,
            :surface_rain_flux, :surface_snow_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test limit helper function
        @test CA.limit(FT(10), FT(2), 5) == FT(1)

        # Test tendency
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # Additional 1M-specific NaN checks
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)
        @test !any(isnan, ᶜYₜ.c.ρq_rai)
        @test !any(isnan, ᶜYₜ.c.ρq_sno)

        # Terminal velocities must be non-negative
        test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ), FT)

        @test all(iszero, parent(ᶜYₜ.c.ρq_tot))

        # Cloud fraction bounds
        test_cloud_fraction_bounds(p, FT)


    end

    @testset "NonEquilibrium moisture + 2-moment" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrecipitatingColumn",
                "microphysics_model" => "2M",
                "config" => "column",
                "output_default_diagnostics" => false,
                "prescribed_aerosols" => ["SSLT01"],
            ),
            job_id = "tendency_2M",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, microphysics_model) = p.atmos

        # Expected cache variables for 2-moment (includes number densities)
        expected_vars = (
            :ᶜSqₗᵐ, :ᶜSqᵢᵐ, :ᶜSqᵣᵐ, :ᶜSqₛᵐ, :ᶜSnₗᵐ, :ᶜSnᵣᵐ,
            :surface_rain_flux, :surface_snow_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₙₗ, :ᶜwₙᵣ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # Additional 2M-specific NaN checks (includes number densities)
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)
        @test !any(isnan, ᶜYₜ.c.ρq_rai)
        @test !any(isnan, ᶜYₜ.c.ρq_sno)
        @test !any(isnan, ᶜYₜ.c.ρn_liq)
        @test !any(isnan, ᶜYₜ.c.ρn_rai)

        # Terminal velocities must be non-negative (includes number-weighted)
        test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₙₗ, :ᶜwₙᵣ), FT)

        @test all(iszero, parent(ᶜYₜ.c.ρq_tot))

        # Cloud fraction bounds
        test_cloud_fraction_bounds(p, FT)


    end
    @testset "NonEquilibrium moisture + 1-moment + SGS quadrature" begin
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "PrecipitatingColumn",
                "microphysics_model" => "1M",
                "config" => "column",
                "use_sgs_quadrature" => true,
                "sgs_distribution" => "gaussian",
                "quadrature_order" => 2,
                "output_default_diagnostics" => false,
            ),
            job_id = "tendency_1M_sgs",
        )
        (; Y, p, params) = generate_test_simulation(config)
        FT = eltype(Y)
        ᶜYₜ = zero(Y)
        (; turbconv_model, microphysics_model) = p.atmos

        # Quadrature-specific cache variables
        expected_vars = (
            :ᶜSqₗᵐ, :ᶜSqᵢᵐ, :ᶜSqᵣᵐ, :ᶜSqₛᵐ,
            :surface_rain_flux, :surface_snow_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency with SGS quadrature
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # 1M-specific NaN checks
        @test !any(isnan, ᶜYₜ.c.ρq_liq)
        @test !any(isnan, ᶜYₜ.c.ρq_ice)
        @test !any(isnan, ᶜYₜ.c.ρq_rai)
        @test !any(isnan, ᶜYₜ.c.ρq_sno)

        # Terminal velocities must be non-negative
        test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ), FT)

        # Cloud fraction bounds
        test_cloud_fraction_bounds(p, FT)

        # ρq_tot tendency should be zero for 1M (no total water tendency)
        @test all(iszero, parent(ᶜYₜ.c.ρq_tot))


    end
end
