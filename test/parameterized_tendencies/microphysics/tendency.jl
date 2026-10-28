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

    # The rain and snow energy fluxes partition the total precipitation energy flux
    (; surface_rain_energy_flux, surface_snow_energy_flux) = p.precomputed
    pet = p.precomputed.col_integrated_precip_energy_tendency
    @test all(isfinite, parent(surface_rain_energy_flux))
    @test all(isfinite, parent(surface_snow_energy_flux))
    @test parent(surface_rain_energy_flux) .+ parent(surface_snow_energy_flux) ≈
          parent(pet) rtol = sqrt(eps(FT)) atol = sqrt(eps(FT))
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
        expected_vars = (
            :ᶜmp_tendency, :ᶜρ_dq_tot_dt, :ᶜρ_de_tot_dt,
            :surface_rain_flux, :surface_snow_flux,
            :surface_rain_energy_flux, :surface_snow_energy_flux, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # Test water budget: total mass tendency = total water tendency
        @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot

        # Verify source equals tendency
        @test ᶜYₜ.c.ρ == p.precomputed.ᶜρ_dq_tot_dt


    end

    @testset "Equilibrium moisture + 0-moment + PrognosticEDMFX updraft budget" begin
        config = CA.AtmosConfig(
            Dict{String, Any}(
                "initial_condition" => "Bomex",
                "FLOAT_TYPE" => "Float64",
                "config" => "column",
                "turbconv" => "prognostic_edmfx",
                "edmfx_entr_model" => "Generalized",
                "edmfx_detr_model" => "Generalized",
                "edmfx_sgs_mass_flux" => true,
                "edmfx_sgs_diffusive_flux" => false,
                "edmfx_vertical_diffusion" => false,
                "edmfx_nh_pressure" => true,
                "implicit_diffusion" => false,
                "prognostic_tke" => false,
                "microphysics_model" => "0M",
                "z_max" => 3000.0,
                "z_elem" => 15,
                "z_stretch" => false,
                "dt" => "10secs",
                "t_end" => "10secs",
                "ode_algo" => "SSP333",
                "toml" =>
                    [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
                "output_default_diagnostics" => false,
            );
            job_id = "tendency_0M_prognostic_edmfx",
        )
        (; Y, p) = generate_test_simulation(config)
        FT = eltype(Y)
        thp = CAP.thermodynamics_params(p.params)
        (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰, ᶜTʲs, ᶜρʲs) = p.precomputed
        (; ᶜΦ) = p.core

        # Only the updraft removes condensate, carrying liquid enthalpy at Tʲ.
        @. ᶜmp_tendency⁰.dq_tot_dt = FT(0)
        @. ᶜmp_tendency⁰.e_tot_hlpr = FT(0)
        ᶜmpʲ = ᶜmp_tendencyʲs.:(1)
        @. ᶜmpʲ.dq_tot_dt = FT(-1e-6)
        @. ᶜmpʲ.e_tot_hlpr = TD.internal_energy_liquid(thp, ᶜTʲs.:(1)) + ᶜΦ
        @. Y.c.sgsʲs.:(1).ρa = FT(0.1) * ᶜρʲs.:(1)

        Yₜ = zero(Y)
        CA.microphysics_tendency!(
            Yₜ, Y, p, FT(0),
            p.atmos.microphysics_model, p.atmos.turbconv_model,
        )

        ρaʲ = Array(parent(Y.c.sgsʲs.:(1).ρa))
        q_totʲ = Array(parent(Y.c.sgsʲs.:(1).q_tot))
        mseʲ = Array(parent(Y.c.sgsʲs.:(1).mse))
        ρaʲₜ = Array(parent(Yₜ.c.sgsʲs.:(1).ρa))
        q_totʲₜ = Array(parent(Yₜ.c.sgsʲs.:(1).q_tot))
        mseʲₜ = Array(parent(Yₜ.c.sgsʲs.:(1).mse))
        ρq_totₜ = Array(parent(Yₜ.c.ρq_tot))
        ρe_totₜ = Array(parent(Yₜ.c.ρe_tot))

        # The updraft loses the water and static energy that the grid mean
        # loses, so the environment residual is unaffected.
        @test maximum(abs, ρe_totₜ) > 0
        @test ρaʲₜ .* q_totʲ .+ ρaʲ .* q_totʲₜ ≈ ρq_totₜ rtol = 1e-10
        @test ρaʲₜ .* mseʲ .+ ρaʲ .* mseʲₜ ≈ ρe_totₜ rtol = 1e-10

        # The analytic ρa solve removes the precipitated mass at the rate
        # dq_tot_dt; every other process in that solve is zeroed.
        fill!(Y.f.sgsʲs.:(1).u₃, CC.Geometry.Covariant3Vector(FT(0)))
        for name in (
            :ᶜρ_diffʲs,
            :ᶜentr_vel_scaleʲs,
            :ᶜentr_nonvel_rateʲs,
            :ᶜarea_bounding_entr_detrʲs,
            :sfc_mass_flux_sourceʲs,
        )
            fill!(getproperty(p.precomputed, name).:(1), FT(0))
        end
        dtγ = FT(10)
        CA.solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)
        stage_area_weighted_density = Array(parent(Y.c.sgsʲs.:(1).ρa))
        @test (stage_area_weighted_density .- ρaʲ) ./ dtγ ≈
              FT(-1e-6) .* stage_area_weighted_density rtol = 1e-8
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
            :ᶜmp_tendency,
            :surface_rain_flux, :surface_snow_flux,
            :surface_rain_energy_flux, :surface_snow_energy_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test limit helper function
        @test CA.limit(FT(10), FT(2), 5) == FT(1)

        # Test tendency
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # Additional 1M-specific NaN checks
        @test !any(isnan, ᶜYₜ.c.ρq_lcl)
        @test !any(isnan, ᶜYₜ.c.ρq_icl)
        @test !any(isnan, ᶜYₜ.c.ρq_rai)
        @test !any(isnan, ᶜYₜ.c.ρq_sno)

        # Terminal velocities must be non-negative
        test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ), FT)

        @test all(iszero, parent(ᶜYₜ.c.ρq_tot))

        # Cloud fraction bounds
        test_cloud_fraction_bounds(p, FT)


    end

    # DISABLED (CloudMicrophysics 0.37 compat): 2-moment microphysics is
    # temporarily blocked by an `@assert` in `precomputed_quantities`
    # (src/cache/precomputed_quantities.jl) - `generate_test_simulation`
    # throws `AssertionError` before this testset's body can run. Re-enable
    # once 2M/2M+P3 compatibility with CloudMicrophysics 0.37 is restored.
    # @testset "NonEquilibrium moisture + 2-moment" begin
    #     config = CA.AtmosConfig(
    #         Dict(
    #             "initial_condition" => "PrecipitatingColumn",
    #             "microphysics_model" => "2M",
    #             "use_sgs_quadrature" => false,
    #             "config" => "column",
    #             "output_default_diagnostics" => false,
    #             "prescribed_aerosols" => ["SSLT01"],
    #         ),
    #         job_id = "tendency_2M",
    #     )
    #     (; Y, p, params) = generate_test_simulation(config)
    #     FT = eltype(Y)
    #     ᶜYₜ = zero(Y)
    #     (; turbconv_model, microphysics_model) = p.atmos
    #
    #     # Expected cache variables for 2-moment (includes number densities)
    #     expected_vars = (
    #         :ᶜmp_tendency,
    #         :surface_rain_flux, :surface_snow_flux,
    #         :surface_rain_energy_flux, :surface_snow_energy_flux,
    #         :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₙₗ, :ᶜwₙᵣ, :ᶜwₜqₜ, :ᶜwₕhₜ,
    #     )
    #     test_precipitation_setup!(Y, p, expected_vars)
    #
    #     # Test tendency
    #     test_microphysics_tendency!(ᶜYₜ, Y, p)
    #
    #     # Additional 2M-specific NaN checks (includes number densities)
    #     @test !any(isnan, ᶜYₜ.c.ρq_lcl)
    #     @test !any(isnan, ᶜYₜ.c.ρq_icl)
    #     @test !any(isnan, ᶜYₜ.c.ρq_rai)
    #     @test !any(isnan, ᶜYₜ.c.ρq_sno)
    #     @test !any(isnan, ᶜYₜ.c.ρn_lcl)
    #     @test !any(isnan, ᶜYₜ.c.ρn_rai)
    #
    #     # Terminal velocities must be non-negative (includes number-weighted)
    #     test_terminal_velocities_nonnegative(p, (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₙₗ, :ᶜwₙᵣ), FT)
    #
    #     @test all(iszero, parent(ᶜYₜ.c.ρq_tot))
    #
    #     # Cloud fraction bounds
    #     test_cloud_fraction_bounds(p, FT)
    # end
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
            :ᶜmp_tendency,
            :surface_rain_flux, :surface_snow_flux,
            :surface_rain_energy_flux, :surface_snow_energy_flux,
            :ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ, :ᶜwₜqₜ, :ᶜwₕhₜ,
        )
        test_precipitation_setup!(Y, p, expected_vars)

        # Test tendency with SGS quadrature
        test_microphysics_tendency!(ᶜYₜ, Y, p)

        # 1M-specific NaN checks
        @test !any(isnan, ᶜYₜ.c.ρq_lcl)
        @test !any(isnan, ᶜYₜ.c.ρq_icl)
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
