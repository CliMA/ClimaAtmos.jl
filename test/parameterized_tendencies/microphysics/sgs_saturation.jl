#=
Unit tests for SGS Saturation Adjustment (sgs_saturation.jl)
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP

@testset "SGS Saturation Adjustment" begin

    @testset "SaturationAdjustmentEvaluator" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.0)

                evaluator = ClimaAtmos.SaturationAdjustmentEvaluator(thp, ρ)

                @testset "Unsaturated conditions" begin
                    # T and q where q < q_sat (unsaturated)
                    T_hat = FT(300.0)
                    q_sat = TD.q_vap_saturation(thp, T_hat, ρ)
                    q_hat = FT(0.5) * q_sat  # Half saturation

                    result = evaluator(T_hat, q_hat)

                    @test haskey(result, :T)
                    @test haskey(result, :q_liq)
                    @test haskey(result, :q_ice)

                    # No condensate when unsaturated
                    @test result.q_liq ≈ FT(0) atol = eps(FT)
                    @test result.q_ice ≈ FT(0) atol = eps(FT)
                    @test result.T == T_hat
                end

                @testset "Saturated conditions (warm)" begin
                    # Warm temperatures: all condensate should be liquid
                    T_hat = FT(290.0)  # Above freezing
                    q_sat = TD.q_vap_saturation(thp, T_hat, ρ)
                    q_hat = FT(1.5) * q_sat  # 50% supersaturated

                    result = evaluator(T_hat, q_hat)

                    # Should have condensate
                    q_cond_expected = q_hat - q_sat
                    @test result.q_liq + result.q_ice ≈ q_cond_expected rtol = FT(1e-5)

                    # At warm temperatures, should be mostly liquid
                    λ = TD.liquid_fraction_ramp(thp, T_hat)
                    @test result.q_liq ≈ λ * q_cond_expected rtol = FT(1e-5)
                end

                @testset "Saturated conditions (cold)" begin
                    # Cold temperatures: all condensate should be ice
                    T_hat = FT(240.0)  # Below freezing
                    q_sat = TD.q_vap_saturation(thp, T_hat, ρ)
                    q_hat = FT(1.5) * q_sat  # 50% supersaturated

                    result = evaluator(T_hat, q_hat)

                    # At cold temperatures, should be mostly ice
                    λ = TD.liquid_fraction_ramp(thp, T_hat)
                    @test λ < FT(0.5)  # Verify it's actually cold enough
                    q_cond_expected = q_hat - q_sat
                    @test result.q_ice ≈ (1 - λ) * q_cond_expected rtol = FT(1e-5)
                end

                @testset "Type stability" begin
                    T_hat = FT(280.0)
                    q_hat = FT(0.01)
                    result = evaluator(T_hat, q_hat)

                    @test eltype(result.T) == FT
                    @test eltype(result.q_liq) == FT
                    @test eltype(result.q_ice) == FT
                end
            end
        end
    end

    @testset "compute_sgs_saturation_adjustment" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)

                ρ = FT(1.0)
                T_mean = FT(280.0)
                q_mean = FT(0.01)

                # Pre-compute grid-mean condensate for comparison
                q_sat_gm = TD.q_vap_saturation(thp, T_mean, ρ)
                q_cond_gm = max(FT(0), q_mean - q_sat_gm)
                λ_gm = TD.liquid_fraction_ramp(thp, T_mean)
                q_liq_gm = λ_gm * q_cond_gm
                q_ice_gm = (FT(1) - λ_gm) * q_cond_gm

                @testset "Zero variance = grid-mean" begin
                    # Zero covariances should give grid-mean result
                    result = ClimaAtmos.compute_sgs_saturation_adjustment(
                        thp, quad, ρ, T_mean, q_mean,
                        FT(0), FT(0), FT(0),
                    )

                    @test result.q_liq ≈ q_liq_gm rtol = FT(1e-5)
                    @test result.q_ice ≈ q_ice_gm rtol = FT(1e-5)
                    @test result.T == T_mean
                end

                @testset "Single quadrature point = grid-mean" begin
                    # Use GaussianSGS: only Gaussian has χ=0 → (μ_T, μ_q)
                    quad_1pt = ClimaAtmos.SGSQuadrature(
                        FT;
                        quadrature_order = 1,
                        distribution = ClimaAtmos.GaussianSGS(),
                    )

                    result = ClimaAtmos.compute_sgs_saturation_adjustment(
                        thp, quad_1pt, ρ, T_mean, q_mean,
                        FT(1.0), FT(1e-6), FT(0),
                    )

                    # Even with variance, single point should give grid-mean
                    @test result.q_liq ≈ q_liq_gm rtol = FT(1e-4)
                    @test result.q_ice ≈ q_ice_gm rtol = FT(1e-4)
                end

                @testset "Non-zero variance gives finite result" begin
                    # With variance
                    result = ClimaAtmos.compute_sgs_saturation_adjustment(
                        thp, quad, ρ, T_mean, q_mean,
                        FT(4.0), FT(1e-5), FT(1e-3),
                    )

                    @test isfinite(result.T)
                    @test isfinite(result.q_liq)
                    @test isfinite(result.q_ice)
                    @test result.q_liq >= FT(0)
                    @test result.q_ice >= FT(0)
                end

                @testset "Condensate non-negativity" begin
                    # Edge case: very dry conditions
                    result = ClimaAtmos.compute_sgs_saturation_adjustment(
                        thp, quad, ρ, FT(300.0), FT(0.001),
                        FT(1.0), FT(1e-8), FT(0),
                    )

                    @test result.q_liq >= FT(0)
                    @test result.q_ice >= FT(0)
                end

                @testset "Type stability" begin
                    result = ClimaAtmos.compute_sgs_saturation_adjustment(
                        thp, quad, ρ, T_mean, q_mean,
                        FT(1.0), FT(1e-6), FT(0),
                    )

                    @test result isa NamedTuple
                    @test eltype(result.T) == FT
                    @test eltype(result.q_liq) == FT
                    @test eltype(result.q_ice) == FT
                end
            end
        end
    end

end
