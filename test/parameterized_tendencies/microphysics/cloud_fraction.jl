#=
Unit tests for cloud_fraction.jl

Tests cover:
1. `_compute_cloud_fraction(q_c, sigma_S_sq, α)` - truncated-Gaussian CF closure
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

@testset "Cloud Fraction" begin

    @testset "`_compute_cloud_fraction`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                α = FT(1)

                @testset "No condensate → zero cloud fraction" begin
                    cf = CA._compute_cloud_fraction(FT(0), FT(1e-8), α)
                    @test cf == FT(0)
                end

                @testset "Condensate present, nonzero sigma → cf > 0" begin
                    cf = CA._compute_cloud_fraction(FT(1e-3), FT(1e-7), α)
                    @test FT(0) < cf <= FT(1)
                end

                @testset "Zero variance, condensate present → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(FT(1e-3), FT(0), α)
                    @test cf > FT(0.99)
                end

                @testset "Large condensate, small variance → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(FT(1e-2), FT(1e-10), α)
                    @test cf > FT(0.99)
                end

                @testset "Type stability" begin
                    cf = CA._compute_cloud_fraction(FT(1e-3), FT(1e-7), α)
                    @test cf isa FT
                end

                @testset "CF monotone in σ_S² at fixed q_c" begin
                    # Broader PDF → more unsaturated points → CF decreases toward 0.5.
                    cfs = [
                        CA._compute_cloud_fraction(FT(1e-3), σ², α) for
                        σ² in FT[1e-9, 1e-8, 1e-7, 1e-6]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Large σ_S², small q_c → cf approaching 0" begin
                    # C = q_c/(α·σ_S) → 0 ⟹ z → −∞ ⟹ CF = Φ(z) → 0.
                    cf = CA._compute_cloud_fraction(FT(1e-6), FT(1e-3), α)
                    @test cf < FT(0.01)
                end

                @testset "σ_S² → 0 limit → cf ≈ 1 when q_c > 0" begin
                    cf = CA._compute_cloud_fraction(FT(2e-3), FT(1e-12), α)
                    @test cf > FT(0.99)
                end
            end
        end
    end

end
