#=
Unit tests for cloud_fraction.jl

Tests cover:
1. `_compute_cloud_fraction(q_c, sigma_S, q_sat, α, ε_rel, σ_abs)` -
   truncated-Gaussian CF closure with the smooth non-equilibrium floor
   `σ_S_floor² = (ε_rel·q_sat)² + σ_abs²` controlled by the two parameters.
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

@testset "Cloud Fraction" begin

    @testset "`_compute_cloud_fraction`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                α = FT(1)
                ε_rel = FT(0.02)
                σ_abs = FT(1e-7)

                @testset "No condensate → zero cloud fraction" begin
                    cf = CA._compute_cloud_fraction(
                        FT(0), FT(1e-4), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf < eps(FT)
                end

                @testset "Condensate present, nonzero sigma → cf > 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), FT(3.16e-4), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test FT(0) < cf <= FT(1)
                end

                @testset "Zero sigma, large condensate → cf ≈ 1" begin
                    # σ_S = 0 ⇒ σ_aug = σ_S_floor = √((ε_rel·q_sat)² + σ_abs²)
                    # ≈ 0.02·5e-5 = 1e-6; C = 1e-3/1e-6 ≫ 1 → CF ≈ 1.
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), FT(0), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Large condensate, small sigma → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-2), FT(1e-5), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Type stability" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), FT(3.16e-4), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf isa FT
                end

                @testset "CF monotone in σ_S at fixed q_c" begin
                    cfs = [
                        CA._compute_cloud_fraction(
                            FT(1e-3), σ, FT(5e-5), α, ε_rel, σ_abs,
                        ) for σ in FT[1e-4, 3.16e-4, 1e-3, 3.16e-3]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Large σ_S, small q_c → cf approaching 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-6), FT(3.16e-2), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf < FT(0.01)
                end

                @testset "Tiny q_c with tiny σ_S → cf stays bounded (smooth floor)" begin
                    # σ_aug ≈ σ_S_floor ≈ 1e-6; C = q_c/σ_aug small ⇒ CF small.
                    cf = CA._compute_cloud_fraction(
                        FT(1e-9), FT(1e-12), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf < FT(0.51)
                end
            end
        end
    end

end
