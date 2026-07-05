#=
Unit tests for cloud_fraction.jl

Tests cover:
1. `_compute_cloud_fraction(q_c, mu_S, sigma_S, q_sat, α, ε_rel, σ_abs)` -
   truncated-Gaussian CF closure with the smooth non-equilibrium floor
   `σ_S_floor² = (D·ε_rel·q_sat)² + σ_abs²` controlled by the two parameters,
   where `D = x/√(1+x²)`, `x = max(−μ_S, 0)/(ε_rel·q_sat)` damps the floor
   as the subdomain mean saturates (overcast limit).
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
                # Well-subsaturated mean: x = 1e-3/(0.02·5e-5) ≫ 1 ⇒ D ≈ 1,
                # floor fully active (legacy behavior).
                μ_sub = FT(-1e-3)

                @testset "No condensate → zero cloud fraction" begin
                    cf = CA._compute_cloud_fraction(
                        FT(0), μ_sub, FT(1e-4), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf < eps(FT)
                end

                @testset "Condensate present, nonzero sigma → cf > 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(3.16e-4), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test FT(0) < cf <= FT(1)
                end

                @testset "Zero sigma, large condensate → cf ≈ 1" begin
                    # σ_S = 0 ⇒ σ_aug = σ_S_floor ≈ √((ε_rel·q_sat)² + σ_abs²)
                    # ≈ 0.02·5e-5 = 1e-6; C = 1e-3/1e-6 ≫ 1 → CF ≈ 1.
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(0), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Large condensate, small sigma → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-2), μ_sub, FT(1e-5), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Type stability" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(3.16e-4), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf isa FT
                end

                @testset "CF monotone in σ_S at fixed q_c" begin
                    cfs = [
                        CA._compute_cloud_fraction(
                            FT(1e-3), μ_sub, σ, FT(5e-5), α, ε_rel, σ_abs,
                        ) for σ in FT[1e-4, 3.16e-4, 1e-3, 3.16e-3]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Large σ_S, small q_c → cf approaching 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-6), μ_sub, FT(3.16e-2), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf < FT(0.01)
                end

                @testset "Tiny q_c with tiny σ_S → cf stays bounded (smooth floor)" begin
                    # σ_aug ≈ σ_S_floor ≈ 1e-6; C = q_c/σ_aug small ⇒ CF small.
                    cf = CA._compute_cloud_fraction(
                        FT(1e-9), μ_sub, FT(1e-12), FT(5e-5), α, ε_rel, σ_abs,
                    )
                    @test cf < FT(0.51)
                end

                @testset "Overcast limit: saturated mean damps the floor → cf ≈ 1" begin
                    # Stratocumulus-like: q_c = 0.25 g/kg ≪ ε_rel·q_sat with
                    # ε_rel = 0.15, q_sat = 10 g/kg (floor = 1.5 g/kg). The
                    # undamped floor caps CF well below 1; with the mean
                    # saturated (μ_S = q_c > 0) the floor collapses to σ_abs
                    # and CF → 1.
                    q_c = FT(2.5e-4)
                    q_sat = FT(1e-2)
                    σ_S = FT(1e-5)
                    cf_sat = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α, FT(0.15), σ_abs,
                    )
                    @test cf_sat > FT(0.99)
                    cf_undamped = CA._compute_cloud_fraction(
                        q_c, FT(-1e-1), σ_S, q_sat, α, FT(0.15), σ_abs,
                    )
                    @test cf_undamped < FT(0.6)
                end

                @testset "CF monotone in μ_S at fixed q_c (floor grows with subsaturation)" begin
                    q_c = FT(2.5e-4)
                    q_sat = FT(1e-2)
                    σ_S = FT(1e-5)
                    cfs = [
                        CA._compute_cloud_fraction(
                            q_c, μ, σ_S, q_sat, α, FT(0.15), σ_abs,
                        ) for μ in FT[0, -1e-4, -1e-3, -1e-2, -1e-1]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Damping continuous across μ_S = 0" begin
                    q_c = FT(2.5e-4)
                    q_sat = FT(1e-2)
                    σ_S = FT(1e-5)
                    cf0 = CA._compute_cloud_fraction(
                        q_c, FT(0), σ_S, q_sat, α, FT(0.15), σ_abs,
                    )
                    cf⁻ = CA._compute_cloud_fraction(
                        q_c, -eps(FT), σ_S, q_sat, α, FT(0.15), σ_abs,
                    )
                    cf⁺ = CA._compute_cloud_fraction(
                        q_c, FT(1e-6), σ_S, q_sat, α, FT(0.15), σ_abs,
                    )
                    @test abs(cf0 - cf⁻) < FT(1e-4)
                    @test abs(cf0 - cf⁺) < FT(1e-4)
                end

                @testset "ε_rel = 0 → guard keeps result finite" begin
                    for μ in FT[0, -1e-3]
                        cf = CA._compute_cloud_fraction(
                            FT(1e-3), μ, FT(3.16e-4), FT(5e-5), α, FT(0), σ_abs,
                        )
                        @test isfinite(cf)
                        @test FT(0) <= cf <= FT(1)
                    end
                end
            end
        end
    end

end
