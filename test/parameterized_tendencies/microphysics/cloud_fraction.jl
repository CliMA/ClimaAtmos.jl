#=
Unit tests for cloud_fraction.jl

Tests cover:
1. `_compute_cloud_fraction(q_c, sigma_S, q_sat, α, ε_rel, σ_abs)` -
   truncated-Gaussian CF closure with the non-equilibrium floor
   `σ_S_floor² = (ε_rel·q_sat)² + σ_abs²`.
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

@testset "Cloud Fraction" begin
    @testset "`sgs_geometric_stability_weight`" begin
        for FT in (Float32, Float64)
            w = CA.sgs_geometric_stability_weight
            N², S², Ri₀ = FT(1e-4), FT(1e-4), FT(0.25)
            @test w(N², S², FT(0)) === one(FT)          # k = 0: exactly 1
            @test w(-N², S², FT(0)) === one(FT)         # k = 0: 1 even at Ri₊ = 0
            @test w(-N², S², Ri₀) === zero(FT)          # unstable: 0
            @test w(zero(FT), S², Ri₀) === zero(FT)     # neutral: 0
            @test w(N², zero(FT), Ri₀) ≈ one(FT)        # no shear: Ri → ∞
            @test w(2 * S² * Ri₀, S², Ri₀) ≈ FT(0.5)    # Ri₊ = Ri₀
            ws = [w(n, S², Ri₀) for n in FT.((1e-6, 1e-5, 1e-4, 1e-3))]
            @test issorted(ws) && all(0 .<= ws .<= 1)
        end
    end


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
                    # σ_S = 0 ⇒ σ_aug = σ_S_floor ≈ √((ε_rel·q_sat)² + σ_abs²)
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

                @testset "ε_rel = 0 → σ_abs keeps result finite" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), FT(3.16e-4), FT(5e-5), α, FT(0), σ_abs,
                    )
                    @test isfinite(cf)
                    @test FT(0) <= cf <= FT(1)
                end

                @testset "Floor caps CF where q_c ≪ ε_rel·q_sat" begin
                    # q_c = 0.25 g/kg, ε_rel·q_sat = 1.5 g/kg, quiescent σ_S:
                    # C = q_c/(α ε_rel q_sat) ≈ 0.17 ⇒ CF well below 1.
                    cf = CA._compute_cloud_fraction(
                        FT(2.5e-4), FT(1e-5), FT(1e-2), α, FT(0.15), σ_abs,
                    )
                    @test cf < FT(0.6)
                end
            end
        end
    end

    @testset "`_compute_z` inversion accuracy" begin
        # z must satisfy the truncated-Gaussian relation z·Φ(z) + φ(z) = C.
        # Two Newton steps leave < 0.7 % residual for C ≥ 0.05 and < 4 % for
        # C ≥ 0.01 (a single step left 14 % and 36 %). Residuals are checked
        # with `normal_cdf` (abs. error 7.5e-8), so tolerances stay well
        # above the CDF approximation error.
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                φ(z) = exp(-z^2 / 2) / sqrt(FT(2) * FT(π))
                h(z) = z * CA.normal_cdf(z) + φ(z)
                for (C, rtol) in (
                    (FT(0.01), FT(0.05)),
                    (FT(0.05), FT(0.01)),
                    (FT(0.1), FT(0.005)),
                    (FT(0.3), FT(1e-3)),
                    (FT(1), FT(1e-3)),
                    (FT(5), FT(1e-3)),
                    (FT(100), FT(1e-3)),
                )
                    z = CA._compute_z(C)
                    @test h(z) ≈ C rtol = rtol
                end
                # C = 0 (no condensate): deeply negative z, negligible CF.
                # Regression for the second-step ϵ_numerics guard, without
                # which Float32 returned z ≈ −2.3 (CF ≈ 1 %).
                z0 = CA._compute_z(FT(0))
                @test CA.normal_cdf(z0) < FT(1e-10)
            end
        end
    end

end
