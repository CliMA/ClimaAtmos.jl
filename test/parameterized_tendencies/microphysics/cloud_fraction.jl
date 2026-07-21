#=
Unit tests for cloud_fraction.jl

Tests cover:
1. `_compute_cloud_fraction(q_c, mu_S, sigma_S, q_sat, α, floor)` -
   truncated-Gaussian CF closure with the smooth non-equilibrium floor
   `σ_S_floor² = (D·ε_rel·q_sat)² + σ_abs²`, where the release factor

       w = √((margin·α)²·(σ_S² + σ_abs²) + (abs_margin·ε_rel·q_sat)²),
       x = max(μ_S, 0)/w,
       D = residual + (1 − residual)·(1 + x²)^(−sharpness/2)

   releases the relative floor only where the subdomain mean is saturated
   by a margin relative to the equilibrium PDF width (overcast limit); for
   μ_S ≤ 0 the floor is fully active for any shape-parameter values.
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

floor_nt(
    ::Type{FT};
    ε_rel = FT(0.02),
    σ_abs = FT(1e-7),
    margin = FT(1),
    abs_margin = FT(0),
    sharpness = FT(1),
    residual = FT(0),
) where {FT} = (; ε_rel, σ_abs, margin, abs_margin, sharpness, residual)

@testset "Cloud Fraction" begin

    @testset "`_compute_cloud_fraction`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                α = FT(1)
                floor = floor_nt(FT)
                # Subsaturated mean: μ_S ≤ 0 ⇒ D = 1, floor fully active
                # (constant-floor behavior).
                μ_sub = FT(-1e-3)

                @testset "No condensate → zero cloud fraction" begin
                    cf = CA._compute_cloud_fraction(
                        FT(0), μ_sub, FT(1e-4), FT(5e-5), α, floor,
                    )
                    @test cf < eps(FT)
                end

                @testset "Condensate present, nonzero sigma → cf > 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(3.16e-4), FT(5e-5), α, floor,
                    )
                    @test FT(0) < cf <= FT(1)
                end

                @testset "Zero sigma, large condensate → cf ≈ 1" begin
                    # σ_S = 0 ⇒ σ_aug = σ_S_floor ≈ √((ε_rel·q_sat)² + σ_abs²)
                    # ≈ 0.02·5e-5 = 1e-6; C = 1e-3/1e-6 ≫ 1 → CF ≈ 1.
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(0), FT(5e-5), α, floor,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Large condensate, small sigma → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-2), μ_sub, FT(1e-5), FT(5e-5), α, floor,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Type stability" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(3.16e-4), FT(5e-5), α, floor,
                    )
                    @test cf isa FT
                end

                @testset "CF monotone in σ_S at fixed q_c" begin
                    cfs = [
                        CA._compute_cloud_fraction(
                            FT(1e-3), μ_sub, σ, FT(5e-5), α, floor,
                        ) for σ in FT[1e-4, 3.16e-4, 1e-3, 3.16e-3]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Large σ_S, small q_c → cf approaching 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-6), μ_sub, FT(3.16e-2), FT(5e-5), α, floor,
                    )
                    @test cf < FT(0.01)
                end

                @testset "Tiny q_c with tiny σ_S → cf stays bounded (smooth floor)" begin
                    # σ_aug ≈ σ_S_floor ≈ 1e-6; C = q_c/σ_aug small ⇒ CF small.
                    cf = CA._compute_cloud_fraction(
                        FT(1e-9), μ_sub, FT(1e-12), FT(5e-5), α, floor,
                    )
                    @test cf < FT(0.51)
                end

                # Stratocumulus-like reference state for the release tests:
                # q_c = 0.25 g/kg ≪ ε_rel·q_sat with ε_rel = 0.15,
                # q_sat = 10 g/kg (floor = 1.5 g/kg), quiescent σ_S = 1e-5.
                q_c = FT(2.5e-4)
                q_sat = FT(1e-2)
                σ_S = FT(1e-5)
                floor_sc = floor_nt(FT; ε_rel = FT(0.15))

                @testset "Overcast limit: saturated margin releases the floor → cf ≈ 1" begin
                    # The unreleased floor caps CF well below 1; with the mean
                    # saturated by many equilibrium widths (μ_S = q_c ⇒
                    # x = 25) the floor is released and CF → 1.
                    cf_sat = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α, floor_sc,
                    )
                    @test cf_sat > FT(0.99)
                    cf_floored = CA._compute_cloud_fraction(
                        q_c, FT(-1e-1), σ_S, q_sat, α, floor_sc,
                    )
                    @test cf_floored < FT(0.6)
                end

                @testset "CF monotone in μ_S at fixed q_c (release grows with margin)" begin
                    cfs = [
                        CA._compute_cloud_fraction(
                            q_c, μ, σ_S, q_sat, α, floor_sc,
                        ) for μ in FT[4e-3, 1e-3, 2.5e-4, 5e-5, 0, -1e-3]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "μ_S ≤ 0 keeps the constant floor for any shape parameters" begin
                    cf_ref = CA._compute_cloud_fraction(
                        q_c, μ_sub, σ_S, q_sat, α, floor_sc,
                    )
                    for shaped in (
                        floor_nt(FT; ε_rel = FT(0.15), margin = FT(10)),
                        floor_nt(FT; ε_rel = FT(0.15), abs_margin = FT(5)),
                        floor_nt(FT; ε_rel = FT(0.15), sharpness = FT(4)),
                        floor_nt(FT; ε_rel = FT(0.15), residual = FT(0.5)),
                    )
                        cf = CA._compute_cloud_fraction(
                            q_c, μ_sub, σ_S, q_sat, α, shaped,
                        )
                        @test cf ≈ cf_ref rtol = sqrt(eps(FT))
                    end
                end

                @testset "Release continuous across μ_S = 0" begin
                    cf0 = CA._compute_cloud_fraction(
                        q_c, FT(0), σ_S, q_sat, α, floor_sc,
                    )
                    cf⁻ = CA._compute_cloud_fraction(
                        q_c, -eps(FT), σ_S, q_sat, α, floor_sc,
                    )
                    cf⁺ = CA._compute_cloud_fraction(
                        q_c, FT(1e-8), σ_S, q_sat, α, floor_sc,
                    )
                    @test abs(cf0 - cf⁻) < FT(1e-4)
                    @test abs(cf0 - cf⁺) < FT(1e-3)
                end

                @testset "margin delays the release" begin
                    cf_default = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α, floor_sc,
                    )
                    cf_wide = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α,
                        floor_nt(FT; ε_rel = FT(0.15), margin = FT(50)),
                    )
                    @test cf_wide < cf_default
                    @test cf_wide < FT(0.6)  # release suppressed → floored CF
                end

                @testset "abs_margin guards against tiny-σ_S release" begin
                    cf_guarded = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α,
                        floor_nt(FT; ε_rel = FT(0.15), abs_margin = FT(1)),
                    )
                    # μ_S = q_c ≪ ε_rel·q_sat: absolute margin not met.
                    @test cf_guarded < FT(0.6)
                end

                @testset "sharpness steepens the transition" begin
                    # x = 2 (μ_S = 2·σ_S): sharper release ⇒ smaller D ⇒
                    # larger CF.
                    μ2 = 2 * σ_S
                    cf_s1 = CA._compute_cloud_fraction(
                        q_c, μ2, σ_S, q_sat, α, floor_sc,
                    )
                    cf_s4 = CA._compute_cloud_fraction(
                        q_c, μ2, σ_S, q_sat, α,
                        floor_nt(FT; ε_rel = FT(0.15), sharpness = FT(4)),
                    )
                    @test cf_s4 > cf_s1
                end

                @testset "residual = 1 recovers the constant floor" begin
                    cf_res = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α,
                        floor_nt(FT; ε_rel = FT(0.15), residual = FT(1)),
                    )
                    cf_const = CA._compute_cloud_fraction(
                        q_c, μ_sub, σ_S, q_sat, α, floor_sc,
                    )
                    @test cf_res ≈ cf_const rtol = sqrt(eps(FT))
                    # Partial residual lies between full release and full floor.
                    cf_half = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α,
                        floor_nt(FT; ε_rel = FT(0.15), residual = FT(0.5)),
                    )
                    cf_full = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, q_sat, α, floor_sc,
                    )
                    @test cf_const < cf_half < cf_full
                end

                @testset "ε_rel = 0 → guard keeps result finite" begin
                    for μ in FT[0, -1e-3]
                        cf = CA._compute_cloud_fraction(
                            FT(1e-3), μ, FT(3.16e-4), FT(5e-5), α,
                            floor_nt(FT; ε_rel = FT(0)),
                        )
                        @test isfinite(cf)
                        @test FT(0) <= cf <= FT(1)
                    end
                end
            end
        end
    end

end
