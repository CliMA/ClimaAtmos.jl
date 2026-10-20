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
import ClimaCore: Fields, Spaces
import ClimaCore.CommonSpaces
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

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

    @testset "`displacement_covariance`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                cov = CA.displacement_covariance
                ℓ_h², ℓ_v² = FT(1e9), FT(1e4)
                a, b = FT(1e-8), FT(2e-5)     # |∇_h q|, |∇_h θ|
                q_z, θ_z = FT(-1e-6), FT(4e-3)
                @test cov(ℓ_h², ℓ_v², a^2, q_z, q_z) ≈ ℓ_h² * a^2 + ℓ_v² * q_z^2
                @test cov(ℓ_h², ℓ_v², a * b, θ_z, q_z) ≈
                      ℓ_h² * a * b + ℓ_v² * θ_z * q_z
                # Cauchy–Schwarz: one displacement for both scalars
                @test cov(ℓ_h², ℓ_v², a * b, θ_z, q_z)^2 <=
                      cov(ℓ_h², ℓ_v², a^2, q_z, q_z) *
                      cov(ℓ_h², ℓ_v², b^2, θ_z, θ_z) * (1 + 10 * eps(FT))
                @test cov(FT(0), ℓ_v², a^2, q_z, q_z) ≈ ℓ_v² * q_z^2
                @test cov(ℓ_h², ℓ_v², a^2, q_z, q_z) isa FT
            end
        end
    end

    @testset "`floor_velocity`: resolution dependence of the floor" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                w = CA.floor_velocity
                w_ref, Δx_ref, p = FT(0.6), FT(1.2e5), FT(2 // 3)
                @test w(Δx_ref, w_ref, Δx_ref, p) == w_ref
                # 60 / 120 / 240 km: 0.38 / 0.6 / 0.95 m/s
                @test w(FT(6e4), w_ref, Δx_ref, p) ≈ FT(0.378) rtol = FT(1e-2)
                @test w(FT(2.4e5), w_ref, Δx_ref, p) ≈ FT(0.952) rtol = FT(1e-2)
                # a −5/3 spectrum: p = 1/3
                @test w(FT(2.4e5), w_ref, Δx_ref, FT(1 // 3)) ≈ w_ref * 2^(1 / 3)
                # a single column has no grid scale and gets the reference value
                @test w(FT(Inf), w_ref, Δx_ref, p) == w_ref
                # the LES limit vanishes
                @test w(FT(100), w_ref, Δx_ref, p) < FT(0.01)
                @test w(FT(6e4), w_ref, Δx_ref, p) isa FT
            end
        end
    end

    @testset "`floor_displacement_length` and `displacement_length`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                l0 = CA.floor_displacement_length
                τ_0 = FT(100)
                Δz = FT(500)
                l(κ_b, N², l_z) = l0(κ_b, N², FT(Inf), τ_0, l_z, Δz)
                κ_b = FT(0.0857)                   # c_b κ_min for w_min = 0.6 m/s
                l_z = FT(5000)
                # stable: √κ_b / N, below both caps
                @test l(κ_b, FT(1e-4), l_z) ≈ sqrt(κ_b) / FT(1e-2)
                # weakly stable: 1/N exceeds τ_0, so the quiescent limit binds
                @test l(κ_b, FT(1e-6), l_z) ≈ sqrt(κ_b) * τ_0
                # neutral and unstable: buoyancy imposes no limit, the quiescent
                # (or turbulent) limit binds, and the floor is continuous at N² = 0
                @test l(κ_b, FT(0), l_z) ≈ sqrt(κ_b) * τ_0
                @test l(κ_b, FT(-1e-4), l_z) ≈ sqrt(κ_b) * τ_0
                @test l(κ_b, FT(1e-12), l_z) ≈ l(κ_b, FT(0), l_z)
                @test l0(κ_b, FT(-1e-4), FT(40), τ_0, l_z, Δz) ≈ sqrt(κ_b) * 40
                @test l0(κ_b, FT(0), FT(40), τ_0, l_z, Δz) ≈
                      l0(κ_b, FT(1e-12), FT(40), τ_0, l_z, Δz)
                # without TKE and with no quiescent limit the caps bind
                @test l0(κ_b, FT(-1e-4), FT(Inf), FT(Inf), l_z, Δz) == Δz
                # turbulence decorrelates sooner than τ_0 or buoyancy in stable air
                @test l0(κ_b, FT(1e-6), FT(40), τ_0, l_z, Δz) ≈ sqrt(κ_b) * 40
                @test l0(κ_b, FT(1e-4), FT(40), τ_0, l_z, Δz) ≈ sqrt(κ_b) * 40
                @test l0(κ_b, FT(1e-4), FT(400), τ_0, l_z, Δz) ≈
                      sqrt(κ_b) / FT(1e-2)
                # caps: cell depth and wall distance in weakly stable air
                @test l0(κ_b, FT(1e-8), FT(Inf), FT(1e4), l_z, Δz) == Δz
                @test l(κ_b, FT(1e-6), FT(10)) == FT(10)
                @test l(κ_b, FT(1e-4), FT(-1)) == FT(0)
                @test l(κ_b, FT(1e-4), l_z) isa FT
                # turbulent Lagrangian time: l_mix / √κ, infinite without TKE
                τ = CA.turbulent_lagrangian_time
                @test τ(FT(200), FT(1)) == FT(200)
                @test τ(FT(0), FT(0)) == FT(Inf)
                @test τ(FT(200), FT(0)) == FT(Inf)
                @test τ(FT(200), FT(1)) isa FT

                params = CA.ClimaAtmosParameters(FT)
                turbconv_params = CAP.turbconv_params(params)
                sf_params = CAP.surface_fluxes_params(params)
                vkc = CAP.von_karman_const(params)
                c_b = FT(CAP.static_stab_coeff(turbconv_params))
                c_σ = FT(CAP.diagnostic_covariance_coeff(params))
                sq = params.sgs_quadrature_params
                w_min = CA.floor_velocity(
                    FT(Inf),
                    FT(CAP.sgs_variance_vertical_velocity_floor(sq)),
                    FT(CAP.sgs_variance_floor_reference_resolution(sq)),
                    FT(CAP.sgs_variance_floor_resolution_exponent(sq)),
                )
                κ_min = w_min^2 / (2 * c_σ * c_b)
                N² = FT(1e-4)
                l_b = l(c_b * κ_min, N², FT(5000))
                # stably stratified, no resolved TKE: 2 c_σ l_b² N² = w_min²
                @test 2 * c_σ * l_b^2 * N² ≈ w_min^2 rtol = FT(1e-5)

                # Even when ustar = 0 or the surface layer is strongly stable
                # (where the MOST wall scale l_W collapses ml.master → 0),
                # taking max(l_mix, l_b) preserves 2 c_σ l_d² N² = w_min².
                for (ustar, L_MO) in ((FT(0), FT(100)), (FT(0.05), FT(10)))
                    ml = CA.mixing_length_lopez_gomez_2020(
                        turbconv_params, sf_params, vkc,
                        ustar, FT(5000), FT(0), FT(Inf), FT(0), N², FT(0),
                        L_MO, FT(1e-10), FT(1), CA.SmoothMinimumBlending(),
                    )
                    l_d = CA.displacement_length(ml.master, l_b, FT(5000))
                    @test 2 * c_σ * l_d^2 * N² ≈ w_min^2 rtol = FT(1e-5)
                end
                # the combination takes max(l_mix, l_b) and never exceeds the wall distance
                @test CA.displacement_length(FT(50), FT(50), FT(50)) == FT(50)
                @test CA.displacement_length(FT(30), FT(40), FT(5000)) == FT(40)
                @test CA.displacement_length(FT(40), FT(30), FT(5000)) == FT(40)
                @test CA.displacement_length(FT(30), FT(40), FT(-1)) == FT(0)
                @test CA.displacement_length(FT(30), FT(40), FT(5000)) isa FT

                # buoyancy_bounded_mixing_length caps l_mix at √(κ_b / N²) when N² > 0 and κ_b > 0
                bml = CA.buoyancy_bounded_mixing_length
                @test bml(FT(200), FT(0.04), FT(1e-4)) ≈ FT(20)
                @test bml(FT(10), FT(0.04), FT(1e-4)) == FT(10)
                @test bml(FT(200), FT(0), FT(1e-4)) == FT(200)
                @test bml(FT(200), FT(0.04), FT(0)) == FT(200)
                @test bml(FT(200), FT(0.04), FT(-1e-4)) == FT(200)
                @test bml(FT(200), FT(0.04), FT(1e-4)) isa FT

                # The composed 2×2 covariance with per-variable scales and a
                # horizontal cross-correlation is positive semi-definite
                # (|ρ| ≤ 1) and its correlation is invariant under the θ→T
                # Jacobian and the σ_q bound rescaling, as in
                # `set_covariance_cache!`.
                cov = CA.displacement_covariance
                ℓ_h_q² = FT(1e9)
                ℓ_v_q² = FT(1e4)
                r_θ = FT(0.7)
                r_vθ = FT(0.4)
                r_h_θq = FT(0.5)
                ℓ_h_θ² = r_θ^2 * ℓ_h_q²
                ℓ_h_θq = r_h_θq * r_θ * ℓ_h_q²
                ℓ_v_θ² = r_vθ^2 * ℓ_v_q²
                ℓ_v_θq = r_vθ * ℓ_v_q²
                for (a, b, c, q_z, θ_z) in (
                    (FT(1e-8), FT(2e-5), FT(1), FT(-1e-6), FT(4e-3)),
                    (FT(1e-8), FT(2e-5), FT(-1), FT(1e-6), FT(4e-3)),
                    (FT(3e-8), FT(1e-6), FT(0.2), FT(-2e-6), FT(-1e-3)),
                    (FT(0), FT(2e-5), FT(0), FT(-1e-6), FT(0)),
                )
                    # |∇_h q| = a, |∇_h θ| = b, cos(angle) = c
                    Σ_qq = cov(ℓ_h_q², ℓ_v_q², a^2, q_z, q_z)
                    Σ_θθ = cov(ℓ_h_θ², ℓ_v_θ², b^2, θ_z, θ_z)
                    Σ_θq = cov(ℓ_h_θq, ℓ_v_θq, a * b * c, θ_z, q_z)
                    @test Σ_qq >= 0
                    @test Σ_θθ >= 0
                    @test Σ_θq^2 <= Σ_θθ * Σ_qq * (1 + 10 * eps(FT))
                    ∂T_∂θ = FT(0.8)
                    r_scale = FT(0.3)
                    T_mean, q_mean = FT(280), FT(1e-2)
                    ρ_θ = CA.sgs_correlation_Tq(
                        Σ_θθ, Σ_qq, Σ_θq, T_mean, q_mean, FT(0.6),
                    )
                    ρ_T = CA.sgs_correlation_Tq(
                        ∂T_∂θ^2 * Σ_θθ,
                        r_scale^2 * Σ_qq,
                        ∂T_∂θ * r_scale * Σ_θq,
                        T_mean,
                        q_mean,
                        FT(0.6),
                    )
                    @test abs(ρ_θ) <= 1
                    @test ρ_T ≈ ρ_θ rtol = sqrt(eps(FT))
                end

                # The fallback applies only where a variance vanishes relative to
                # its grid mean, not where the variances are small because the air
                # is dry: a dry upper-tropospheric cell (q_tot = 1e-5) with
                # T′T′ q′q′ below eps(Float32)² keeps its diagnosed correlation.
                ρ_ut = FT(-0.3)
                vT_ut, vq_ut = FT(0.05), FT(4e-14)
                c_ut = ρ_ut * sqrt(vT_ut) * sqrt(vq_ut)
                @test vT_ut * vq_ut < eps(Float32)^2
                @test CA.sgs_correlation_Tq(
                    vT_ut, vq_ut, c_ut, FT(220), FT(1e-5), FT(0.6),
                ) ≈ ρ_ut rtol = FT(1e-4)
                # Vanishing variances (exactly zero, or below the round-off floor
                # of the mean) fall back to the prescribed value.
                @test CA.sgs_correlation_Tq(
                    FT(0.05), FT(0), FT(0), FT(220), FT(1e-5), FT(0.6),
                ) == FT(0.6)
                @test CA.sgs_correlation_Tq(
                    FT(0), FT(4e-14), FT(0), FT(220), FT(1e-5), FT(0.6),
                ) == FT(0.6)
                @test CA.sgs_correlation_Tq(
                    FT(0.05), (eps(FT) * FT(1e-5))^2 / 2, FT(0), FT(220),
                    FT(1e-5), FT(0.6),
                ) == FT(0.6)

                # Where l_mix is at the buoyancy scale √(κ_b / N²),
                # turbulent_lagrangian_time(l_mix, κ_b) equals 1 / N.
                κ_b_test = FT(0.04)
                N²_test = FT(1e-4)
                l_buoy_test = bml(FT(200), κ_b_test, N²_test)
                @test τ(l_buoy_test, κ_b_test) ≈ inv(sqrt(N²_test))
            end
        end
    end

    @testset "`set_displacement_buoyancy_gradient!`: dry face-maximum N² across an inversion" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                cspace = CommonSpaces.ColumnSpace(
                    FT;
                    z_min = 0, z_max = 1000, z_elem = 4,
                    staggering = CommonSpaces.CellCenter(),
                )
                fspace = Spaces.face_space(cspace)
                thermo_params = CAP.thermodynamics_params(CA.ClimaAtmosParameters(FT))
                coeffs_val = CA.buoyancy_gradient_coefficients(
                    thermo_params, FT(290), FT(1), FT(0.01), FT(1e-4), FT(0),
                )
                ᶜbg_coeffs = Fields.Field(typeof(coeffs_val), cspace)
                fill!(ᶜbg_coeffs, coeffs_val)
                ᶠ∂θli∂z = Fields.zeros(fspace)
                ᶠ∂qt∂z = Fields.zeros(fspace)
                # Sharp inversion at face 2 + half (between cells 2 and 3):
                # lower faces are neutral (0), face 2+half has a strong jump.
                Fields.level(ᶠ∂θli∂z, 2 + Fields.half) .= FT(0.02)
                Fields.level(ᶠ∂qt∂z, 2 + Fields.half) .= FT(-1e-5)
                ᶜgrad_θ = Fields.zeros(CA.Geometry.Covariant3Vector{FT}, cspace)
                ᶜgrad_q = Fields.zeros(CA.Geometry.Covariant3Vector{FT}, cspace)
                CA.ᶜVanLeer_gradient!(ᶜgrad_θ, ᶠ∂θli∂z)
                CA.ᶜVanLeer_gradient!(ᶜgrad_q, ᶠ∂qt∂z)
                # Van Leer harmonic mean vanishes in cell 2 because its lower face is 0:
                @test Array(parent(Fields.level(ᶜgrad_θ, 2)))[1] == FT(0)
                # Face-maximum dry N² in cell 2 and cell 3 sees the full inversion jump:
                ᶜN²_disp = Fields.zeros(cspace)
                CA.set_displacement_buoyancy_gradient!(
                    ᶜN²_disp, ᶜbg_coeffs, ᶠ∂θli∂z, ᶠ∂qt∂z,
                )
                N²_inv = CA.dry_N²(coeffs_val, FT(0.02), FT(-1e-5))
                @test N²_inv > FT(0)
                @test Array(parent(Fields.level(ᶜN²_disp, 2)))[1] ≈ N²_inv
                @test Array(parent(Fields.level(ᶜN²_disp, 3)))[1] ≈ N²_inv
                @test Array(parent(Fields.level(ᶜN²_disp, 1)))[1] == FT(0)
            end
        end
    end

    @testset "`_precip_fraction_sweep!`: max-random overlap down a column" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                space = CommonSpaces.ColumnSpace(
                    FT;
                    z_min = 0, z_max = 10000, z_elem = 10,
                    staggering = CommonSpaces.CellCenter(),
                )
                nz = Spaces.nlevels(space)
                cf = Fields.zeros(space)
                qp = Fields.zeros(space)
                a_p = Fields.zeros(space)
                setlev!(f, k, v) = (Fields.level(f, k) .= v)
                col(f) = [Array(parent(Fields.level(f, k)))[1] for k in 1:nz]
                q_min = FT(1e-10)
                # cloud at levels 8 (cover 0.4) and 6 (cover 0.2), precipitation
                # from level 8 down to level 3, none at 1-2 and above 8
                for k in 1:nz
                    setlev!(cf, k, k == 8 ? FT(0.4) : k == 6 ? FT(0.2) : FT(0))
                    setlev!(qp, k, 3 <= k <= 8 ? FT(1e-5) : FT(0))
                end
                CA._precip_fraction_sweep!(a_p, cf, qp, FT(1), q_min)
                a = col(a_p)
                @test a[9:10] == [FT(0), FT(0)]            # no precipitation above the cloud
                @test a[8] == FT(0.4)                       # seeded by its own cover
                @test a[7] == FT(0.4)                       # inherited (max overlap)
                @test a[6] == FT(0.4)                       # max(own 0.2, above 0.4)
                @test all(a[3:5] .== FT(0.4))               # below cloud base, inherited
                @test a[1:2] == [FT(0), FT(0)]              # shaft closed: no precipitation
                @test all(col(a_p) .>= col(cf) .* (col(qp) .> q_min))
                # decay shrinks the inherited fraction per level, own cover re-seeds
                CA._precip_fraction_sweep!(a_p, cf, qp, FT(0.5), q_min)
                a = col(a_p)
                @test a[8] == FT(0.4) && a[7] == FT(0.2) && a[6] == FT(0.2)
                @test a[5] == FT(0.1) && a[3] == FT(0.025)
                # a precipitation-free level in the middle resets the recursion
                setlev!(qp, 5, FT(0))
                CA._precip_fraction_sweep!(a_p, cf, qp, FT(1), q_min)
                a = col(a_p)
                @test a[5] == FT(0) && a[4] == FT(0) && a[3] == FT(0)
                @test a[6] == FT(0.4)
                # an overcast layer caps the fraction at 1 and gives 1 below it
                setlev!(cf, 8, FT(1))
                setlev!(qp, 5, FT(1e-5))
                CA._precip_fraction_sweep!(a_p, cf, qp, FT(1), q_min)
                @test all(col(a_p)[3:8] .== FT(1))
            end
        end
    end

end
