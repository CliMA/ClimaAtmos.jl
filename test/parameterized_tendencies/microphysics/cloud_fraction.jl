#=
Unit tests for cloud_fraction.jl

Tests cover:
1. `_compute_cloud_fraction(q_c, mu_S, sigma_S, α, floor)` -
   truncated-Gaussian CF closure with the augmented floor
   `σ_aug = max(sqrt((α · σ_S)² + σ_abs²), ϵ_numerics(FT))`, which keeps
   subsaturated air with trace condensate patchy while vanishing smoothly
   (`C¹` across `μ_S = 0`) in saturated cloud decks (`μ_S ≥ 0`).
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces
import ClimaCore.CommonSpaces
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

floor_nt(::Type{FT}; σ_abs = FT(1e-7)) where {FT} =
    CA.CloudFractionFloorParams(; σ_abs)

@testset "Cloud Fraction" begin
    @testset "`_compute_cloud_fraction`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                α = FT(1)
                floor = floor_nt(FT)
                μ_sub = FT(-1e-3)

                @testset "No condensate → zero cloud fraction" begin
                    cf = CA._compute_cloud_fraction(
                        FT(0), μ_sub, FT(1e-4), α, floor,
                    )
                    @test cf < eps(FT)
                end

                @testset "Condensate present, nonzero sigma → cf > 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(3.16e-4), α, floor,
                    )
                    @test FT(0) < cf <= FT(1)
                end

                @testset "Zero sigma, large condensate → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-2), μ_sub, FT(0), α, floor,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Non-equilibrium excess normalized by subsaturation deficit" begin
                    # When σ_S = 0 and μ_S = -1e-3, q_c_eq = 0 so q_c = 1e-3 is
                    # purely non-equilibrium condensate normalized by
                    # σ_neq = √(σ_aug² + δ_sub²) ≈ 1e-3 (C = 1, cf = Φ(z(1)) ≈ 0.816),
                    # whereas at saturation (μ_S = 0) δ_sub = 0 and cf > 0.99.
                    cf_neq = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(0), α, floor,
                    )
                    cf_eq = CA._compute_cloud_fraction(
                        FT(1e-3), FT(0), FT(0), α, floor,
                    )
                    @test FT(0.7) < cf_neq < FT(0.9)
                    @test cf_eq > FT(0.99)
                end

                @testset "Large condensate, small sigma → cf ≈ 1" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-2), μ_sub, FT(1e-5), α, floor,
                    )
                    @test cf > FT(0.99)
                end

                @testset "Type stability" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-3), μ_sub, FT(3.16e-4), α, floor,
                    )
                    @test cf isa FT
                end

                @testset "CF monotone in σ_S at fixed q_c" begin
                    cfs = [
                        CA._compute_cloud_fraction(
                            FT(1e-3), μ_sub, σ, α, floor,
                        ) for σ in FT[1e-4, 3.16e-4, 1e-3, 3.16e-3]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Large σ_S, small q_c → cf approaching 0" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-6), μ_sub, FT(3.16e-2), α, floor,
                    )
                    @test cf < FT(0.01)
                end

                @testset "Tiny q_c with tiny σ_S → cf stays bounded (subsaturation floor)" begin
                    cf = CA._compute_cloud_fraction(
                        FT(1e-9), μ_sub, FT(1e-12), α, floor,
                    )
                    @test cf < FT(1e-3)
                end

                # Stratocumulus vs. subsaturated cumulus regime:
                # q_c = 0.25 g/kg, quiescent σ_S = 1e-5.
                q_c = FT(2.5e-4)
                σ_S = FT(1e-5)
                floor_sc = floor_nt(FT)

                @testset "Overcast vs. subsaturated limit" begin
                    # Saturated deck (μ_S ≥ 0): δ_sub = 0, so CF → 1.
                    cf_sat = CA._compute_cloud_fraction(
                        q_c, q_c, σ_S, α, floor_sc,
                    )
                    @test cf_sat > FT(0.99)
                    # Subsaturated air (μ_S = -1e-2): δ_sub = 1e-2, CF < 0.6.
                    cf_floored = CA._compute_cloud_fraction(
                        q_c, FT(-1e-2), σ_S, α, floor_sc,
                    )
                    @test cf_floored < FT(0.6)
                end

                @testset "CF monotone in μ_S at fixed q_c" begin
                    cfs = [
                        CA._compute_cloud_fraction(
                            q_c, μ, σ_S, α, floor_sc,
                        ) for μ in FT[4e-3, 1e-3, 2.5e-4, 0, -5e-4, -2e-3, -1e-2]
                    ]
                    for i in 2:length(cfs)
                        @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                    end
                end

                @testset "Saturated regime (μ_S ≥ 0) is independent of subsaturation deficit" begin
                    cf_ref = CA._compute_cloud_fraction(
                        q_c, FT(0), σ_S, α, floor_sc,
                    )
                    for μ_pos in FT[0, 1e-4, 2.5e-4, 1e-3]
                        cf = CA._compute_cloud_fraction(
                            q_c, μ_pos, σ_S, α, floor_sc,
                        )
                        @test cf ≈ cf_ref rtol = sqrt(eps(FT))
                    end
                end

                @testset "Smooth C¹ transition across μ_S = 0" begin
                    cf0 = CA._compute_cloud_fraction(
                        q_c, FT(0), σ_S, α, floor_sc,
                    )
                    cf⁻ = CA._compute_cloud_fraction(
                        q_c, -FT(1e-6), σ_S, α, floor_sc,
                    )
                    cf⁺ = CA._compute_cloud_fraction(
                        q_c, FT(1e-6), σ_S, α, floor_sc,
                    )
                    @test abs(cf0 - cf⁻) < FT(1e-4)
                    @test cf0 == cf⁺
                end

                @testset "Subsaturation floor guard keeps result finite" begin
                    for μ in FT[0, -1e-3]
                        cf = CA._compute_cloud_fraction(
                            FT(1e-3), μ, FT(3.16e-4), α,
                            floor_sc,
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

                # buoyancy_bounded_mixing_length bounds l_mix smoothly by
                # √(κ_b / N²) when N² > 0 and κ_b > 0, and is continuous down
                # to N² = 0 (including across N² = eps(Float32)).
                bml = CA.buoyancy_bounded_mixing_length
                @test bml(FT(200), FT(0.04), FT(1e-4)) ≈
                      FT(200) / sqrt(FT(101))
                @test bml(FT(20), FT(0.04), FT(1e-4)) ≈ FT(20) / sqrt(FT(2))
                @test bml(FT(2000), FT(0.04), FT(1e-4)) < FT(20)
                @test bml(FT(10), FT(0.04), FT(1e-4)) ≈
                      FT(10) / sqrt(FT(1.25))
                @test bml(FT(200), FT(0), FT(1e-4)) == FT(200)
                @test bml(FT(200), FT(0.04), FT(0)) == FT(200)
                @test bml(FT(200), FT(0.04), FT(-1e-4)) == FT(200)
                @test bml(FT(500), FT(4e-3), eps(FT)) ≈
                      FT(500) / sqrt(one(FT) + eps(FT) * FT(500)^2 / FT(4e-3))
                @test bml(FT(200), FT(1), FT(1e-8)) ≈ FT(200) rtol = FT(1e-3)
                @test bml(FT(200), FT(0.04), FT(1e-4)) isa FT

                # With κ_b = (c_b / (2 c_σ)) tke:
                # 1) in neutral turbulence (N² ≤ 0), 2 c_σ l_d² = 2 c_σ l_mix²
                #    so diagnostic_covariance_coeff (c_σ) remains active;
                # 2) in a strong capping inversion (2 c_σ N² l_mix² ≫ c_b tke),
                #    2 c_σ l_disp² → c_b tke / N² (the adiabatic excursion).
                tke = FT(0.5)
                l_mix = FT(300)
                κ_b_turb = (c_b / (2 * c_σ)) * tke
                l_disp_neutral = bml(l_mix, κ_b_turb, FT(0))
                l_d_neutral = CA.displacement_length(
                    l_disp_neutral,
                    l(c_b * κ_min, FT(0), FT(5000)),
                    FT(5000),
                )
                @test 2 * c_σ * l_d_neutral^2 ≈ 2 * c_σ * l_mix^2
                N²_inv = FT(1e-2)
                l_disp_inv = bml(l_mix, κ_b_turb, N²_inv)
                @test 2 * c_σ * l_disp_inv^2 ≈ c_b * tke / N²_inv rtol = FT(1e-2)

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

                # In the buoyancy-dominated limit l_mix ≫ √(κ_b / N²),
                # turbulent_lagrangian_time(l_disp, κ_b) approaches 1 / N.
                κ_b_test = FT(0.04)
                N²_test = FT(1e-4)
                l_buoy_test = bml(FT(200), κ_b_test, N²_test)
                @test τ(l_buoy_test, κ_b_test) ≈
                      inv(sqrt(N²_test)) / sqrt(FT(1.01))
                @test τ(bml(FT(2000), κ_b_test, N²_test), κ_b_test) ≈
                      inv(sqrt(N²_test)) rtol = FT(1e-4)
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

    @testset "`_updraft_cloud_fraction`" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                config = CA.AtmosConfig(Dict("FLOAT_TYPE" => string(FT)))
                params = CA.ClimaAtmosParameters(config)
                thermo_params = CAP.thermodynamics_params(params)
                σ_abs = CAP.cloud_fraction_sigma_abs(params)
                ρ = FT(1.1)
                T = FT(295)
                q_sat = TD.q_vap_saturation(thermo_params, T, ρ)

                # Sub-cloud subsaturated updraft with trace entrained
                # condensate: 0M uses binary has_condensate, 1M uses smooth
                # subsaturation ratio.
                q_c_trace = FT(5e-7)
                q_tot_sub = FT(0.85) * q_sat + q_c_trace
                cf_0m_sub = CA._updraft_cloud_fraction(
                    thermo_params,
                    T,
                    ρ,
                    q_tot_sub,
                    q_c_trace,
                    FT(0),
                    q_c_trace,
                    FT(0),
                    σ_abs,
                    CA.EquilibriumMicrophysics0M(),
                )
                cf_1m_sub = CA._updraft_cloud_fraction(
                    thermo_params,
                    T,
                    ρ,
                    q_tot_sub,
                    q_c_trace,
                    FT(0),
                    q_c_trace,
                    FT(0),
                    σ_abs,
                    CA.NonEquilibriumMicrophysics1M(),
                )
                @test cf_0m_sub == FT(1)
                @test cf_1m_sub < FT(1e-3)

                # Saturated in-cloud updraft: both 0M and 1M give CF ≈ 1.
                q_c_cloud = FT(3e-4)
                q_tot_sat = q_sat + q_c_cloud
                cf_1m_sat = CA._updraft_cloud_fraction(
                    thermo_params,
                    T,
                    ρ,
                    q_tot_sat,
                    q_c_cloud,
                    FT(0),
                    q_c_cloud,
                    FT(0),
                    σ_abs,
                    CA.NonEquilibriumMicrophysics1M(),
                )
                @test cf_1m_sat > FT(0.999)

                # Zero condensate: both give CF == 0.
                @test CA._updraft_cloud_fraction(
                    thermo_params,
                    T,
                    ρ,
                    q_sat,
                    FT(0),
                    FT(0),
                    FT(0),
                    FT(0),
                    σ_abs,
                    CA.NonEquilibriumMicrophysics1M(),
                ) == FT(0)
            end
        end
    end

    @testset "Subsaturated environment tail truncation and detrained fraction" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                config = CA.AtmosConfig(Dict("FLOAT_TYPE" => string(FT)))
                params = CA.ClimaAtmosParameters(config)
                thermo_params = CAP.thermodynamics_params(params)

                # Upper-tail truncation c_hi < c reduces equilibrium condensate
                # monotonically and vanishes when mu_S <= -s * c_hi * sigma_S.
                mu_S = FT(-3e-4)
                sigma_S = FT(2e-4)
                c = FT(3)
                q_sym = CA.truncated_displacement_condensate(mu_S, sigma_S, c)
                @test CA.truncated_displacement_condensate(mu_S, sigma_S, c, c) ==
                      q_sym
                q_t2 = CA.truncated_displacement_condensate(
                    mu_S,
                    sigma_S,
                    c,
                    FT(2.5),
                )
                q_t1 = CA.truncated_displacement_condensate(
                    mu_S,
                    sigma_S,
                    c,
                    FT(1.8),
                )
                @test q_sym > q_t2 > q_t1 >= FT(0)
                @test CA.truncated_displacement_condensate(
                    mu_S,
                    sigma_S,
                    c,
                    FT(1.2),
                ) == FT(0)

                # Subsaturated cumulus switch vanishes at saturation and across
                # strong inversions, and approaches 1 in warm subsaturated air.
                ρ = FT(1.1)
                T = FT(295)
                q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
                rh_trans = FT(0.03)
                gamma_crit = FT(0.0115)
                @test CA._env_subsat_cu_switch(
                    thermo_params,
                    T,
                    ρ,
                    q_sat,
                    rh_trans,
                    FT(0.002),
                    gamma_crit,
                ) == FT(0)
                w_cu_sub = CA._env_subsat_cu_switch(
                    thermo_params,
                    T,
                    ρ,
                    FT(0.85) * q_sat,
                    rh_trans,
                    FT(0.002),
                    gamma_crit,
                )
                w_cu_inv = CA._env_subsat_cu_switch(
                    thermo_params,
                    T,
                    ρ,
                    FT(0.85) * q_sat,
                    rh_trans,
                    FT(0.05),
                    gamma_crit,
                )
                @test w_cu_sub > FT(0.95)
                @test w_cu_inv < FT(5e-3)

                c_hi_0 = CA._env_tail_c_hi_from_sums(
                    FT(0.01),
                    FT(0),
                    c,
                    FT(4),
                )
                c_hi_cu = CA._env_tail_c_hi_from_sums(
                    FT(0.01),
                    FT(1),
                    c,
                    FT(4),
                )
                @test c_hi_0 == c
                @test c_hi_cu < c
                @test c_hi_cu ≈ CA.normal_cdf_inv(FT(0.96)) rtol = FT(1e-3)

                # Detrained condensate fraction is 0 at or below equilibrium and
                # (q_c - q_c_eq) / q_c above equilibrium.
                q_eq = CA.truncated_gaussian_condensate(mu_S, sigma_S)
                @test CA.env_detrained_condensate_fraction(
                    FT(0),
                    FT(0),
                    sigma_S,
                    mu_S,
                ) == FT(0)
                @test CA.env_detrained_condensate_fraction(
                    FT(0.5) * q_eq,
                    FT(0),
                    sigma_S,
                    mu_S,
                ) == FT(0)
                @test CA.env_detrained_condensate_fraction(
                    FT(2) * q_eq,
                    FT(0),
                    sigma_S,
                    mu_S,
                ) ≈ FT(0.5) rtol = FT(1e-5)
                @test CA.env_detrained_condensate_fraction(
                    FT(1e-4),
                    FT(0),
                    sigma_S,
                    FT(-Inf),
                ) == FT(1)
            end
        end
    end

    @testset "`compute_effective_env_displacement_length` (A2 and A2′)" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                l_d = FT(150)
                l_iso = FT(50)
                c_σ = FT(0.5)
                r_v_θ = FT(0.38)
                ∂q∂z = FT(-3e-3)
                ∂θ∂z = FT(4e-3)
                ρ = FT(1.15)
                q_tot_0 = FT(14e-3)
                θ_li_0 = FT(300)

                # 1. Identity recovery at a_up = 0 (ρa_up = 0)
                l_d_env_a2_0 = CA.compute_effective_env_displacement_length(
                    l_d, l_iso, c_σ, r_v_θ, ∂q∂z, ∂θ∂z, ρ, FT(0),
                    FT(16e-3), FT(302), q_tot_0, θ_li_0, false,
                )
                l_d_env_a2p_0 = CA.compute_effective_env_displacement_length(
                    l_d, l_iso, c_σ, r_v_θ, ∂q∂z, ∂θ∂z, ρ, FT(0),
                    FT(16e-3), FT(302), q_tot_0, θ_li_0, true,
                )
                @test l_d_env_a2_0 == l_d
                @test l_d_env_a2p_0 == l_d

                # 2. Updraft presence attenuates displacement length
                ρa_up = FT(0.1 * 1.15)
                q_tot_up = FT(16e-3)
                θ_li_up = FT(301.5)

                l_d_env_a2 = CA.compute_effective_env_displacement_length(
                    l_d, l_iso, c_σ, r_v_θ, ∂q∂z, ∂θ∂z, ρ, ρa_up,
                    q_tot_up, θ_li_up, q_tot_0, θ_li_0, false,
                )
                l_d_env_a2p = CA.compute_effective_env_displacement_length(
                    l_d, l_iso, c_σ, r_v_θ, ∂q∂z, ∂θ∂z, ρ, ρa_up,
                    q_tot_up, θ_li_up, q_tot_0, θ_li_0, true,
                )

                # Both respect Loewner order: l_d_env <= l_d
                @test l_d_env_a2 < l_d
                @test l_d_env_a2p < l_d
                @test l_d_env_a2 >= l_iso
                # A2' attenuates the full vertical displacement, so l_d_env_a2p < l_d_env_a2
                @test l_d_env_a2p < l_d_env_a2
                @test l_d_env_a2p > FT(0)

                # 3. Exact formula verification for A2' (full_attenuation = true):
                # l_d_env^2 = l_d^2 / (1 + ξ_v)
                w1 = ρa_up / ρ
                w0 = one(FT) - w1
                Σ_qq = w0 * w1 * (q_tot_up - q_tot_0)^2
                Σ_θθ = (r_v_θ)^2 * w0 * w1 * (θ_li_up - θ_li_0)^2
                denom_q = 2 * c_σ * l_d^2 * (∂q∂z)^2
                denom_θ = 2 * c_σ * (r_v_θ)^2 * l_d^2 * (∂θ∂z)^2
                ξ_v_expected = max(Σ_qq / denom_q, Σ_θθ / denom_θ)
                l_d_env_a2p_expected = l_d / sqrt(one(FT) + ξ_v_expected)
                @test l_d_env_a2p ≈ l_d_env_a2p_expected rtol = FT(1e-5)
            end
        end

        @testset "SGS variance fidelity mode 2 (equilibrium-modulated, unclamped)" begin
            for FT in (Float32, Float64)
                A = FT(1.2)
                τ_mp = FT(100)
                c_d = FT(0.1)
                tke = FT(0.5)
                l_mix = FT(50)
                w_eq = FT(0.5)
                q_min = FT(1e-7)
                mu_S = FT(-0.001)
                sigma_S = FT(0.002)

                inv_τ_turb = c_d * sqrt(tke) / l_mix
                Da = A * τ_mp * inv_τ_turb
                α_Da = 1 / (1 + Da)
                q_c_eq = CA.equilibrium_condensate(mu_S, sigma_S, FT(0), FT(0), FT(Inf32))

                # Case A: moderate condensate matches exact α_Da * α_eq
                q_c_actual = FT(1e-4)
                r_A = (q_c_actual + q_min) / (q_c_eq + q_min)
                α_eq_A = r_A / (w_eq + (1 - w_eq) * r_A)
                α_A = CA.sgs_variance_fidelity(
                    A, τ_mp, c_d, tke, l_mix, w_eq, q_c_actual, mu_S, sigma_S, q_min,
                )
                @test α_A ≈ α_Da * α_eq_A rtol = FT(1e-6)

                # Case B: trace condensate (q_c << q_c_eq) showing α can drop below 0.05 (unclamped)
                q_c_trace = FT(1e-8)
                r_B = (q_c_trace + q_min) / (q_c_eq + q_min)
                α_eq_B = r_B / (w_eq + (1 - w_eq) * r_B)
                α_B = CA.sgs_variance_fidelity(
                    A, τ_mp, c_d, tke, l_mix, w_eq, q_c_trace, mu_S, sigma_S, q_min,
                )
                @test α_B ≈ α_Da * α_eq_B rtol = FT(1e-6)
                @test α_B < FT(0.05)
            end
        end
    end

end
