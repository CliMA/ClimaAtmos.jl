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

    @testset "`along_surface_q_variance`: displacement on the θ surface" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                var0 = CA.along_surface_q_variance
                ℓ_h = FT(1e5)
                ℓ_v = FT(500)
                q_z = FT(-1e-6)
                θ_z = FT(4e-3)
                a = FT(1e-8)                       # |∇_h q|
                # a large adiabatic q_sat gradient leaves the bound inactive
                var(args...) = var0(args[1:5]..., FT(1), args[6:7]...)
                # no surface at all: the unconstrained box
                @test var(a^2, FT(0), FT(0), q_z, FT(0), ℓ_h, ℓ_v) ≈
                      ℓ_h^2 * a^2 + ℓ_v^2 * q_z^2
                # vertical surface with ∇_h q ∥ ∇_h θ: q changes with the
                # vertical displacement only
                gθ = FT(1e-6)
                @test var(a^2, a * gθ, gθ^2, q_z, FT(0), ℓ_h, ℓ_v) ≈
                      ℓ_v^2 * q_z^2
                # vertical surface with ∇_h q ⊥ ∇_h θ: horizontal gradient kept
                @test var(a^2, FT(0), gθ^2, q_z, FT(0), ℓ_h, ℓ_v) ≈
                      ℓ_h^2 * a^2 + ℓ_v^2 * q_z^2
                # parallel horizontal gradients: the along-surface gradient
                # |∇_h q − r ∇_h θ|², r = q_z / θ_z, times the displacement
                # cap 1 / (1 + (s ℓ_h / ℓ_v)²), s = |∇_h θ| / θ_z
                for s_ in (FT(1e-3), FT(3e-3), FT(1e-2))
                    gθ = s_ * θ_z
                    r = q_z / θ_z
                    x = s_ * ℓ_h / ℓ_v
                    expected = ℓ_h^2 * (a - r * gθ)^2 / (1 + x^2)
                    @test var(a^2, a * gθ, gθ^2, q_z, θ_z, ℓ_h, ℓ_v) ≈ expected rtol =
                        FT(1e-4)
                end
                # bounded by the unconstrained variance, non-negative, finite
                for θz in (FT(-2e-3), FT(0), FT(1e-6), FT(4e-3)),
                    gθ in (FT(0), FT(1e-9), FT(1e-6))

                    v = var(a^2, a * gθ / 2, gθ^2, q_z, θz, ℓ_h, ℓ_v)
                    @test FT(0) <= v <= ℓ_h^2 * a^2 + ℓ_v^2 * q_z^2
                    @test isfinite(v)
                end
                # no horizontal displacement: zero on a tilted surface, the
                # vertical step on a vertical one
                @test var(a^2, FT(0), FT(0), q_z, θ_z, FT(0), ℓ_v) ≈ FT(0) atol =
                    eps(FT) * ℓ_v^2 * q_z^2
                @test var(a^2, FT(0), FT(0), q_z, FT(0), FT(0), ℓ_v) ≈ ℓ_v^2 * q_z^2
                @test var(a^2, FT(0), FT(0), q_z, θ_z, ℓ_h, ℓ_v) isa FT
            end
        end
    end

    @testset "`along_surface_q_variance`: bounded by the q_sat variance" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                var = CA.along_surface_q_variance
                ℓ_h = FT(1e5)
                ℓ_v = FT(500)
                q_z = FT(-1e-6)
                θ_z = FT(4e-3)
                a = FT(1e-8)
                q_sat_z = FT(-1e-6)               # 1.7 g/kg at −6e-4 /m
                # flat surface: no vertical displacement along it, so no
                # saturation excess and the term vanishes
                @test var(a^2, FT(0), FT(0), q_z, θ_z, q_sat_z, ℓ_h, ℓ_v) ===
                      FT(0)
                # vertical surface and no surface: bounded by (q_sat_z ℓ_v)²
                gθ = FT(1e-6)
                @test var(a^2, FT(0), gθ^2, q_z, FT(0), q_sat_z, ℓ_h, ℓ_v) ≈
                      (q_sat_z * ℓ_v)^2
                @test var(a^2, FT(0), FT(0), q_z, FT(0), q_sat_z, ℓ_h, ℓ_v) ≈
                      (q_sat_z * ℓ_v)^2
                # tilted surface: the bound is the along-surface vertical
                # variance ℓ_v² x² / (1 + x²), x = s ℓ_h / ℓ_v, times q_sat_z²
                s_ = FT(3e-3)
                gθ = s_ * θ_z
                x = s_ * ℓ_h / ℓ_v
                bound = q_sat_z^2 * ℓ_v^2 * x^2 / (1 + x^2)
                v = var(a^2, a * gθ, gθ^2, q_z, θ_z, q_sat_z, ℓ_h, ℓ_v)
                @test v ≈ bound
                # the bound is inactive where the q_tot spread is smaller
                @test var(a^2, a * gθ, gθ^2, q_z, θ_z, FT(1), ℓ_h, ℓ_v) ≈
                      var(a^2, a * gθ, gθ^2, q_z, θ_z, FT(10), ℓ_h, ℓ_v)
                @test v isa FT
            end
        end
    end

    @testset "`adiabatic_q_sat_gradient`: q_sat falls along a dry adiabat" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                thermo_params = CAP.thermodynamics_params(CA.ClimaAtmosParameters(FT))
                T = FT(270)
                ρ = FT(1)
                q_tot = FT(5e-3)
                q_sat = TD.q_vap_saturation(thermo_params, T, ρ)
                dq = CA.adiabatic_q_sat_gradient(thermo_params, T, ρ, q_tot, FT(0), FT(0))
                @test dq < FT(0)
                # about −6e-4 q_sat per metre at 270 K
                @test -FT(8e-4) * q_sat < dq < -FT(4e-4) * q_sat
                @test dq isa FT
            end
        end
    end

    @testset "`across_surface_q_gradient`: q change per vertical step" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                qn = CA.across_surface_q_gradient
                q_z = FT(-1e-6)
                θ_z = FT(4e-3)
                a = FT(1e-8)
                gθ = FT(1e-6)
                @test qn(FT(0), FT(0), q_z, θ_z) ≈ q_z            # flat surface
                @test qn(a * gθ, gθ^2, q_z, FT(0)) === FT(0)      # vertical surface
                @test qn(FT(0), FT(0), q_z, FT(0)) === FT(0)      # no surface
                # tilted surface, ∇_h q ⊥ ∇_h θ: same sign as q_z, smaller
                v = qn(FT(0), gθ^2, q_z, θ_z)
                @test abs(v) < abs(q_z)
                @test sign(v) == sign(q_z)
                @test qn(a * gθ, gθ^2, q_z, θ_z) isa FT
            end
        end
    end


    @testset "`along_surface_q_variance` with a constant depth H" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                along0 = CA.along_surface_q_variance
                # a large adiabatic q_sat gradient leaves the bound inactive
                along(args...) = along0(args[1:5]..., FT(1), args[6:7]...)
                ℓ_h = FT(2.5e5)
                Δz = FT(500)
                q_z = FT(-2e-6)
                θ_z = FT(4e-3)
                a = FT(2e-8)   # parallel horizontal gradients of q and θ
                b = FT(1e-5)
                inv_q, inv_θ, inv_θq = a^2, b^2, a * b
                r = q_z / θ_z
                s = b / θ_z
                # with the depth H as ℓ_v the conditional variance is the
                # invariant times the cap H² / (H² + (s ℓ_h)²)
                for H in (FT(200), FT(1000), FT(2000))
                    expected = ℓ_h^2 * (a - r * b)^2 / (1 + (s * ℓ_h / H)^2)
                    @test along(inv_q, inv_θq, inv_θ, q_z, θ_z, ℓ_h, H) ≈ expected rtol =
                        1e-3
                end
                # a deeper displacement samples more of the surface; the
                # 1-km default keeps most of the uncapped value on this slope
                @test along(inv_q, inv_θq, inv_θ, q_z, θ_z, ℓ_h, FT(1000)) >
                      along(inv_q, inv_θq, inv_θ, q_z, θ_z, ℓ_h, Δz)
                @test along(inv_q, inv_θq, inv_θ, q_z, θ_z, ℓ_h, FT(1000)) >
                      FT(0.7) * ℓ_h^2 * (a - r * b)^2
                @test along(inv_q, inv_θq, inv_θ, q_z, θ_z, ℓ_h, FT(1000)) isa FT
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
                setlev!(cf, 8, FT(1));
                setlev!(qp, 5, FT(1e-5))
                CA._precip_fraction_sweep!(a_p, cf, qp, FT(1), q_min)
                @test all(col(a_p)[3:8] .== FT(1))
            end
        end
    end

end
