#=
Checks `displacement_condensate` against brute-force integration of the
quadratic displacement model and its exact limits (zero curvature is the
truncated Gaussian; no horizontal part is `quadratic_condensate`, bounded by
the maximum of the quadratic), that it is type-stable and continuous in the
curvature, and that `_compute_sgs_moments` and
`compute_sgs_saturation_adjustment` use it.
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP

const CA = ClimaAtmos

# Shortening factor `s` of `displacement_condensate`,
# `s² lin² + s⁴ curv²/2 = σ²` where the vertical part alone exceeds `σ²`,
# else 1.
function shortening(σ, lin, curv)
    v_lin, v_curv = lin^2, curv^2 / 2
    u = 2σ^2 / (v_lin + sqrt(v_lin^2 + 4v_curv * σ^2))
    return sqrt(min(1.0, u))
end

# Brute-force reference: midpoint rule in the vertical variable x of the
# exact horizontal expectation, with the same shortening and horizontal width
# as `displacement_condensate`.
function reference_condensate(mu_S, sigma_S, lin, curv; n = 40001)
    σ = max(sigma_S, 1e-10)
    s = curv == 0 ? 1.0 : shortening(σ, lin, curv)
    l = s * lin
    c = s^2 * curv
    σ_h = sqrt(max(σ^2 - l^2 - c^2 / 2, 0.0))
    xs = range(-10, 10; length = n)
    dx = step(xs)
    return sum(
        CA.truncated_gaussian_condensate(mu_S + l * x + c * (x^2 - 1) / 2, σ_h) *
        exp(-x^2 / 2) / sqrt(2π) * dx for x in xs
    )
end

@testset "Curvature-aware displacement condensate" begin

    @testset "displacement_curvature_coefficients" begin
        for FT in (Float32, Float64)
            ℓ_v, χ = FT(300), FT(2e-4)
            ∂q, ∂θ, ∂²q, ∂²θ = FT(-3e-6), FT(4e-3), FT(-5e-9), FT(2e-6)
            (; lin, curv) =
                CA.displacement_curvature_coefficients(ℓ_v, χ, ∂q, ∂θ, ∂²q, ∂²θ)
            @test lin ≈ ℓ_v * (∂q - χ * ∂θ)
            @test curv ≈ ℓ_v^2 * (∂²q - χ * ∂²θ)
            @test lin isa FT && curv isa FT
        end
    end

    @testset "Zero curvature is the truncated Gaussian, bit for bit" begin
        for FT in (Float32, Float64)
            σ = FT(1e-3)
            for zμ in (FT(-2), FT(-0.5), FT(0.5)), fl in (FT(0), FT(0.5), FT(1), FT(3))
                μ = zμ * σ
                @test CA.displacement_condensate(μ, σ, fl * σ, FT(0)) ==
                      CA.truncated_gaussian_condensate(μ, σ)
            end
        end
    end

    @testset "Type stability" begin
        for FT in (Float32, Float64)
            σ = FT(1e-3)
            for fl in (FT(0.8), FT(0.2)), fc in (FT(-0.5), FT(0.5))
                @test (@inferred CA.displacement_condensate(
                    FT(-1.4) * σ, σ, fl * σ, fc * σ,
                )) isa FT
            end
            @test (@inferred CA.quadratic_condensate(-σ, σ, -σ)) isa FT
        end
    end

    @testset "Closed-form vertical model" begin
        for FT in (Float32, Float64)
            σ = FT(1e-3)
            # Without horizontal part (lin² + curv²/2 = σ²), the expectation
            # is the closed form, which the brute-force integral reproduces.
            for zμ in (FT(-2), FT(-1.4), FT(-0.5), FT(0.5)),
                fc in (FT(-1.2), FT(-0.8), FT(-0.3), FT(0.3), FT(0.8), FT(1.2))

                μ = zμ * σ
                c = fc * σ
                l = sqrt(σ^2 - c^2 / 2)
                E = CA.displacement_condensate(μ, σ, l, c)
                ref = reference_condensate(Float64(μ), 1e-3, Float64(l), Float64(c))
                @test E ≈ ref rtol = sqrt(eps(FT)) atol = 1e-3 * σ
                @test E ≈ CA.quadratic_condensate(μ, l, c) rtol = 1e-4 atol = eps(FT)
            end
            # Negative curvature bounds the saturation excess by the maximum
            # of the quadratic.
            for zμ in (FT(-1.4), FT(-0.5), FT(0.5)), fc in (FT(-1.2), FT(-0.8), FT(-0.3))
                μ = zμ * σ
                c = fc * σ
                l = sqrt(σ^2 - c^2 / 2)
                bound = max(FT(0), μ + (l^2 + c^2) / (2 * abs(c)))
                @test CA.displacement_condensate(μ, σ, l, c) <=
                      bound * (1 + 1e-4) + eps(FT)
            end
            # No real roots: one sign everywhere, so the expectation is the
            # mean for positive and zero for negative curvature.
            @test CA.quadratic_condensate(FT(5e-4), FT(0), FT(1e-3)) ≈ FT(5e-4) rtol =
                1e-6
            @test CA.quadratic_condensate(FT(-5e-4), FT(0), FT(-1e-3)) ≈ FT(0) atol =
                1e-8
            # Linear limit of the closed form.
            @test CA.quadratic_condensate(FT(-1e-3), FT(7e-4), FT(0)) ==
                  CA.truncated_gaussian_condensate(FT(-1e-3), FT(7e-4))
        end
    end

    @testset "Mixed vertical and horizontal parts against brute force" begin
        σ = 1e-3
        errors = Float64[]
        for zμ in (-2.0, -1.4, -0.5, 0.5),
            fl in (0.9, 0.8, 0.6, 0.4, 0.2, 0.05),
            fc in (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0)

            μ = zμ * σ
            ref = reference_condensate(μ, σ, fl * σ, fc * σ)
            E = CA.displacement_condensate(μ, σ, fl * σ, fc * σ)
            @test E ≈ ref rtol = 0.3 atol = 1e-3 * σ
            ref > 1e-2 * σ && push!(errors, abs(E - ref) / ref)
            # Float32 agrees with Float64.
            E32 = CA.displacement_condensate(
                Float32(μ), Float32(σ), Float32(fl * σ), Float32(fc * σ),
            )
            @test E32 ≈ E rtol = 1e-3 atol = 1e-4 * σ
        end
        # The error distribution of the Gauss–Hermite correction: most states
        # are resolved to well below a percent, and the tail of the
        # distribution stays within the per-state bound above.
        sort!(errors)
        @test errors[cld(length(errors), 2)] < 1e-2
        @test errors[cld(9 * length(errors), 10)] < 0.1
    end

    @testset "Shortened displacement where the vertical part exceeds σ_S" begin
        for FT in (Float32, Float64)
            σ = FT(1e-3)
            μ = FT(-0.5e-3)
            for (lin, curv) in ((FT(1.5) * σ, FT(-0.5) * σ), (σ, -σ), (FT(0), FT(-2) * σ))
                s = FT(shortening(Float64(σ), Float64(lin), Float64(curv)))
                @test s < 1
                # The shortened pair has no horizontal part, so the result is
                # the closed form at `(s lin, s² curv)`.
                closed_form = CA.quadratic_condensate(μ, s * lin, s^2 * curv)
                @test closed_form > FT(0)
                @test CA.displacement_condensate(μ, σ, lin, curv) ≈ closed_form rtol =
                    1e-4
            end
        end
    end

    @testset "Continuity in the curvature" begin
        for FT in (Float32, Float64)
            σ = FT(1e-3)
            μ = FT(-1.4e-3)
            E0 = CA.truncated_gaussian_condensate(μ, σ)
            for lin in (FT(0.8) * σ, σ), δ in (FT(1e-6) * σ, FT(1e-4) * σ)
                @test abs(CA.displacement_condensate(μ, σ, lin, δ) - E0) <= 50 * δ
                @test abs(CA.displacement_condensate(μ, σ, lin, -δ) - E0) <= 50 * δ
            end
        end
    end

    @testset "1M moments: λ_eq is the linearized mean excess" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            corr, ρ, T = FT(0.6), FT(1.1), FT(288)
            T′T′, q′q′ = FT(0.25), FT(2.5e-7)
            q_c = FT(5e-5)
            quad = CA.SGSQuadrature(FT; quadrature_order = 3)
            q_sat = TD.q_vap_saturation(thp, T, ρ)
            for dq in (FT(-1e-3), FT(-3e-4), FT(2e-4)), fc in (FT(-1), FT(0.5)),
                α in (FT(0.5), FT(1))

                q_tot = q_sat + dq
                m0 = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_c, quad, T′T′, q′q′, corr, α,
                )
                lin = FT(0.8) * m0.sigma_S
                curv = fc * m0.sigma_S
                m = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_c, quad, T′T′, q′q′, corr, α,
                    q_tot, lin, curv,
                )
                @test m.sigma_S == m0.sigma_S
                @test m.λ_lagrange == m0.λ_lagrange
                mu_S = q_tot - q_sat
                q_c_eq = CA.displacement_condensate(mu_S, m.sigma_S, lin, curv)
                # Under Simplification Q1, λ_eq is the linearized mean saturation excess mu_S.
                @test m.λ_eq == mu_S
                # Negative curvature with a subsaturated mean lowers the
                # equilibrium condensate below the Gaussian value.
                fc < 0 && dq < 0 &&
                    @test q_c_eq < CA.truncated_gaussian_condensate(mu_S, m.sigma_S)
            end
        end
    end

    @testset "0M saturation adjustment picks the curvature up" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            quad = CA.SGSQuadrature(FT; quadrature_order = 3)
            ρ, T = FT(1), FT(280)
            σ_q = FT(1e-3)
            q_sub = TD.q_vap_saturation(thp, T, ρ) - FT(1.4) * σ_q
            base = CA.compute_sgs_saturation_adjustment(
                thp, quad, ρ, T, q_sub, FT(0), σ_q^2, FT(0),
            )
            same = CA.compute_sgs_saturation_adjustment(
                thp, quad, ρ, T, q_sub, FT(0), σ_q^2, FT(0), σ_q, FT(0),
            )
            curved = CA.compute_sgs_saturation_adjustment(
                thp, quad, ρ, T, q_sub, FT(0), σ_q^2, FT(0), σ_q, -σ_q,
            )
            @test same.q_liq + same.q_ice == base.q_liq + base.q_ice
            @test curved.q_liq + curved.q_ice < base.q_liq + base.q_ice
            @test curved.q_liq + curved.q_ice >= FT(0)
            @test curved.T == T
        end
    end
end
