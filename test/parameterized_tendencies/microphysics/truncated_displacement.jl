#=
Checks `truncated_displacement_condensate`, the equilibrium condensate under
the Gaussian displacement truncated at `c` standard deviations and stretched
to keep its variance: against brute-force integration, its limits (zero below
the support, the mean above it, the Gaussian for `c = Inf`), that the stretch
keeps unit variance, and that the 1M moments and the 0M saturation adjustment
use it. The tolerances follow from `normal_cdf`, the polynomial approximation of
the normal CDF in Abramowitz and Stegun (1964), Handbook of Mathematical
Functions, Eq. 26.2.17, whose absolute error is below 7.5e-8.
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

const CA = ClimaAtmos

# Midpoint rule over the truncated, stretched standard normal on [−c, c].
function reference_condensate(mu_S, sigma_S, c; n = 200001)
    s = CA.truncated_displacement_scale(c)
    xs = range(-c, c; length = n)
    w = [exp(-x^2 / 2) for x in xs]
    return sum(max(0.0, mu_S + s * sigma_S * x) * wi for (x, wi) in zip(xs, w)) /
           sum(w)
end

@testset "Truncated displacement condensate" begin

    @testset "Stretch keeps unit variance" begin
        for c in (1.5, 2.0, 3.0, 4.0)
            s = CA.truncated_displacement_scale(c)
            xs = range(-c, c; length = 200001)
            w = [exp(-x^2 / 2) for x in xs]
            # Limited by the 7.5e-8 absolute error of `normal_cdf`.
            @test sum((s * x)^2 * wi for (x, wi) in zip(xs, w)) / sum(w) ≈ 1 rtol = 1e-5
            @test s > 1
        end
        @test CA.truncated_displacement_scale(Inf) == 1
        @test CA.truncated_displacement_scale(3.0f0) isa Float32
    end

    @testset "Closed form against brute force" begin
        σ = 1e-3
        for c in (2.0, 3.0, 4.0), m in (-4.0, -3.2, -2.5, -1.0, 0.0, 0.7, 2.0, 3.5)
            # Within 2e-3 of the brute-force value: the 7.5e-8 absolute error
            # of `normal_cdf` is 1e-3 of the deepest tail probability here.
            @test CA.truncated_displacement_condensate(m * σ, σ, c) ≈
                  reference_condensate(m * σ, σ, c) rtol = 2e-3 atol = 1e-12
        end
    end

    @testset "Limits" begin
        for FT in (Float32, Float64)
            σ = FT(1e-3)
            c = FT(3)
            s = CA.truncated_displacement_scale(c)
            # Below the support no displaced parcel reaches saturation; the
            # Gaussian tail at μ_S = −3.6 σ_S still condenses.
            for m in (FT(-3.6), -s * c, FT(-10))
                @test CA.truncated_displacement_condensate(m * σ, σ, c) == FT(0)
            end
            @test CA.truncated_gaussian_condensate(FT(-3.6) * σ, σ) > FT(0)
            # Above the support every displaced parcel is saturated.
            @test CA.truncated_displacement_condensate(FT(3.5) * σ, σ, c) ≈
                  FT(3.5) * σ rtol = 10 * eps(FT)
            # `c = Inf` is the Gaussian, bit for bit.
            for m in (FT(-5), FT(-1), FT(0), FT(2))
                @test CA.truncated_displacement_condensate(m * σ, σ, FT(Inf)) ==
                      CA.truncated_gaussian_condensate(m * σ, σ)
            end
            # The parameter keeps its own type; the result has the state's.
            for c_param in (3.0f0, Inf32)
                @test @inferred(
                    CA.truncated_displacement_condensate(FT(-1e-3), σ, c_param),
                ) isa FT
            end
        end
    end

    @testset "Continuous and nondecreasing in the mean" begin
        σ = 1e-3
        ms = range(-4, 4; length = 4001)
        E = [CA.truncated_displacement_condensate(m * σ, σ, 3.0) for m in ms]
        @test all(diff(E) .>= -1e-15)
        @test maximum(abs.(diff(E))) < 3 * step(ms) * σ
    end

    @testset "equilibrium_condensate dispatch" begin
        σ = 1e-3
        for m in (-3.6, -1.0, 0.5)
            @test CA.equilibrium_condensate(m * σ, σ, 0.0, 0.0, 3.0) ==
                  CA.truncated_displacement_condensate(m * σ, σ, 3.0)
            @test CA.equilibrium_condensate(m * σ, σ, 0.8σ, -σ, 3.0) ==
                  CA.displacement_condensate(m * σ, σ, 0.8σ, -σ)
        end
    end

    @testset "1M moments: λ_eq matches the truncated condensate" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            corr, ρ, T = FT(-0.9), FT(0.9), FT(280)
            T′T′, q′q′ = FT(0.02), FT(1e-6)
            quad = CA.SGSQuadrature(FT; quadrature_order = 3)
            q_sat = TD.q_vap_saturation(thp, T, ρ)
            for dq in (FT(-4e-3), FT(-2e-3), FT(-5e-4), FT(3e-4))
                q_tot = q_sat + dq
                mu_S = q_tot - q_sat
                m_default = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, FT(0), quad, T′T′, q′q′, corr, FT(1),
                )
                m_inf = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, FT(0), quad, T′T′, q′q′, corr, FT(1),
                    q_tot, FT(0), FT(0), FT(Inf),
                )
                @test m_inf == m_default
                # The untruncated Gaussian's multiplier is the mean excess.
                @test m_inf.λ_eq == mu_S
                m = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, FT(0), quad, T′T′, q′q′, corr, FT(1),
                    q_tot, FT(0), FT(0), FT(3),
                )
                @test m.sigma_S == m_default.sigma_S
                @test m.λ_lagrange == m_default.λ_lagrange
                # Under default sgs_lambda_eq_mode = 0, λ_eq is the linearized mean saturation excess mu_S.
                @test m.λ_eq == mu_S
                q_c_eq = CA.truncated_displacement_condensate(mu_S, m.sigma_S, FT(3))
                mu_S < -FT(3.1) * m.sigma_S && @test q_c_eq == FT(0)

                # Under sgs_lambda_eq_mode = 1, λ_eq matches the truncated condensate.
                m_fitted = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, FT(0), quad, T′T′, q′q′, corr, FT(1),
                    q_tot, FT(0), FT(0), FT(3), 1,
                )
                if q_c_eq > FT(0)
                    @test CA.truncated_gaussian_condensate(
                        m_fitted.λ_eq,
                        m_fitted.sigma_S,
                    ) ≈
                          q_c_eq atol = (FT == Float32 ? 1e-6 : 1e-7) * m_fitted.sigma_S
                else
                    @test m_fitted.λ_eq == FT(-Inf)
                end
            end
        end
    end

    # `λ_eq = −Inf` stands for "no equilibrium condensate". Every reader of
    # `λ_eq` must tolerate it: a product with it, such as the `s λ` scaling of
    # the actual multiplier, would turn `0 · Inf` into NaN.
    @testset "1M tendencies with no equilibrium condensate (λ_eq = −Inf)" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(
                FT; quadrature_order = 3, distribution = CA.GaussianSGS(),
            )
            ρ, T, w = FT(1.1), FT(285), FT(0.1)
            T′T′, q′q′, corr, α = FT(0.25), FT(2.5e-7), FT(-0.9), FT(0.7)
            σ_S = CA._sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr)
            # Below the truncated support (`s c ≈ 3.04` for `c = 3`).
            mu_S = -FT(4) * σ_S
            q_tot = TD.q_vap_saturation(thp, T, ρ) + mu_S
            # Clear sky; rain falling into it; an evaporating cloud with rain
            # placed in a shaft, which sets all three thresholds of the rule.
            for (q_lcl, q_rai, β_p) in (
                (FT(0), FT(0), FT(0)),
                (FT(0), FT(1e-4), FT(0)),
                (FT(2e-5), FT(1e-4), FT(0.5)),
            )
                mom = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_lcl, quad, T′T′, q′q′, corr, α,
                    q_tot - q_rai, FT(0), FT(0), FT(3), 1,
                )
                @test mom.λ_eq == FT(-Inf)
                tendencies(λ_eq) = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, w,
                    q_tot, q_lcl, FT(0), q_rai, FT(0), T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, FT(0)), mu_S,
                    β_p, FT(0.4), mom.sigma_S, FT(0.1), λ_eq,
                )
                out = tendencies(mom.λ_eq)
                @test all(isfinite, values(out))
                ref = tendencies(-FT(20) * mom.sigma_S)
                @test all(
                    isapprox.(
                        values(out), values(ref);
                        rtol = sqrt(eps(FT)), atol = eps(FT) * mom.sigma_S / FT(180),
                    ),
                )
            end
        end
    end

    @testset "1M tendencies in subsaturated tail" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(
                FT; quadrature_order = 3, distribution = CA.GaussianSGS(),
            )
            ρ, T, w = FT(1.1), FT(285), FT(0.1)
            T′T′, q′q′, corr, α = FT(0.25), FT(2.5e-7), FT(-0.9), FT(0.7)
            σ_S = CA._sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr)
            # Below the truncated support (`s c ≈ 3.04` for `c = 3`).
            mu_S = -FT(4) * σ_S
            q_tot = TD.q_vap_saturation(thp, T, ρ) + mu_S
            # Clear sky; rain falling into it; an evaporating cloud with rain
            # placed in a shaft.
            for (q_lcl, q_rai, β_p) in (
                (FT(0), FT(0), FT(0)),
                (FT(0), FT(1e-4), FT(0)),
                (FT(2e-5), FT(1e-4), FT(0.5)),
            )
                mom = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_lcl, quad, T′T′, q′q′, corr, α,
                    q_tot - q_rai, FT(0), FT(0), FT(3),
                )
                @test mom.λ_eq ≈ mu_S - q_rai atol =
                    64 * eps(FT) * TD.q_vap_saturation(thp, T, ρ)
                out = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, w,
                    q_tot, q_lcl, FT(0), q_rai, FT(0), T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, FT(0)), mu_S,
                    β_p, FT(0.4), mom.sigma_S, FT(0.1),
                )
                @test all(isfinite, values(out))
            end
        end
    end

    @testset "0M saturation adjustment uses the truncation" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            quad = CA.SGSQuadrature(FT; quadrature_order = 3)
            ρ, T = FT(1), FT(280)
            σ_q = FT(1e-3)
            q_sub = TD.q_vap_saturation(thp, T, ρ) - FT(3.6) * σ_q
            gaussian = CA.compute_sgs_saturation_adjustment(
                thp, quad, ρ, T, q_sub, FT(0), σ_q^2, FT(0),
            )
            truncated = CA.compute_sgs_saturation_adjustment(
                thp, quad, ρ, T, q_sub, FT(0), σ_q^2, FT(0), FT(0), FT(0), FT(3),
            )
            @test gaussian.q_liq + gaussian.q_ice > FT(0)
            @test truncated.q_liq + truncated.q_ice == FT(0)
            @test truncated.T == T
        end
    end
end
