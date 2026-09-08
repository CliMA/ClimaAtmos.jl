#=
Unit tests for the SGS quadrature infrastructure.

Tests `SGSVarianceEvaluator` and `_sgs_saturation_moments`, verifying:
  - μ_S equals the linearized analytic mean q_tot_mean − q_sat(T_mean, ρ)
  - `SGSVarianceEvaluator` evaluated at the mean point gives zero
  - σ_S ≥ ϵ_numerics in the zero-variance limit (T′T′ = q′q′ = 0)
  - σ_S is monotonically increasing with σ_q
  - grid-mean fallback: nothing and GridMeanSGS give the same result
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP

const CA = ClimaAtmos

# Functor computing E[max(0, λ + α·S′)] over the quadrature, replicating the
# condensate diagnosis of `Microphysics1MEvaluator` (defined at top level:
# structs cannot be declared inside a testset).
struct CondensateMeanProbe{TPS, FT}
    tps::TPS
    ρ::FT
    mu_S::FT
    λ_lagrange::FT
    α::FT
end
function (p::CondensateMeanProbe)(T_hat, q_hat)
    FT = typeof(p.ρ)
    S′ = max(FT(0), q_hat) - TD.q_vap_saturation(p.tps, T_hat, p.ρ) - p.mu_S
    return max(FT(0), p.λ_lagrange + p.α * S′)
end

@testset "SGS Moments" begin

    @testset "SGSVarianceEvaluator: (S − μ_S)² with linearized μ_S" begin
        # μ_S is set analytically to q_tot_mean − q_sat(T_mean, ρ).
        # At the mean state, S = μ_S, so the evaluator returns 0.
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(2e-3)
                mu_S = q_tot_mean - q_sat_mean

                eval = CA.SGSVarianceEvaluator(thp, ρ, mu_S)
                @test eval(T_mean, q_tot_mean) ≈ FT(0) atol = eps(FT)

                # Off-mean point gives (S − μ_S)² > 0.
                δq = FT(1e-4)
                out = eval(T_mean, q_tot_mean + δq)
                @test out ≈ δq^2 rtol = FT(1e-5)

                # `_sgs_saturation_moments` exposes μ_S = analytic value.
                mom = CA._sgs_saturation_moments(
                    thp, ρ, T_mean, q_tot_mean, nothing, FT(0), FT(0), FT(0),
                )
                @test mom.mu_S ≈ mu_S rtol = FT(1e-6)
            end
        end
    end

    @testset "_sgs_saturation_moments: zero-variance limit (σ_S → ϵ_numerics)" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(2e-3)

                for quad in (
                    CA.SGSQuadrature(
                        FT;
                        distribution = CA.GaussianSGS(),
                        quadrature_order = 3,
                    ),
                    CA.SGSQuadrature(
                        FT;
                        distribution = CA.LogNormalSGS(),
                        quadrature_order = 3,
                    ),
                )
                    mom = CA._sgs_saturation_moments(
                        thp, ρ, T_mean, q_tot_mean, quad, FT(0), FT(0), FT(0),
                    )
                    @test mom.sigma_S ≥ 0
                    @test mom.sigma_S < FT(1e-10)
                end
            end
        end
    end

    @testset "_sgs_saturation_moments: σ_S increases monotonically with σ_q" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean - FT(1e-4)
                quad = CA.SGSQuadrature(
                    FT;
                    distribution = CA.GaussianSGS(),
                    quadrature_order = 3,
                )

                prev = FT(-1)
                for σ_q in FT[1e-4, 5e-4, 1e-3, 2e-3]
                    mom = CA._sgs_saturation_moments(
                        thp, ρ, T_mean, q_tot_mean, quad, FT(0), σ_q^2, FT(0),
                    )
                    @test mom.sigma_S ≥ 0
                    @test isfinite(mom.sigma_S)
                    @test mom.sigma_S > prev
                    prev = mom.sigma_S
                end
            end
        end
    end

    @testset "_sgs_saturation_moments: nothing / GridMeanSGS give the same result" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.0)
                T_mean = FT(285.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(1e-3)

                m1 = CA._sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                    nothing, FT(1), FT(1e-6), FT(0))
                m2 = CA._sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                    CA.GridMeanSGS(), FT(1), FT(1e-6), FT(0))

                @test isfinite(m1.sigma_S)
                @test m1.sigma_S ≥ 0
                @test m2.sigma_S ≈ m1.sigma_S rtol = FT(1e-6)
            end
        end
    end

    @testset "Discrete Lagrange-multiplier fit: exact mass constraint" begin
        # λ_lagrange from `_compute_sgs_moments` must satisfy
        # ⟨max(0, λ + α·S′)⟩ = q_c exactly under the quadrature measure the
        # microphysics evaluator integrates over — including regimes where
        # the analytic Gaussian fit errs by up to a factor ~2.5 (cold thin
        # cloud, q_c ≪ σ_S) and where the q ≥ 0 sampling clamp shifts the
        # sampled mean (σ_q comparable to q_tot).
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                α = FT(1)
                corr = FT(0.6)
                ρ = FT(1.1)
                # (T, saturation offset, q_c, T′T′, q′q′)
                regimes = (
                    (FT(288), FT(2e-3), FT(3e-4), FT(1), FT(1e-6)),
                    (FT(288), FT(-2e-3), FT(5e-5), FT(1), FT(1e-6)),
                    (FT(233), FT(1e-4), FT(2e-5), FT(1), FT(1e-7)),
                    (FT(288), FT(0), FT(2e-4), FT(9), FT(4e-6)),
                )
                for order in (1, 3, 5), (T, dq, q_c, T′T′, q′q′) in regimes
                    quad = CA.SGSQuadrature(FT; quadrature_order = order)
                    q_tot = TD.q_vap_saturation(thp, T, ρ) + dq
                    m = CA._compute_sgs_moments(
                        thp, ρ, T, q_tot, q_c, quad, T′T′, q′q′, corr, α,
                    )
                    mu_S = q_tot - TD.q_vap_saturation(thp, T, ρ)
                    probe = CondensateMeanProbe(thp, ρ, mu_S, m.λ_lagrange, α)
                    qc_mean = CA.integrate_over_sgs(
                        probe, quad, q_tot, T, q′q′, T′T′, corr,
                    )
                    @test qc_mean ≈ q_c rtol = sqrt(eps(FT))
                end
            end
        end
    end

    @testset "Discrete Lagrange-multiplier fit: limits and edge cases" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                α = FT(1)
                corr = FT(0.6)
                ρ = FT(1.1)
                T = FT(288)
                q_tot = TD.q_vap_saturation(thp, T, ρ) + FT(1e-3)
                q_c = FT(3e-4)
                quad = CA.SGSQuadrature(FT; quadrature_order = 3)

                # No SGS sampling (nothing / GridMeanSGS): λ = q_c directly.
                for sq in (nothing, CA.GridMeanSGS(),
                    CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()))
                    m = CA._compute_sgs_moments(
                        thp, ρ, T, q_tot, q_c, sq, FT(1), FT(1e-6), corr, α,
                    )
                    @test m.λ_lagrange == q_c
                end

                # Zero-variance limit of the sampled branch: λ → q_c.
                m0 = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_c, quad, FT(0), FT(0), corr, α,
                )
                @test m0.λ_lagrange ≈ q_c rtol = sqrt(eps(FT))

                # q_c = 0: every sampled point must reconstruct zero
                # condensate (λ at or below all kinks).
                mz = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, FT(0), quad, FT(1), FT(1e-6), corr, α,
                )
                mu_S = q_tot - TD.q_vap_saturation(thp, T, ρ)
                probe = CondensateMeanProbe(thp, ρ, mu_S, mz.λ_lagrange, α)
                qc_mean = CA.integrate_over_sgs(
                    probe, quad, q_tot, T, FT(1e-6), FT(1), corr,
                )
                @test qc_mean ≤ eps(FT)

                # Seed far below all kinks (deeply subsaturated + tiny q_c):
                # the solver must recover via the dg = 0 jump.
                q_tot_dry = TD.q_vap_saturation(thp, T, ρ) - FT(5e-3)
                q_c_tiny = FT(1e-8)
                md = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot_dry, q_c_tiny, quad,
                    FT(1), FT(1e-6), corr, α,
                )
                mu_S_dry = q_tot_dry - TD.q_vap_saturation(thp, T, ρ)
                probe_dry =
                    CondensateMeanProbe(thp, ρ, mu_S_dry, md.λ_lagrange, α)
                qc_dry = CA.integrate_over_sgs(
                    probe_dry, quad, q_tot_dry, T, FT(1e-6), FT(1), corr,
                )
                @test qc_dry ≈ q_c_tiny rtol = sqrt(eps(FT))
            end
        end
    end

end
