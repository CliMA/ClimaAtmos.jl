#=
Unit tests for SGS Moments pre-pass (Stage A of the scale-aware unified
microphysics framework, see docs/src/sgs_microphysics_redesign.md).

Tests the SGSMomentsEvaluator functor and the compute_sgs_moments driver,
verifying:
  - moment finiteness and non-negativity (M_l, M_i ≥ 0; σ_S² ≥ 0)
  - high-resolution limit: at σ → 0, moments collapse to grid-mean values
  - low-resolution behavior: σ_S² > 0 and M_l grows as the PDF widens
  - distribution dispatch: Gaussian uses linear S = q_tot - q_sat; lognormal
    uses log S = log(q_tot / q_sat)
  - grid-mean fallback: works with GridMeanSGS / nothing
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import CloudMicrophysics.Parameters as CMP
import ClimaParams as CP

const CA = ClimaAtmos

@testset "SGS Moments" begin

    @testset "SGSMomentsEvaluator dispatch and basic properties" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)

                # Construct an in-equilibrium grid state
                q_tot_mean = q_sat_mean + FT(2e-3)
                excess = q_tot_mean - q_sat_mean
                λ = FT(1.0)            # warm: pure liquid
                q_rai = FT(1e-4)
                q_sno = FT(1e-5)
                q_min = FT(1e-10)

                # Build evaluators for each distribution
                eval_g = CA.SGSMomentsEvaluator(
                    CA.GaussianSGS(), thp, ρ, λ, q_rai, q_sno, q_min,
                )
                eval_ln = CA.SGSMomentsEvaluator(
                    CA.LogNormalSGS(), thp, ρ, λ, q_rai, q_sno, q_min,
                )
                eval_gm = CA.SGSMomentsEvaluator(
                    CA.GridMeanSGS(), thp, ρ, λ, q_rai, q_sno, q_min,
                )

                # Evaluate at the grid mean point — should give well-defined results
                out_g = eval_g(T_mean, q_tot_mean)
                out_ln = eval_ln(T_mean, q_tot_mean)
                out_gm = eval_gm(T_mean, q_tot_mean)

                # Gaussian / GridMean: S = q_tot - q_sat (linear)
                @test out_g.mu_S ≈ excess rtol = FT(1e-5)
                @test out_gm.mu_S ≈ excess rtol = FT(1e-5)
                # Lognormal: S = log(q_tot / q_sat)
                @test out_ln.mu_S ≈ log(q_tot_mean / q_sat_mean) rtol = FT(1e-5)

                # M_l and M_i should be non-negative
                @test out_g.M_l ≥ 0
                @test out_g.M_i ≥ 0
                @test out_ln.M_l ≥ 0
                @test out_ln.M_i ≥ 0

                # When the max(0, ·) clamp does not bind, the equilibrium
                # partition is mass-conserving against the saturation excess:
                # M_l + q_rai = λ · excess.  Here λ = 1 (pure liquid), so the
                # liquid relation holds.  M_i is clamped to zero in this
                # pure-liquid case (since (1−λ)·excess = 0 but q_sno > 0).
                @test out_g.M_l + q_rai ≈ λ * excess rtol = FT(1e-5)
                @test out_g.M_i == FT(0)
            end
        end
    end

    @testset "compute_sgs_moments high-resolution limit (σ → 0)" begin
        # At zero variance, the quadrature collapses to a single point at the
        # grid mean, and moments reduce to: μ_S = S(T_mean, q_tot_mean),
        # σ_S² = 0, M_l = q_lcl_eq_at_mean, M_i = q_icl_eq_at_mean.
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(2e-3)
                q_lcl = FT(1.5e-3)
                q_icl = FT(0)
                q_rai = FT(1e-4)
                q_sno = FT(0)
                q_min = FT(1e-10)
                T′T′ = FT(0)
                q′q′ = FT(0)
                corr_Tq = FT(0)

                quad_g = CA.SGSQuadrature(
                    FT; distribution = CA.GaussianSGS(), quadrature_order = 3,
                )
                quad_ln = CA.SGSQuadrature(
                    FT; distribution = CA.LogNormalSGS(), quadrature_order = 3,
                )

                # Gaussian
                m_g = CA.compute_sgs_moments(
                    thp, ρ, T_mean, q_tot_mean,
                    q_lcl, q_icl, q_rai, q_sno,
                    quad_g, T′T′, q′q′, corr_Tq, q_min,
                )
                @test m_g.mu_S ≈ q_tot_mean - q_sat_mean rtol = FT(1e-5)
                @test m_g.sigma_S_sq ≥ 0
                @test m_g.sigma_S_sq < FT(1e-10)  # essentially zero at σ → 0
                # M_l at grid mean: max(0, λ × excess − q_rai)
                λ_mean = TD.liquid_fraction(thp, T_mean, q_lcl, q_icl)
                excess = q_tot_mean - q_sat_mean
                M_l_expected = max(FT(0), λ_mean * excess - q_rai)
                @test m_g.M_l ≈ M_l_expected rtol = FT(1e-5)

                # Lognormal: same structure but μ_S is log-ratio
                m_ln = CA.compute_sgs_moments(
                    thp, ρ, T_mean, q_tot_mean,
                    q_lcl, q_icl, q_rai, q_sno,
                    quad_ln, T′T′, q′q′, corr_Tq, q_min,
                )
                @test m_ln.mu_S ≈ log(q_tot_mean / q_sat_mean) rtol = FT(1e-5)
                @test m_ln.sigma_S_sq ≥ 0
                @test m_ln.sigma_S_sq < FT(1e-10)
                # M_l does not depend on the SGS distribution choice in the
                # σ → 0 limit (single-point evaluation): linear excess used.
                @test m_ln.M_l ≈ M_l_expected rtol = FT(1e-5)
            end
        end
    end

    @testset "compute_sgs_moments low-resolution behavior (σ > 0)" begin
        # At nonzero variance, σ_S² should be positive; M_l should generally
        # increase with σ because more of the PDF tail extends into the
        # saturated regime.
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean - FT(1e-4)  # slightly subsaturated
                q_lcl = FT(0)
                q_icl = FT(0)
                q_rai = FT(0)
                q_sno = FT(0)
                q_min = FT(1e-10)
                corr_Tq = FT(0)

                quad = CA.SGSQuadrature(
                    FT; distribution = CA.GaussianSGS(), quadrature_order = 3,
                )

                # Sweep over σ_q
                σ_q_vals = FT[1e-4, 5e-4, 1e-3, 2e-3]
                M_l_vals = FT[]
                for σ_q in σ_q_vals
                    q′q′ = σ_q * σ_q
                    m = CA.compute_sgs_moments(
                        thp, ρ, T_mean, q_tot_mean,
                        q_lcl, q_icl, q_rai, q_sno,
                        quad, FT(0), q′q′, corr_Tq, q_min,
                    )
                    @test m.sigma_S_sq ≥ 0
                    @test m.M_l ≥ 0
                    @test m.M_i ≥ 0
                    @test isfinite(m.mu_S)
                    @test isfinite(m.sigma_S_sq)
                    push!(M_l_vals, m.M_l)
                end
                # Monotonically nondecreasing as σ grows (more of PDF crosses
                # saturation)
                for i in 2:length(M_l_vals)
                    @test M_l_vals[i] >= M_l_vals[i - 1] - FT(1e-12)
                end
            end
        end
    end

    @testset "compute_sgs_moments handles nothing / GridMean SGS" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.0)
                T_mean = FT(285.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(1e-3)
                q_lcl = FT(1e-3)
                q_icl = FT(0)
                q_rai = FT(0)
                q_sno = FT(0)
                q_min = FT(1e-10)

                # Should not error with nothing
                m1 = CA.compute_sgs_moments(
                    thp, ρ, T_mean, q_tot_mean,
                    q_lcl, q_icl, q_rai, q_sno,
                    nothing, FT(1), FT(1e-6), FT(0), q_min,
                )
                @test isfinite(m1.mu_S)
                @test m1.sigma_S_sq ≥ 0
                @test m1.M_l ≥ 0

                # GridMeanSGS — should give the same as nothing (single-point
                # evaluation at the mean)
                m2 = CA.compute_sgs_moments(
                    thp, ρ, T_mean, q_tot_mean,
                    q_lcl, q_icl, q_rai, q_sno,
                    CA.GridMeanSGS(), FT(1), FT(1e-6), FT(0), q_min,
                )
                @test m2.mu_S ≈ m1.mu_S rtol = FT(1e-6)
                @test m2.M_l ≈ m1.M_l rtol = FT(1e-6)
            end
        end
    end

    @testset "compute_sgs_moments fully-clear / fully-cloudy limits" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_lcl = FT(0)
                q_icl = FT(0)
                q_rai = FT(0)
                q_sno = FT(0)
                q_min = FT(1e-10)

                quad = CA.SGSQuadrature(
                    FT; distribution = CA.GaussianSGS(), quadrature_order = 3,
                )

                # Deep subsaturation, small σ: M_l should be ≈ 0
                m_dry = CA.compute_sgs_moments(
                    thp, ρ, T_mean, q_sat_mean - FT(5e-3),
                    q_lcl, q_icl, q_rai, q_sno,
                    quad, FT(0), FT(1e-8), FT(0), q_min,
                )
                @test m_dry.M_l < FT(1e-6)
                @test m_dry.mu_S < 0

                # Deeply saturated, small σ: M_l should be ≈ excess
                excess = FT(5e-3)
                m_wet = CA.compute_sgs_moments(
                    thp, ρ, T_mean, q_sat_mean + excess,
                    q_lcl, q_icl, q_rai, q_sno,
                    quad, FT(0), FT(1e-8), FT(0), q_min,
                )
                # λ_at_grid_mean uses (0, 0) condensate → T-ramp fallback
                λ_mean = TD.liquid_fraction(thp, T_mean, q_lcl, q_icl)
                @test m_wet.M_l ≈ λ_mean * excess rtol = FT(1e-3)
                @test m_wet.mu_S > 0
            end
        end
    end
end
