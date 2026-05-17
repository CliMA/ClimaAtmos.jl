using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.Parameters as CMP

const WF_GH_ORDER = 5
const WF_BULK_RTOL = 1e-2
# Explicit Riemann reference on a χ₁×χ₂ grid (test only; not production GH).
const WF_REF_GRID = 128

"""GH integral of condensate (production path, one N² pass)."""
function wf_gh_bulk(
    thp, SG, ρ, T_mean, q_tot_mean, q′q′, T′T′, corr, q_cond_mean, lf, α; iters = 1,
)
    FT = eltype(q_cond_mean)
    q_liq = lf * q_cond_mean
    q_ice = q_cond_mean - q_liq
    res = ClimaAtmos.compute_sgs_condensate_water_filling(
        thp, SG, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr, q_liq, q_ice; α, iters,
    )
    return res.q_liq + res.q_ice, res
end

"""
Explicit reference for the same 2D integral production GH approximates:

``⟨\\hat{q}_c⟩ = ∫∫ \\hat{q}_c(T(χ_1),q(χ_2))\\, e^{-(χ_1^2+χ_2^2)}/π\\, dχ_1 dχ_2``

via a uniform Riemann sum on ``χ_i ∈ [-4,4]`` (`WF_REF_GRID`² points). This is **not**
higher-order `gauss_hermite` (ClimaAtmos only ships N≤5); it brute-forces the integral
in Julia to tighten test tolerances. Same [`PhysicalPointTransform`](@ref) and
[`qcond_hat_water_filling`](@ref) as `integrate_over_sgs`.
"""
function wf_dense_ref_bulk(
    thp, ρ, T_mean, q_tot_mean, q′q′, T′T′, corr, μ, λ, α, lf, scale_pos;
    n = WF_REF_GRID,
)
    FT = eltype(T_mean)
    σ_q, σ_T, corr_c = ClimaAtmos.sgs_stddevs_and_correlation(q′q′, T′T′, corr)
    dist = ClimaAtmos.GaussianSGS()
    transform = ClimaAtmos.PhysicalPointTransform(
        dist, T_mean, q_tot_mean, σ_T, σ_q, corr_c, FT(150), FT(0.1),
    )
    χ = range(-FT(4), FT(4); length = n)
    dχ = χ[2] - χ[1]
    acc = zero(FT)
    invπ = one(FT) / FT(π)
    for χ1 in χ, χ2 in χ
        T_hat, q_hat = transform(χ1, χ2)
        qc = ClimaAtmos.qcond_hat_water_filling(
            thp, ρ, q_hat, T_hat, μ, λ, α; scale_pos = scale_pos,
        )
        acc += qc * exp(-χ1 * χ1 - χ2 * χ2) * dχ * dχ * invπ
    end
    return acc
end

@testset "Water-filling" begin
    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            toml = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml)
            ρ = FT(1.0)
            T_mean = FT(280.0)
            q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
            μ_moist = FT(0.01) - q_sat_mean
            q_tot_moist = FT(0.01)

            σq = FT(1e-5)
            q′q′ = σq * σq
            T′T′ = FT(0)
            corr = FT(0)
            lf = FT(0.5)
            A_on = ClimaAtmos.compute_sigma_S(thp, ρ, q_tot_moist, T_mean, q′q′, T′T′, corr; lf)
            SG5 = ClimaAtmos.SGSQuadrature(FT; quadrature_order = WF_GH_ORDER)

            @testset "lambda_from_qc_1d at A = 0" begin
                q_cond = FT(2e-6)
                λ_inv = ClimaAtmos.lambda_from_qc_1d(
                    thp, ρ, q_tot_moist, T_mean, q′q′, T′T′, corr, μ_moist, q_cond;
                    α = zero(FT), iters = 1, lf,
                )
                @test isapprox(λ_inv, q_cond; atol = FT(0), rtol = FT(0))
            end

            @testset "α = 0: GH bulk equals prescribed q_cond" begin
                q_star = ClimaAtmos.qc_from_lambda_1d(μ_moist, A_on)
                cases = (
                    (μ_moist, q_star + FT(0.1) * A_on, q_tot_moist),
                    (FT(-3e-3), FT(1e-6), q_sat_mean + FT(-3e-3)),
                )
                for (μ_c, q_c, q_tot_c) in cases
                    qmean, res = wf_gh_bulk(
                        thp, SG5, ρ, T_mean, q_tot_c, q′q′, T′T′, corr, q_c, lf, zero(FT),
                    )
                    @test isapprox(res.λ, q_c; atol = FT(0), rtol = FT(0))
                    @test abs(qmean - q_c) / max(q_c, FT(1e-20)) ≤ WF_BULK_RTOL
                end
            end

            @testset "α = 1: 1D closure inverts shift targets" begin
                q_star = ClimaAtmos.qc_from_lambda_1d(μ_moist, A_on)
                q_shift = q_star + FT(0.05) * A_on
                λ = ClimaAtmos.lambda_from_qc_1d(
                    thp, ρ, q_tot_moist, T_mean, q′q′, T′T′, corr, μ_moist, q_shift;
                    α = one(FT), iters = 1, lf,
                )
                qrec = ClimaAtmos.qc_from_lambda_1d(λ, A_on)
                @test abs(qrec - q_shift) / q_shift ≤ WF_BULK_RTOL

                μ_dry = FT(-3e-3)
                q_trace = FT(1e-6)
                λ_d = ClimaAtmos.lambda_from_qc_1d(
                    thp, ρ, q_sat_mean + μ_dry, T_mean, q′q′, T′T′, corr, μ_dry, q_trace;
                    α = one(FT), iters = 1, lf,
                )
                qrec_d = ClimaAtmos.qc_from_lambda_1d(λ_d, A_on)
                @test abs(qrec_d - q_trace) / q_trace ≤ WF_BULK_RTOL
            end

            @testset "α = 1: GH bulk ≈ q_cond (moist / subcapacity / μ≈0)" begin
                q_star = ClimaAtmos.qc_from_lambda_1d(μ_moist, A_on)
                μ_sub = FT(2e-3)
                q_sub = FT(0.5) * ClimaAtmos.qc_from_lambda_1d(μ_sub, A_on)
                cases = (
                    (μ_moist, q_star + FT(0.1) * A_on, q_tot_moist),
                    (μ_sub, q_sub, q_sat_mean + μ_sub),
                    (zero(FT), ClimaAtmos.qc_from_lambda_1d(zero(FT), A_on) + FT(0.05) * A_on, q_tot_moist),
                )
                for (μ_c, q_c, q_tot_c) in cases
                    qmean, = wf_gh_bulk(
                        thp, SG5, ρ, T_mean, q_tot_c, q′q′, T′T′, corr, q_c, lf, one(FT),
                    )
                    @test abs(qmean - q_c) / max(q_c, FT(1e-20)) ≤ WF_BULK_RTOL
                end
            end

            @testset "GH N=$WF_GH_ORDER vs explicit χ-grid ($(WF_REF_GRID)^2 pts, moist shift)" begin
                q_star = ClimaAtmos.qc_from_lambda_1d(μ_moist, A_on)
                q_c = q_star + FT(0.1) * A_on
                λ, scale_pos = ClimaAtmos.water_filling_λ_scale_pos(
                    q_c, μ_moist, A_on, one(FT), thp, ρ, q_tot_moist, T_mean, q′q′, T′T′, corr;
                    iters = 1, lf,
                )
                eval = ClimaAtmos.WaterFillingSGSEvaluator(thp, ρ, μ_moist, λ, one(FT), lf, scale_pos)
                q_gh = ClimaAtmos.integrate_over_sgs(eval, SG5, q_tot_moist, T_mean, q′q′, T′T′, corr)
                q_gh = q_gh.q_liq + q_gh.q_ice
                q_ref = wf_dense_ref_bulk(
                    thp, ρ, T_mean, q_tot_moist, q′q′, T′T′, corr, μ_moist, λ, one(FT), lf, scale_pos,
                )
                @test abs(q_gh - q_ref) / max(q_ref, FT(1e-20)) ≤ WF_BULK_RTOL
            end

            @testset "dry trace: 1D closure matches explicit χ-grid; GH N=$WF_GH_ORDER can overshoot" begin
                μ_dry = FT(-3e-3)
                q_trace = FT(1e-6)
                q_tot_dry = q_sat_mean + μ_dry
                λ, scale_pos = ClimaAtmos.water_filling_λ_scale_pos(
                    q_trace, μ_dry, A_on, one(FT), thp, ρ, q_tot_dry, T_mean, q′q′, T′T′, corr;
                    iters = 1, lf,
                )
                q_ref = wf_dense_ref_bulk(
                    thp, ρ, T_mean, q_tot_dry, q′q′, T′T′, corr, μ_dry, λ, one(FT), lf, scale_pos,
                )
                @test abs(q_ref - q_trace) / q_trace ≤ WF_BULK_RTOL
                eval = ClimaAtmos.WaterFillingSGSEvaluator(thp, ρ, μ_dry, λ, one(FT), lf, scale_pos)
                q_gh = ClimaAtmos.integrate_over_sgs(eval, SG5, q_tot_dry, T_mean, q′q′, T′T′, corr)
                q_gh = q_gh.q_liq + q_gh.q_ice
                # λ inverts the 1D closure; clipped q̂_c is not smooth — N=5 GH need not match the grid ref.
                @test q_gh > q_trace
            end

            @testset "α = 0 uniform nodes" begin
                q_cond = FT(2e-6)
                λ_u, scale_u = ClimaAtmos.water_filling_λ_scale_pos(
                    q_cond, μ_moist, zero(FT), zero(FT), thp, ρ, q_tot_moist, T_mean, q′q′, T′T′, corr;
                    lf,
                )
                for T_hat in (T_mean - FT(1), T_mean, T_mean + FT(1))
                    qc = ClimaAtmos.qcond_hat_water_filling(
                        thp, ρ, q_tot_moist, T_hat, μ_moist, λ_u, zero(FT); scale_pos = scale_u,
                    )
                    @test isapprox(qc, q_cond; atol = FT(0), rtol = FT(0))
                end
            end

            @testset "qc_from_lambda_1d: large-a tail asymptotic" begin
                # λ < 0 path uses Q(a) with a = -λ/A; asymptotic fires once the direct φ,Φ form underflows.
                function _Q_tail_ref(a::Float64; n = 80_000)
                    setprecision(80)
                    β = big(1) / sqrt(2 * big(π))
                    a = big(a)
                    hi = a + big(40)
                    h = (hi - a) / n
                    s = big(0)
                    for k in 0:n
                        t = a + k * h
                        w = (k == 0 || k == n) ? 1 : 2
                        s += w * (t - a) * β * exp(-t^2 / 2)
                    end
                    return Float64(s * h / 2)
                end
                # a ≥ 7 uses the log-asymptotic branch (Winitzki Φ is unreliable above a ≈ 3).
                for a in (FT(8), FT(10), FT(15))
                    A = FT(1)
                    λ = -a * A
                    q = ClimaAtmos.qc_from_lambda_1d(λ, A)
                    q_ref = FT(_Q_tail_ref(Float64(a)) * Float64(A))
                    @test isapprox(q, q_ref; rtol = FT(0.01), atol = zero(FT))
                end
            end

            @testset "microphysics_tendencies_1m smoke" begin
                mp = CMP.Microphysics1MParams(toml)
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3, α_max = FT(1))
                q_cond = ClimaAtmos.qc_from_lambda_1d(μ_moist, A_on) + FT(0.05) * A_on
                tend = ClimaAtmos.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, mp, thp, ρ,
                    T_mean, q_tot_moist, lf * q_cond, q_cond - lf * q_cond, FT(0), FT(0),
                    FT(0.5), q′q′, FT(0.6), FT(60), 1,
                )
                @test isfinite(tend.dq_lcl_dt)
            end
        end
    end
end
