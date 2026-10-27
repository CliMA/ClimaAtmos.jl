#=
Unit tests for cloud_inhomogeneity.jl: the effective-thickness factor
`χ = exp(E[ln q]) / E[q]` of the in-cloud condensate that scales the water
paths passed to radiation.
=#

using Test
import ClimaAtmos as CA

@testset "Cloud inhomogeneity: $FT" for FT in (Float32, Float64)
    γ_E = FT(Base.MathConstants.eulergamma)
    rtol_L = FT == Float32 ? FT(1e-5) : FT(1e-8)
    rtol = FT(3e-4)

    @testset "ψ(ν) − ln ν" begin
        L = CA.log_gamma_effective_thickness
        @test L(FT(1)) ≈ -γ_E rtol = rtol_L
        @test L(FT(2)) ≈ 1 - γ_E - log(FT(2)) rtol = rtol_L
        @test L(FT(0.5)) ≈ -γ_E - log(FT(2)) rtol = rtol_L
        @test L(FT(Inf)) == 0
        @test L(FT(1e4)) ≈ -1 / FT(2e4) rtol = FT(1e-3)
        @test L(FT(0.1)) < L(FT(1)) < L(FT(10)) < 0
    end

    @testset "Truncated-Gaussian relative variance" begin
        rv = CA.truncated_gaussian_relative_variance
        # Exact values of v / m² for the positive tail of N(Φ⁻¹(cf), 1).
        @test rv(FT(0)) == 1
        @test rv(FT(1e-4)) ≈ FT(1 / 1.100840) rtol = rtol
        @test rv(FT(0.08)) ≈ FT(1 / 1.302724) rtol = rtol
        @test rv(FT(0.5)) ≈ FT((π - 2) / 2) rtol = rtol
        @test rv(FT(0.92)) ≈ FT(1 / 3.286448) rtol = rtol
        # Continuous on both sides of the tail branch and constant above the
        # upper clamp.
        @test rv(FT(0.00819)) ≈ rv(FT(0.00821)) rtol = FT(1e-3)
        @test rv(FT(1)) == rv(one(FT) - FT(1e-4))
    end

    @testset "Single subdomain" begin
        χ(cf, rv_floor) = CA.cloud_optical_depth_factor(
            cf, one(FT), FT(1e-4), zero(FT), zero(FT), zero(FT), zero(FT),
            rv_floor,
        )
        L = CA.log_gamma_effective_thickness
        # With `rv_floor = 0`, χ is the Gamma factor of the truncated Gaussian.
        @test χ(FT(0.5), FT(0)) ≈ FT(0.732150) rtol = rtol
        @test χ(FT(1e-6), FT(0)) ≈ exp(-γ_E) rtol = FT(0.05)
        # The floor composes multiplicatively with the Gaussian variance.
        rv_floor = FT(0.25)
        rv_tot = (1 + CA.truncated_gaussian_relative_variance(FT(0.5))) *
                 (1 + rv_floor) - 1
        @test χ(FT(0.5), rv_floor) ≈ exp(L(1 / rv_tot)) rtol = rtol
        # Overcast layers keep the floor inhomogeneity, with no jump at the
        # upper clamp of the cloud fraction.
        @test χ(FT(1), rv_floor) < FT(0.9)
        @test χ(FT(1), rv_floor) == χ(one(FT) - FT(1e-4), rv_floor)
        # Independent of the condensate amount; 1 with no condensate.
        @test CA.cloud_optical_depth_factor(
            FT(0.5), one(FT), FT(3e-3), zero(FT), zero(FT), zero(FT), zero(FT),
            rv_floor,
        ) ≈ χ(FT(0.5), rv_floor) rtol = FT(1e-5)
        @test CA.cloud_optical_depth_factor(
            FT(0.5), one(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT),
            rv_floor,
        ) == 1
        @test χ(FT(0), rv_floor) == 1

        cfs = FT[1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9999]
        for rv_f in (FT(0), rv_floor)
            χs = χ.(cfs, rv_f)
            @test eltype(χs) == FT
            @test all(diff(χs) .> 0)
            @test all(0 .< χs .<= 1)
        end
    end

    @testset "Environment and updraft mixture" begin
        L = CA.log_gamma_effective_thickness
        χ_mix(cf, a_env, q_env, as, qs, rv_floor) =
            CA.cloud_optical_depth_factor(
                cf, a_env, q_env, sum(as), sum(as .* qs), sum(as .* qs .^ 2),
                sum(as .* log.(qs)), rv_floor,
            )
        χ_single(cf, rv_floor) = CA.cloud_optical_depth_factor(
            cf, one(FT), FT(1e-4), zero(FT), zero(FT), zero(FT), zero(FT),
            rv_floor,
        )
        # Two homogeneous updrafts (`q_env = 0`, `rv_floor = 0`): the
        # larger of the geometric-mean ratio and the moment-matched Gamma.
        as, qs = FT[0.05, 0.15], FT[1e-3, 4e-3]
        A = sum(as)
        mean_q = sum(as .* qs) / A
        χ_geo = exp(sum(as .* log.(qs)) / A) / mean_q
        rv = sum(as .* qs .^ 2) / A / mean_q^2 - 1
        χ_mom = exp(L(1 / rv))
        @test χ_mix(A, FT(0.8), zero(FT), as, qs, zero(FT)) ≈
              max(χ_geo, χ_mom) rtol = FT(1e-4)
        # Identical homogeneous updrafts give χ = 1.
        @test χ_mix(FT(0.2), FT(0.8), zero(FT), FT[0.1, 0.1], FT[1e-3, 1e-3],
            zero(FT)) ≈ 1 rtol = FT(1e-4)

        rv_floor = FT(0.25)
        # An updraft with trace condensate beside an overcast deck is nearly
        # clear; it barely changes the deck's factor, whereas the geometric
        # mean alone would collapse it.
        a_up, q_trace = FT(0.05), FT(1e-10)
        cf_env = FT(0.9)
        a_env = 1 - a_up
        q_env = cf_env * FT(1e-4)
        cf = a_env * cf_env + a_up
        χ_deck = χ_single(cf_env, rv_floor)
        χ_trace = χ_mix(cf, a_env, q_env, [a_up], [q_trace], rv_floor)
        @test χ_trace > FT(0.85) * χ_deck
        # A small dense updraft core beside a thin environment cloud makes the
        # cell more inhomogeneous than the environment cloud alone, by an
        # amount set by its area (the geometric mean).
        a_core, q_core = FT(0.01), FT(1e-2)
        cf_env = FT(0.5)
        a_env = 1 - a_core
        q_env = cf_env * FT(1e-4)
        cf = a_env * cf_env + a_core
        χ_env = χ_single(cf_env, rv_floor)
        χ_core = χ_mix(cf, a_env, q_env, [a_core], [q_core], rv_floor)
        @test FT(0.2) < χ_core < χ_env
        for args in (
            (FT(0.3), FT(0.9), FT(1e-4), [FT(0.1)], [FT(2e-3)]),
            (FT(0.06), FT(0.94), FT(3e-8), [FT(0.058)], [FT(1e-9)]),
        )
            χ = χ_mix(args..., rv_floor)
            @test χ isa FT
            @test 0 <= χ <= 1
        end
    end
end
