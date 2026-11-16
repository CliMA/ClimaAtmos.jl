#=
Unit tests of the EDMF tail-entrainment point functions:
- `environment_tail_shift`: the mean of the top fraction `f` of a Gaussian
  `(T′, q′)` distribution, ranked by a linear selector, matches Monte Carlo
  for positive and negative `T′`–`q′` correlation, and vanishes for `f = 1` and with the variance.
- `tail_entrainment_increments`: zero for fraction 1, the moist-static-energy
  tail for `f < 1`, and the fraction widened to `M / ρa⁰` when one step entrains
  more mass `M` than the top fraction holds.
- `tail_entrainment_fields`: empty (tail off) for fraction 1 and without the
  SGS covariance cache.
=#
using Test
import Random
import Statistics: mean, quantile
import ClimaAtmos as CA
import ClimaParams as CP
import Thermodynamics as TD

const FT = Float64
const thp = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(FT))

# Correlated Gaussian sample with standard deviations σT, σq and correlation r
function sample_Tq(rng, n, σT, σq, r)
    z1, z2 = randn(rng, n), randn(rng, n)
    return σT .* z1, σq .* (r .* z1 .+ sqrt(1 - r^2) .* z2)
end

@testset "environment_tail_shift" begin
    rng = Random.MersenneTwister(1)
    σT, σq = 0.4, 9e-4
    α_T, α_q = TD.Parameters.cp_d(thp), TD.Parameters.LH_v0(thp)
    # r < 0: the moist-static-energy tail is moister but cooler than the mean
    for r in (0.3, -0.6)
        Tp, qp = sample_Tq(rng, 1_000_000, σT, σq, r)
        s = α_T .* Tp .+ α_q .* qp
        for f in (0.5, 0.1)
            top = s .> quantile(s, 1 - f)
            ΔT, Δq = CA.environment_tail_shift(σT^2, σq^2, r, α_T, α_q, f)
            @test ΔT ≈ mean(Tp[top]) rtol = 0.02
            @test Δq ≈ mean(qp[top]) rtol = 0.02
            @test Δq > 0
            @test sign(ΔT) == sign(r)
        end
    end
    @test CA.environment_tail_shift(σT^2, σq^2, 0.3, α_T, α_q, 1.0) == (0.0, 0.0)
    @test CA.environment_tail_shift(0.0, 0.0, 0.3, α_T, α_q, 0.5) == (0.0, 0.0)
end

@testset "tail_entrainment_increments" begin
    T, q_tot, σT, σq, r = 295.0, 0.015, 0.4, 9e-4, 0.3
    α_T, α_q = TD.Parameters.cp_d(thp), TD.Parameters.LH_v0(thp)
    shift(f) = CA.environment_tail_shift(σT^2, σq^2, r, α_T, α_q, f)
    # ρa⁰ = 1, ρaʲ = 0.01, dt = 120: the mass entrained per step is M = 1.2 entr
    args(f, entr) = (thp, f, T, q_tot, 0.0, 0.0, σT^2, σq^2, r, 1.0, 0.01, entr, 120.0)
    entr_for(M) = M / (0.01 * 120)
    # fraction 1: the environment mean
    @test CA.tail_entrainment_increments(args(1.0, 1e-3)...) ==
          (; mse = 0.0, q_tot = 0.0)
    # small entrainment: the moist-static-energy tail of the fraction itself
    for f in (0.5, 0.1)
        tail_f = CA.tail_entrainment_increments(args(f, 1e-3)...)
        @test tail_f.q_tot ≈ shift(f)[2]
        @test tail_f.mse > 0
    end
    # M = 3/4 ρa⁰ exceeds the top half: the step draws the top 3/4
    tail = CA.tail_entrainment_increments(args(0.5, entr_for(0.75))...)
    @test tail.q_tot ≈ shift(0.75)[2]
    @test 0 < tail.q_tot < shift(0.5)[2]
    # the same M leaves a wider fraction unchanged
    @test CA.tail_entrainment_increments(args(0.9, entr_for(0.75))...).q_tot ≈
          shift(0.9)[2]
    # M ≥ ρa⁰: the step would entrain the whole environment, so its mean
    @test CA.tail_entrainment_increments(args(0.5, entr_for(1.5))...) ==
          (; mse = 0.0, q_tot = 0.0)
end

@testset "tail_entrainment_fields" begin
    # Off for fraction 1, and without the SGS covariance cache, where the
    # environment has no subgrid variance and its tail is its mean.
    no_covariances = (; precomputed = (;))
    @test CA.tail_entrainment_fields(no_covariances, 0.5) == ()
    @test CA.tail_entrainment_fields(no_covariances, 1.0) == ()
end
