#=
Unit tests of the EDMF updraft mass-supply point functions:
- `environment_tail_shift`: the mean of the upper half of a Gaussian `(T′, q′)`
  distribution along a linear selector matches Monte Carlo, and vanishes with
  the variance.
- `tail_entrainment_increments`: zero for weight 0, the moist-static-energy
  tail for weight 1, and the weight reduced so that one step does not entrain
  more than the tail's mass.
=#
using Test
import Random
import Statistics: mean
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
    σT, σq, r = 0.4, 9e-4, 0.3
    Tp, qp = sample_Tq(rng, 1_000_000, σT, σq, r)
    α_T, α_q = CA.mse_selector_coefficients(thp)
    upper = α_T .* Tp .+ α_q .* qp .> 0
    ΔT, Δq = CA.environment_tail_shift(σT^2, σq^2, r, α_T, α_q)
    @test ΔT ≈ mean(Tp[upper]) rtol = 0.02
    @test Δq ≈ mean(qp[upper]) rtol = 0.02
    @test CA.environment_tail_shift(0.0, 0.0, r, α_T, α_q) == (0.0, 0.0)
end

@testset "tail_entrainment_increments" begin
    T, q_tot, σT, σq, r = 295.0, 0.015, 0.4, 9e-4, 0.3
    args(w, entr) = (thp, w, T, q_tot, 0.0, 0.0, σT^2, σq^2, r, 1.0, 0.01, entr, 120.0)
    # weight 0: the environment mean
    @test CA.tail_entrainment_increments(args(0.0, 1e-3)...) == (0.0, 0.0)
    # weight 1, small entrainment: the full moist-static-energy tail
    α_T, α_q = CA.mse_selector_coefficients(thp)
    ΔT, Δq = CA.environment_tail_shift(σT^2, σq^2, r, α_T, α_q)
    Δmse, Δq_tot = CA.tail_entrainment_increments(args(1.0, 1e-3)...)
    @test Δq_tot ≈ Δq
    @test Δmse > 0
    # entrained mass per step entr ρaʲ dt = 2 ρa⁰ = 4 × the tail's mass: weight / 4
    Δmse4, Δq_tot4 = CA.tail_entrainment_increments(args(1.0, 2 / (0.01 * 120))...)
    @test Δq_tot4 ≈ Δq / 4
    @test Δmse4 ≈ Δmse / 4 rtol = 1e-3
end
