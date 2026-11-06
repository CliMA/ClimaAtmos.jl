#=
Unit tests of the EDMF updraft mass-supply point functions:
- `environment_tail_shift`: the mean of the upper half of a Gaussian `(T′, q′)`
  distribution along a linear selector matches Monte Carlo, and vanishes with
  the variance.
- `tail_entrainment_increments`: zero for weight 0, the moist-static-energy
  tail for weight 1, and the weight reduced so that one step does not entrain
  more than the tail's mass.
- `buoyancy_selected_fraction`: the top fraction of a standard normal whose
  conditional mean is the target matches Monte Carlo; limits.
- `satinst_source_increments`: the conditional mean of the most buoyant top
  fraction matches Monte Carlo, equals the half-space tail at `f = 1/2`, and
  vanishes as `f → 1`.
- `satinst_source_rate`: zero for `N²_sat ≥ 0`, proportional to `√(−N²_sat)`,
  and capped at half the moved fraction per step.
=#
using Test
import Random
import Statistics: mean
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP

const FT = Float64
const params = CA.ClimaAtmosParameters(FT)
const thp = CAP.thermodynamics_params(params)
const turbconv_params = CAP.turbconv_params(params)

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

@testset "buoyancy_selected_fraction" begin
    z = sort(randn(Random.MersenneTwister(2), 2_000_000); rev = true)
    for x in (0.3, 0.8, 1.5, 3.0)
        f = CA.buoyancy_selected_fraction(x, 1.0)
        @test mean(z[1:max(1, round(Int, f * length(z)))]) ≈ x rtol = 0.02
    end
    @test CA.buoyancy_selected_fraction(-0.1, 1.0) == 1
    @test CA.buoyancy_selected_fraction(0.1, 0.0) == 0
    @test CA.buoyancy_selected_fraction(5.0, 1.0) < 5e-3
end

@testset "satinst_source_increments" begin
    T, q_tot, σT, σq, r = 290.0, 0.012, 0.4, 9e-4, 0.3
    Tp, qp = sample_Tq(Random.MersenneTwister(3), 1_000_000, σT, σq, r)
    α_T, α_q = CA.buoyancy_selector_coefficients(thp, T)
    s = α_T .* Tp .+ α_q .* qp
    increments(f) =
        CA.satinst_source_increments(thp, T, q_tot, 0.0, 0.0, σT^2, σq^2, r, f)
    for f in (0.5, 0.2, 0.05)
        top = s .> sort(s; rev = true)[round(Int, f * length(s))]
        @test increments(f)[2] ≈ mean(qp[top]) rtol = 0.03
    end
    Δq½ = CA.environment_tail_shift(σT^2, σq^2, r, α_T, α_q)[2]
    @test increments(0.5)[2] ≈ Δq½ rtol = 1e-3
    @test abs(increments(1.0)[2]) < 0.01 * abs(Δq½)
end

@testset "satinst_source_rate" begin
    rate(c_s, N²) = CA.satinst_source_rate(c_s, N², 1.0, 0.5, 0.01, turbconv_params, 120.0)
    @test rate(0.1, 1e-5) == 0
    @test rate(0.1, -4e-5) / rate(0.1, -1e-5) ≈ 2
    @test rate(1e3, -4e-5) ≈ 0.5 * 0.5 / 120
end
