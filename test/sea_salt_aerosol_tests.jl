#=
Unit tests for the prognostic sea-salt hygroscopic growth, gravitational
settling, ocean dry-deposition, and activation-seam physics in
  src/parameterized_tendencies/aerosols/hygroscopic_growth.jl
  src/parameterized_tendencies/aerosols/sea_salt_activation.jl
(the growth/deposition plan, docs/sea_salt_growth_deposition_plan.md).

These exercise the pure physics functions; the precompute/tendency wiring is
integration-tested separately (interactive-aerosol config on GPU/server).
=#

using Test
import ClimaAtmos as CA
import ClimaParams as CP
import CloudMicrophysics.AerosolModel as CMAM
import SurfaceFluxes.UniversalFunctions as UF

const FT = Float64

# Sea-salt reference parameters (ClimaParams defaults).
const ρ_s = FT(2170)     # dry salt density [kg m⁻³]
const ρ_w = FT(1000)     # water density [kg m⁻³]
const κ = FT(1.12)       # hygroscopicity
const σ = FT(1.8)        # geometric std dev
const rh_cap = FT(0.99)
const R_D = FT(287)
const G = FT(9.81)
# dry (number-median) radii of the five MERRA2 sea-salt bins [m]
const R_DRYS = (FT(7.9e-8), FT(3.16e-7), FT(1.119e-6), FT(2.818e-6), FT(7.772e-6))

@testset "Hygroscopic growth factor" begin
    gf(rh) = CA.sea_salt_growth_factor(κ, rh, rh_cap)
    # GF(0.8) in the accepted marine range, GF(0) = 1
    @test gf(0.0) == 1
    @test 1.7 ≤ gf(0.8) ≤ 1.9
    # monotone increasing in RH
    @test gf(0.5) < gf(0.8) < gf(0.95)
    # finite at / above the cap (no a_w → 1 blow-up)
    @test isfinite(gf(1.0))
    @test gf(1.0) == gf(rh_cap)      # clamped
    @test gf(-0.1) == gf(0.0)        # clamped below
    # Lewis 2008 alternative: monotone, larger than κ-Köhler near saturation
    lg(rh) = CA.sea_salt_lewis2008_growth_factor(rh, rh_cap, FT(1.08), FT(1.10))
    @test lg(0.7) < lg(0.9)
    @test isfinite(lg(1.0))
end

@testset "Wet density" begin
    # ρ_wet between water and salt densities, → ρ_s as GF → 1, → ρ_w as GF → ∞
    for gf in (FT(1.0), FT(1.5), FT(2.0), FT(5.0))
        ρ = CA.sea_salt_wet_density(ρ_s, ρ_w, gf)
        @test ρ_w ≤ ρ ≤ ρ_s
    end
    @test CA.sea_salt_wet_density(ρ_s, ρ_w, FT(1)) == ρ_s
    @test CA.sea_salt_wet_density(ρ_s, ρ_w, FT(1e6)) ≈ ρ_w rtol = 1e-6
end

@testset "Air viscosity (Sutherland)" begin
    μ288 = CA.air_dynamic_viscosity(FT(288))
    @test 1.7e-5 < μ288 < 1.9e-5                      # ≈ 1.79e-5 Pa s
    @test CA.air_dynamic_viscosity(FT(250)) < μ288    # μ increases with T
end

@testset "Cunningham slip correction" begin
    # Cc → 1 in the continuum (coarse-mode) limit, grows for fine particles
    @test CA.cunningham_slip_correction(FT(1e-3)) ≈ 1 atol = 2e-3
    @test CA.cunningham_slip_correction(FT(1.0)) > CA.cunningham_slip_correction(FT(1e-2))
end

@testset "Stokes settling velocity" begin
    v(rw, ρwet = FT(1200)) =
        CA.sea_salt_settling_velocity(rw, ρwet, FT(1.2), FT(288), R_D, G, FT(1))
    # monotone increasing in wet radius
    @test v(FT(1e-6)) < v(FT(1e-5)) < v(FT(3e-5))
    # positive and finite; coarse-mode order cm/s
    @test v(FT(1e-5)) > 0
    @test 1e-3 < v(FT(1e-5)) < 1e-1
    # mass-weighting factor scales it up
    vw(rw) = CA.sea_salt_settling_velocity(rw, FT(1200), FT(1.2), FT(288), R_D, G, FT(2))
    @test vw(FT(1e-5)) ≈ 2 * v(FT(1e-5)) rtol = 1e-12
    # denser (drier) particle settles faster at fixed size
    @test v(FT(1e-5), FT(2000)) > v(FT(1e-5), FT(1100))
end

@testset "Zhang dry deposition velocity" begin
    uf_params = UF.GryanikParams(CP.create_toml_dict(FT))
    κ_vk = FT(0.4)
    Vd(vg, rw; T = FT(290), L = FT(-50), u★ = FT(0.3)) =
        CA.sea_salt_dry_deposition_velocity(
            vg, rw, FT(1.2), T, FT(30), L, FT(1e-4), u★, uf_params, κ_vk, R_D,
        )
    Vd_coarse = Vd(FT(0.02), FT(1e-5))
    Vd_fine = Vd(FT(1e-5), FT(1e-7))
    @test Vd_coarse > 0 && isfinite(Vd_coarse) && Vd_coarse < 1
    @test Vd_fine > 0 && isfinite(Vd_fine)
    # calm surface (u★ = 0) => zero
    @test Vd(FT(0.02), FT(1e-5); u★ = FT(0)) == 0
    # strongly-unstable profile: R_a is floored, so V_d stays finite and ≥ 0
    # (no huge/negative deposition velocity from a near-zero R_a + R_s)
    Vd_unstable = Vd(FT(0.02), FT(1e-5); T = FT(300), L = FT(-1), u★ = FT(0.5))
    @test isfinite(Vd_unstable) && Vd_unstable ≥ 0
    # For a Brownian-regime (fine) particle, higher friction velocity ⇒ faster
    # turbulent deposition. (For coarse particles the Zhang rebound term
    # R₁ = exp(-√St) suppresses the turbulent part at high St; total deposition
    # is still dominated by gravitational settling there.)
    @test Vd(FT(1e-5), FT(1e-7); u★ = FT(0.6)) >
          Vd(FT(1e-5), FT(1e-7); u★ = FT(0.2))
end

@testset "Bins → aerosol distribution bridge" begin
    # mass ↔ number round-trip is exact
    for r_dry in R_DRYS
        N0 = FT(1e7)
        v̄ = CA.sea_salt_mean_particle_volume(r_dry, σ)
        M = N0 * ρ_s * v̄
        @test CA.sea_salt_number_concentration(M, ρ_s, r_dry, σ) ≈ N0 rtol = 1e-10
    end
    # N ∝ M (timestep-invariant)
    Na = CA.sea_salt_number_concentration(FT(2e-9), ρ_s, R_DRYS[1], σ)
    Nb = CA.sea_salt_number_concentration(FT(4e-9), ρ_s, R_DRYS[1], σ)
    @test Nb ≈ 2Na rtol = 1e-12
    # negative mass clamps to zero number
    @test CA.sea_salt_number_concentration(FT(-1), ρ_s, R_DRYS[1], σ) == 0

    masses = ntuple(k -> FT(1e-9) * k, 5)
    dist = CA.bins_to_aerosol_distribution(masses, R_DRYS, σ, κ, ρ_s)
    @test dist isa CMAM.AerosolDistribution
    @test length(dist.modes) == 5
    @test dist.modes[1].kappa == (κ,)
    @test dist.modes[1].r_dry == R_DRYS[1]
    # zero mass => zero-number mode (no NaN/negative)
    dist0 = CA.bins_to_aerosol_distribution(ntuple(_ -> FT(0), 5), R_DRYS, σ, κ, ρ_s)
    @test dist0.modes[1].N == 0
end
