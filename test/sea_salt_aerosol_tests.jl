#=
Unit tests for the prognostic sea-salt hygroscopic growth, gravitational
settling, ocean dry-deposition, activation-seam, and Gong-spectrum-moment
physics in
  src/parameterized_tendencies/aerosols/sea_salt.jl
  src/parameterized_tendencies/aerosols/hygroscopic_growth.jl
  src/parameterized_tendencies/aerosols/sea_salt_activation.jl
(the growth/deposition plan, docs/sea_salt_growth_deposition_plan.md).

These exercise the pure physics functions with an explicit parameter
NamedTuple mirroring the ClimaParams defaults, so they need no parameter
files; the precompute/tendency wiring is integration-tested separately
(prognostic-aerosol config on GPU/server).
=#

using Test
import ClimaAtmos as CA
# Reuse ClimaAtmos's own imports so the test needs no extra direct deps.
const CP = CA.CP
const CMAM = CA.CMAM
const UF = CA.UF

const FT = Float64

# Mirror of the ClimaParams defaults for the prognostic-aerosol bundle.
const AP = (;
    k_B = FT(1.380649e-23),
    ρ_water = FT(1000),
    μ_air_ref = FT(1.716e-5),
    T_μ_ref = FT(273.15),
    S_μ = FT(110.4),
    cunningham_C = (FT(1.257), FT(0.4), FT(1.1)),
    ssa_bin_edges = FT.((0.03e-6, 0.1e-6, 0.5e-6, 1.5e-6, 5.0e-6, 10.0e-6)),
    ssa_r_ref = FT(1e-6),
    r80_per_dry = FT(2),
    gong_theta = FT(30),
    gong_A = (FT(4.7), FT(-0.017), FT(-1.44)),
    gong_B = FT(0.433),
    gong_F = (FT(1.373), FT(0.057), FT(3.45), FT(1.607)),
    gong_dim_factor = FT(1e6),
    gong_wind_exp = FT(3.41),
    jaegle_C = (FT(0.3), FT(0.1), FT(0.0076), FT(0.00021)),
    rh_cap = FT(0.99),
    settling_courant_max = FT(0.5),
    zhang_ε0 = FT(3),
    zhang_β = FT(2),
    zhang_α_water = FT(100),
    zhang_γ_water = FT(0.56),
)
const ρ_s = FT(2170)     # dry salt density [kg m⁻³]
const PARAMS = (;
    prognostic_aerosol_params = AP,
    prescribed_aerosol_params = (; seasalt_density = ρ_s),
)
const κ = FT(1.12)       # hygroscopicity
const R_D = FT(287)
const G = FT(9.81)
const NBINS = length(AP.ssa_bin_edges) - 1

@testset "Gong spectrum moments" begin
    flux_scales = CA.sea_salt_bin_flux_scales(PARAMS, FT)
    masses = CA.sea_salt_particle_masses(PARAMS, FT)
    radii = CA.sea_salt_bin_settling_radii(PARAMS, FT)
    fits = CA.sea_salt_bin_lognormal_fits(PARAMS, FT)
    @test length(flux_scales) == NBINS

    # total number flux at u10 = 9 m/s in the whitecap-derived range
    total = sum(flux_scales) * FT(9)^AP.gong_wind_exp
    @test 1e4 < total < 1e7

    # per-particle mass and settling radius grow monotonically with bin size
    @test issorted(masses) && issorted(radii)
    for i in 1:NBINS
        lo, hi = AP.ssa_bin_edges[i], AP.ssa_bin_edges[i + 1]
        # moment radii of the sub-bin spectrum lie inside the dry bin bounds
        @test lo < radii[i] < hi
        r_g, σ_g = fits[i]
        @test lo < r_g < hi
        @test 1 < σ_g < 3
        # mean particle mass within the single-particle mass bounds of the bin
        vol(r) = FT(4π / 3) * r^3
        @test ρ_s * vol(lo) < masses[i] < ρ_s * vol(hi)
    end

    # the r80 convention matters: collapsing r80 = r_dry changes the scales
    ap_no80 = merge(AP, (; r80_per_dry = FT(1)))
    scales_no80 = CA.sea_salt_bin_flux_scales(
        (; prognostic_aerosol_params = ap_no80), FT,
    )
    @test !(collect(scales_no80) ≈ collect(flux_scales))
end

@testset "Hygroscopic growth factor" begin
    gf(rh) = CA.sea_salt_growth_factor(κ, rh, AP.rh_cap)
    @test gf(0.0) == 1
    @test 1.7 ≤ gf(0.8) ≤ 1.9
    @test gf(0.5) < gf(0.8) < gf(0.95)
    @test isfinite(gf(1.0))
    @test gf(1.0) == gf(AP.rh_cap)   # clamped above
    @test gf(-0.1) == gf(0.0)        # clamped below
    lg(rh) = CA.sea_salt_lewis2008_growth_factor(rh, AP.rh_cap, FT(1.08), FT(1.10))
    @test lg(0.7) < lg(0.9)
    @test isfinite(lg(1.0))
end

@testset "Wet density" begin
    for gf in (FT(1.0), FT(1.5), FT(2.0), FT(5.0))
        ρ = CA.sea_salt_wet_density(ρ_s, AP.ρ_water, gf)
        @test AP.ρ_water ≤ ρ ≤ ρ_s
    end
    @test CA.sea_salt_wet_density(ρ_s, AP.ρ_water, FT(1)) == ρ_s
    @test CA.sea_salt_wet_density(ρ_s, AP.ρ_water, FT(1e6)) ≈ AP.ρ_water rtol = 1e-6
end

@testset "Air viscosity (Sutherland)" begin
    μ288 = CA.air_dynamic_viscosity(FT(288), AP)
    @test 1.7e-5 < μ288 < 1.9e-5                        # ≈ 1.79e-5 Pa s
    @test CA.air_dynamic_viscosity(FT(250), AP) < μ288  # μ increases with T
end

@testset "Cunningham slip correction" begin
    @test CA.cunningham_slip_correction(FT(1e-3), AP) ≈ 1 atol = 2e-3
    @test CA.cunningham_slip_correction(FT(1.0), AP) >
          CA.cunningham_slip_correction(FT(1e-2), AP)
end

@testset "Stokes settling velocity" begin
    v(rw, ρwet = FT(1200)) =
        CA.sea_salt_settling_velocity(rw, ρwet, FT(1.2), FT(288), R_D, G, AP)
    @test v(FT(1e-6)) < v(FT(1e-5)) < v(FT(3e-5))   # monotone in wet radius
    @test v(FT(1e-5)) > 0
    @test 1e-3 < v(FT(1e-5)) < 1e-1                 # coarse mode ~ cm/s
    @test v(FT(1e-5), FT(2000)) > v(FT(1e-5), FT(1100))  # denser falls faster
end

@testset "Zhang dry deposition velocity" begin
    uf_params = UF.GryanikParams(CP.create_toml_dict(FT))
    κ_vk = FT(0.4)
    Vd(vg, rw; T = FT(290), L = FT(-50), u★ = FT(0.3)) =
        CA.sea_salt_dry_deposition_velocity(
            vg, rw, FT(1.2), T, FT(30), L, FT(1e-4), u★, uf_params, κ_vk, R_D, AP,
        )
    Vd_coarse = Vd(FT(0.02), FT(1e-5))
    Vd_fine = Vd(FT(1e-5), FT(1e-7))
    @test Vd_coarse > 0 && isfinite(Vd_coarse) && Vd_coarse < 1
    @test Vd_fine > 0 && isfinite(Vd_fine)
    # calm surface (u★ = 0) => zero
    @test Vd(FT(0.02), FT(1e-5); u★ = FT(0)) == 0
    # strongly-unstable profile: R_a is floored, so V_d stays finite and ≥ 0
    Vd_unstable = Vd(FT(0.02), FT(1e-5); T = FT(300), L = FT(-1), u★ = FT(0.5))
    @test isfinite(Vd_unstable) && Vd_unstable ≥ 0
    # Brownian-regime (fine) particle: higher u★ ⇒ faster turbulent deposition
    @test Vd(FT(1e-5), FT(1e-7); u★ = FT(0.6)) >
          Vd(FT(1e-5), FT(1e-7); u★ = FT(0.2))
end

@testset "Bins → aerosol distribution bridge" begin
    masses_per_particle = CA.sea_salt_particle_masses(PARAMS, FT)
    fits = CA.sea_salt_bin_lognormal_fits(PARAMS, FT)

    # mass ↔ number round-trip is exact and linear in M
    for k in 1:NBINS
        N0 = FT(1e7)
        M = N0 * masses_per_particle[k]
        @test CA.sea_salt_number_concentration(M, masses_per_particle[k]) ≈ N0 rtol = 1e-10
    end
    Na = CA.sea_salt_number_concentration(FT(2e-9), masses_per_particle[1])
    Nb = CA.sea_salt_number_concentration(FT(4e-9), masses_per_particle[1])
    @test Nb ≈ 2Na rtol = 1e-12
    @test CA.sea_salt_number_concentration(FT(-1), masses_per_particle[1]) == 0

    bin_masses = ntuple(k -> FT(1e-9) * k, NBINS)
    dist = CA.bins_to_aerosol_distribution(bin_masses, κ, masses_per_particle, fits)
    @test dist isa CMAM.AerosolDistribution
    @test length(dist.modes) == NBINS
    @test dist.modes[1].kappa == (κ,)
    @test dist.modes[1].r_dry == fits[1][1]
    # zero mass => zero-number mode (no NaN/negative)
    dist0 = CA.bins_to_aerosol_distribution(
        ntuple(_ -> FT(0), NBINS), κ, masses_per_particle, fits,
    )
    @test dist0.modes[1].N == 0
end
