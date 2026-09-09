#=
Unit tests for the prognostic sea-salt hygroscopic growth and gravitational
settling physics in
  src/parameterized_tendencies/aerosols/sea_salt.jl
  src/parameterized_tendencies/aerosols/hygroscopic_growth.jl
  src/parameterized_tendencies/aerosols/settling.jl

These exercise the pure physics functions with the prognostic-aerosol
parameter bundle read straight from ClimaParams (the ssa_* keys must exist
in the ClimaParams release the environment resolves to); the tendency
assembly is integration-tested in sea_salt_subdomain_tests.jl.
=#

using Test
import ClimaAtmos as CA

const FT = Float64

# Parameters straight from ClimaParams, as the model reads them.
const PARAMS = CA.ClimaAtmosParameters(FT; has_prognostic_aerosols = true)
const AP = CA.CAP.prognostic_aerosol_params(PARAMS)
const ρ_s = CA.CAP.prescribed_aerosol_params(PARAMS).seasalt_density  # dry salt density [kg m⁻³]
const R_V = CA.CAP.R_v(PARAMS)
const R_D = FT(287)
const G = FT(9.81)
const NBINS = length(AP.ssa_bin_edges) - 1

@testset "Bin spectrum moments (closed form vs brute force)" begin
    moments = CA.sslt_bin_moments(PARAMS, FT)
    @test length(moments) == NBINS
    @test all(m -> length(m) == CA.SSLT_MAX_MOMENT + 1, moments)
    @test all(m -> all(>(0), m), moments)

    # brute-force check of every cached order against a direct log-radius sum
    modes = CA.sslt_lognormal_modes(AP)
    for i in 1:NBINS
        r̂_lo = AP.ssa_bin_edges[i] / AP.ssa_r_ref
        r̂_hi = AP.ssa_bin_edges[i + 1] / AP.ssa_r_ref
        n = 100_000
        x = range(log(r̂_lo), log(r̂_hi); length = n)
        r̂s = exp.(x)
        w = [CA._ssa_mode_spectrum(r̂, modes) * r̂ for r̂ in r̂s] .* step(x)  # dF/d(lnr̂) · dlnr̂
        for k in 0:CA.SSLT_MAX_MOMENT
            @test CA.sslt_bin_moment(moments[i], k) ≈ sum(w .* r̂s .^ k) rtol = 1e-3
        end
    end
end

@testset "Mass-weighted settling radii" begin
    moments = CA.sslt_bin_moments(PARAMS, FT)
    radii = CA.sslt_settling_radii(moments, PARAMS)
    @test length(radii) == NBINS
    @test issorted(radii)
    for i in 1:NBINS
        # the settling radius of the sub-bin spectrum lies inside the dry bin bounds
        @test AP.ssa_bin_edges[i] < radii[i] < AP.ssa_bin_edges[i + 1]
        M̂5 = CA.sslt_bin_moment(moments[i], 5)
        M̂3 = CA.sslt_bin_moment(moments[i], 3)
        @test radii[i] ≈ AP.ssa_r_ref * sqrt(M̂5 / M̂3)
    end
end

@testset "κ-Köhler and Lewis 33 growth factors" begin
    ξ_κ(rh) = CA.sslt_kappa_kohler_growth_factor(FT(rh), AP)
    @test ξ_κ(0.0) == 1
    @test 1.7 ≤ ξ_κ(0.8) ≤ 1.9
    @test ξ_κ(0.5) < ξ_κ(0.8) < ξ_κ(0.95)
    @test isfinite(ξ_κ(1.0))
    @test ξ_κ(1.0) == ξ_κ(AP.rh_cap)   # clamped above
    ξ_33(rh) = CA.sslt_lewis33_growth_factor(FT(rh), AP)
    @test ξ_33(0.7) < ξ_33(0.9)
    @test ξ_33(1.0) == ξ_33(AP.rh_cap)
end

@testset "Lewis 34 growth factor" begin
    (; lewis_a, lewis_b, σ_w, ρ_water) = AP
    ξ_34(rh, ε) = CA.sslt_lewis34_growth_factor(FT(rh), ε, AP)
    ξ_33(rh) = CA.sslt_lewis33_growth_factor(FT(rh), AP)

    # Kelvin coefficient reproduces Lewis's r_σ,0 ≈ 1.1 nm and Eq. 34.
    T = FT(293.15)
    r_dry = FT(50e-9)
    C = CA.sslt_kelvin_coefficient(r_dry, lewis_a, σ_w, ρ_water, R_V)
    ε = CA.sslt_kelvin_shift(C, T)
    ξ_σ0 = 2 * σ_w / (ρ_water * R_V * T * r_dry)
    @test 1.0e-9 < ξ_σ0 * r_dry < 1.2e-9
    @test ε ≈ (ξ_σ0 / lewis_a)^(3 // 2) rtol = 1e-12
    for rh in (0.5, 0.9, 0.99, 0.999, 1.0)
        ξ_ref = lewis_a * cbrt(lewis_b + 1 / (1 - rh + (ξ_σ0 / lewis_a)^(3 // 2)))
        @test ξ_34(rh, ε) ≈ ξ_ref rtol = 1e-12
    end

    # Reduces to Eq. 33 as the Kelvin shift vanishes, lies below it otherwise;
    # smaller particles grow less; finite and monotone up to RH = 1 without a cap.
    @test ξ_34(0.8, FT(0)) == ξ_33(0.8)
    C_big = CA.sslt_kelvin_coefficient(FT(1e-6), lewis_a, σ_w, ρ_water, R_V)
    ε_big = CA.sslt_kelvin_shift(C_big, T)
    @test ξ_34(0.8, ε_big) ≈ ξ_33(0.8) rtol = 1e-3
    @test ξ_34(0.8, ε) < ξ_34(0.8, ε_big) < ξ_33(0.8)
    @test isfinite(ξ_34(1.0, ε)) && ξ_34(1.05, ε) == ξ_34(1.0, ε)
    @test ξ_34(0.5, ε) < ξ_34(0.9, ε) < ξ_34(1.0, ε)
    @test ξ_34(1.0, ε) ≈ lewis_a * cbrt(lewis_b + (lewis_a / ξ_σ0)^(3 // 2))

    # The consumer-facing growth factor is Lewis 34 with the efflorescence floor.
    @test CA.sslt_growth_factor(FT(0.8), ε, AP) == ξ_34(0.8, ε)
    @test CA.sslt_growth_factor(AP.rh_effl - FT(0.01), ε, AP) == 1

    # Float32 kernel agrees with the Float64 oracle up to saturation.
    AP32 = map(x -> Float32.(x), AP)
    for rh in (0.5f0, 0.99f0, 0.999f0, 1.0f0)
        ξ32 = CA.sslt_growth_factor(rh, CA.sslt_kelvin_shift(Float32(C), 293.15f0), AP32)
        @test ξ32 isa Float32
        @test ξ32 ≈ ξ_34(rh, ε) rtol = 1e-5
    end
end

@testset "Wet density" begin
    for gf in (FT(1.0), FT(1.5), FT(2.0), FT(5.0))
        ρ = CA.sslt_wet_density(ρ_s, AP.ρ_water, gf)
        @test AP.ρ_water ≤ ρ ≤ ρ_s
    end
    @test CA.sslt_wet_density(ρ_s, AP.ρ_water, FT(1)) == ρ_s
    @test CA.sslt_wet_density(ρ_s, AP.ρ_water, FT(1e6)) ≈ AP.ρ_water rtol = 1e-6
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
        CA.sslt_settling_velocity(rw, ρwet, FT(1.2), FT(288), R_D, G, AP)
    @test v(FT(1e-6)) < v(FT(1e-5)) < v(FT(3e-5))   # monotone in wet radius
    @test v(FT(1e-5)) > 0
    @test 1e-3 < v(FT(1e-5)) < 1e-1                 # coarse mode ~ cm/s
    @test v(FT(1e-5), FT(2000)) > v(FT(1e-5), FT(1100))  # denser falls faster
end

@testset "Emission flux tables ↔ mode fit consistency" begin
    # The per-bin emission flux scales in ClimaParams
    # (`ssa_gong_logfit_bin_{0M,3M}_flux`) and the bin moments evaluated at
    # cache construction from `ssa_gong_logfit_mode1..3` were generated from the
    # same fit; anchor the tables to the modes so a partial regeneration of
    # either cannot silently give emission and removal different spectra.
    # Mirrors of the ClimaParams table values (4 significant figures).
    bin_0M_flux = (48.37, 41.92, 5.903, 1.122, 0.03885)
    bin_3M_flux = (1.589e-16, 3.575e-15, 4.557e-14, 1.238e-13, 1.32e-13)
    moments = CA.sslt_bin_moments(PARAMS, FT)
    for i in 1:NBINS
        M̂0 = CA.sslt_bin_moment(moments[i], 0)
        M̂3 = CA.sslt_bin_moment(moments[i], 3)
        @test M̂0 ≈ bin_0M_flux[i] rtol = 1e-3
        @test FT(4π / 3) * ρ_s * AP.ssa_r_ref^3 * M̂3 ≈ bin_3M_flux[i] rtol = 1e-3
    end
end
