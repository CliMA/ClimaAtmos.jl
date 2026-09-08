#=
Unit tests for the prognostic sea-salt hygroscopic-growth physics in
  src/parameterized_tendencies/aerosols/hygroscopic_growth.jl

These exercise the pure physics functions with the parameter bundles read
straight from ClimaParams, as the model reads them (the ssa_* keys must exist
in the ClimaParams release the environment resolves to); the precompute
wiring is integration-tested in sea_salt_subdomain_tests.jl.
=#

using Test
import ClimaAtmos as CA

const FT = Float64

const PARAMS = CA.ClimaAtmosParameters(FT; has_prognostic_aerosols = true)
const AP = CA.CAP.prognostic_aerosol_params(PARAMS)
const R_V = CA.CAP.R_v(PARAMS)

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
