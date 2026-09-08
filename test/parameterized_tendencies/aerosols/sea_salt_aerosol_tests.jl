#=
Unit tests for the prognostic sea-salt hygroscopic growth, gravitational
settling, dry-deposition, and below-cloud washout physics in
  src/parameterized_tendencies/aerosols/sea_salt.jl
  src/parameterized_tendencies/aerosols/lognormal_moments.jl
  src/parameterized_tendencies/aerosols/hygroscopic_growth.jl
  src/parameterized_tendencies/aerosols/settling.jl
  src/parameterized_tendencies/aerosols/dry_deposition.jl
  src/parameterized_tendencies/aerosols/wet_deposition.jl

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
const THP = CA.CAP.thermodynamics_params(PARAMS)

# Cell air state at (T, ρ) as the caches build it, for dry air: the mean free
# path and viscosity are humidity-independent, so the tests below that only
# need (μ, λ) pass a dry state.
air_state(T, ρ) =
    CA._aerosol_air_state(THP, T, ρ * R_D * T, FT(0), FT(0), FT(0), ρ, R_D, AP)

const NBINS = length(AP.ssa_bin_edges) - 1
const MAX_MOMENT = 6

@testset "Bin spectrum moments (closed form vs brute force)" begin
    moments = CA.sslt_bin_moments(PARAMS, MAX_MOMENT, FT)
    @test length(moments) == NBINS
    @test all(m -> length(m) == MAX_MOMENT + 1, moments)
    @test all(m -> all(>(0), m), moments)

    # brute-force check of every cached order against a direct log-radius sum
    modes = CA.sslt_lognormal_modes(AP)
    for i in 1:NBINS
        r̂_lo = AP.ssa_bin_edges[i] / AP.aerosol_r_ref
        r̂_hi = AP.ssa_bin_edges[i + 1] / AP.aerosol_r_ref
        n = 100_000
        x = range(log(r̂_lo), log(r̂_hi); length = n)
        r̂s = exp.(x)
        w = [CA.lognormal_mode_spectrum(r̂, modes) * r̂ for r̂ in r̂s] .* step(x)  # dF/d(lnr̂) · dlnr̂
        for k in 0:MAX_MOMENT
            @test CA.spectrum_moment(moments[i], k) ≈ sum(w .* r̂s .^ k) rtol = 1e-3
        end
    end
end

@testset "Mass-weighted settling radii" begin
    moments = CA.sslt_bin_moments(PARAMS, MAX_MOMENT, FT)
    radii = CA.sslt_settling_radii(moments, AP)
    @test length(radii) == NBINS
    @test issorted(radii)
    for i in 1:NBINS
        # the settling radius of the sub-bin spectrum lies inside the dry bin bounds
        @test AP.ssa_bin_edges[i] < radii[i] < AP.ssa_bin_edges[i + 1]
        M̂5 = CA.spectrum_moment(moments[i], 5)
        M̂3 = CA.spectrum_moment(moments[i], 3)
        @test radii[i] ≈ AP.aerosol_r_ref * sqrt(M̂5 / M̂3)
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
        ρ = CA.wet_density(ρ_s, AP.ρ_water, gf)
        @test AP.ρ_water ≤ ρ ≤ ρ_s
    end
    @test CA.wet_density(ρ_s, AP.ρ_water, FT(1)) == ρ_s
    @test CA.wet_density(ρ_s, AP.ρ_water, FT(1e6)) ≈ AP.ρ_water rtol = 1e-6
end

@testset "Air viscosity (Seinfeld & Pandis Eq. 9.7)" begin
    μ288 = CA.air_dynamic_viscosity(FT(288), AP)
    @test 1.7e-5 < μ288 < 1.9e-5                        # ≈ 1.79e-5 Pa s
    @test CA.air_dynamic_viscosity(FT(250), AP) < μ288  # μ increases with T
end

@testset "Air state (Seinfeld & Pandis Eq. 9.6 mean free path)" begin
    T, ρ = FT(288), FT(1.2)
    air = air_state(T, ρ)
    @test air.μ == CA.air_dynamic_viscosity(T, AP)
    # density form ≡ pressure form λ = 2μ / (p √(8 M/(π R T))) with p = ρ R_d T
    p_air = ρ * R_D * T
    @test air.λ ≈ 2 * air.μ / (p_air * sqrt(8 / (FT(π) * R_D * T))) rtol = 1e-12
    @test 5e-8 < air.λ < 8e-8                          # ≈ 0.065 µm at sea level
    @test air_state(T, ρ / 3).λ ≈ 3 * air.λ  # λ ∝ 1/ρ at fixed T
end

@testset "Cunningham slip correction" begin
    @test CA.cunningham_slip_correction(FT(1e-3), AP.cunningham_C) ≈ 1 atol = 2e-3
    @test CA.cunningham_slip_correction(FT(1.0), AP.cunningham_C) >
          CA.cunningham_slip_correction(FT(1e-2), AP.cunningham_C)
end

@testset "Stokes settling velocity" begin
    air = air_state(FT(288), FT(1.2))
    v(rw, ρwet = FT(1200)) =
        CA.settling_velocity(rw, FT(1), ρwet, FT(1.2), air.μ, air.λ, G, AP)
    @test v(FT(1e-6)) < v(FT(1e-5)) < v(FT(3e-5))   # monotone in wet radius
    @test v(FT(1e-5)) > 0
    @test 1e-3 < v(FT(1e-5)) < 1e-1                 # coarse mode ~ cm/s
    @test v(FT(1e-5), FT(2000)) > v(FT(1e-5), FT(1100))  # denser falls faster
end

@testset "Dry deposition velocity" begin
    # the model's own SurfaceFluxes parameter path (sea_salt.jl), not a mirror
    sfp = CA.CAP.surface_fluxes_params(PARAMS)
    uf_params = CA.SFP.uf_params(sfp)
    κ_vk = CA.SFP.von_karman_const(sfp)
    function Vd(vg, rw; T = FT(290), L = FT(-50), u★ = FT(0.3))
        (; μ, λ) = air_state(T, FT(1.2))
        return CA.sslt_dry_deposition_velocity(
            vg, rw, FT(1), FT(1.2), T, μ, λ, FT(30), L, FT(1e-4), u★, uf_params,
            κ_vk, G, AP,
        )
    end
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

@testset "Emission flux tables ↔ mode fit consistency" begin
    # The per-bin emission flux scales in ClimaParams
    # (`ssa_gong_logfit_bin_{0M,3M}_flux`) and the bin moments evaluated at
    # cache construction from `ssa_gong_logfit_mode1..3` were generated from the
    # same fit; anchor the tables to the modes so a partial regeneration of
    # either cannot silently give emission and removal different spectra.
    # Mirrors of the ClimaParams table values (4 significant figures).
    bin_0M_flux = (48.37, 41.92, 5.903, 1.122, 0.03885)
    bin_3M_flux = (1.589e-16, 3.575e-15, 4.557e-14, 1.238e-13, 1.32e-13)
    moments = CA.sslt_bin_moments(PARAMS, MAX_MOMENT, FT)
    for i in 1:NBINS
        M̂0 = CA.spectrum_moment(moments[i], 0)
        M̂3 = CA.spectrum_moment(moments[i], 3)
        @test M̂0 ≈ bin_0M_flux[i] rtol = 1e-3
        @test FT(4π / 3) * ρ_s * AP.aerosol_r_ref^3 * M̂3 ≈ bin_3M_flux[i] rtol = 1e-3
    end
end

@testset "Rain swept-volume collection rate" begin
    toml = CA.CP.create_toml_dict(FT)
    cmp = CA.CM.Parameters.Microphysics1MParams(toml)
    rain = cmp.precip.rain
    vel = cmp.terminal_velocity.rain
    cloud_liquid = CA.CM.Parameters.CloudLiquid(toml)
    ρ_air = FT(1.1)

    # Λ/E is zero without rain and increases with rain water content.
    @test CA.rain_swept_collection_rate(FT(0), ρ_air, rain, vel) == 0
    Λs = map(
        q -> CA.rain_swept_collection_rate(FT(q), ρ_air, rain, vel),
        (1e-6, 1e-5, 1e-4, 1e-3),
    )
    @test all(>(0), Λs) && issorted(collect(Λs))

    for q_rai in (FT(1e-5), FT(1e-4), FT(1e-3))
        Λ = CA.rain_swept_collection_rate(q_rai, ρ_air, rain, vel)

        # Matches the numerical swept-volume integral ∫ a(r)·v(r)·n(r) dr
        # over the same Marshall-Palmer distribution (composite Simpson).
        λ_inv = CA.CM1.lambda_inverse(rain.pdf, rain.mass, q_rai, ρ_air)
        v0 = CA.CM1.get_v0(vel, ρ_air)
        r0 = rain.mass.r0
        integrand(r) =
            rain.pdf.n0 *
            exp(-r / λ_inv) *
            rain.area.a0 *
            rain.area.χa *
            (r / r0)^(rain.area.ae + rain.area.Δa) *
            v0 *
            vel.χv *
            (r / r0)^(vel.ve + vel.Δv)
        N, r_hi = 4000, 60 * λ_inv
        h = r_hi / N
        Λ_num =
            h / 3 * (
                integrand(eps(FT)) +
                integrand(r_hi) +
                sum(j -> (isodd(j) ? 4 : 2) * integrand(j * h), 1:(N - 1))
            )
        @test Λ ≈ Λ_num rtol = 1e-4

        # Reproduces the CloudMicrophysics accretion kernel to rounding:
        # with unit cloud water, accretion = q_clo · E · Λ (the factor
        # ordering differs, so equality holds to a few ulps, not bitwise).
        E = FT(0.8)
        n0 = rain.pdf.n0
        accr = CA.CM1.accretion(
            cloud_liquid, rain, vel, E, one(FT), q_rai, ρ_air, n0, v0, λ_inv,
        )
        @test E * Λ ≈ accr rtol = 8 * eps(FT)
    end

    # The DSD closes Λ ∝ R^((ae+ve+1)/(me+ve+1)) = R^(7/9) analytically
    # (Feng 2007 fits 0.79-0.80 for coarse marine aerosol).
    q_lo, q_hi = FT(1e-5), FT(1e-3)
    R(q) = ρ_air * q * CA.CM1.terminal_velocity(rain, vel, ρ_air, q)
    slope =
        log(
            CA.rain_swept_collection_rate(q_hi, ρ_air, rain, vel) /
            CA.rain_swept_collection_rate(q_lo, ρ_air, rain, vel),
        ) / log(R(q_hi) / R(q_lo))
    @test slope ≈ 7 / 9 rtol = 1e-6
end

@testset "Collection efficiency defaults (Greenfield gap)" begin
    E_coll = AP.ssa_E_coll
    @test length(E_coll) == NBINS
    # Accumulation bins collect far less efficiently than coarse bins,
    # and efficiency is bounded by 1.
    @test E_coll[1] == E_coll[2] < E_coll[3] == E_coll[4] < E_coll[5] <= 1
end

@testset "Power-law washout rate (0M)" begin
    a, b = FT(2.2e-4), FT(0.79)
    @test CA.power_law_washout_rate(FT(0), a, b) == 0
    @test CA.power_law_washout_rate(FT(-1e-12), a, b) == 0   # round-off rain
    @test CA.power_law_washout_rate(FT(1), a, b) ≈ a
    @test CA.power_law_washout_rate(FT(4), a, b) ≈ a * 4^b
    @test CA.power_law_washout_rate(FT(4), a, b) >
          CA.power_law_washout_rate(FT(1), a, b)
    # 1 kg m⁻² s⁻¹ of liquid water is 3600 mm h⁻¹.
    @test CA.precipitation_rate_mm_h(FT(1), FT(1000)) ≈ 3600
    @test CA.precipitation_rate_mm_h(FT(1) / 3600, FT(1000)) ≈ 1

    # Defaults: sub-Greenfield-gap accumulation bins wash out orders of
    # magnitude slower than coarse bins; exponents in the 0.6–0.8 range.
    @test length(AP.ssa_washout_a) == length(AP.ssa_washout_b) == NBINS
    @test AP.ssa_washout_a[1] == AP.ssa_washout_a[2] <
          AP.ssa_washout_a[3] == AP.ssa_washout_a[4] < AP.ssa_washout_a[5]
    @test all(b -> 0.6 <= b <= 0.8, AP.ssa_washout_b)
end
