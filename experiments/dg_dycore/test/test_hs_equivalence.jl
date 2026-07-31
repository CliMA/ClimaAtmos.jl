# The reused ClimaAtmos Held–Suarez functions must reproduce the ClimaCore
# examples' inline HS block (baroclinic_wave_fddg_fluxform.jl) under the
# parity TOML: same σ = p/p₀ (flat topography), same constants.
@testset "Held–Suarez ≡ examples' block" begin
    m = DG.DGModel(
        BaroclinicWaveFDDG(; helem = 2, zelem = 5, dt = 60.0, perturb = true),
    )
    c = m.c
    Y = DG.initial_state_fddg(m)
    (; eE1, eE2, eE3, eN1, eN2, eN3, ᶜΦ, ccoords) = m.fields
    ρ = Y.c.ρ
    uE = @. (Y.c.ρu1 * eE1 + Y.c.ρu2 * eE2 + Y.c.ρu3 * eE3) / ρ
    uN = @. (Y.c.ρu1 * eN1 + Y.c.ρu2 * eN2 + Y.c.ρu3 * eN3) / ρ
    u1 = @. Y.c.ρu1 / ρ
    u2 = @. Y.c.ρu2 / ρ
    u3 = @. Y.c.ρu3 / ρ
    K = @. (uE^2 + uN^2) / 2
    p = @. DG.pres_ρe(c, Y.c.ρe, K, ᶜΦ, ρ)

    # (a) via the reused ClimaAtmos functions
    dY_a = similar(Y)
    fill!(parent(dY_a.c), 0)
    DG.hs_forcing_fddg!(dY_a.c, ρ, p, u1, u2, u3, m)

    # (b) the examples' inline block (constants as hardcoded there)
    day = 86400.0
    k_a = 1 / (40 * day)
    k_f = 1 / day
    k_s = 1 / (4 * day)
    ΔT_y, Δθ_z, T_equator, T_min, σ_b = 60.0, 10.0, 315.0, 200.0, 7 / 10
    φ = @. deg2rad(ccoords.lat)
    σ = @. p / c.p_0
    hf = @. max(0, (σ - σ_b) / (1 - σ_b))
    ΔρT = @. (k_a + (k_s - k_a) * hf * cos(φ)^4) *
       ρ *
       (
           p / (ρ * c.R_d) - max(
               T_min,
               (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) *
               σ^(c.R_d / c.cp_d),
           )
       )
    dρe_b = @. -ΔρT * c.cv_d
    dρu1_b = @. -k_f * hf * ρ * u1

    rel(a, b) = maximum(abs, parent(a) .- parent(b)) /
                max(maximum(abs, parent(b)), 1e-30)
    @test rel(dY_a.c.ρe, dρe_b) ≤ 1e-11
    @test rel(dY_a.c.ρu1, dρu1_b) ≤ 1e-11
end
