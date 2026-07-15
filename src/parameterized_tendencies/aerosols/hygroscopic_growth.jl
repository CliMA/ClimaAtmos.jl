# Hygroscopic growth and particle-scale physics for prognostic sea salt.
#
# The "wet-size seam" of the growth/deposition plan
# (docs/sea_salt_growth_deposition_plan.md): once per explicit-precompute step,
# `set_sea_salt_growth_factor!` fills the κ-Köhler growth factor
# `ᶜsslt_GF = r_wet / r_dry` in `p.precomputed`. GF is bin-independent (it
# depends only on RH and κ, with no Kelvin term below the RH cap), so one
# cached field serves every consumer, each scaling its own dry moment radius:
# settling and dry deposition the mass-flux-weighted settling radius
# (`sea_salt_bin_settling_radii`), the activation seam the lognormal fit, and
# (later) optics the effective radius. Wet density and settling speed are
# cheap functions of GF with few readers, so they are computed where used.
#
# All physics lives here (in ClimaAtmos) rather than in CloudMicrophysics,
# which ships neither hygroscopic growth nor a slip-corrected Stokes velocity;
# upstreaming both is a documented follow-up. All constants come from
# ClimaParams via `params.prognostic_aerosol_params` (`ap`); pointwise
# functions take `ap` and are broadcast with the `(ap,)` 1-tuple idiom.

"""
    sea_salt_growth_factor(κ, RH, rh_cap)

κ-Köhler radius growth factor `GF = r_wet / r_dry`, neglecting the Kelvin term
(< 1–2% below the cap; owned by activation):

    GF = (1 + κ · a_w / (1 - a_w))^(1/3),   a_w = clamp(RH, 0, rh_cap)

`GF(0.8) ≈ 1.76` for sea salt (κ ≈ 1.12), monotonically increasing in RH.
"""
function sea_salt_growth_factor(κ, RH, rh_cap)
    a_w = min(max(RH, zero(RH)), rh_cap)
    return cbrt(1 + κ * a_w / (1 - a_w))
end

"""
    sea_salt_lewis2008_growth_factor(RH, rh_cap, a, b)

Alternative near-saturation growth factor (Lewis 2008, Eq. 33), accurate up to
RH = 1 without the κ-Köhler cap sensitivity: `GF = a · (b + 1/(1-RH))^(1/3)`,
with NaCl `a = 1.08`, `b = 1.10`. Config-selectable option; the default is
[`sea_salt_growth_factor`](@ref) (single κ shared with activation).
"""
function sea_salt_lewis2008_growth_factor(RH, rh_cap, a, b)
    a_w = min(max(RH, zero(RH)), rh_cap)
    return a * cbrt(b + 1 / (1 - a_w))
end

"""
    sea_salt_wet_density(ρ_s, ρ_w, GF)

Volume-weighted wet-particle density for a dry salt core (density `ρ_s`)
coated with condensed water (density `ρ_w`):
`ρ_wet = (ρ_s + (GF³ - 1) · ρ_w) / GF³`, tending to `ρ_s` as `GF → 1` and to
`ρ_w` as `GF → ∞`.
"""
function sea_salt_wet_density(ρ_s, ρ_w, GF)
    gf3 = GF^3
    return (ρ_s + (gf3 - 1) * ρ_w) / gf3
end

"""
    air_dynamic_viscosity(T, ap)

Dynamic viscosity of air μ(T) (Pa s) from Sutherland's law, with the reference
viscosity/temperature and Sutherland constant from ClimaParams
(`μ_air_ref`, `T_μ_ref`, `S_μ`).
"""
function air_dynamic_viscosity(T, ap)
    (; μ_air_ref, T_μ_ref, S_μ) = ap
    FT = typeof(T)
    return μ_air_ref * (T / T_μ_ref)^FT(1.5) * (T_μ_ref + S_μ) / (T + S_μ)
end

"""
    cunningham_slip_correction(Kn, ap)

Cunningham slip-correction factor `Cc(Kn) = 1 + Kn(A + B·exp(-C/Kn))` with
coefficients `ap.cunningham_C = [A, B, C]`. `Cc → 1` for the coarse bins
(continuum regime) and grows for fine bins where the particle size approaches
the mean free path.
"""
function cunningham_slip_correction(Kn, ap)
    C = ap.cunningham_C
    return 1 + Kn * (C[1] + C[2] * exp(-C[3] / Kn))
end

"""
    sea_salt_settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)

Slip-corrected Stokes terminal velocity of a wet sea salt particle (positive
downward, m s⁻¹): `v_g = (2/9)·(ρ_wet - ρ_air)·g·r_wet²·Cc/μ`, with the mean
free path `λ = μ/(0.499·ρ_air·v̄)`, `v̄ = √(8 R_d T/π)`, `Kn = λ/r_wet`, and
`Cc` from [`cunningham_slip_correction`](@ref).

RADIUS BASIS: `r_wet` must be the moment radius of the flux being computed —
for the dry-**mass** tracers, the wet mass-flux-weighted radius
`GF(RH) · √(⟨r_dry⁵⟩/⟨r_dry³⟩)` ([`sea_salt_bin_settling_radii`](@ref)), whose
Stokes speed carries the bin's settling mass flux (v ∝ r², mass ∝ r³).
"""
function sea_salt_settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, ap)
    FT = typeof(r_wet)
    μ = air_dynamic_viscosity(T, ap)
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = μ / (FT(0.499) * ρ_air * v̄)
    C_c = cunningham_slip_correction(λ / r_wet, ap)
    v_g = FT(2 / 9) * (ρ_wet - ρ_air) * grav * r_wet^2 * C_c / μ
    return max(v_g, zero(FT))
end

"""
    set_sea_salt_growth_factor!(Y, p)

Fill the κ-Köhler growth factor `ᶜsslt_GF = r_wet / r_dry` in `p.precomputed`
from the current relative humidity. No-op unless sea salt is prognostic.
Reads the grid-mean thermodynamic state and sets only the diagnostic size —
never the prognostic mass. Called from `set_explicit_precomputed_quantities!`
once the grid-mean `ᶜT`/`ᶜp` are current.
"""
set_sea_salt_growth_factor!(Y, p) =
    set_sea_salt_growth_factor!(Y, p, p.atmos.seasalt)
set_sea_salt_growth_factor!(Y, p, ::Union{Nothing, PrescribedSeaSalt}) =
    nothing
function set_sea_salt_growth_factor!(Y, p, ::PrognosticSeaSalt)
    (; ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜsslt_GF) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    κ = p.params.prescribed_aerosol_params.seasalt_kappa
    rh_cap = p.params.prognostic_aerosol_params.rh_cap

    @. ᶜsslt_GF = sea_salt_growth_factor(
        κ,
        TD.relative_humidity(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
        rh_cap,
    )
    return nothing
end

"""
    sea_salt_dry_deposition_velocity(
        V_g, r_wet, ρ_air, T, z_R, L, z₀, ustar, uf_params, κ_vk, R_d, ap,
    )

Turbulent dry-deposition velocity `V_d,turb = 1/(R_a + R_s)` (m s⁻¹), Zhang
et al. (2001). Carries **only** the turbulent removal — the gravitational
contribution `V_g` is deposited by the settling term's free-outflow boundary,
so the two sum to the full deposition velocity without double counting.

  - `R_a = F_h / (κ_vk · u★)` from the MOST heat-transport dimensionless
    profile at reference height `z_R` (floored at 0 to guard degenerate
    strongly-unstable profiles).
  - `R_s = 1 / [ε₀ · u★ · (E_B + E_IM + E_IN) · R₁]` with Brownian collection
    `E_B = Sc^(-γ)`, impaction `E_IM = (St/(α+St))^β`, interception
    `E_IN = 0` over water, rebound `R₁ = exp(-√St)`, smooth-surface Stokes
    number `St = V_g·u★²/ν`, and `Sc = ν/D_B`,
    `D_B = k_B·T·Cc/(6π·μ·r_wet)` (Stokes–Einstein).

Every surface currently uses the water/ocean land-use category
(`zhang_α_water`, `zhang_γ_water`) — exact over ocean, an approximation over
land (TODO: per-land-use parameters from the coupler). Zero for
calm/degenerate surface states. RADIUS BASIS: `r_wet` is the bin's wet
mass-flux-weighted radius, the same working radius as the settling term.
"""
function sea_salt_dry_deposition_velocity(
    V_g,
    r_wet,
    ρ_air,
    T,
    z_R,
    L,
    z₀,
    ustar,
    uf_params,
    κ_vk,
    R_d,
    ap,
)
    FT = typeof(V_g)
    (ustar <= 0 || r_wet <= 0) && return zero(FT)
    ζ = iszero(L) ? zero(FT) : z_R / L
    F_h = UF.dimensionless_profile(uf_params, z_R, ζ, z₀, UF.HeatTransport())
    R_a = max(F_h / (κ_vk * ustar), zero(FT))

    μ = air_dynamic_viscosity(T, ap)
    ν = μ / ρ_air
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = μ / (FT(0.499) * ρ_air * v̄)
    C_c = cunningham_slip_correction(λ / r_wet, ap)
    D_B = ap.k_B * T * C_c / (6 * FT(π) * μ * r_wet)
    Sc = ν / D_B

    St = V_g * ustar^2 / ν
    E_B = Sc^(-ap.zhang_γ_water)
    E_IM = (St / (ap.zhang_α_water + St))^ap.zhang_β
    # Interception needs a collector radius; ≈ 0 over water. The rebound
    # correction R₁ suppresses high-St collection; over water (sticky) it is
    # arguably too aggressive, but coarse-mode deposition is settling-dominated
    # anyway (TODO: R₁ = 1 for the water category).
    R_1 = exp(-sqrt(St))
    R_s = 1 / (ap.zhang_ε0 * ustar * (E_B + E_IM) * R_1)

    return max(1 / (R_a + R_s), zero(FT))
end
