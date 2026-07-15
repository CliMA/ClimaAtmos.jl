# Hygroscopic growth and gravitational settling of prognostic sea-salt aerosol.
#
# This file implements the "wet-size seam" of the sea-salt growth/deposition
# plan (docs/sea_salt_growth_deposition_plan.md). Once per explicit-precompute
# step `set_sea_salt_wet_radius!` fills the per-bin wet radius `ᶜsslt_r_wet` in
# `p.precomputed` — the one quantity with several read-only consumers (settling,
# dry deposition, and — later — activation and optics). The wet particle density
# and slip-corrected settling speed are cheap functions of `r_wet`, so they are
# computed where used rather than cached (see `sea_salt.jl`). Because per-bin dry
# radius `r_dry` is a fixed parameter (not derived from the mass field), the
# growth factor depends only on RH and κ, so emission, growth, settling, and
# deposition all use one consistent particle size.
#
# All physics lives here (in ClimaAtmos) rather than in CloudMicrophysics:
# CloudMicrophysics 0.36 ships neither hygroscopic growth nor a slip-corrected
# Stokes velocity, and editing it would be a dependency change. Upstreaming the
# slip-corrected settling and a growth scheme is a documented follow-up.

# --- Tunable module constants (see plan §3, §4, §7) --------------------------
const SEA_SALT_RH_CAP = 0.99
const SEA_SALT_WATER_DENSITY = 1000.0
# Lognormal geometric standard deviation is read from params (`seasalt_std`).

# Sutherland's law for the dynamic viscosity of air μ(T) [Pa s].
const AIR_VISCOSITY_REF = 1.716e-5   # μ at T_ref
const AIR_VISCOSITY_T_REF = 273.15   # reference temperature [K]
const AIR_VISCOSITY_SUTHERLAND = 110.4  # Sutherland constant [K]

# Cunningham slip-correction coefficients: Cc = 1 + Kn(A + B·exp(-C/Kn)).
const CUNNINGHAM_A = 1.257
const CUNNINGHAM_B = 0.4
const CUNNINGHAM_C = 1.1

# Explicit-stability safeguard: the settling speed is capped so that a particle
# falls at most this Courant number of a cell per step. Sea-salt settling is
# applied explicitly (like the grid-mean vertical advection of passive
# tracers), so an uncapped coarse-mode speed near the RH cap could exceed the
# explicit CFL limit at large time steps / coarse near-surface layers. Capping
# conserves mass (excess deposition is spread over a few steps) and keeps the
# scheme robust across configurations.
const SEA_SALT_SETTLING_COURANT_MAX = 0.5

# --- Hygroscopic growth ------------------------------------------------------

"""
    sea_salt_growth_factor(κ, RH, rh_cap)

κ-Köhler diameter/radius growth factor `GF = r_wet / r_dry`, neglecting the
Kelvin term (< 1–2% below the cap; owned by activation):

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
RH = 1 without the κ-Köhler cap sensitivity:

    GF = a · (b + 1 / (1 - RH))^(1/3)

with NaCl `a = 1.08`, `b = 1.10`. Provided as a config-selectable option; the
default scheme is `sea_salt_growth_factor` (single κ shared with activation).
"""
function sea_salt_lewis2008_growth_factor(RH, rh_cap, a, b)
    a_w = min(max(RH, zero(RH)), rh_cap)
    return a * cbrt(b + 1 / (1 - a_w))
end

"""
    sea_salt_wet_density(ρ_s, ρ_w, GF)

Volume-weighted wet-particle density for a dry salt core (density `ρ_s`) coated
with condensed water (density `ρ_w`) at growth factor `GF = r_wet / r_dry`:

    ρ_wet = (ρ_s + (GF³ - 1) · ρ_w) / GF³

which tends to `ρ_s` as `GF → 1` and to `ρ_w` as `GF → ∞`.
"""
function sea_salt_wet_density(ρ_s, ρ_w, GF)
    gf3 = GF^3
    return (ρ_s + (gf3 - 1) * ρ_w) / gf3
end

# --- Gravitational settling velocity -----------------------------------------

"""
    air_dynamic_viscosity(T)

Dynamic viscosity of air μ(T) [Pa s] from Sutherland's law.
"""
function air_dynamic_viscosity(T)
    FT = typeof(T)
    μ_ref = FT(AIR_VISCOSITY_REF)
    T_ref = FT(AIR_VISCOSITY_T_REF)
    S = FT(AIR_VISCOSITY_SUTHERLAND)
    return μ_ref * (T / T_ref)^FT(1.5) * (T_ref + S) / (T + S)
end

"""
    cunningham_slip_correction(Kn)

Cunningham slip-correction factor `Cc(Kn) = 1 + Kn(A + B·exp(-C/Kn))`. `Cc → 1`
for coarse bins (continuum regime) and grows for the fine bins where the
particle size approaches the mean free path.
"""
function cunningham_slip_correction(Kn)
    FT = typeof(Kn)
    return 1 + Kn * (FT(CUNNINGHAM_A) + FT(CUNNINGHAM_B) * exp(-FT(CUNNINGHAM_C) / Kn))
end

"""
    sea_salt_settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, mass_weight)

Slip-corrected Stokes terminal velocity of a wet sea-salt particle
(positive downward) [m s⁻¹]:

    v_g = (2/9) · (ρ_wet - ρ_air) · g · r_wet² · Cc / μ

with the mean free path `λ = μ / (0.499 · ρ_air · v̄)`, `v̄ = √(8 R_d T / π)`,
`Kn = λ / r_wet`, and `Cc` from [`cunningham_slip_correction`](@ref). The
`mass_weight` factor `exp(2·ln²σ)` converts the mean-size velocity to the
mass-weighted mean over the sub-bin lognormal (σ tunable via `seasalt_std`),
matching the Marshall–Palmer mass-weighting used for hydrometeors.
"""
function sea_salt_settling_velocity(r_wet, ρ_wet, ρ_air, T, R_d, grav, mass_weight)
    FT = typeof(r_wet)
    μ = air_dynamic_viscosity(T)
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = μ / (FT(0.499) * ρ_air * v̄)
    Kn = λ / r_wet
    C_c = cunningham_slip_correction(Kn)
    v_g = FT(2 / 9) * (ρ_wet - ρ_air) * grav * r_wet^2 * C_c / μ
    return max(v_g * mass_weight, zero(FT))
end

# --- Precompute --------------------------------------------------------------

"""
    set_sea_salt_wet_radius!(Y, p)

Fill the per-bin wet (deliquesced) radius `ᶜsslt_r_wet` in `p.precomputed` from
the current relative humidity, `r_wet = r_dry · GF(RH, κ)`. No-op unless
interactive sea-salt bins are configured. Reads the grid-mean thermodynamic
state; does **not** modify any prognostic mass (growth sets only the diagnostic
size, never `ρSSLTxx`).

The wet radius is cached because it has several read-only consumers (settling,
dry deposition, and — later — activation and radiation optics). The wet density
and settling speed are deliberately **not** cached: they are cheap functions of
`r_wet` with at most two readers, so they are computed where used (the settling
velocity into scratch in `sea_salt_settling_tendency!`, and the surface values
inline in `sea_salt_dry_deposition_tendency!`).

Called from `set_explicit_precomputed_quantities!` after the emission flux is
set, so the grid-mean `ᶜT`/`ᶜp` are already current.
"""
function set_sea_salt_wet_radius!(Y, p)
    interactive_aerosol_names = _aerosol_names(p.atmos.interactive_aerosols)
    isempty(interactive_aerosol_names) && return

    FT = eltype(Y)
    (; ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; ᶜsslt_r_wet) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    aero_params = p.params.prescribed_aerosol_params

    κ = FT(aero_params.seasalt_kappa)
    rh_cap = FT(SEA_SALT_RH_CAP)
    bin_radii = (
        aero_params.SSLT01_radius,
        aero_params.SSLT02_radius,
        aero_params.SSLT03_radius,
        aero_params.SSLT04_radius,
        aero_params.SSLT05_radius,
    )

    ᶜRH = @. lazy(
        TD.relative_humidity(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
    )
    # The growth factor is bin-independent (depends only on RH and κ), so
    # materialize it once and scale per bin. ᶜtemp_scalar is written and
    # consumed entirely within this call.
    ᶜGF = p.scratch.ᶜtemp_scalar
    @. ᶜGF = sea_salt_growth_factor(κ, ᶜRH, rh_cap)
    for (bin_index, name) in enumerate(interactive_aerosol_names)
        r_dry = FT(bin_radii[bin_index])
        r_wet = getproperty(ᶜsslt_r_wet, name)
        @. r_wet = r_dry * ᶜGF
    end
    return nothing
end

# --- Turbulent dry deposition: Zhang et al. (2001) (plan §5, §7) --------------

# Boltzmann constant [J K⁻¹], for the Brownian diffusivity (Stokes–Einstein).
const BOLTZMANN_CONSTANT = 1.380649e-23

# Zhang et al. (2001) surface-resistance parameters. `ε₀` and the impaction
# exponent `β` are universal; `α` (impaction) and `γ` (Brownian) are land-use
# dependent — the values here are the smooth water/ocean category (Zhang 2001,
# Table 3). TODO: source per-land-use-category α, γ, and the collector radius A
# (for the interception term E_IN, zero over water) from the coupler so land
# points use their own surface characteristics instead of ocean values.
const ZHANG_EPS0 = 3.0
const ZHANG_IMPACTION_BETA = 2.0
const ZHANG_IMPACTION_ALPHA_WATER = 100.0
const ZHANG_BROWNIAN_GAMMA_WATER = 0.56

"""
    sea_salt_dry_deposition_velocity(
        V_g, r_wet, ρ_air, T, z_R, L, z₀, ustar, uf_params, κ_vk, R_d,
    )

Turbulent dry-deposition velocity `V_{d,turb} = 1 / (R_a + R_s)` [m s⁻¹] from
the Zhang et al. (2001) scheme. This carries **only** the turbulent removal; the
gravitational contribution `V_g` is deposited separately by the settling term's
free-outflow bottom boundary, so the two are not double counted (their sum is
the full deposition velocity `V_g + 1/(R_a+R_s)`).

  - `R_a = F_h / (κ · u★)` from the MOST heat-transport dimensionless profile
    `F_h` (aerodynamic resistance), floored at 0 to guard degenerate
    strongly-unstable profiles.
  - `R_s = 1 / [ε₀ · u★ · (E_B + E_IM + E_IN) · R₁]`, the Zhang surface
    resistance, with collection efficiencies for Brownian diffusion
    `E_B = Sc^(-γ)`, impaction `E_IM = (St/(α+St))^β`, interception
    `E_IN` (zero over water), the sticking/rebound correction
    `R₁ = exp(-√St)`, smooth-surface Stokes number `St = V_g·u★²/ν`, and
    `Sc = ν/D_B`, `ν = μ/ρ_air`, `D_B = k_B·T·Cc/(6π μ r_wet)`.

Applied everywhere: for now every surface uses the water/ocean land-use
category (`α`, `γ`), which is exact over ocean and an approximation over land
(TODO: per-land-use-category parameters from the coupler). Returns zero for
calm/degenerate surface states.
"""
function sea_salt_dry_deposition_velocity(
    V_g, r_wet, ρ_air, T, z_R, L, z₀, ustar, uf_params, κ_vk, R_d,
)
    FT = typeof(V_g)
    (ustar <= 0 || r_wet <= 0) && return zero(FT)
    ζ = iszero(L) ? zero(FT) : z_R / L
    F_h = UF.dimensionless_profile(uf_params, z_R, ζ, z₀, UF.HeatTransport())
    # Floor R_a at 0: an unphysical negative aerodynamic resistance in a
    # strongly-unstable profile would otherwise make 1/(R_a+R_s) huge or
    # negative.
    R_a = max(F_h / (κ_vk * ustar), zero(FT))

    μ = air_dynamic_viscosity(T)
    ν = μ / ρ_air
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = μ / (FT(0.499) * ρ_air * v̄)
    C_c = cunningham_slip_correction(λ / r_wet)
    D_B = FT(BOLTZMANN_CONSTANT) * T * C_c / (6 * FT(π) * μ * r_wet)
    Sc = ν / D_B

    # Zhang (2001) surface resistance, smooth-surface (water) land-use category.
    St = V_g * ustar^2 / ν                          # smooth-surface Stokes number
    E_B = Sc^(-FT(ZHANG_BROWNIAN_GAMMA_WATER))      # Brownian diffusion
    α = FT(ZHANG_IMPACTION_ALPHA_WATER)
    E_IM = (St / (α + St))^FT(ZHANG_IMPACTION_BETA) # impaction
    E_IN = zero(FT)                                 # interception (needs collector radius A; ≈0 over water)
    # Rebound/sticking correction. This suppresses the turbulent collection of
    # high-Stokes-number (coarse) particles; over water, where impacting
    # particles stick, it is arguably too aggressive (R₁ → 1 would be more
    # physical), but coarse-mode deposition is dominated by gravitational
    # settling anyway. TODO: consider R₁ = 1 for the water land-use category.
    R_1 = exp(-sqrt(St))
    R_s = 1 / (FT(ZHANG_EPS0) * ustar * (E_B + E_IM + E_IN) * R_1)

    V_d = 1 / (R_a + R_s)
    return max(V_d, zero(FT))
end
