import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

"""
    rain_swept_collection_rate(q_rai, ρ, rain, vel)

First-order collection rate `Λ/E` [s⁻¹] of the Marshall–Palmer rain
population sweeping through unit-efficiency collectors — the swept-volume
integral `∫ a(r)·v(r)·n(r) dr` in the same closed form as `CM1.accretion`
(which evaluates `q_clo · E` times this expression), so that below-cloud
aerosol scavenging, in-cloud accretion, and the rain rate all derive from
one size distribution:

    Λ/E = n₀ · a₀χₐ · v₀χᵥ · λ⁻¹ · Γ(ae+ve+Δa+Δv+1) · (λ⁻¹/r₀)^(ae+ve+Δa+Δv)

with `λ⁻¹` from the exported `CM1.lambda_inverse` and the Γ factor
precomputed as `vel.gamma_accr`. Multiplying by a per-particle collection
efficiency `E` gives the below-cloud scavenging coefficient; with the
ClimaParams-default geometric area parameters this is exactly
`E · ∫ (π/4)D²·v(D)·n(D) dD`. Zero when `q_rai` is negligible, mirroring the
`ϵ_numerics` gate of the accretion kernel.
"""
function rain_swept_collection_rate(q_rai, ρ, rain, vel)
    FT = typeof(q_rai)
    λ_inv = CM1.lambda_inverse(rain.pdf, rain.mass, q_rai, ρ)
    n0 = rain.pdf.n0
    v0 = CM1.get_v0(vel, ρ)
    (; r0) = rain.mass
    (; a0, ae, χa, Δa) = rain.area
    (; χv, ve, Δv, gamma_accr) = vel
    rate =
        n0 * a0 * v0 * χa * χv * λ_inv * gamma_accr /
        (r0 / λ_inv)^(ae + ve + Δa + Δv)
    return ifelse(q_rai > ϵ_numerics(FT), rate, zero(FT))
end

"""
    rain_representative_drop_radius(q_rai, ρ, rain, vel)

Radius `r_D` [m] of the single raindrop that stands in for the whole
Marshall–Palmer population when the collection efficiency is evaluated (one
drop per cell, not a size-distribution integral of `E`): the swept-volume
weighted mean radius

    r_D = ∫ r·a(r)·v(r)·n(r) dr / ∫ a(r)·v(r)·n(r) dr = (k + 1) · λ⁻¹,
    k = ae + ve + Δa + Δv,

i.e. `Γ(k+2)/Γ(k+1) = k + 1` for the exponential `n(r) = n₀·exp(−r/λ⁻¹)`,
with the same area and fall-speed power laws as
[`rain_swept_collection_rate`](@ref) so the drop is the one whose sweeping
that rate already counts. Zero when `q_rai` is below the `ϵ_numerics` gate
of that rate.
"""
function rain_representative_drop_radius(q_rai, ρ, rain, vel)
    FT = typeof(q_rai)
    λ_inv = CM1.lambda_inverse(rain.pdf, rain.mass, q_rai, ρ)
    (; ae, Δa) = rain.area
    (; ve, Δv) = vel
    r_D = (ae + ve + Δa + Δv + 1) * λ_inv
    return ifelse(q_rai > ϵ_numerics(FT), r_D, zero(FT))
end

"""
    rain_drop_fall_speed(r_D, ρ, rain, vel)

Fall speed `V_D = χᵥ·v₀(ρ)·(r_D/r₀)^(ve+Δv)` [m s⁻¹] of a raindrop of radius
`r_D`, the same power-law velocity of the 1-moment scheme that the swept
collection rate and the rain sedimentation use. Zero for `r_D = 0` (no rain).
"""
function rain_drop_fall_speed(r_D, ρ, rain, vel)
    FT = typeof(r_D)
    v0 = CM1.get_v0(vel, ρ)
    (; r0) = rain.mass
    (; χv, ve, Δv) = vel
    return ifelse(r_D > zero(FT), χv * v0 * (r_D / r0)^(ve + Δv), zero(FT))
end

"""
    rain_representative_drop(q_rai, ρ, rain, vel)

`(r_D, V_D)` of the representative raindrop,
[`rain_representative_drop_radius`](@ref) and [`rain_drop_fall_speed`](@ref)
together; the drivers call the two separately so each can be hoisted into
its own scratch field.
"""
function rain_representative_drop(q_rai, ρ, rain, vel)
    r_D = rain_representative_drop_radius(q_rai, ρ, rain, vel)
    return (r_D, rain_drop_fall_speed(r_D, ρ, rain, vel))
end

"""
    slinn_collection_efficiency(
        r_wet, ρ_wet, V_g, D_B, r_D, V_D, ρ_air, μ_air, grav, ap,
    )

Raindrop–aerosol collection efficiency `E ∈ [0, 1]` of Slinn (1984) as
reproduced in Feng (2007, Atmos. Environ., doi:10.1016/j.atmosenv.2007.04.046),
for a wet particle of radius `r_wet`, density `ρ_wet`, settling speed `V_g`,
and Brownian diffusivity `D_B` collected by one drop of radius `r_D` falling
at `V_D` through air of density `ρ_air` and viscosity `μ_air`:

    E = E_Br + E_int + E_imp

    E_Br  = 4 / (Re·Sc) · (1 + C₁·Re^½·Sc^⅓ + C₂·Re^½·Sc^½)        Brownian diffusion
    E_int = 4·φ · (1/ω + (1 + 2·Re^½)·φ)                            interception
    E_imp = ((St − S*) / (St − S* + 2/3))^(3/2) · (ρ_w/ρ_wet)^½    inertial impaction, St > S* only

with the radius-based drop Reynolds number `Re = r_D·V_D·ρ_air/μ_air`
(Slinn's convention, `D·V_D·ρ_air/(2μ_air)`), the particle Schmidt number
`Sc = μ_air/(ρ_air·D_B)`, the size ratio `φ = r_wet/r_D`, the viscosity ratio
`ω = μ_water/μ_air`, the Stokes number `St = τ·(V_D − V_g)/r_D` with the
relaxation time `τ = V_g/g`, and the critical Stokes number
`S* = (a + b·ln(1+Re)) / (1 + ln(1+Re))`. `C₁, C₂` are `ap.slinn_C_B`,
`a, b` are `ap.slinn_S_star_C`, the impaction exponent is
`ap.slinn_impaction_exp`, and `μ_water` is `ap.μ_water`.

The drop is the single representative drop of
[`rain_representative_drop`](@ref) rather than an integral of `E` over the
size distribution, so `E` multiplies the swept-volume rate of the whole
population. Zero where there is no drop (`r_D = 0` or `V_D = 0`); the
denominators are guarded so the discarded branch stays finite. Kernel safe:
no allocations, `ifelse` for the impaction branch.
"""
function slinn_collection_efficiency(
    r_wet,
    ρ_wet,
    V_g,
    D_B,
    r_D,
    V_D,
    ρ_air,
    μ_air,
    grav,
    ap,
)
    FT = typeof(r_wet)
    has_drop = (r_D > zero(FT)) & (V_D > zero(FT))
    # Evaluate on a unit drop where there is none, so every denominator below
    # is finite, then discard that value.
    r_D⁺ = ifelse(has_drop, r_D, one(FT))
    V_D⁺ = ifelse(has_drop, V_D, one(FT))

    Re = r_D⁺ * V_D⁺ * ρ_air / μ_air
    sqrt_Re = sqrt(Re)
    Sc = μ_air / (ρ_air * D_B)
    C_B = ap.slinn_C_B
    E_brownian =
        4 / (Re * Sc) *
        (1 + C_B[1] * sqrt_Re * cbrt(Sc) + C_B[2] * sqrt_Re * sqrt(Sc))

    φ = r_wet / r_D⁺
    ω = ap.μ_water / μ_air
    E_interception = 4 * φ * (1 / ω + (1 + 2 * sqrt_Re) * φ)

    τ = V_g / grav
    St = τ * (V_D⁺ - V_g) / r_D⁺
    log_Re = log1p(Re)
    C_S = ap.slinn_S_star_C
    S_star = (C_S[1] + C_S[2] * log_Re) / (1 + log_Re)
    # The floor keeps the base of the power non-negative in the discarded
    # `St ≤ S*` branch (a negative base to a fractional power is NaN). The
    # 2/3 here, like the 4s of E_Br and E_int above, is a structural constant
    # of Slinn's functional form and stays a literal; the fitted coefficients
    # (C_B, S* coefficients, impaction exponent) are the ClimaParams keys.
    ΔSt = max(St - S_star, zero(FT))
    E_impaction = ifelse(
        St > S_star,
        (ΔSt / (ΔSt + FT(2 / 3)))^ap.slinn_impaction_exp *
        sqrt(ap.ρ_water / ρ_wet),
        zero(FT),
    )

    E = clamp(E_brownian + E_interception + E_impaction, zero(FT), one(FT))
    return ifelse(has_drop, E, zero(FT))
end

"""
    sslt_bin_washout_rate(
        air, T, r_dry, C_kelvin, ρ_s, ρ_air, r_D, V_D, Λ, grav, ap,
    )

Below-cloud washout rate `k = E · Λ` [s⁻¹] of one bin at its dry settling
radius `r_dry`, given the cell's bin-independent air state `air = (; RH, μ, λ)`,
the representative raindrop `(r_D, V_D)` of the same subdomain, and the
unit-efficiency swept collection rate `Λ` ([`rain_swept_collection_rate`](@ref)).
The growth factor, wet density, and settling speed come from
[`bin_wet_particle`](@ref) (the chain the settling and dry-deposition terms
evaluate), the diffusivity from [`brownian_diffusivity`](@ref) at the wet
radius, and `E` from [`slinn_collection_efficiency`](@ref).
"""
function sslt_bin_washout_rate(
    air,
    T,
    r_dry,
    C_kelvin,
    ρ_s,
    ρ_air,
    r_D,
    V_D,
    Λ,
    grav,
    ap,
)
    (; μ, λ) = air
    ξ, ρ_wet, V_g =
        bin_wet_particle(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, ap)
    r_wet = r_dry * ξ
    C_c = cunningham_slip_correction(λ / r_wet, ap.cunningham_C)
    D_B = brownian_diffusivity(T, C_c, μ, r_wet, ap.k_B)
    E = slinn_collection_efficiency(
        r_wet,
        ρ_wet,
        V_g,
        D_B,
        r_D,
        V_D,
        ρ_air,
        μ,
        grav,
        ap,
    )
    return E * Λ
end

"""
    rain_mixing_ratio_from_flux(P, ρ, rain, vel::CMP.Blk1MVelTypeRain)

Rain mixing ratio `q_rai` [kg kg⁻¹] whose 1-moment Marshall–Palmer
population carries the downward rain mass flux `P` [kg m⁻² s⁻¹] through air
of density `ρ`, i.e. the inverse of `P = ρ · q · v_term(ρ, q)` with
`v_term = CM1.terminal_velocity(rain, vel, ρ, q)`. Lets the 0-moment
precipitation shadow ([`set_sslt_precipitation_shadow!`](@ref)), which has a
rain flux but no rain state, drive the same washout chain as the 1-moment
scheme.

With the power-law fall speed `v ∝ r^(ve+Δv)` and mass `m ∝ r^(me+Δm)`,
`λ⁻¹ ∝ (ρq)^(1/(me+Δm+1))` and the mass-weighted fall speed scales as
`λ⁻¹^(ve+Δv)`, so at fixed `ρ` the flux is an exact power law `P = C(ρ) · qⁿ`,

    n = 1 + (ve + Δv) / (me + Δm + 1),

and the inverse is closed form against a reference state evaluated with the
same `terminal_velocity` call:

    q = q_ref · (P / P_ref)^(1/n),   P_ref = ρ · q_ref · v_term(ρ, q_ref).

`q_ref = 10⁻³` (a typical rain mixing ratio) only anchors the power law;
the result does not depend on it beyond round-off. Zero for
`P ≤ ϵ_numerics` (no rain, or a negative round-off flux from the shadow's
partial integral).
"""
function rain_mixing_ratio_from_flux(P, ρ, rain, vel::CMP.Blk1MVelTypeRain)
    FT = typeof(P)
    q_ref = FT(1e-3)
    (; me, Δm) = rain.mass
    (; ve, Δv) = vel
    n = 1 + (ve + Δv) / (me + Δm + 1)
    P_ref = ρ * q_ref * CM1.terminal_velocity(rain, vel, ρ, q_ref)
    q = q_ref * (max(P, zero(FT)) / P_ref)^(1 / n)
    return ifelse(P > ϵ_numerics(FT), q, zero(FT))
end
rain_mixing_ratio_from_flux(P, ρ, rain, vel) = error(
    "rain_mixing_ratio_from_flux inverts the Blk1MVelTypeRain power-law rain \
     flux in closed form, but got $(typeof(vel)). The sea-salt washout chain \
     (rain_swept_collection_rate, rain_representative_drop_radius, \
     rain_drop_fall_speed) already assumes the Blk1M rain velocity.",
)

"""
    set_sslt_precipitation_shadow!(ᶜP, Y, p)

Fill `ᶜP` [kg m⁻² s⁻¹] with the downward flux of rain through each
cell under 0-moment microphysics, which removes precipitation from the
column the instant it forms and so carries no rain state. The flux through
height `z` is the rain formed above it,

    P(z) = ∫_z^{z_top} max(0, −ρ ∂q_tot/∂t)|_{T ≥ T_freeze} dz′
         = −surface_rain_flux − ∫_0^z (…) dz′,

built from the same per-cell integrand as `set_precipitation_surface_fluxes!`,
whose `surface_rain_flux` (the whole column integral, upward-positive) must
therefore already be current. Because the two share one integrand the partial
integral is exact, so no clamp is needed here; a negative rain flux from a
positive sink washes nothing out through the `P ≤ 0` gate of
[`rain_mixing_ratio_from_flux`](@ref). The face flux is interpolated to cell
centers.
"""
function set_sslt_precipitation_shadow!(ᶜP, Y, p)
    FT = eltype(Y)
    thp = CAP.thermodynamics_params(p.params)
    T_freeze = TD.Parameters.T_freeze(thp)
    (; ᶜT, ᶜρ_dq_tot_dt, surface_rain_flux) = p.precomputed
    ᶠ∫_0_z = p.scratch.ᶠtemp_scalar
    ᶜrain_formation =
        @. lazy(ifelse(ᶜT >= T_freeze, -(ᶜρ_dq_tot_dt), zero(FT)))
    Operators.column_integral_indefinite!(ᶠ∫_0_z, ᶜrain_formation)
    @. ᶠ∫_0_z = -surface_rain_flux - ᶠ∫_0_z
    @. ᶜP = ᶜinterp(ᶠ∫_0_z)
    return nothing
end

"""
    WetDepositionMicrophysics

The microphysics models under which sea salt wet removal is active and the
rate cache is populated: `NonEquilibriumMicrophysics1M` (rain size
distribution available) and `EquilibriumMicrophysics0M` (in-cloud driver from
the 0M sink and equilibrium condensate; washout from the precipitation shadow).
Every other model leaves the rates unread and wet deposition off.
"""
const WetDepositionMicrophysics =
    Union{EquilibriumMicrophysics0M, NonEquilibriumMicrophysics1M}

"""
    cloud_precip_formation_rate(cmp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno)

Rate at which cloud condensate of either phase is converted to precipitation
[kg kg⁻¹ s⁻¹, ≥ 0] on the given (subdomain) state: every sink of `q_lcl` and
`q_icl` that lands in rain or snow, from a dedicated `InstantaneousVerbose`
microphysics evaluation —

    S_acnv_lcl_rai + S_accr_lcl_rai + S_accr_lcl_sno_warm + S_accr_lcl_sno_cold

  - S_acnv_icl_sno + S_accr_icl_rai + S_accr_icl_sno

Using individual process terms rather than net `dq_*_dt` tendencies keeps
condensation/evaporation, melt/freeze bookkeeping, and rain evaporation out of
the scavenging driver; `S_melt_icl_lcl` is excluded because it moves condensate
within the cloud rather than into precipitation. Pairing this with the
all-condensate divisor `q_lcl + q_icl` means a glaciated subdomain is
scavenged at the rate its ice converts to snow, and matches the phase-blind
cloud indicator and the 0-moment driver. Unused source terms are
dead-code-eliminated inside the broadcast, as in the `mp1m_` diagnostics.
"""
@inline function cloud_precip_formation_rate(
    cmp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
)
    S = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(), BMT.Microphysics1Moment(),
        cmp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
    )
    formation =
        S.S_acnv_lcl_rai + S.S_accr_lcl_rai +
        S.S_accr_lcl_sno_warm + S.S_accr_lcl_sno_cold +
        S.S_acnv_icl_sno + S.S_accr_icl_rai + S.S_accr_icl_sno
    return max(formation, zero(formation))
end

"""
    sslt_in_cloud_scavenging_rate(F, f_act, Q, q_cld, dt)

In-cloud (nucleation) scavenging rate `F · f_act · min(Q/q_cld, 1/dt)` [s⁻¹]
of one aerosol bin. Within the precipitating-area proxy `F`, the activated
fraction `f_act` of the aerosol is dissolved in cloud condensate and leaves at
the intensive rate at which that condensate converts to precipitation,
`c₁ = Q/q_cld` (area dilution cancels in the ratio because `Q` and `q_cld` are
diluted alike). Neither driver bounds `Q` against the condensate reservoir —
`InstantaneousVerbose` applies no timestep limiter, and the 0-moment limiter
bounds only against `q_tot` — so the `1/dt` cap is a real bound; the term is
gated to zero where cloud condensate is negligible.
"""
function sslt_in_cloud_scavenging_rate(F, f_act, Q, q_cld, dt)
    FT = typeof(F)
    c₁ = ifelse(q_cld > ϵ_numerics(FT), min(Q / q_cld, 1 / dt), zero(FT))
    return F * f_act * c₁
end

"""
    sslt_below_cloud_scavenging_rate(F, f_act, Λ)

Below-cloud (impaction) washout rate `(1 − F · f_act) · Λ` [s⁻¹] of one
aerosol bin, at the unit-efficiency collection rate `Λ`. Washout acts on the
aerosol that is *not* dissolved in cloud condensate: all of the cloud-free
area `(1 − F)`, plus the interstitial fraction `F · (1 − f_act)` of the cloudy
area. The droplet-borne remainder is already removed by
[`sslt_in_cloud_scavenging_rate`](@ref), whose driver `Q` contains the
rain-accretes-cloud-liquid arm, so weighting washout by the full area instead
would count that channel twice. `f_act ≡ 1` for the shipped mass-only sea salt
bins (all bins activate at marine supersaturations), which reduces the weight
to `1 − F`; the argument is the seam for an activation-derived per-bin
fraction.
"""
sslt_below_cloud_scavenging_rate(F, f_act, Λ) = (1 - F * f_act) * Λ

"""
    set_sslt_washout_rain_fields!(ᶜΛ̃, ᶜr_D, ᶜV_D, ᶜF, ᶜq_rai, ᶜρ, rain, vel)

Fill the bin-independent washout fields of one subdomain from its rain
mixing ratio `ᶜq_rai` (a field or a lazy expression) and air density `ᶜρ`:
the below-cloud swept rate `ᶜΛ̃ = (1 − F) Λ`
([`sslt_below_cloud_scavenging_rate`](@ref) of
[`rain_swept_collection_rate`](@ref)) and the representative raindrop
`ᶜr_D` ([`rain_representative_drop_radius`](@ref)), `ᶜV_D`
([`rain_drop_fall_speed`](@ref)) that [`sslt_bin_washout_rate`](@ref) then
reads per bin. Shared by the 1-moment drivers (prognostic `q_rai`) and the
0-moment drivers (`q_rai` inverted from the precipitation shadow by
[`rain_mixing_ratio_from_flux`](@ref)) so both wash out through one chain.
The three outputs must be distinct from every field `ᶜF`, `ᶜq_rai`, and
`ᶜρ` read, and `ᶜr_D` is written before `ᶜV_D` reads it.
"""
function set_sslt_washout_rain_fields!(
    ᶜΛ̃,
    ᶜr_D,
    ᶜV_D,
    ᶜF,
    ᶜq_rai,
    ᶜρ,
    rain,
    vel,
)
    FT = eltype(ᶜΛ̃)
    @. ᶜΛ̃ = sslt_below_cloud_scavenging_rate(
        ᶜF,
        one(FT),
        rain_swept_collection_rate(ᶜq_rai, ᶜρ, rain, vel),
    )
    @. ᶜr_D = rain_representative_drop_radius(ᶜq_rai, ᶜρ, rain, vel)
    @. ᶜV_D = rain_drop_fall_speed(ᶜr_D, ᶜρ, rain, vel)
    return nothing
end

"""
    sslt_env_cloud_fraction!(ᶜa_cld, ᶜa⁰, Y, p, n)

Environment cloud fraction `F⁰` (lazy) under `PrognosticEDMFX`, recovered
from the area-weighted `ᶜcloud_fraction = a⁰F⁰ + Σⱼ aʲ·1[condʲ]` (see
`_apply_edmf_cloud_weighting!`) by subtracting the binary updraft cloud
areas, accumulated into the scratch `ᶜa_cld`, and dividing by the
environment area `ᶜa⁰`. The binary updraft cloud check uses the cloud
condensate only (`_updraft_cloud_condensate`: rain and snow are not cloud),
exactly as the cloud fraction does, so a raining but cloud-free updraft
washes out rather than nucleation-scavenges.
"""
function sslt_env_cloud_fraction!(ᶜa_cld, ᶜa⁰, Y, p, n)
    FT = eltype(Y)
    thp = CAP.thermodynamics_params(p.params)
    (; ᶜρʲs, ᶜcloud_fraction) = p.precomputed
    microphysics_model = p.atmos.microphysics_model
    @. ᶜa_cld = zero(FT)
    for j in 1:n
        ᶜq_lclʲ, ᶜq_iclʲ = _updraft_cloud_condensate(Y, p, j, microphysics_model)
        @. ᶜa_cld += ifelse(
            TD.has_condensate(thp, max(zero(FT), ᶜq_lclʲ + ᶜq_iclʲ)),
            draft_area(max(zero(FT), Y.c.sgsʲs.:($$j).ρa), ᶜρʲs.:($$j)),
            zero(FT),
        )
    end
    return @. lazy(
        ifelse(
            ᶜa⁰ > ϵ_numerics(FT),
            min(max((ᶜcloud_fraction - ᶜa_cld) / ᶜa⁰, zero(FT)), one(FT)),
            zero(FT),
        ),
    )
end

"""
    set_sslt_wet_deposition_rates!(Y, p)
    set_sslt_wet_deposition_rates!(Y, p, sslt, microphysics_model)

Fill the per-bin first-order wet-removal rates `k` [s⁻¹] in
`p.tracers.sslt_wetdep_rates` for [`aerosol_wet_deposition_tendency!`](@ref)
and the opt-in `wetss` diagnostic. Called from
`set_explicit_precomputed_quantities!` after the microphysics cache update,
so `ᶜcloud_fraction` and the subdomain thermodynamic states are current.
No-op unless sea salt is prognostic and the microphysics is one of
[`WetDepositionMicrophysics`](@ref); with any other microphysics the rates
are never read and wet deposition is off.

Every ingredient of the two scavenging rates is evaluated per
subdomain, mirroring the subdomain split of the
microphysics process rates: the environment uses the residual
`ᶜspecific_env_value` water species, the environment cloud fraction `F⁰`
recovered from the area-weighted `ᶜcloud_fraction` by subtracting the
(binary) updraft cloud areas, and its own `Q⁰`, `Λ⁰`, temperature and air
state; each updraft uses its own `q_*`, a binary cloud indicator
(condensate present), `Qʲ`, `Λʲ`, temperature and air state. The
environment `Q⁰` is a mean-state verbose call even when the environment
microphysics uses SGS quadrature; the inconsistency is accepted and
documented in `docs/sea_salt_wet_deposition_immediate_plan.md` (§3a,
"Cache plumbing").

Under `NonEquilibriumMicrophysics1M` the washout part of a subdomain's rate
is [`sslt_bin_washout_rate`](@ref) evaluated on the below-cloud swept rate
`Λ̃ = (1 − F) Λ` ([`sslt_below_cloud_scavenging_rate`](@ref) of
[`rain_swept_collection_rate`](@ref)): the Slinn (1984) collection
efficiency [`slinn_collection_efficiency`](@ref) of the bin's wet settling
radius (dry settling radius times the Lewis Eq. 34 growth factor at the
subdomain's RH) by the representative raindrop
[`rain_representative_drop`](@ref) of the subdomain's rain, times `Λ̃`. Two
sets of rates are cached:

  - `p.tracers.sslt_wetdep_rates[bin]`, the grid-mean rate
    `k = (ρa⁰χ⁰k⁰ + Σⱼ ρaʲχʲkʲ) / (ρa⁰χ⁰ + Σⱼ ρaʲχʲ)`, mass-weighted so the
    grid-scale sink is the sum of the subdomain sinks (to first order in
    `kΔt`); where the bin carries no mass the environment rate stands in;
  - `p.tracers.sslt_wetdep_ratesʲs[bin][j]`, updraft `j`'s own rate, which
    the tendency applies to the updraft tracer so precipitating updrafts
    scavenge their own aerosol rather than the grid-mean share.

Under `EquilibriumMicrophysics0M` the same two-term rate is assembled from
the 0-moment ingredients: the in-cloud driver is
[`precipitation_conversion_rate_0m`](@ref) of the cached total-water sink
over the equilibrium condensate `q_liq + q_ice` (all condensate, no phase
resolution), and the washout is the same Slinn chain as for 1M
([`set_sslt_washout_rain_fields!`](@ref), then
[`sslt_bin_washout_rate`](@ref) per bin), evaluated on the rain mixing ratio
[`rain_mixing_ratio_from_flux`](@ref) that carries the column precipitation
shadow from [`set_sslt_precipitation_shadow!`](@ref) as a 1-moment
Marshall–Palmer population. The in-cloud term and the cloudy area are per
subdomain exactly as for 1M (the environment from `ᶜmp_tendency⁰` and
`ᶜq_liq⁰ + ᶜq_ice⁰`, each updraft from its own `ᶜmp_tendencyʲs` and
`ᶜq_liqʲs + ᶜq_iceʲs`), while the shadow is a property of the whole column —
0-moment precipitation leaves the grid column the instant it forms, with no
record of which subdomain it fell through — so every subdomain is washed by
the same rain mass flux, inverted at its own air density.
"""
set_sslt_wet_deposition_rates!(Y, p) = set_sslt_wet_deposition_rates!(
    Y, p, p.atmos.seasalt, p.atmos.microphysics_model,
)
set_sslt_wet_deposition_rates!(Y, p, ::Nothing, _) = nothing
set_sslt_wet_deposition_rates!(Y, p, ::PrognosticSeaSalt, _) = nothing
"""
    sslt_mass_weighted_rate!(ᶜk, ᶜk⁰, ᶜkʲs, χ_name, Y, p, ᶜρa⁰, ᶜρaχ, ᶜρaχE, dt)

Grid-mean wet-removal rate of one bin under `PrognosticEDMFX`, defined by the
mass-weighted *survival* fraction rather than the mass-weighted rate:

    e^{-k Δt} = (ρa⁰χ⁰ e^{-k⁰Δt} + Σⱼ ρaʲχʲ e^{-kʲΔt}) / (ρa⁰χ⁰ + Σⱼ ρaʲχʲ)

so that the grid-mean mass removed per step by the exponential sink of
[`aerosol_wet_deposition_tendency!`](@ref) equals the sum of the subdomain
removals exactly, for any `kΔt`. Weighting the rate instead would make the
grid mean over-remove (`k ↦ 1 - e^{-kΔt}` is concave), and the residual
environment would absorb the difference. Where the bin carries no mass the
environment rate stands in. `ᶜρaχ` and `ᶜρaχE` are scratch accumulators.
"""
function sslt_mass_weighted_rate!(
    ᶜk,
    ᶜk⁰,
    ᶜkʲs,
    χ_name,
    Y,
    p,
    ᶜρa⁰,
    ᶜρaχ,
    ᶜρaχE,
    dt,
)
    FT = eltype(Y)
    ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
    @. ᶜρaχ = ᶜρa⁰ * max(zero(FT), ᶜχ⁰)
    @. ᶜρaχE = ᶜρaχ * expm1(-(ᶜk⁰ * dt))
    for j in 1:length(ᶜkʲs)
        ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
        ᶜkʲ = ᶜkʲs[j]
        ᶜρaʲχʲ = @. lazy(
            max(zero(FT), Y.c.sgsʲs.:($$j).ρa) * max(zero(FT), ᶜχʲ),
        )
        @. ᶜρaχ += ᶜρaʲχʲ
        @. ᶜρaχE += ᶜρaʲχʲ * expm1(-(ᶜkʲ * dt))
    end
    @. ᶜk = ifelse(
        ᶜρaχ > ϵ_numerics(FT),
        -log1p(max(ᶜρaχE / ᶜρaχ, -one(FT) + eps(FT))) / dt,
        ᶜk⁰,
    )
    return nothing
end

function set_sslt_wet_deposition_rates!(
    Y,
    p,
    sslt::PrognosticSeaSalt,
    ::NonEquilibriumMicrophysics1M,
)
    turbconv_model = p.atmos.turbconv_model
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    rain = cmp.precip.rain
    vel = cmp.terminal_velocity.rain
    grav = FT(CAP.grav(p.params))
    ρ_s = CAP.prescribed_aerosol_params(p.params).seasalt_density
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs) = p.precomputed
    (; sslt_air_state⁰, sslt_air_stateʲs) = p.tracers
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    n = n_mass_flux_subdomains(turbconv_model)
    dt = float(p.dt)
    microphysics_model = p.atmos.microphysics_model
    ᶜρa⁰ = @. lazy(max(zero(FT), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))

    # Each subdomain's rate is `c + E_bin · Λ̃`, with `c` the in-cloud part
    # ([`sslt_in_cloud_scavenging_rate`](@ref)) and `Λ̃` the unit-efficiency
    # washout part ([`sslt_below_cloud_scavenging_rate`](@ref)) — both
    # bin-independent — and `E_bin` the per-cell Slinn collection efficiency
    # inside [`sslt_bin_washout_rate`](@ref). The bin-independent fields
    # (`c`, `Λ̃`, and the representative raindrop `r_D`, `V_D`) are hoisted
    # out of each subdomain's bin loop. The updrafts go first so the
    # environment's fields can then stay in place through the bin loop that
    # composes the grid-mean rate.

    # The air state (RH, μ, λ) of every subdomain, read by the growth factor
    # and the Brownian diffusivity, is refilled here rather than reused from
    # the settling tendency: this runs in the precomputed update, before the
    # tendencies, so that fill is a stage stale (unset at the first).
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    set_sslt_air_states!(Y, p, ᶜρ⁰)

    bins = sslt_settling_bin_props(p, sslt)

    # 1. Updrafts: own state, binary cloud indicator, own rate per bin.
    ᶜcʲ = p.scratch.ᶜtemp_scalar_2
    ᶜΛ̃ʲ = p.scratch.ᶜtemp_scalar_3
    ᶜr_Dʲ = p.scratch.ᶜtemp_scalar_4
    ᶜV_Dʲ = p.scratch.ᶜtemp_scalar_5
    for j in 1:n
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜairʲ = sslt_air_stateʲs[j]
        ᶜq_raiʲ = @. lazy(max(zero(FT), Y.c.sgsʲs.:($$j).q_rai))
        ᶜq_lclʲ, ᶜq_iclʲ = _updraft_cloud_condensate(Y, p, j, microphysics_model)
        ᶜFʲ = @. lazy(
            ifelse(
                TD.has_condensate(thp, max(zero(FT), ᶜq_lclʲ + ᶜq_iclʲ)),
                one(FT),
                zero(FT),
            ),
        )
        @. ᶜcʲ = sslt_in_cloud_scavenging_rate(
            ᶜFʲ,
            one(FT),
            cloud_precip_formation_rate(
                cmp, thp, ᶜρʲ, ᶜTʲ, ᶜq_tot_nonnegʲs.:($$j),
                Y.c.sgsʲs.:($$j).q_lcl, Y.c.sgsʲs.:($$j).q_icl,
                Y.c.sgsʲs.:($$j).q_rai, Y.c.sgsʲs.:($$j).q_sno,
            ),
            Y.c.sgsʲs.:($$j).q_lcl + Y.c.sgsʲs.:($$j).q_icl,
            dt,
        )
        set_sslt_washout_rain_fields!(
            ᶜΛ̃ʲ, ᶜr_Dʲ, ᶜV_Dʲ, ᶜFʲ, ᶜq_raiʲ, ᶜρʲ, rain, vel,
        )
        cache_sslth_updraft_exposure_rates!(p, j, ᶜcʲ, ᶜΛ̃ʲ, p.atmos.seasalt_history)
        MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
            ᶜkʲ = ratesʲs[MatrixFields.extract_first(ρχ_name)][j]
            @. ᶜkʲ =
                ᶜcʲ + sslt_bin_washout_rate(
                    ᶜairʲ,
                    ᶜTʲ,
                    r_settle,
                    C_kelvin,
                    ρ_s,
                    ᶜρʲ,
                    ᶜr_Dʲ,
                    ᶜV_Dʲ,
                    ᶜΛ̃ʲ,
                    grav,
                    (ap,),
                )
        end
    end

    # 2. Environment: residual water species, cloud fraction, and the
    # bin-independent fields. Scratch is tight (seven slots), so each
    # species slot is handed on as soon as its last reader has run.
    ᶜq_lcl⁰ = p.scratch.ᶜtemp_scalar_2
    ᶜq_icl⁰ = p.scratch.ᶜtemp_scalar_3
    ᶜq_rai⁰ = p.scratch.ᶜtemp_scalar_4
    ᶜq_sno⁰ = p.scratch.ᶜtemp_scalar_5
    ᶜq_lcl⁰ .= ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜq_icl⁰ .= ᶜspecific_env_value(@name(q_icl), Y, p)
    ᶜq_rai⁰ .= ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ .= ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜa_cld = p.scratch.ᶜtemp_scalar_6
    ᶜa⁰ = @. lazy(draft_area(ᶜρa⁰, ᶜρ⁰))
    ᶜF⁰ = sslt_env_cloud_fraction!(ᶜa_cld, ᶜa⁰, Y, p, n)
    ᶜc⁰ = p.scratch.ᶜtemp_scalar_7
    @. ᶜc⁰ = sslt_in_cloud_scavenging_rate(
        ᶜF⁰,
        one(FT),
        cloud_precip_formation_rate(
            cmp, thp, ᶜρ⁰, ᶜT⁰, ᶜq_tot_nonneg⁰,
            ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
        ),
        ᶜq_lcl⁰ + ᶜq_icl⁰,
        dt,
    )
    # The cloud species are no longer needed, so their scratch takes Λ̃⁰ and
    # the representative raindrop; then q_rai⁰ and the cloud area go too.
    ᶜΛ̃⁰ = p.scratch.ᶜtemp_scalar_2
    ᶜr_D⁰ = p.scratch.ᶜtemp_scalar_3
    ᶜV_D⁰ = p.scratch.ᶜtemp_scalar_5
    set_sslt_washout_rain_fields!(
        ᶜΛ̃⁰, ᶜr_D⁰, ᶜV_D⁰, ᶜF⁰, ᶜq_rai⁰, ᶜρ⁰, rain, vel,
    )
    cache_sslth_env_exposure_rates!(p, ᶜc⁰, ᶜΛ̃⁰, p.atmos.seasalt_history)

    # 3. Grid mean per bin: mass-weighted over the subdomains.
    ᶜρaχ = p.scratch.ᶜtemp_scalar_4
    ᶜρaχE = p.scratch.ᶜtemp_scalar_6
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
        bin = MatrixFields.extract_first(ρχ_name)
        ᶜk⁰ = @. lazy(
            ᶜc⁰ + sslt_bin_washout_rate(
                sslt_air_state⁰,
                ᶜT⁰,
                r_settle,
                C_kelvin,
                ρ_s,
                ᶜρ⁰,
                ᶜr_D⁰,
                ᶜV_D⁰,
                ᶜΛ̃⁰,
                grav,
                (ap,),
            ),
        )
        sslt_mass_weighted_rate!(
            rates[bin], ᶜk⁰, ratesʲs[bin], specific_tracer_name(ρχ_name),
            Y, p, ᶜρa⁰, ᶜρaχ, ᶜρaχE, dt,
        )
    end
    return nothing
end

"""
    precipitation_conversion_rate_0m(dq_tot_dt)

In-cloud driver `Q` [kg kg⁻¹ s⁻¹, ≥ 0] under 0-moment microphysics: the
rate at which condensate is converted to precipitation is the (negative)
total-water sink `dq_tot_dt` of the cached `MP0_NT`, sign-flipped. The
0-moment scheme resolves no phase or process, so — unlike the liquid-only
process-resolved arms of [`cloud_precip_formation_rate`](@ref) — the whole condensate
(`q_liq + q_ice`, the equilibrium cloud water) is the reservoir it drains,
and the activated aerosol share follows it regardless of phase.
"""
precipitation_conversion_rate_0m(dq_tot_dt) = max(zero(dq_tot_dt), -dq_tot_dt)

function set_sslt_wet_deposition_rates!(
    Y,
    p,
    sslt::PrognosticSeaSalt,
    microphysics_model::EquilibriumMicrophysics0M,
)
    turbconv_model = p.atmos.turbconv_model
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    # The 1M rain parameters describe the raindrop population the washout
    # chain sweeps with; `ClimaAtmosParameters` keeps them under 0M whenever
    # prognostic aerosols are on.
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    rain = cmp.precip.rain
    vel = cmp.terminal_velocity.rain
    grav = FT(CAP.grav(p.params))
    ρ_s = CAP.prescribed_aerosol_params(p.params).seasalt_density
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜmp_tendency⁰) =
        p.precomputed
    (; ᶜρʲs, ᶜTʲs, ᶜmp_tendencyʲs) = p.precomputed
    (; sslt_air_state⁰, sslt_air_stateʲs) = p.tracers
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    n = n_mass_flux_subdomains(turbconv_model)
    dt = float(p.dt)
    ᶜρa⁰ = @. lazy(max(zero(FT), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))

    # Same structure as the 1M driver: each subdomain's rate is
    # `c + E_bin · Λ̃`, with the bin-independent `c`, `Λ̃`, `r_D`, `V_D`
    # hoisted out of the bin loop. The washout runs through the 1M chain on
    # the rain mixing ratio that carries the column precipitation shadow
    # (`rain_mixing_ratio_from_flux`); the shadow is a column quantity, so
    # every subdomain sees the same flux `P`, but the inversion uses each
    # subdomain's own density, so `q_rai`, `Λ̃` and the raindrop differ.
    ᶜP = p.scratch.ᶜtemp_scalar
    set_sslt_precipitation_shadow!(ᶜP, Y, p)

    # The Slinn chain reads the air state (RH, μ, λ) of every subdomain;
    # refilled here for the same staleness reason as in the 1M driver.
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar_2
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    set_sslt_air_states!(Y, p, ᶜρ⁰)

    bins = sslt_settling_bin_props(p, sslt)

    # 1. Updrafts: own condensate and sink, binary cloud indicator, own rain
    # mixing ratio from the shadow at the updraft density.
    ᶜq_raiʲ = p.scratch.ᶜtemp_scalar_3
    ᶜΛ̃ʲ = p.scratch.ᶜtemp_scalar_4
    ᶜr_Dʲ = p.scratch.ᶜtemp_scalar_5
    ᶜV_Dʲ = p.scratch.ᶜtemp_scalar_6
    for j in 1:n
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜairʲ = sslt_air_stateʲs[j]
        ᶜq_liqʲ, ᶜq_iceʲ = _updraft_cloud_condensate(Y, p, j, microphysics_model)
        ᶜq_cldʲ = @. lazy(max(zero(FT), ᶜq_liqʲ + ᶜq_iceʲ))
        ᶜFʲ = @. lazy(
            ifelse(TD.has_condensate(thp, ᶜq_cldʲ), one(FT), zero(FT)),
        )
        ᶜQʲ = @. lazy(
            precipitation_conversion_rate_0m(ᶜmp_tendencyʲs.:($$j).dq_tot_dt),
        )
        ᶜcʲ = @. lazy(
            sslt_in_cloud_scavenging_rate(ᶜFʲ, one(FT), ᶜQʲ, ᶜq_cldʲ, dt),
        )
        @. ᶜq_raiʲ = rain_mixing_ratio_from_flux(ᶜP, ᶜρʲ, rain, vel)
        set_sslt_washout_rain_fields!(
            ᶜΛ̃ʲ, ᶜr_Dʲ, ᶜV_Dʲ, ᶜFʲ, ᶜq_raiʲ, ᶜρʲ, rain, vel,
        )
        cache_sslth_updraft_exposure_rates!(p, j, ᶜcʲ, ᶜΛ̃ʲ, p.atmos.seasalt_history)
        MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
            ᶜkʲ = ratesʲs[MatrixFields.extract_first(ρχ_name)][j]
            @. ᶜkʲ =
                ᶜcʲ + sslt_bin_washout_rate(
                    ᶜairʲ,
                    ᶜTʲ,
                    r_settle,
                    C_kelvin,
                    ρ_s,
                    ᶜρʲ,
                    ᶜr_Dʲ,
                    ᶜV_Dʲ,
                    ᶜΛ̃ʲ,
                    grav,
                    (ap,),
                )
        end
    end

    # 2. Environment: equilibrium condensate, 0M sink, cloud fraction, and
    # the bin-independent washout fields. The cloud-area accumulator is
    # dead once `F⁰` is materialized, so its slot then takes `q_rai⁰`.
    ᶜa_cld = p.scratch.ᶜtemp_scalar_3
    ᶜa⁰ = @. lazy(draft_area(ᶜρa⁰, ᶜρ⁰))
    ᶜF⁰ = p.scratch.ᶜtemp_scalar_4
    ᶜF⁰ .= sslt_env_cloud_fraction!(ᶜa_cld, ᶜa⁰, Y, p, n)
    ᶜQ⁰ = @. lazy(precipitation_conversion_rate_0m(ᶜmp_tendency⁰.dq_tot_dt))
    ᶜq_cld⁰ = @. lazy(ᶜq_liq⁰ + ᶜq_ice⁰)
    ᶜc⁰ = @. lazy(
        sslt_in_cloud_scavenging_rate(ᶜF⁰, one(FT), ᶜQ⁰, ᶜq_cld⁰, dt),
    )
    ᶜq_rai⁰ = p.scratch.ᶜtemp_scalar_3
    @. ᶜq_rai⁰ = rain_mixing_ratio_from_flux(ᶜP, ᶜρ⁰, rain, vel)
    ᶜΛ̃⁰ = p.scratch.ᶜtemp_scalar_5
    ᶜr_D⁰ = p.scratch.ᶜtemp_scalar_6
    ᶜV_D⁰ = p.scratch.ᶜtemp_scalar_7
    set_sslt_washout_rain_fields!(
        ᶜΛ̃⁰, ᶜr_D⁰, ᶜV_D⁰, ᶜF⁰, ᶜq_rai⁰, ᶜρ⁰, rain, vel,
    )
    cache_sslth_env_exposure_rates!(p, ᶜc⁰, ᶜΛ̃⁰, p.atmos.seasalt_history)

    # 3. Grid mean per bin: mass-weighted over the subdomains. The shadow
    # and q_rai⁰ have no readers left, so their slots take the accumulators.
    ᶜρaχ = p.scratch.ᶜtemp_scalar
    ᶜρaχE = p.scratch.ᶜtemp_scalar_3
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
        bin = MatrixFields.extract_first(ρχ_name)
        ᶜk⁰ = @. lazy(
            ᶜc⁰ + sslt_bin_washout_rate(
                sslt_air_state⁰,
                ᶜT⁰,
                r_settle,
                C_kelvin,
                ρ_s,
                ᶜρ⁰,
                ᶜr_D⁰,
                ᶜV_D⁰,
                ᶜΛ̃⁰,
                grav,
                (ap,),
            ),
        )
        sslt_mass_weighted_rate!(
            rates[bin], ᶜk⁰, ratesʲs[bin], specific_tracer_name(ρχ_name),
            Y, p, ᶜρa⁰, ᶜρaχ, ᶜρaχE, dt,
        )
    end
    return nothing
end

"""
    aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt)
    aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, sslt, microphysics_model)

Explicit wet removal of the prognostic sea salt bins — a pointwise
first-order sink at the cached per-bin rate `k` from
[`set_sslt_wet_deposition_rates!`](@ref), applied in the
unconditionally stable exponential form

    ∂(ρχ)/∂t += ρχ · expm1(−k·Δt) / Δt

which removes exactly the fraction `1 − e^{−kΔt} ≤ 1` of the tracer per
step for any `k ≥ 0`. In the lowest cell the same mass is also being drained
by the dry-deposition surface flux, so the sink there acts only on the
fraction `1 − min(V_d Δt/Δz₁, 1)` that dry deposition leaves behind
(`p.tracers.sslt_drydep_velocities`), rather than on the full cell again. Each updraft tracer is scavenged in the same form at its own cached rate
`p.tracers.sslt_wetdep_ratesʲs[bin][j]`, so detrainment returns depleted
updraft air, and the grid-mean rate ([`sslt_mass_weighted_rate!`](@ref)) is
defined so the grid-scale sink equals the sum of the subdomain sinks. No-op unless the
microphysics is one of [`WetDepositionMicrophysics`](@ref), matching the
rate cache.
"""
aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, sslt::PrognosticSeaSalt) =
    aerosol_wet_deposition_tendency!(
        Yₜ, Y, p, t, sslt, p.atmos.microphysics_model,
    )
aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, ::PrognosticSeaSalt, _) = nothing
function aerosol_wet_deposition_tendency!(
    Yₜ, Y, p, t, sslt::PrognosticSeaSalt, ::WetDepositionMicrophysics,
)
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    velocities = p.tracers.sslt_drydep_velocities
    settling_velocities = p.tracers.sslt_settling_velocities

    MatrixFields.unrolled_foreach(aerosol_state_names(sslt)) do ρχ_name
        bin = MatrixFields.extract_first(ρχ_name)
        wet_remove_tracer!(
            Yₜ,
            Y,
            p,
            ρχ_name,
            rates[bin],
            ratesʲs[bin],
            velocities[bin],
            settling_velocities[bin],
        )
    end
    return nothing
end

"""
    wet_remove_tracer!(Yₜ, Y, p, ρχ_name, ᶜk, ᶜkʲs, V_d, V_s)

The wet-removal operator of [`aerosol_wet_deposition_tendency!`](@ref) for one
grid-mean tracer `Y.c.<ρχ_name>` and its updraft companions: the exponential
sink at the grid-mean rate `ᶜk` and at each updraft's own rate `ᶜkʲs[j]`,
with level 1 scavenged only on the fraction that the dry-deposition (`V_d`)
and gravitational-settling (`V_s`) surface fall speeds leave behind this
stage. Factored out so the sea-salt history tracers (`history_tracers.jl`)
are scavenged by exactly the operator the mass bins use, with their own
mass-weighted rates and fall speeds. `V_d` and `V_s` are surface fields.
"""
function wet_remove_tracer!(Yₜ, Y, p, ρχ_name, ᶜk, ᶜkʲs, V_d, V_s)
    FT = eltype(Y)
    dt = float(p.dt)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    level1(f) = Fields.field_values(Fields.level(f, 1))
    Δz1_values = level1(Fields.Δz_field(Y.c))

    # Grid mean at the mass-weighted rate, then each updraft at its own.
    χ_name = specific_tracer_name(ρχ_name)
    ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
    ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
    @. ᶜρχₜ += ᶜρχ * expm1(-(ᶜk * dt)) / dt
    for j in 1:n
        ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
        ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
        ᶜkʲ = ᶜkʲs[j]
        @. ᶜχʲₜ += ᶜχʲ * expm1(-(ᶜkʲ * dt)) / dt
    end

    # Level 1 is also drained by the dry-deposition AND gravitational-
    # settling surface fluxes this stage, so scavenge only the fraction
    # that survives both. Both are downward level-1 fall speeds, so their
    # Courant numbers add. Every level view is hoisted out of the
    # broadcasts below. (Settling reuses the grid-mean fall speed for the
    # updraft correction too, matching how V_d is reused there.)
    V_d_values = Fields.field_values(V_d)
    Vs_values = Fields.field_values(V_s)
    k1_values = level1(ᶜk)
    ρχ1_values = level1(ᶜρχ)
    ρχ1ₜ_values = level1(ᶜρχₜ)
    @. ρχ1ₜ_values -=
        min((V_d_values + Vs_values) * dt / Δz1_values, one(FT)) *
        ρχ1_values *
        expm1(-(k1_values * dt)) / dt
    for j in 1:n
        χʲ1_values = level1(MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name))
        χʲ1ₜ_values = level1(MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name))
        kʲ1_values = level1(ᶜkʲs[j])
        @. χʲ1ₜ_values -=
            min((V_d_values + Vs_values) * dt / Δz1_values, one(FT)) *
            χʲ1_values *
            expm1(-(kʲ1_values * dt)) / dt
    end
    return nothing
end
