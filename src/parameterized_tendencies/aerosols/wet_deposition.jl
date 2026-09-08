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
    precipitation_rate_mm_h(P, ρ_water)

Rain rate `R` [mm h⁻¹] of a downward liquid precipitation mass flux `P`
[kg m⁻² s⁻¹], the unit the washout power law is fitted in.
"""
precipitation_rate_mm_h(P, ρ_water) = P * 3600 * 1000 / ρ_water

"""
    power_law_washout_rate(R, a, b)

Below-cloud scavenging coefficient `Λ = a · R^b` [s⁻¹] of one aerosol bin
under the rain rate `R` [mm h⁻¹], the empirical form of Feng (2007) with the
per-bin prefactor `a` [s⁻¹ at 1 mm h⁻¹] and exponent `b` (0.6–0.8 across
the marine size range). Used where no rain size distribution exists to
evaluate [`rain_swept_collection_rate`](@ref), i.e. under 0-moment
microphysics. Negative rain rates (round-off) wash nothing out.
"""
power_law_washout_rate(R, a, b) = a * max(R, zero(R))^b

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
integral is exact, so no clamp is needed here; a negative rain rate from a
positive sink washes nothing out through the `max(R, 0)` of
[`power_law_washout_rate`](@ref). The face flux is interpolated to cell
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
(binary) updraft cloud areas, and its own `Q⁰` and `Λ⁰`; each updraft uses
its own `q_*`, a binary cloud indicator (condensate present), `Qʲ`, and
`Λʲ`. The environment `Q⁰` is a mean-state verbose call even when the
environment microphysics uses SGS quadrature; the inconsistency is accepted
and documented in `docs/sea_salt_wet_deposition_immediate_plan.md` (§3a,
"Cache plumbing"). Two sets of rates are cached:

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
resolution), and the washout is the per-bin power law
[`power_law_washout_rate`](@ref) (`ssa_washout_a[bin]`, `ssa_washout_b[bin]`)
in the rain rate of the column precipitation shadow from
[`set_sslt_precipitation_shadow!`](@ref). The in-cloud term and the cloudy
area are per subdomain exactly as for 1M (the
environment from `ᶜmp_tendency⁰` and `ᶜq_liq⁰ + ᶜq_ice⁰`, each updraft from
its own `ᶜmp_tendencyʲs` and `ᶜq_liqʲs + ᶜq_iceʲs`), while the shadow is a
property of the whole column — 0-moment precipitation leaves the grid column
the instant it forms, with no record of which subdomain it fell through — so
every subdomain is washed at the same rain rate.
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
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜcloud_fraction) =
        p.precomputed
    (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs) = p.precomputed
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    n = n_mass_flux_subdomains(turbconv_model)
    dt = float(p.dt)
    microphysics_model = p.atmos.microphysics_model
    ᶜρa⁰ = @. lazy(max(zero(FT), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))

    # Each subdomain's rate is `c + E_bin · Λ̃`, with `c` the in-cloud part
    # ([`sslt_in_cloud_scavenging_rate`](@ref)) and `Λ̃` the unit-efficiency
    # washout part ([`sslt_below_cloud_scavenging_rate`](@ref)) — both
    # bin-independent; only the collection efficiency E_bin varies across bins.

    # 1. Environment: residual water species and density.
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar
    ᶜq_lcl⁰ = p.scratch.ᶜtemp_scalar_2
    ᶜq_icl⁰ = p.scratch.ᶜtemp_scalar_3
    ᶜq_rai⁰ = p.scratch.ᶜtemp_scalar_4
    ᶜq_sno⁰ = p.scratch.ᶜtemp_scalar_5
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
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
    # `ᶜq_lcl⁰` is no longer needed, so its scratch takes Λ̃⁰.
    ᶜΛ̃⁰ = p.scratch.ᶜtemp_scalar_2
    @. ᶜΛ̃⁰ = sslt_below_cloud_scavenging_rate(
        ᶜF⁰,
        one(FT),
        rain_swept_collection_rate(ᶜq_rai⁰, ᶜρ⁰, rain, vel),
    )

    ρχ_names = aerosol_state_names(sslt)
    bins = ntuple(Val(length(ρχ_names))) do i
        (ρχ_names[i], FT(ap.ssa_E_coll[i]))
    end

    # 2. Updrafts: own state, binary cloud indicator, own rate per bin.
    ᶜcʲ = p.scratch.ᶜtemp_scalar_3
    ᶜΛ̃ʲ = p.scratch.ᶜtemp_scalar_4
    for j in 1:n
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
                cmp, thp, ᶜρʲs.:($$j), ᶜTʲs.:($$j), ᶜq_tot_nonnegʲs.:($$j),
                Y.c.sgsʲs.:($$j).q_lcl, Y.c.sgsʲs.:($$j).q_icl,
                Y.c.sgsʲs.:($$j).q_rai, Y.c.sgsʲs.:($$j).q_sno,
            ),
            Y.c.sgsʲs.:($$j).q_lcl + Y.c.sgsʲs.:($$j).q_icl,
            dt,
        )
        @. ᶜΛ̃ʲ = sslt_below_cloud_scavenging_rate(
            ᶜFʲ,
            one(FT),
            rain_swept_collection_rate(
                max(zero(FT), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                rain,
                vel,
            ),
        )
        MatrixFields.unrolled_foreach(bins) do (ρχ_name, E_bin)
            ᶜkʲ = ratesʲs[MatrixFields.extract_first(ρχ_name)][j]
            @. ᶜkʲ = ᶜcʲ + E_bin * ᶜΛ̃ʲ
        end
    end

    # 3. Grid mean per bin: mass-weighted over the subdomains.
    ᶜρaχ = p.scratch.ᶜtemp_scalar_5
    ᶜρaχE = p.scratch.ᶜtemp_scalar_6
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, E_bin)
        bin = MatrixFields.extract_first(ρχ_name)
        ᶜk⁰ = @. lazy(ᶜc⁰ + E_bin * ᶜΛ̃⁰)
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
    thp = CAP.thermodynamics_params(p.params)
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜmp_tendency⁰) =
        p.precomputed
    (; ᶜmp_tendencyʲs) = p.precomputed
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    n = n_mass_flux_subdomains(turbconv_model)
    dt = float(p.dt)
    ᶜρa⁰ = @. lazy(max(zero(FT), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))

    # The column shadow washes every subdomain at the same rain rate; only
    # the in-cloud term and the cloudy area differ between subdomains.
    ᶜR = p.scratch.ᶜtemp_scalar
    set_sslt_precipitation_shadow!(ᶜR, Y, p)
    @. ᶜR = precipitation_rate_mm_h(ᶜR, ap.ρ_water)

    # 1. Environment: equilibrium condensate, 0M sink, and cloud fraction.
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar_2
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜa_cld = p.scratch.ᶜtemp_scalar_3
    ᶜa⁰ = @. lazy(draft_area(ᶜρa⁰, ᶜρ⁰))
    ᶜF⁰ = p.scratch.ᶜtemp_scalar_4
    ᶜF⁰ .= sslt_env_cloud_fraction!(ᶜa_cld, ᶜa⁰, Y, p, n)
    ᶜQ⁰ = @. lazy(precipitation_conversion_rate_0m(ᶜmp_tendency⁰.dq_tot_dt))
    ᶜq_cld⁰ = @. lazy(ᶜq_liq⁰ + ᶜq_ice⁰)

    ρχ_names = aerosol_state_names(sslt)
    bins = ntuple(Val(length(ρχ_names))) do i
        (ρχ_names[i], FT(ap.ssa_washout_a[i]), FT(ap.ssa_washout_b[i]))
    end

    # 2. Updrafts: own condensate and sink, binary cloud indicator.
    for j in 1:n
        ᶜq_liqʲ, ᶜq_iceʲ = _updraft_cloud_condensate(Y, p, j, microphysics_model)
        ᶜq_cldʲ = @. lazy(max(zero(FT), ᶜq_liqʲ + ᶜq_iceʲ))
        ᶜFʲ = @. lazy(
            ifelse(TD.has_condensate(thp, ᶜq_cldʲ), one(FT), zero(FT)),
        )
        ᶜQʲ = @. lazy(
            precipitation_conversion_rate_0m(ᶜmp_tendencyʲs.:($$j).dq_tot_dt),
        )
        MatrixFields.unrolled_foreach(bins) do (ρχ_name, a, b)
            ᶜkʲ = ratesʲs[MatrixFields.extract_first(ρχ_name)][j]
            @. ᶜkʲ =
                sslt_in_cloud_scavenging_rate(ᶜFʲ, one(FT), ᶜQʲ, ᶜq_cldʲ, dt) +
                sslt_below_cloud_scavenging_rate(
                    ᶜFʲ,
                    one(FT),
                    power_law_washout_rate(ᶜR, a, b),
                )
        end
    end

    # 3. Grid mean per bin: mass-weighted over the subdomains.
    ᶜρaχ = p.scratch.ᶜtemp_scalar_5
    ᶜρaχE = p.scratch.ᶜtemp_scalar_6
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, a, b)
        bin = MatrixFields.extract_first(ρχ_name)
        ᶜk⁰ = @. lazy(
            sslt_in_cloud_scavenging_rate(ᶜF⁰, one(FT), ᶜQ⁰, ᶜq_cld⁰, dt) +
            sslt_below_cloud_scavenging_rate(
                ᶜF⁰,
                one(FT),
                power_law_washout_rate(ᶜR, a, b),
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
    FT = eltype(Y)
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    velocities = p.tracers.sslt_drydep_velocities
    dt = float(p.dt)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    level1(f) = Fields.field_values(Fields.level(f, 1))
    Δz1_values = level1(Fields.Δz_field(Y.c))

    # Grid mean at the mass-weighted rate, then each updraft at its own.
    MatrixFields.unrolled_foreach(aerosol_state_names(sslt)) do ρχ_name
        χ_name = specific_tracer_name(ρχ_name)
        bin = MatrixFields.extract_first(ρχ_name)
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        ᶜk = rates[bin]
        @. ᶜρχₜ += ᶜρχ * expm1(-(ᶜk * dt)) / dt
        for j in 1:n
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            ᶜkʲ = ratesʲs[bin][j]
            @. ᶜχʲₜ += ᶜχʲ * expm1(-(ᶜkʲ * dt)) / dt
        end

        # Level 1 is also drained by the dry-deposition surface flux this
        # stage, so scavenge only the fraction that survives it. Every level
        # view is hoisted out of the broadcasts below.
        V_d_values = Fields.field_values(velocities[bin])
        k1_values = level1(ᶜk)
        ρχ1_values = level1(ᶜρχ)
        ρχ1ₜ_values = level1(ᶜρχₜ)
        @. ρχ1ₜ_values -=
            min(V_d_values * dt / Δz1_values, one(FT)) *
            ρχ1_values *
            expm1(-(k1_values * dt)) / dt
        for j in 1:n
            χʲ1_values = level1(MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name))
            χʲ1ₜ_values =
                level1(MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name))
            kʲ1_values = level1(ratesʲs[bin][j])
            @. χʲ1ₜ_values -=
                min(V_d_values * dt / Δz1_values, one(FT)) *
                χʲ1_values *
                expm1(-(kʲ1_values * dt)) / dt
        end
    end
    return nothing
end
