import CloudMicrophysics.Microphysics1M as CM1

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
distribution available) and `EquilibriumMicrophysics0M` (precipitation
shadow only). Every other model leaves the rates unread and wet deposition
off.
"""
const WetDepositionMicrophysics =
    Union{EquilibriumMicrophysics0M, NonEquilibriumMicrophysics1M}

"""
    set_sslt_wet_deposition_rates!(Y, p)
    set_sslt_wet_deposition_rates!(Y, p, sslt, microphysics_model)

Fill the per-bin first-order wet-removal rates `k` [s⁻¹] in
`p.tracers.sslt_wetdep_rates` for [`aerosol_wet_deposition_tendency!`](@ref)
and the opt-in `wetss` diagnostic: each bin's below-cloud washout rate. Called from
`set_explicit_precomputed_quantities!` after the microphysics cache and
surface precipitation flux updates. No-op unless sea salt is prognostic and
the microphysics is one of [`WetDepositionMicrophysics`](@ref); with any
other microphysics the rates are never read and wet deposition is off.

Under `NonEquilibriumMicrophysics1M` the rate is `ssa_E_coll[bin]` times
[`rain_swept_collection_rate`](@ref) on the rain state. Prognostic sea salt
requires `PrognosticEDMFX`, so the rate is always evaluated per subdomain — `Λ⁰` from the environment
rain `q_rai⁰` (the residual `ᶜspecific_env_value`) and density, `Λʲ` from
each updraft's own `q_rai` and density — mirroring the subdomain split of
the microphysics process rates. Two sets are cached:

  - `p.tracers.sslt_wetdep_rates[bin]`, the grid-mean rate
    `k = (ρa⁰χ⁰k⁰ + Σⱼ ρaʲχʲkʲ) / (ρa⁰χ⁰ + Σⱼ ρaʲχʲ)`, mass-weighted so the
    grid-scale sink is the sum of the subdomain sinks (to first order in
    `kΔt`); where the bin carries no mass the environment rate stands in;
  - `p.tracers.sslt_wetdep_ratesʲs[bin][j]`, updraft `j`'s own rate, which
    the tendency applies to the updraft tracer so precipitating updrafts
    scavenge their own aerosol rather than the grid-mean share.

Under `EquilibriumMicrophysics0M` there is no rain state, so the rate is the
per-bin power law [`power_law_washout_rate`](@ref) (`ssa_washout_a[bin]`,
`ssa_washout_b[bin]`) in the rain rate of the column precipitation shadow
from [`set_sslt_precipitation_shadow!`](@ref). The shadow is a property of
the whole column — 0-moment precipitation is removed from the grid column
the instant it forms, with no record of which subdomain it fell through —
so under `PrognosticEDMFX` the same rate is written to the grid-mean and to
every updraft slot; the mass-weighted composition then reduces to it
identically.
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
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜρʲs) = p.precomputed
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    n = n_mass_flux_subdomains(turbconv_model)

    # Bin-independent environment quantities, hoisted: density, rain, and the
    # unit-efficiency collection rate Λ⁰ (the bins differ only through E).
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar
    ᶜq_rai⁰ = p.scratch.ᶜtemp_scalar_2
    ᶜΛ⁰ = p.scratch.ᶜtemp_scalar_3
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜq_rai⁰ .= ᶜspecific_env_value(@name(q_rai), Y, p)
    @. ᶜΛ⁰ = rain_swept_collection_rate(ᶜq_rai⁰, ᶜρ⁰, rain, vel)
    ᶜρa⁰ = @. lazy(max(zero(FT), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))
    dt = float(p.dt)
    # Scratch accumulators of the mass and the survived mass per bin.
    ᶜρaχ = p.scratch.ᶜtemp_scalar_4
    ᶜρaχE = p.scratch.ᶜtemp_scalar_5

    ρχ_names = aerosol_state_names(sslt)
    bins = ntuple(Val(length(ρχ_names))) do i
        (ρχ_names[i], FT(ap.ssa_E_coll[i]))
    end
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, E_bin)
        bin = MatrixFields.extract_first(ρχ_name)
        for j in 1:n
            @. ratesʲs[bin][j] =
                E_bin * rain_swept_collection_rate(
                    max(zero(FT), Y.c.sgsʲs.:($$j).q_rai),
                    ᶜρʲs.:($$j),
                    rain,
                    vel,
                )
        end
        ᶜk⁰ = @. lazy(E_bin * ᶜΛ⁰)
        sslt_mass_weighted_rate!(
            rates[bin],
            ᶜk⁰,
            ratesʲs[bin],
            specific_tracer_name(ρχ_name),
            Y,
            p,
            ᶜρa⁰,
            ᶜρaχ,
            ᶜρaχE,
            dt,
        )
    end
    return nothing
end

function set_sslt_wet_deposition_rates!(
    Y,
    p,
    sslt::PrognosticSeaSalt,
    microphysics_model::EquilibriumMicrophysics0M,
)
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    rates = p.tracers.sslt_wetdep_rates
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    # The precipitation shadow is a column quantity (0M rain leaves the
    # column as it forms), so one rain rate washes every subdomain alike.
    ᶜR = p.scratch.ᶜtemp_scalar
    set_sslt_precipitation_shadow!(ᶜR, Y, p)
    @. ᶜR = precipitation_rate_mm_h(ᶜR, ap.ρ_water)

    ρχ_names = aerosol_state_names(sslt)
    bins = ntuple(Val(length(ρχ_names))) do i
        (ρχ_names[i], FT(ap.ssa_washout_a[i]), FT(ap.ssa_washout_b[i]))
    end
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, a, b)
        bin = MatrixFields.extract_first(ρχ_name)
        ᶜk = rates[bin]
        @. ᶜk = power_law_washout_rate(ᶜR, a, b)
        for j in 1:n
            @. ratesʲs[bin][j] = ᶜk
        end
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
