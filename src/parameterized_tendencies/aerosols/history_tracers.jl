#####
##### Sea-salt history tracers (`SeaSaltHistory`, config "SSLTH")
#####
#
# Passive companions of the prognostic sea-salt mass bins. Each history tracer
# `H` is carried like a mass tracer (grid-mean `ρH`, per-updraft `H`,
# transported and diffused by the model) with its own mass-proportional source
# and the SAME removal sinks as the total sea-salt mass `M = Σ_bins ρχ_i`:
#
#   - settling and dry deposition at the bin-mass-weighted velocity
#     `V_H = Σ_i V_i ρχ_i / Σ_i ρχ_i`, through `settle_tracer!` and
#     `aerosol_surface_flux_tendency!` (the bins' own operators), and
#   - wet removal at, in each subdomain, the survival-weighted rate
#     `k_H,s = −log(Σ_i m_i,s e^{−k_i,s Δt} / Σ_i m_i,s) / Δt` over that
#     subdomain's bin masses, so the removed FRACTION per step matches the
#     bins' exactly (weighting the rate instead would over-remove, as
#     `sslt_mass_weighted_rate!` explains); the grid-mean rate of each tracer
#     then composes the subdomain rates with that tracer's OWN subdomain split
#     (`sslt_mass_weighted_rate!` on `H⁰`, `Hʲ`), so the grid-mean removed
#     amount equals the sum of the subdomain removed amounts for each history
#     tracer, to the same approximation as the bins. Both go through
#     `wet_remove_tracer!`.
#
# Settling, dry deposition and the subdomain wet rates are evaluated per cell
# and per subdomain (environment and each updraft) from that subdomain's own
# bin masses; where a subdomain holds no sea salt the first bin's value
# stands in, as the bins' grid-mean weightings fall back to the environment.
# Every bin velocity or rate is the Courant/budget-capped one the bins
# actually use (the settling cap inside the bin loop, `sslt_drydep_velocities`
# for dry deposition, `sslt_wetdep_rates⁰`/`sslt_wetdep_ratesʲs` for wet
# removal), so the caps are applied once and reused rather than re-derived on
# the mass-weighted value. The tracers never feed back.
#
# Known approximation at level 1 under wet removal: the bins' lowest-level wet
# sink is `Σ_i (1 − c_i) ρχ_i (1 − e^{−k_i Δt})` with the per-bin survivor
# factor `c_i` of the surface sinks, whereas the history tracer applies the
# separately mass-weighted `c_H` and `k_H`; the two agree exactly only when
# `V_d,i`/`V_s,i` and `k_i` do not co-vary across bins. Interior levels match
# per subdomain exactly, and the grid mean is the subdomain sum as above.
# Regressions that need level-1 parity should weight the bins' joint level-1
# removal directly. The grid-mean composition's no-tracer fallback compares
# the `H` weights against `ϵ_numerics` in `H`'s own units (as the bins do in
# mass units); below it the environment rate is used, which only matters
# where `H` is negligible anyway.

"""
    settling_exposure_rate(air, T, ρ_air, grav, Δz, V_H, ap)

Rate `g_s` [s⁻¹] at which the sea-salt population in a cell accumulates
"settling exposure": the `SSHSET` history tracer integrates `ρM · g_s` and
`SSHSET2` integrates `2 ρα_s · g_s` (`α_s = SSHSET`), so that offline
`SSHSET/M` and the variance `SSHSET2/M − (SSHSET/M)²` describe how much
size-selective gravitational sorting the particles have been subjected to.

The choice space, to be settled against the thin-bin truth runs: settling is
a flux divergence, not a local rate, so any per-cell exposure needs a length
scale `ℓ_s` with `g_s = A / ℓ_s`, where `A` is the size-independent Stokes
prefactor of the fall speed (`V_g = A · r̂²`, with `A ∝ (ρ_wet − ρ_air) g ξ² C_c / μ` at the reference radius — bin-independent away from the Kelvin
regime, so it is a property of the air state, not of the bins). Candidates
for `ℓ_s` are the local `Δz` (frozen a priori, the simplest choice that is
identifiable), a fixed reference height, or the boundary-layer depth; the
alternative is to normalize by the realised mass-weighted fall speed `V_H`
instead of the reference-radius prefactor.

Inputs available in-kernel: the subdomain air state `air = (; RH, μ, λ)`
([`_aerosol_air_state`](@ref)), its temperature `T` and density `ρ_air`,
`grav`, the cell thickness `Δz`, the subdomain's mass-weighted (capped)
settling velocity `V_H` from the settling tendency, and the parameter bundle
`ap` (bin radii via `ap.ssa_bin_edges`, `ap.aerosol_r_ref`, `ρ_water`,
`cunningham_C`, the Lewis growth-factor constants read by
[`sslt_growth_factor`](@ref)).

Returns zero until the choice is made, so the tracers load and ride along
without recording settling exposure.
"""
function settling_exposure_rate(air, T, ρ_air, grav, Δz, V_H, ap)
    FT = typeof(T)
    # TODO(human): choose g_s (see the docstring) and evaluate it here.
    return zero(FT)
end

"""
    cache_sslth_env_exposure_rates!(p, ᶜc⁰, ᶜΛ̃⁰, history_model)
    cache_sslth_updraft_exposure_rates!(p, j, ᶜcʲ, ᶜΛ̃ʲ, history_model)

Copy the bin-independent in-cloud rate `c` and below-cloud exposure rate `Λ̃`
of the environment (or of updraft `j`) that [`set_sslt_wet_deposition_rates!`](@ref)
holds in scratch into `p.tracers.sslt_in_cloud_exposure_rate⁰` /
`sslt_washout_exposure_rate⁰` (`…ʲs[j]`), where the history source tendency
reads them. `Λ̃` is the unit-efficiency below-cloud swept-volume rate
`(1 − F) Λ` [s⁻¹] under both 1-moment and 0-moment microphysics (the latter
on the rain mixing ratio inverted from the precipitation shadow), so
`SSHWET/M` is the same dimensionless sweep count either way. No-op without
the history tracers.
"""
cache_sslth_env_exposure_rates!(p, ᶜc⁰, ᶜΛ̃⁰, ::Nothing) = nothing
function cache_sslth_env_exposure_rates!(p, ᶜc⁰, ᶜΛ̃⁰, ::SeaSaltHistory)
    p.tracers.sslt_in_cloud_exposure_rate⁰ .= ᶜc⁰
    p.tracers.sslt_washout_exposure_rate⁰ .= ᶜΛ̃⁰
    return nothing
end
cache_sslth_updraft_exposure_rates!(p, j, ᶜcʲ, ᶜΛ̃ʲ, ::Nothing) = nothing
function cache_sslth_updraft_exposure_rates!(p, j, ᶜcʲ, ᶜΛ̃ʲ, ::SeaSaltHistory)
    p.tracers.sslt_in_cloud_exposure_rateʲs[j] .= ᶜcʲ
    p.tracers.sslt_washout_exposure_rateʲs[j] .= ᶜΛ̃ʲ
    return nothing
end

#####
##### Mass weighting helpers
#####

"""
    sslth_mass_weighted(numerator, mass, fallback)

Pointwise `numerator / mass` where the bin mass sum `mass` is above the
numerics floor, else `fallback` (the first bin's value), the floor the bins'
own grid-mean weightings use.
"""
sslth_mass_weighted(numerator, mass, fallback) =
    ifelse(mass > ϵ_numerics(typeof(mass)), numerator / mass, fallback)

"""
    sslth_bin_mass(ᶜρχ_or_χ)

Nonnegative mass weight of one bin, lazily.
"""
sslth_bin_mass(ᶜm) = @. lazy(max(zero(ᶜm), ᶜm))

#####
##### Emission (SSHUEM)
#####

"""
    set_sslth_emission_flux!(Y, p, u₁₀_ocean, history_model)

Surface source of the emission-wind tracer `SSHUEM`: the total sea-salt
emission mass flux (the per-bin fluxes just written to
`p.tracers.sslt_sfc_fluxes`) times the emission wind `u₁₀_ocean`, stored as
the `ρSSHUEM` entry of `p.tracers.sslth_sfc_fluxes` (the other entries stay
zero: emission adds mass with zero age and zero exposure, the implicit
age-0 reset). Called at the end of [`set_sslt_surface_fluxes!`](@ref), so
both the coupler and the standalone column driver feed it.
"""
set_sslth_emission_flux!(Y, p, u₁₀_ocean, ::Nothing) = nothing
function set_sslth_emission_flux!(Y, p, u₁₀_ocean, ::SeaSaltHistory)
    sfc_flux_uem = p.tracers.sslth_sfc_fluxes.ρSSHUEM
    @. sfc_flux_uem = zero(sfc_flux_uem)
    for sfc_flux in values(p.tracers.sslt_sfc_fluxes)
        @. sfc_flux_uem += sfc_flux
    end
    # Multiply once after the sum: in the standalone column driver `u₁₀_ocean`
    # is a lazy MOST profile evaluation, which fusing into the per-bin loop
    # would re-evaluate N times.
    @. sfc_flux_uem *= u₁₀_ocean
    return nothing
end

aerosol_emission_tendency!(Yₜ, Y, p, t, sslth::SeaSaltHistory) =
    aerosol_surface_flux_tendency!(Yₜ, Y, p, sslth, p.tracers.sslth_sfc_fluxes)

#####
##### Gravitational settling
#####

"""
    sslth_capped_bin_settling_velocity(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, cap, ap)

One bin's settling velocity exactly as the bin loop of
[`aerosol_settling_tendency!`](@ref) evaluates it: [`bin_settling_velocity`](@ref)
Courant-capped at `cap = settling_courant_max · Δz / Δt`.
"""
sslth_capped_bin_settling_velocity(
    air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, cap, ap,
) = min(
    bin_settling_velocity(air, T, r_dry, C_kelvin, ρ_s, ρ_air, grav, ap),
    cap,
)

"""
    set_sslth_subdomain_settling_velocity!(
        ᶜV_H, ᶜden, ᶜw, ᶜw₁, bin_mass, ᶜair, ᶜT, ᶜρ_air, Y, p, bins,
    )

Fill `ᶜV_H` with one subdomain's bin-mass-weighted capped settling velocity
`Σ_i w_i m_i / Σ_i m_i`, where `bin_mass(χ_name)` returns the subdomain's
(lazy, nonnegative) mass weight of a bin and `w_i` is
[`sslth_capped_bin_settling_velocity`](@ref) on the subdomain's air state.
`ᶜden`, `ᶜw`, and `ᶜw₁` are scratch (mass sum, current bin speed, first-bin
speed kept as the no-mass fallback).
"""
function set_sslth_subdomain_settling_velocity!(
    ᶜV_H,
    ᶜden,
    ᶜw,
    ᶜw₁,
    bin_mass,
    ᶜair,
    ᶜT,
    ᶜρ_air,
    Y,
    p,
    bins,
)
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    grav = FT(CAP.grav(p.params))
    ρ_s = CAP.prescribed_aerosol_params(p.params).seasalt_density
    ᶜΔz = Fields.Δz_field(Y.c)
    dt = float(p.dt)

    capped_speed!(ᶜout, r_settle, C_kelvin) = @. ᶜout =
        sslth_capped_bin_settling_velocity(
            ᶜair,
            ᶜT,
            r_settle,
            C_kelvin,
            ρ_s,
            ᶜρ_air,
            grav,
            ap.settling_courant_max * ᶜΔz / dt,
            (ap,),
        )

    @. ᶜV_H = zero(FT)
    @. ᶜden = zero(FT)
    MatrixFields.unrolled_foreach(bins) do (ρχ_name, r_settle, C_kelvin)
        capped_speed!(ᶜw, r_settle, C_kelvin)
        ᶜm = bin_mass(specific_tracer_name(ρχ_name))
        @. ᶜden += ᶜm
        @. ᶜV_H += ᶜm * ᶜw
    end
    # No-mass fallback: the first (smallest, slowest) bin's speed.
    (_, r_settle₁, C_kelvin₁) = first(bins)
    capped_speed!(ᶜw₁, r_settle₁, C_kelvin₁)
    @. ᶜV_H = sslth_mass_weighted(ᶜV_H, ᶜden, ᶜw₁)
    return nothing
end

"""
    aerosol_settling_tendency!(Yₜ, Y, p, t, sslth::SeaSaltHistory)

Settle every history tracer with [`settle_tracer!`](@ref) — the bins'
operator — at the bin-mass-weighted capped fall speed of each subdomain
([`set_sslth_subdomain_settling_velocity!`](@ref)), cached in
`p.tracers.sslth_settling_velocity⁰` / `sslth_settling_velocityʲs` so the
source tendency can hand `V_H` to [`settling_exposure_rate`](@ref). The
environment's weights are its residual bin masses `ρa⁰χ⁰_i`, each updraft's
its own `ρaʲχʲ_i`. Refills the subdomain air states like the bins do.
"""
function aerosol_settling_tendency!(Yₜ, Y, p, t, sslth::SeaSaltHistory)
    turbconv_model = p.atmos.turbconv_model
    FT = eltype(Y)
    thp = CAP.thermodynamics_params(p.params)
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜTʲs, ᶜρʲs) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    n = n_mass_flux_subdomains(turbconv_model)
    (; sslt_air_state⁰, sslt_air_stateʲs) = p.tracers
    (; sslth_settling_velocity⁰, sslth_settling_velocityʲs) = p.tracers

    ᶜρ⁰ = p.scratch.ᶜtemp_scalar_5
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜρa⁰ = @. lazy(max(zero(Y.c.ρ), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))
    ᶠρ = p.scratch.ᶠtemp_scalar_4
    @. ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    set_sslt_air_states!(Y, p, ᶜρ⁰)

    # 1. Mass-weighted fall speed of each subdomain. The scratch slots are
    # free here: `settle_tracer!` (step 2) takes `ᶜtemp_scalar_2/3/6` only
    # after every velocity is in its cache field.
    bins = sslt_settling_bin_props(p, p.atmos.seasalt)
    ᶜden = p.scratch.ᶜtemp_scalar_2
    ᶜw = p.scratch.ᶜtemp_scalar_4
    ᶜw₁ = p.scratch.ᶜtemp_scalar_3
    function env_bin_mass(χ_name)
        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
        return @. lazy(ᶜρa⁰ * max(zero(FT), ᶜχ⁰))
    end
    set_sslth_subdomain_settling_velocity!(
        sslth_settling_velocity⁰,
        ᶜden,
        ᶜw,
        ᶜw₁,
        env_bin_mass,
        sslt_air_state⁰,
        ᶜT⁰,
        ᶜρ⁰,
        Y,
        p,
        bins,
    )
    for j in 1:n
        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
        function updraft_bin_mass(χ_name)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            return @. lazy(max(zero(FT), ᶜρaʲ) * max(zero(FT), ᶜχʲ))
        end
        set_sslth_subdomain_settling_velocity!(
            sslth_settling_velocityʲs[j],
            ᶜden,
            ᶜw,
            ᶜw₁,
            updraft_bin_mass,
            sslt_air_stateʲs[j],
            ᶜTʲs.:($j),
            ᶜρʲs.:($j),
            Y,
            p,
            bins,
        )
    end

    # 2. The bins' operator, once per history tracer.
    updraft_fall_speed(j) = sslth_settling_velocityʲs[j]
    MatrixFields.unrolled_foreach(aerosol_state_names(sslth)) do ρH_name
        settle_tracer!(
            Yₜ,
            Y,
            p,
            ρH_name,
            sslth_settling_velocity⁰,
            updraft_fall_speed,
            ᶜρ⁰,
            ᶜρa⁰,
            ᶠρ,
        )
    end
    return nothing
end

#####
##### Dry deposition
#####

"""
    set_sslth_dry_deposition_fluxes!(Y, p, history_model)

Bin-mass-weighted lowest-level dry-deposition velocity
`V_d,H = Σ_i V_d,i ρχ_i|₁ / Σ_i ρχ_i|₁` from the budget-capped bin
velocities `p.tracers.sslt_drydep_velocities` (so the joint settling +
dry-deposition cap of [`sslt_bin_budgeted_deposition_velocity`](@ref) is
applied once, in the bins' pass, and reused here), into
`p.tracers.sslth_drydep_velocity`; the level-1 settling fall speeds
`sslt_settling_velocities` are weighted the same way into
`sslth_settling_velocity_sfc` for the wet-removal survivor fraction (note:
nothing writes `sslt_settling_velocities` yet, a pre-existing WIP gap of the
bins' settling pass, so the bins and the history tracers both currently use
`V_s = 0` there; once the bins' pass fills it, this weighting follows). Then
each history tracer's surface flux `−V_d,H · ρH|₁` goes into
`p.tracers.sslth_drydep_fluxes`, mirroring [`set_sslt_dry_deposition_fluxes!`](@ref)
(whose fluxes must already be current). Surface and level-1 fields live on
different spaces, so everything is assembled over their data values.
"""
set_sslth_dry_deposition_fluxes!(Y, p, ::Nothing) = nothing
function set_sslth_dry_deposition_fluxes!(Y, p, sslth::SeaSaltHistory)
    FT = eltype(Y)
    sslt = p.atmos.seasalt
    bin_velocities = p.tracers.sslt_drydep_velocities
    bin_settling_velocities = p.tracers.sslt_settling_velocities
    (; sslth_drydep_velocity, sslth_settling_velocity_sfc, sslth_sfc_mass) =
        p.tracers
    fluxes = p.tracers.sslth_drydep_fluxes

    level1(f) = Fields.field_values(Fields.level(f, 1))
    sfc_lg_values =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.f), Fields.half))
    V_d_values = Fields.field_values(sslth_drydep_velocity)
    V_s_values = Fields.field_values(sslth_settling_velocity_sfc)
    mass_values = Fields.field_values(sslth_sfc_mass)

    @. V_d_values = zero(FT)
    @. V_s_values = zero(FT)
    @. mass_values = zero(FT)
    MatrixFields.unrolled_foreach(aerosol_state_names(sslt)) do ρχ_name
        bin = MatrixFields.extract_first(ρχ_name)
        ρχ1_values = level1(MatrixFields.get_field(Y.c, ρχ_name))
        V_d_bin_values = Fields.field_values(bin_velocities[bin])
        V_s_bin_values = Fields.field_values(bin_settling_velocities[bin])
        @. mass_values += max(zero(FT), ρχ1_values)
        @. V_d_values += V_d_bin_values * max(zero(FT), ρχ1_values)
        @. V_s_values += V_s_bin_values * max(zero(FT), ρχ1_values)
    end
    first_bin = MatrixFields.extract_first(first(aerosol_state_names(sslt)))
    V_d_first_values = Fields.field_values(bin_velocities[first_bin])
    V_s_first_values = Fields.field_values(bin_settling_velocities[first_bin])
    @. V_d_values = sslth_mass_weighted(V_d_values, mass_values, V_d_first_values)
    @. V_s_values = sslth_mass_weighted(V_s_values, mass_values, V_s_first_values)

    MatrixFields.unrolled_foreach(aerosol_state_names(sslth)) do ρH_name
        ρH1_values = level1(MatrixFields.get_field(Y.c, ρH_name))
        sfc_flux_values =
            Fields.field_values(fluxes[MatrixFields.extract_first(ρH_name)])
        @. sfc_flux_values = C3(
            -V_d_values *
            max(zero(FT), ρH1_values) *
            unit_basis_vector_data(C3, sfc_lg_values),
        )
    end
    return nothing
end

aerosol_dry_deposition_tendency!(Yₜ, Y, p, t, sslth::SeaSaltHistory) =
    aerosol_surface_flux_tendency!(
        Yₜ,
        Y,
        p,
        sslth,
        p.tracers.sslth_drydep_fluxes,
    )

#####
##### Wet removal
#####

"""
    set_sslth_wet_deposition_rates!(Y, p, history_model)
    set_sslth_wet_deposition_rates!(Y, p, history_model, microphysics_model)

Wet-removal rates of the history tracers, built in two stages so that each
subdomain's history removal matches the bins' and the grid-mean removal is
the sum of the subdomain removals.

 1. Subdomain rates, shared by all history tracers (the bin rates do not
    depend on the tracer): in each subdomain `s` the bin-mass-weighted
    survival fraction (the definition of [`sslt_mass_weighted_rate!`](@ref),
    applied bin-to-history)

        e^{-k_H,s Δt} = Σ_i m_i,s e^{-k_i,s Δt} / Σ_i m_i,s

    with the cached capped bin rates `p.tracers.sslt_wetdep_ratesʲs[·][j]` and
    weights `χʲ_i` for updraft `j` (into `p.tracers.sslth_wetdep_rateʲs[j]`),
    and the environment rates `sslt_wetdep_rates⁰` with the residual
    environment masses `ρa⁰ max(χ⁰_i, 0)` for the environment (scratch only:
    no tendency applies an environment rate). Where a subdomain holds no sea
    salt the first bin's rate stands in.

 2. Grid-mean rate, per history tracer `H`: the subdomain rates composed
    with `H`'s OWN subdomain split, `sslt_mass_weighted_rate!` with weights
    `ρa⁰ max(H⁰, 0)` and `ρaʲ max(Hʲ, 0)`, into
    `p.tracers.sslth_wetdep_rates[ρH_name]` (falling back to the
    environment rate where `H` is absent, as the bins do).

Composing the grid mean from the bins' grid-mean rates instead would mix the
subdomain rates in proportion to sea-salt MASS, not to `H`; where `H/M`
differs between the updrafts and the environment (e.g. `SSHINC`, whose
source and sink both live in cloudy updrafts) the grid-mean removal would
then differ from the sum of the subdomain removals and the diagnosed
environment `ρH − Σⱼ ρaʲHʲ` would absorb the error. Must run after
[`set_sslt_wet_deposition_rates!`](@ref); a no-op (rates stay zero) unless the
microphysics is one of [`WetDepositionMicrophysics`](@ref), like the bins'
rate cache.

Scratch: `ᶜtemp_scalar` and `ᶜtemp_scalar_2` (survival accumulators, reused
by step 2), `ᶜtemp_scalar_3` (the environment history rate `k_H⁰`).
"""
set_sslth_wet_deposition_rates!(Y, p, sslth) =
    set_sslth_wet_deposition_rates!(Y, p, sslth, p.atmos.microphysics_model)
set_sslth_wet_deposition_rates!(Y, p, _, _) = nothing
function set_sslth_wet_deposition_rates!(
    Y,
    p,
    sslth::SeaSaltHistory,
    ::WetDepositionMicrophysics,
)
    FT = eltype(Y)
    sslt = p.atmos.seasalt
    turbconv_model = p.atmos.turbconv_model
    rates⁰ = p.tracers.sslt_wetdep_rates⁰
    ratesʲs = p.tracers.sslt_wetdep_ratesʲs
    ᶜk_Hs = p.tracers.sslth_wetdep_rates
    ᶜk_Hʲs = p.tracers.sslth_wetdep_rateʲs
    dt = float(p.dt)
    n = n_mass_flux_subdomains(turbconv_model)
    ρχ_names = aerosol_state_names(sslt)
    first_bin = MatrixFields.extract_first(first(ρχ_names))
    ᶜmass = p.scratch.ᶜtemp_scalar
    ᶜmassE = p.scratch.ᶜtemp_scalar_2
    ᶜk_H⁰ = p.scratch.ᶜtemp_scalar_3
    ᶜρa⁰ = @. lazy(max(zero(FT), ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)))

    # 1. One subdomain at a time; `survival_weighted_rate!` folds the bins.
    survival_weighted_rate!(ᶜk, bin_rate, bin_mass) = begin
        @. ᶜmass = zero(FT)
        @. ᶜmassE = zero(FT)
        MatrixFields.unrolled_foreach(ρχ_names) do ρχ_name
            ᶜm = bin_mass(specific_tracer_name(ρχ_name))
            ᶜk_i = bin_rate(MatrixFields.extract_first(ρχ_name))
            @. ᶜmass += ᶜm
            @. ᶜmassE += ᶜm * expm1(-(ᶜk_i * dt))
        end
        ᶜk₁ = bin_rate(first_bin)
        @. ᶜk = ifelse(
            ᶜmass > ϵ_numerics(FT),
            -log1p(max(ᶜmassE / ᶜmass, -one(FT) + eps(FT))) / dt,
            ᶜk₁,
        )
    end
    # Environment weights are the residual bin masses, clamped like the
    # settling weights (`env_bin_mass` in `aerosol_settling_tendency!`).
    function env_bin_mass(χ_name)
        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
        return @. lazy(ᶜρa⁰ * max(zero(FT), ᶜχ⁰))
    end
    survival_weighted_rate!(ᶜk_H⁰, bin -> rates⁰[bin], env_bin_mass)
    for j in 1:n
        survival_weighted_rate!(
            ᶜk_Hʲs[j],
            bin -> ratesʲs[bin][j],
            χ_name -> sslth_bin_mass(
                MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name),
            ),
        )
    end

    # 2. Grid mean per tracer, weighted by the tracer's own subdomain split.
    # The step-1 accumulators are dead, so their slots are reused.
    MatrixFields.unrolled_foreach(aerosol_state_names(sslth)) do ρH_name
        sslt_mass_weighted_rate!(
            ᶜk_Hs[MatrixFields.extract_first(ρH_name)],
            ᶜk_H⁰,
            ᶜk_Hʲs,
            specific_tracer_name(ρH_name),
            Y,
            p,
            ᶜρa⁰,
            ᶜmass,
            ᶜmassE,
            dt,
        )
    end
    return nothing
end

aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, sslth::SeaSaltHistory) =
    aerosol_wet_deposition_tendency!(
        Yₜ, Y, p, t, sslth, p.atmos.microphysics_model,
    )
aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, ::SeaSaltHistory, _) = nothing
function aerosol_wet_deposition_tendency!(
    Yₜ, Y, p, t, sslth::SeaSaltHistory, ::WetDepositionMicrophysics,
)
    (; sslth_wetdep_rates, sslth_wetdep_rateʲs) = p.tracers
    (; sslth_drydep_velocity, sslth_settling_velocity_sfc) = p.tracers
    MatrixFields.unrolled_foreach(aerosol_state_names(sslth)) do ρH_name
        wet_remove_tracer!(
            Yₜ,
            Y,
            p,
            ρH_name,
            sslth_wetdep_rates[MatrixFields.extract_first(ρH_name)],
            sslth_wetdep_rateʲs,
            sslth_drydep_velocity,
            sslth_settling_velocity_sfc,
        )
    end
    return nothing
end

#####
##### Sources
#####

"""
    aerosol_history_source_tendency!(Yₜ, Y, p, t)
    aerosol_history_source_tendency!(Yₜ, Y, p, t, species_model)

Volume sources of the sea-salt history tracers, per subdomain with that
subdomain's own sea-salt mass `M_s = Σ_i χ_i,s` and own rates: for the
environment (`s = 0`, bin masses by residual) and each updraft `j`,

    SSHAGE : ρM                      (grid mean only: dτ/dt = 1 per unit mass,
                                      updrafts get Mʲ)
    SSHSET : ρa_s M_s g_s,s          g_s from `settling_exposure_rate`
    SSHSET2: 2 ρa_s α_s,s g_s,s      α_s = SSHSET of the subdomain
    SSHWET : ρa_s M_s Λ̃_s            cached `sslt_washout_exposure_rate⁰/ʲs`
    SSHINC : ρa_s M_s c_s            cached `sslt_in_cloud_exposure_rate⁰/ʲs`

summed over subdomains into the grid-mean tendencies and applied as specific
tendencies to each updraft tracer. `SSHUEM` has no volume source (its surface
source is [`set_sslth_emission_flux!`](@ref)). Reads the subdomain air states
and mass-weighted fall speeds left by the history settling tendency, which
runs earlier in the same `remaining_tendency!`. Dispatches over the species
models like the other aerosol tendencies; no-op for species without history.
"""
aerosol_history_source_tendency!(Yₜ, Y, p, t, _) = nothing
aerosol_history_source_tendency!(Yₜ, Y, p, t) = unrolled_foreach(
    model -> aerosol_history_source_tendency!(Yₜ, Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)
function aerosol_history_source_tendency!(Yₜ, Y, p, t, ::SeaSaltHistory)
    turbconv_model = p.atmos.turbconv_model
    FT = eltype(Y)
    ap = CAP.prognostic_aerosol_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    grav = FT(CAP.grav(p.params))
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜTʲs, ᶜρʲs) = p.precomputed
    (; sslt_air_state⁰, sslt_air_stateʲs) = p.tracers
    (; sslth_settling_velocity⁰, sslth_settling_velocityʲs) = p.tracers
    (; sslt_washout_exposure_rate⁰, sslt_washout_exposure_rateʲs) = p.tracers
    (; sslt_in_cloud_exposure_rate⁰, sslt_in_cloud_exposure_rateʲs) = p.tracers
    ᶜΔz = Fields.Δz_field(Y.c)
    n = n_mass_flux_subdomains(turbconv_model)
    ρχ_names = aerosol_state_names(p.atmos.seasalt)

    ᶜρSSHAGEₜ = Yₜ.c.ρSSHAGE
    ᶜρSSHSETₜ = Yₜ.c.ρSSHSET
    ᶜρSSHSET2ₜ = Yₜ.c.ρSSHSET2
    ᶜρSSHWETₜ = Yₜ.c.ρSSHWET
    ᶜρSSHINCₜ = Yₜ.c.ρSSHINC

    # Grid-mean sea-salt mass; the environment's share and its settling
    # exposure are what is left after the updrafts are subtracted.
    ᶜρM = p.scratch.ᶜtemp_scalar
    ᶜMʲ = p.scratch.ᶜtemp_scalar_2
    ᶜρaM⁰ = p.scratch.ᶜtemp_scalar_3
    ᶜρaα⁰ = p.scratch.ᶜtemp_scalar_4
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar_5
    # Unrolled over compile-time names (not a runtime `Symbol` loop) so each
    # per-bin broadcast is statically dispatched in this hot tendency path.
    @. ᶜρM = zero(FT)
    MatrixFields.unrolled_foreach(ρχ_names) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        @. ᶜρM += max(zero(FT), ᶜρχ)
    end
    @. ᶜρSSHAGEₜ += ᶜρM
    ᶜρaM⁰ .= ᶜρM
    ᶜρaα⁰ .= Y.c.ρSSHSET

    for j in 1:n
        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
        ᶜρʲ = ᶜρʲs.:($j)
        sgsʲ = Y.c.sgsʲs.:($j)
        sgsʲₜ = Yₜ.c.sgsʲs.:($j)
        @. ᶜMʲ = zero(FT)
        MatrixFields.unrolled_foreach(ρχ_names) do ρχ_name
            ᶜχʲ = MatrixFields.get_field(sgsʲ, specific_tracer_name(ρχ_name))
            @. ᶜMʲ += max(zero(FT), ᶜχʲ)
        end
        ᶜα_sʲ = sgsʲ.SSHSET
        ᶜgʲ = @. lazy(
            settling_exposure_rate(
                sslt_air_stateʲs[$j],
                ᶜTʲs.:($$j),
                ᶜρʲ,
                grav,
                ᶜΔz,
                sslth_settling_velocityʲs[$j],
                (ap,),
            ),
        )
        ᶜΛ̃ʲ = sslt_washout_exposure_rateʲs[j]
        ᶜcʲ = sslt_in_cloud_exposure_rateʲs[j]

        @. sgsʲₜ.SSHAGE += ᶜMʲ
        @. sgsʲₜ.SSHSET += ᶜMʲ * ᶜgʲ
        @. sgsʲₜ.SSHSET2 += 2 * ᶜα_sʲ * ᶜgʲ
        @. sgsʲₜ.SSHWET += ᶜMʲ * ᶜΛ̃ʲ
        @. sgsʲₜ.SSHINC += ᶜMʲ * ᶜcʲ

        ᶜρaʲMʲ = @. lazy(max(zero(FT), ᶜρaʲ) * ᶜMʲ)
        @. ᶜρSSHSETₜ += ᶜρaʲMʲ * ᶜgʲ
        @. ᶜρSSHSET2ₜ += 2 * max(zero(FT), ᶜρaʲ) * ᶜα_sʲ * ᶜgʲ
        @. ᶜρSSHWETₜ += ᶜρaʲMʲ * ᶜΛ̃ʲ
        @. ᶜρSSHINCₜ += ᶜρaʲMʲ * ᶜcʲ
        @. ᶜρaM⁰ -= ᶜρaʲMʲ
        @. ᶜρaα⁰ -= max(zero(FT), ᶜρaʲ) * ᶜα_sʲ
    end

    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜg⁰ = @. lazy(
        settling_exposure_rate(
            sslt_air_state⁰,
            ᶜT⁰,
            ᶜρ⁰,
            grav,
            ᶜΔz,
            sslth_settling_velocity⁰,
            (ap,),
        ),
    )
    @. ᶜρSSHSETₜ += max(zero(FT), ᶜρaM⁰) * ᶜg⁰
    @. ᶜρSSHSET2ₜ += 2 * max(zero(FT), ᶜρaα⁰) * ᶜg⁰
    @. ᶜρSSHWETₜ += max(zero(FT), ᶜρaM⁰) * sslt_washout_exposure_rate⁰
    @. ᶜρSSHINCₜ += max(zero(FT), ᶜρaM⁰) * sslt_in_cloud_exposure_rate⁰
    return nothing
end
