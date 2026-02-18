# ============================================================================
# Unified Microphysics Tendencies
# ============================================================================
# Single entry point for all microphysics tendency calculations.
# Combines cloud condensation/evaporation and precipitation processes.
#
# For NonEquilMoistModel, cloud condensation/evaporation is computed first,
# then precipitation processes (autoconversion, accretion, evaporation).
#
# In EDMF modes (PrognosticEDMFX/DiagnosticEDMFX), tendencies are computed
# per-subdomain in the EDMF precomputed quantities files and applied here
# with area weighting.

import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

#####
##### No Microphysics
#####

microphysics_tendency!(Yₜ, Y, p, t, _, ::NoPrecipitation, _) = nothing

#####
##### 0-Moment Microphysics (EquilMoistModel only)
#####

function microphysics_tendency!(Yₜ, Y, p, t, ::DryModel, ::Microphysics0Moment, _)
    error("Microphysics0Moment is incompatible with DryModel")
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilMoistModel, ::Microphysics0Moment, _,
)
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed

    @. Yₜ.c.ρq_tot += ᶜS_ρq_tot
    @. Yₜ.c.ρ += ᶜS_ρq_tot
    @. Yₜ.c.ρe_tot += ᶜS_ρe_tot

    return nothing
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics0Moment, _,
)
    error("Microphysics0Moment is not supported with NonEquilMoistModel")
end

#####
##### 1-Moment Microphysics
#####

function microphysics_tendency!(Yₜ, Y, p, t,
    ::DryModel, ::Microphysics1Moment, _,
)
    error("Microphysics1Moment is incompatible with DryModel")
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilMoistModel, ::Microphysics1Moment, _,
)
    error("Microphysics1Moment is not implemented for EquilMoistModel")
end

"""
    microphysics_tendency!(Yₜ, Y, p, t, moisture_model, microphysics_model, turbconv_model)

Unified entry point for all microphysics tendency calculations.

For `NonEquilMoistModel` with 1M/2M microphysics (without EDMF): Applies microphysics 
tendencies from precomputed cache

For EDMF modes, tendencies are computed per-subdomain in precomputed quantities
and applied here with area weighting.

Modifies `Yₜ.c.ρq_liq`, `Yₜ.c.ρq_ice`, `Yₜ.c.ρq_rai`, `Yₜ.c.ρq_sno` in place.
"""
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics1Moment, _,
)
    (; ᶜSqₗᵐ, ᶜSqᵢᵐ, ᶜSqᵣᵐ, ᶜSqₛᵐ) = p.precomputed

    # Tendencies are already limited in the cache; just scale by density
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜSqₗᵐ
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜSqᵢᵐ
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜSqᵣᵐ
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜSqₛᵐ

    return nothing
end

# 1M with DiagnosticEDMFX
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics1Moment, ::DiagnosticEDMFX,
)
    # Source terms from EDMFX environment
    (; ᶜSqₗᵐ⁰, ᶜSqᵢᵐ⁰, ᶜSqᵣᵐ⁰, ᶜSqₛᵐ⁰) = p.precomputed
    # Source terms from EDMFX updrafts
    (; ᶜSqₗᵐʲs, ᶜSqᵢᵐʲs, ᶜSqᵣᵐʲs, ᶜSqₛᵐʲs) = p.precomputed
    (; ᶜρaʲs) = p.precomputed

    # Tendencies are already limited in the cache; just scale by density
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    turbconv_model = p.atmos.turbconv_model
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, p.precomputed.ᶜρaʲs, turbconv_model))

    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜSqₗᵐ⁰
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜSqᵢᵐ⁰
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜSqᵣᵐ⁰
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜSqₛᵐ⁰

    # Update from the updraft microphysics tendencies
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += ᶜρaʲs.:($$j) * ᶜSqₗᵐʲs.:($$j)
        @. Yₜ.c.ρq_ice += ᶜρaʲs.:($$j) * ᶜSqᵢᵐʲs.:($$j)
        @. Yₜ.c.ρq_rai += ᶜρaʲs.:($$j) * ᶜSqᵣᵐʲs.:($$j)
        @. Yₜ.c.ρq_sno += ᶜρaʲs.:($$j) * ᶜSqₛᵐʲs.:($$j)
    end
end

# 1M with PrognosticEDMFX
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics1Moment, turbconv_model::PrognosticEDMFX,
)
    # Source terms from EDMFX updrafts
    (; ᶜSqₗᵐʲs, ᶜSqᵢᵐʲs, ᶜSqᵣᵐʲs, ᶜSqₛᵐʲs) = p.precomputed
    # Source terms from EDMFX environment
    (; ᶜSqₗᵐ⁰, ᶜSqᵢᵐ⁰, ᶜSqᵣᵐ⁰, ᶜSqₛᵐ⁰) = p.precomputed

    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    # Tendencies are already limited in the cache; just scale by density
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜSqₗᵐ⁰
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜSqᵢᵐ⁰
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜSqᵣᵐ⁰
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜSqₛᵐ⁰

    # Update from the updraft microphysics tendencies
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += Y.c.sgsʲs.:($$j).ρa * ᶜSqₗᵐʲs.:($$j)
        @. Yₜ.c.ρq_ice += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵢᵐʲs.:($$j)
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵣᵐʲs.:($$j)
        @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜSqₛᵐʲs.:($$j)
    end
end

#####
##### QuadratureMicrophysics
#####

# Generic fallback: delegate to base model
function microphysics_tendency!(Yₜ, Y, p, t,
    moisture_model, qm::QuadratureMicrophysics, turbconv_model,
)
    return microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        moisture_model,
        qm.base_model,
        turbconv_model,
    )
end

function microphysics_tendency!(Yₜ, Y, p, t,
    moisture_model::NonEquilMoistModel, qm::QuadratureMicrophysics{Microphysics1Moment},
    turbconv_model::DiagnosticEDMFX,
)
    return microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        moisture_model,
        qm.base_model,
        turbconv_model,
    )
end
function microphysics_tendency!(Yₜ, Y, p, t,
    moisture_model::NonEquilMoistModel, qm::QuadratureMicrophysics{Microphysics1Moment},
    turbconv_model::PrognosticEDMFX,
)
    return microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        moisture_model,
        qm.base_model,
        turbconv_model,
    )
end

#####
##### 2-Moment Microphysics
#####

function microphysics_tendency!(Yₜ, Y, p, t,
    ::DryModel, ::Microphysics2Moment, _,
)
    error("Microphysics2Moment is incompatible with DryModel")
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::EquilMoistModel, ::Microphysics2Moment, _,
)
    error("Microphysics2Moment is not implemented for EquilMoistModel")
end

function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics2Moment, _,
)
    (; ᶜT, ᶜp, ᶜu) = p.precomputed
    (; ᶜSqₗᵐ, ᶜSqᵢᵐ, ᶜSqᵣᵐ, ᶜSqₛᵐ) = p.precomputed
    (; ᶜSnₗᵐ, ᶜSnᵣᵐ) = p.precomputed
    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_2m_params(params)

    # --- Aerosol activation (ARG 2000) — requires prescribed aerosols ---
    # Cloud condensation is already included in the fused BMT cache tendencies.
    if hasproperty(p, :tracers) &&
       hasproperty(p.tracers, :prescribed_aerosols_field)
        seasalt_num = p.scratch.ᶜtemp_scalar
        seasalt_mean_radius = p.scratch.ᶜtemp_scalar_2
        sulfate_num = p.scratch.ᶜtemp_scalar_3

        compute_prescribed_aerosol_properties!(
            seasalt_num, seasalt_mean_radius, sulfate_num,
            p.tracers.prescribed_aerosols_field, params.prescribed_aerosol_params,
        )

        aerosol_params = params.prescribed_aerosol_params
        act_params = CAP.microphysics_cloud_params(params).activation
        ᶜw = @. lazy(w_component(Geometry.WVector(ᶜu)))
        @. Yₜ.c.ρn_liq +=
            Y.c.ρ * aerosol_activation_sources(
                (act_params,),
                seasalt_num, seasalt_mean_radius, sulfate_num,
                specific(Y.c.ρq_tot, Y.c.ρ),
                specific(Y.c.ρq_liq + Y.c.ρq_rai, Y.c.ρ),
                specific(Y.c.ρq_ice + Y.c.ρq_sno, Y.c.ρ),
                specific(Y.c.ρn_liq + Y.c.ρn_rai, Y.c.ρ),
                Y.c.ρ,
                ᶜw,
                (cmp,), thp, ᶜT, ᶜp, dt,
                (aerosol_params,),
            )
    end

    # --- Microphysics tendencies ---
    # Tendencies are already limited in the cache; just scale by density
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜSqₗᵐ
    @. Yₜ.c.ρn_liq += Y.c.ρ * ᶜSnₗᵐ
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜSqᵣᵐ
    @. Yₜ.c.ρn_rai += Y.c.ρ * ᶜSnᵣᵐ

    # Ice and snow (zero in warm-rain 2M, will be nonzero when cold processes are added)
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜSqᵢᵐ
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜSqₛᵐ

    return nothing
end

# 2M with DiagnosticEDMFX
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics2Moment, ::DiagnosticEDMFX,
)
    error("Microphysics2Moment is not implemented for DiagnosticEDMFX")
end

# 2M with PrognosticEDMFX
function microphysics_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Microphysics2Moment, turbconv_model::PrognosticEDMFX,
)
    # Source terms from EDMFX updrafts
    (; ᶜSqₗᵐʲs, ᶜSqᵢᵐʲs, ᶜSqᵣᵐʲs, ᶜSqₛᵐʲs, ᶜSnₗᵐʲs, ᶜSnᵣᵐʲs) = p.precomputed
    # Source terms from EDMFX environment
    (; ᶜSqₗᵐ⁰, ᶜSqᵢᵐ⁰, ᶜSqᵣᵐ⁰, ᶜSqₛᵐ⁰, ᶜSnₗᵐ⁰, ᶜSnᵣᵐ⁰) = p.precomputed

    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    # Tendencies are already limited in the cache; just scale by density
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜSqₗᵐ⁰
    @. Yₜ.c.ρn_liq += ᶜρa⁰ * ᶜSnₗᵐ⁰
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜSqᵣᵐ⁰
    @. Yₜ.c.ρn_rai += ᶜρa⁰ * ᶜSnᵣᵐ⁰

    # Ice and snow (zero in warm-rain 2M)
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜSqᵢᵐ⁰
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜSqₛᵐ⁰

    # Update from the updraft microphysics tendencies
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        # Tendencies already limited in cache; just scale by density
        @. Yₜ.c.ρq_liq += Y.c.sgsʲs.:($$j).ρa * ᶜSqₗᵐʲs.:($$j)
        @. Yₜ.c.ρn_liq += Y.c.sgsʲs.:($$j).ρa * ᶜSnₗᵐʲs.:($$j)
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵣᵐʲs.:($$j)
        @. Yₜ.c.ρn_rai += Y.c.sgsʲs.:($$j).ρa * ᶜSnᵣᵐʲs.:($$j)

        # Ice and snow (zero in warm-rain 2M)
        @. Yₜ.c.ρq_ice += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵢᵐʲs.:($$j)
        @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜSqₛᵐʲs.:($$j)
    end
end

function microphysics_tendency!(Yₜ, Y, p, t,
    moisture_model::NonEquilMoistModel, qm::QuadratureMicrophysics{Microphysics2Moment},
    turbconv_model::DiagnosticEDMFX,
)
    return microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        moisture_model,
        qm.base_model,
        turbconv_model,
    )
end
function microphysics_tendency!(Yₜ, Y, p, t,
    moisture_model::NonEquilMoistModel, qm::QuadratureMicrophysics{Microphysics2Moment},
    turbconv_model::PrognosticEDMFX,
)
    return microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        moisture_model,
        qm.base_model,
        turbconv_model,
    )
end

#####
##### 2-Moment Microphysics with P3 Ice Scheme
#####

function microphysics_tendency!(Yₜ, Y, p, t,
    ne::NonEquilMoistModel, ::Microphysics2MomentP3, ::Nothing,
)
    (; ᶜScoll) = p.precomputed

    # 2 moment scheme (warm)
    microphysics_tendency!(Yₜ, Y, p, t, ne, Microphysics2Moment(), nothing)

    # P3 scheme (cold) - collisions
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜScoll.∂ₜq_c
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜScoll.∂ₜq_r
    @. Yₜ.c.ρn_liq += ᶜScoll.∂ₜN_c
    @. Yₜ.c.ρn_rai += ᶜScoll.∂ₜN_r
    @. Yₜ.c.ρq_rim += ᶜScoll.∂ₜL_rim
    @. Yₜ.c.ρq_ice += ᶜScoll.∂ₜL_ice
    @. Yₜ.c.ρb_rim += ᶜScoll.∂ₜB_rim

    return nothing
end
