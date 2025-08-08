#####
##### DryModel, EquilMoistModel
#####

cloud_condensate_tendency!(Yₜ, Y, p, _, _, _) = nothing

#####
##### NonEquilMoistModel
#####

function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Union{NoPrecipitation, Microphysics0Moment},
    _,
)
    error(
        "NonEquilMoistModel can only be run with Microphysics1Moment or Microphysics2Moment precipitation",
    )
end

function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Microphysics1Moment,
    _,
)
    (; ᶜts) = p.precomputed
    (; params, dt) = p
    FT = eltype(params)
    thp = CAP.thermodynamics_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    Tₐ = @. lazy(TD.air_temperature(thp, ᶜts))

    @. Yₜ.c.ρq_liq +=
        Y.c.ρ * cloud_sources(
            cmc.liquid,
            thp,
            specific(Y.c.ρq_tot, Y.c.ρ),
            specific(Y.c.ρq_liq, Y.c.ρ),
            specific(Y.c.ρq_ice, Y.c.ρ),
            specific(Y.c.ρq_rai, Y.c.ρ),
            specific(Y.c.ρq_sno, Y.c.ρ),
            Y.c.ρ,
            Tₐ,
            dt,
        )
    @. Yₜ.c.ρq_ice +=
        Y.c.ρ * cloud_sources(
            cmc.ice,
            thp,
            specific(Y.c.ρq_tot, Y.c.ρ),
            specific(Y.c.ρq_liq, Y.c.ρ),
            specific(Y.c.ρq_ice, Y.c.ρ),
            specific(Y.c.ρq_rai, Y.c.ρ),
            specific(Y.c.ρq_sno, Y.c.ρ),
            Y.c.ρ,
            Tₐ,
            dt,
        )
end

function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Microphysics2Moment,
    _,
)
    (; ᶜts, ᶜu) = p.precomputed
    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_2m_params(params)

    Tₐ = @. lazy(TD.air_temperature(thp, ᶜts))

    @. Yₜ.c.ρq_liq +=
        Y.c.ρ * cloud_sources(
            cmp.liquid,
            thp,
            specific(Y.c.ρq_tot, Y.c.ρ),
            specific(Y.c.ρq_liq, Y.c.ρ),
            specific(Y.c.ρq_ice, Y.c.ρ),
            specific(Y.c.ρq_rai, Y.c.ρ),
            specific(Y.c.ρq_sno, Y.c.ρ),
            Y.c.ρ,
            Tₐ,
            dt,
        )
    @. Yₜ.c.ρq_ice +=
        Y.c.ρ * cloud_sources(
            cmp.ice,
            thp,
            specific(Y.c.ρq_tot, Y.c.ρ),
            specific(Y.c.ρq_liq, Y.c.ρ),
            specific(Y.c.ρq_ice, Y.c.ρ),
            specific(Y.c.ρq_rai, Y.c.ρ),
            specific(Y.c.ρq_sno, Y.c.ρ),
            Y.c.ρ,
            Tₐ,
            dt,
        )

    # Aerosol activation using prescribed aerosol (Sea salt and sulfate)
    if !(:tracers in propertynames(p)) ||
       !(:prescribed_aerosols_field in propertynames(p.tracers))
        return
    end
    seasalt_num = p.scratch.ᶜtemp_scalar
    seasalt_mean_radius = p.scratch.ᶜtemp_scalar_2
    sulfate_num = p.scratch.ᶜtemp_scalar_3
    compute_prescribed_aerosol_properties!(
        seasalt_num,
        seasalt_mean_radius,
        sulfate_num,
        p.tracers.prescribed_aerosols_field,
        cmp.aerosol,
    )

    # Compute aerosol activation (ARG 2000)
    @. Yₜ.c.ρn_liq +=
        Y.c.ρ * aerosol_activation_sources(
            seasalt_num,
            seasalt_mean_radius,
            sulfate_num,
            specific(Y.c.ρq_tot, Y.c.ρ),
            specific(Y.c.ρq_liq + Y.c.ρq_rai, Y.c.ρ),
            specific(Y.c.ρq_ice + Y.c.ρq_sno, Y.c.ρ),
            specific(Y.c.ρn_liq + Y.c.ρn_rai, Y.c.ρ),
            Y.c.ρ,
            w_component.(Geometry.WVector.(ᶜu)),
            (cmp,),
            thp,
            ᶜts,
            dt,
        )
end

#####
##### PrognosticEDMF and DiagnosticEDMF
#####

function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Union{NoPrecipitation, Microphysics0Moment},
    ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    nothing
end
function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Microphysics1Moment,
    ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    nothing
end
function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Microphysics2Moment,
    ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    nothing
end
