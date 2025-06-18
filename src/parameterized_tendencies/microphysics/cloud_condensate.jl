#####
##### DryModel, EquilMoistModel
#####

cloud_condensate_tendency!(Yₜ, Y, p, _, _) = nothing

#####
##### NonEquilMoistModel
#####

function cloud_condensate_tendency!(
    Yₜ,
    Y,
    p,
    ::NonEquilMoistModel,
    ::Union{NoPrecipitation, Microphysics0Moment},
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

    @. Yₜ.c.ρn_liq +=
        Y.c.ρ * aerosol_activation_sources(
            cmc.liquid,
            thp,
            Y.c.ρ,
            Tₐ,
            specific(Y.c.ρq_tot, Y.c.ρ),
            specific(Y.c.ρq_liq, Y.c.ρ),
            specific(Y.c.ρq_ice, Y.c.ρ),
            specific(Y.c.ρq_rai, Y.c.ρ),
            specific(Y.c.ρq_sno, Y.c.ρ),
            specific(Y.c.ρn_liq, Y.c.ρ),
            cmc.N_cloud_liquid_droplets,
            dt,
        )
end
