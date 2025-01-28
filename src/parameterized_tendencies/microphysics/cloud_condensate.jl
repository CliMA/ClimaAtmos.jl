#####
##### DryModel, EquilMoistModel
#####

cloud_condensate_tendency!(Yₜ, p, _, _) = nothing

#####
##### NonEquilMoistModel
#####

function cloud_condensate_tendency!(
    Yₜ,
    p,
    ::NonEquilMoistModel,
    ::Union{NoPrecipitation, Microphysics0Moment},
)
    error(
        "NonEquilMoistModel can only be run with Microphysics1Moment precipitation",
    )
end
function cloud_condensate_tendency!(
    Yₜ,
    p,
    ::NonEquilMoistModel,
    ::Microphysics1Moment,
)
    (; ᶜts) = p.precomputed
    (; q_rai, q_sno) = p.precomputed.ᶜspecific
    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    @. Yₜ.c.ρq_liq += cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. Yₜ.c.ρq_ice += cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)
end
