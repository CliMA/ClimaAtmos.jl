#####
##### DryModel, EquilMoistModel
#####

cloud_condensate_tendency!(Yₜ, p, colidx, _) = nothing

#####
##### NonEquilMoistModel
#####

function cloud_condensate_tendency!(Yₜ, p, colidx, ::NonEquilMoistModel)

    (; ᶜts) = p.precomputed
    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    @. Yₜ.c.ρq_liq[colidx] += cloud_sources(cmc.liquid, thp, ᶜts[colidx], dt)
    @. Yₜ.c.ρq_ice[colidx] += cloud_sources(cmc.ice, thp, ᶜts[colidx], dt)
end
