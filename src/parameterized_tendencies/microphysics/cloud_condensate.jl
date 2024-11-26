#####
##### DryModel, EquilMoistModel
#####

cloud_condensate_tendency!(Yₜ, p, _) = nothing

#####
##### NonEquilMoistModel
#####

function cloud_condensate_tendency!(Yₜ, p, ::NonEquilMoistModel)

    (; ᶜts) = p.precomputed
    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    @fused_direct begin
        @. Yₜ.c.ρq_liq += cloud_sources(cmc.liquid, thp, ᶜts, dt)
        @. Yₜ.c.ρq_ice += cloud_sources(cmc.ice, thp, ᶜts, dt)
    end
end
