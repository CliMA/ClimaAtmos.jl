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
    (; q_rai, q_sno) = p.precomputed.ᶜspecific
    FT = eltype(params)
    thp = CAP.thermodynamics_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    @. Yₜ.c.ρq_liq += Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. Yₜ.c.ρq_ice += Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)
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
    (; q_rai, q_sno) = p.precomputed.ᶜspecific
    FT = eltype(params)
    thp = CAP.thermodynamics_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    @. Yₜ.c.ρq_liq += Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. Yₜ.c.ρq_ice += Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    @. Yₜ.c.N_liq += aerosol_activation_sources(cmc.liquid, thp, ᶜts, Y.c.ρ, q_rai, Y.c.N_ice, cmc.N_cloud_liquid_droplets, dt)
    @. Yₜ.c.N_ice += aerosol_activation_sources(cmc.ice, thp, ᶜts, Y.c.ρ, q_sno, Y.c.N_ice, cmc.N_cloud_liquid_droplets, dt)
end
