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
        "NonEquilMoistModel can only be run with Microphysics1Moment precipitation",
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

    @assert sum(isnan, Y.c.ρq_liq) == 0
    @assert sum(isnan, Y.c.ρq_ice) == 0
    @assert sum(isnan, Yₜ.c.ρq_liq) == 0
    @assert sum(isnan, Yₜ.c.ρq_ice) == 0

    @. p.scratch.tmp_cloud_liquid_src = Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. p.scratch.tmp_cloud_ice_src = Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    @. Yₜ.c.ρq_liq += Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. Yₜ.c.ρq_ice += Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    @assert sum(isnan, Y.c.ρq_liq) == 0
    @assert sum(isnan, Y.c.ρq_ice) == 0
    @assert sum(isnan, Yₜ.c.ρq_liq) == 0
    @assert sum(isnan, Yₜ.c.ρq_ice) == 0

end
