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

    T = p.scratch.ᶜtemp_scalar
    pᵥ_sat_liq = p.scratch.ᶜtemp_scalar_2
    qᵥ_sat_liq = p.scratch.ᶜtemp_scalar_3
    dqsldT = p.scratch.ᶜtemp_scalar_4
    Γₗ = p.scratch.ᶜtemp_scalar_5

    @. T = TD.air_temperature(thp, ᶜts)
    @assert sum(isnan, T) == 0
    if minimum(T) < FT(200)
        @info(" ", TD.air_density(thp, ᶜts), extrema(T))
        @info(" ", extrema(TD.PhasePartition(thp, ᶜts).tot), extrema(TD.PhasePartition(thp, ᶜts).liq), extrema(TD.PhasePartition(thp, ᶜts).ice))
        @info(" ", extrema(q_rai), extrema(q_sno))
    end
    @assert minimum(T) > FT(0)

    @. pᵥ_sat_liq = TD.saturation_vapor_pressure(thp, T, TD.Liquid())
    @assert sum(isnan, pᵥ_sat_liq) == 0

    @. qᵥ_sat_liq = TD.q_vap_saturation_from_density(thp, T, TD.air_density(thp, ᶜts), pᵥ_sat_liq)
    @assert sum(isnan, qᵥ_sat_liq) == 0

    @. dqsldT = qᵥ_sat_liq * (TD.latent_heat_vapor(thp, T) / (TD.Parameters.R_v(thp) * T^2) - 1 / T)
    @assert sum(isnan, dqsldT) == 0

    @. Γₗ = FT(1) + (TD.latent_heat_vapor(thp, T) / TD.cp_m(thp, TD.PhasePartition(thp, ᶜts))) * dqsldT
    @assert sum(isnan, Γₗ) == 0

    @. p.scratch.tmp_cloud_liquid_src = Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. p.scratch.tmp_cloud_ice_src = Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    @. Yₜ.c.ρq_liq += Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. Yₜ.c.ρq_ice += Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    @assert sum(isnan, Y.c.ρq_liq) == 0
    @assert sum(isnan, Y.c.ρq_ice) == 0
    @assert sum(isnan, Yₜ.c.ρq_liq) == 0
    @assert sum(isnan, Yₜ.c.ρq_ice) == 0

end
