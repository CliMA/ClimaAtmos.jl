#####
##### DryModel, EquilMoistModel
#####
import ClimaCore: Spaces, Quadratures, Fields

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

function column_iterator_indices(field)
    axes(field) isa Union{Spaces.PointSpace, Spaces.FiniteDifferenceSpace} &&
        return ((1, 1, 1),)
    horz_space = Spaces.horizontal_space(axes(field))
    qs = 1:Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
    hs = Spaces.eachslabindex(horz_space)
    return horz_space isa Spaces.SpectralElementSpace1D ?
           Iterators.product(qs, hs) : Iterators.product(qs, qs, hs)
end

column_iterator_indices(field_vector::Fields.FieldVector) =
    column_iterator_indices(first(Fields._values(field_vector)))

column_iterator(iterable) =
    Iterators.map(column_iterator_indices(iterable)) do (indices...,)
        Fields.column(iterable, indices...)
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
    @. T = TD.air_temperature(thp, ᶜts)
    @assert sum(isnan, T) == 0

    #if minimum(T) < FT(200)
    #    ρ = p.scratch.ᶜtemp_scalar_2
    #    @. ρ = TD.air_density(thp, ᶜts)

    #    q_tot = p.scratch.ᶜtemp_scalar_3
    #    @. q_tot = TD.PhasePartition(thp, ᶜts).tot

    #    q_liq = p.scratch.ᶜtemp_scalar_4
    #    @. q_liq = TD.PhasePartition(thp, ᶜts).liq

    #    q_ice = p.scratch.ᶜtemp_scalar_5
    #    @. q_ice = TD.PhasePartition(thp, ᶜts).ice

    #    @info(" ", extrema(ρ), extrema(T))
    #    @info(" ", extrema(q_tot), extrema(q_liq), extrema(q_ice))
    #    @info(" ", extrema(q_rai), extrema(q_sno))
    #end
    @assert minimum(T) > FT(0)

    pᵥ_sat_liq = p.scratch.ᶜtemp_scalar_2
    @. pᵥ_sat_liq = TD.saturation_vapor_pressure(thp, T, TD.Liquid())
    @assert sum(isnan, pᵥ_sat_liq) == 0

    qᵥ_sat_liq = p.scratch.ᶜtemp_scalar_3
    @. qᵥ_sat_liq = TD.q_vap_saturation_from_density(thp, T, TD.air_density(thp, ᶜts), pᵥ_sat_liq)
    @assert sum(isnan, qᵥ_sat_liq) == 0

    dqsldT = p.scratch.ᶜtemp_scalar_4
    @. dqsldT = qᵥ_sat_liq * (TD.latent_heat_vapor(thp, T) / (TD.Parameters.R_v(thp) * T^2) - 1 / T)
    @assert sum(isnan, dqsldT) == 0

    Γₗ = p.scratch.ᶜtemp_scalar_5
    @. Γₗ = FT(1) + (TD.latent_heat_vapor(thp, T) / TD.cp_m(thp, TD.PhasePartition(thp, ᶜts))) * dqsldT
    @assert sum(isnan, Γₗ) == 0

    @. p.scratch.tmp_cloud_liquid_src = Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. p.scratch.tmp_cloud_ice_src = Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    for (Y_col, ql_src_col, qi_src_col ,ts_col, c_spec_col) in zip(
        column_iterator(Y),
        column_iterator(p.scratch.tmp_cloud_liquid_src),
        column_iterator(p.scratch.tmp_cloud_ice_src),
        column_iterator(ᶜts),
        column_iterator(ᶜspecific),
   )
       if minimum(TD.air_temperature.(thp, ts_col)) < FT(170)
          @show(Fields.coordinate_field(Y_col.c.ρ))
          @show(TD.air_temperature.(thp, ts_col))
          @show(TD.air_density.(thp, ts_col))
          @show(TD.PhasePartition.(thp, ts_col).tot)
          @show(TD.PhasePartition.(thp, ts_col).liq)
          @show(TD.PhasePartition.(thp, ts_col).ice)
          @show(c_spec_col.q_rai)
          @show(c_spec_col.q_sno)
          @show(ql_src_col)
          @show(qi_src_col)
      end
    end

    @. Yₜ.c.ρq_liq += Y.c.ρ * cloud_sources(cmc.liquid, thp, ᶜts, q_rai, dt)
    @. Yₜ.c.ρq_ice += Y.c.ρ * cloud_sources(cmc.ice, thp, ᶜts, q_sno, dt)

    @assert sum(isnan, Y.c.ρq_liq) == 0
    @assert sum(isnan, Y.c.ρq_ice) == 0
    @assert sum(isnan, Yₜ.c.ρq_liq) == 0
    @assert sum(isnan, Yₜ.c.ρq_ice) == 0

end
