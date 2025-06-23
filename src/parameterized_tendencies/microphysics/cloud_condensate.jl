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
    if !(:prescribed_aerosols_field in propertynames(p.tracers))
        return
    end
    seasalt_num = p.scratch.ᶜtemp_scalar
    seasalt_mean_radius = p.scratch.ᶜtemp_scalar_2
    sulfate_num = p.scratch.ᶜtemp_scalar_3
    @. seasalt_num = 0
    @. seasalt_mean_radius = 0
    @. sulfate_num = 0
    # Get aerosol concentrations if available
    seasalt_names = [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
    sulfate_names = [:SO4]
    aerosol_field = p.tracers.prescribed_aerosols_field
    for aerosol_name in propertynames(aerosol_field)
        if aerosol_name in seasalt_names
            seasalt_particle_radius = getproperty(
                cmp.aerosol,
                Symbol(string(aerosol_name) * "_radius"),
            )
            seasalt_particle_mass =
                FT(4 / 3 * pi) *
                seasalt_particle_radius^3 *
                cmp.aerosol.seasalt_density
            seasalt_mass = getproperty(aerosol_field, aerosol_name)
            @. seasalt_num += seasalt_mass / seasalt_particle_mass
            @. seasalt_mean_radius +=
                seasalt_mass / seasalt_particle_mass *
                log(seasalt_particle_radius)
        elseif aerosol_name in sulfate_names
            sulfate_particle_mass =
                FT(4 / 3 * pi) *
                cmp.aerosol.sulfate_radius^3 *
                cmp.aerosol.sulfate_density
            sulfate_mass = getproperty(aerosol_field, aerosol_name)
            @. sulfate_num += sulfate_mass / sulfate_particle_mass
        end
    end
    # Compute geometric mean radius of the log-normal distribution:
    # exp(weighted average of log(radius))
    @. seasalt_mean_radius =
        ifelse(seasalt_num == 0, 0, exp(seasalt_mean_radius / seasalt_num))

    # Compute aerosol activation (ARG 2000)
    (; ᶜu) = p.precomputed
    ᶜw = p.scratch.ᶜtemp_scalar_4
    Snₗ = p.scratch.ᶜtemp_scalar_5
    @. ᶜw = max(0, w_component.(Geometry.WVector.(ᶜu)))
    @. Snₗ = aerosol_activation_sources(
        seasalt_num,
        seasalt_mean_radius,
        sulfate_num,
        specific(Y.c.ρq_tot, Y.c.ρ),
        specific(Y.c.ρq_liq + Y.c.ρq_rai, Y.c.ρ),
        specific(Y.c.ρq_ice + Y.c.ρq_sno, Y.c.ρ),
        specific(Y.c.ρn_liq + Y.c.ρn_rai, Y.c.ρ),
        Y.c.ρ,
        ᶜw,
        (cmp,),
        thp,
        ᶜts,
        dt,
    )
    @. Yₜ.c.ρn_liq += Y.c.ρ * Snₗ
end
