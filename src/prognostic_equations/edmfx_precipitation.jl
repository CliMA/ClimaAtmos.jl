#####
##### EDMF precipitation
#####

edmfx_precipitation_tendency!(Yₜ, Y, p, t, turbconv_model, precip_model) =
    nothing

function edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    precip_model::Microphysics0Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜSqₜᵖʲs, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    grav = TDP.grav(thermo_params)
    ᶜz = Fields.coordinate_field(axes(Y.c)).z

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)

        @. Yₜ.c.sgsʲs.:($$j).mse +=
            ᶜSqₜᵖʲs.:($$j) * (
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜtsʲs.:($$j),
                    Φ(grav, ᶜz),
                ) - TD.internal_energy(thermo_params, ᶜtsʲs.:($$j))
            )

        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            ᶜSqₜᵖʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_tot)
    end
    return nothing
end

function edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    precip_model::Microphysics1Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜtsʲs) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)

        @. Yₜ.c.sgsʲs.:($$j).mse +=
            ᶜSeₜᵖʲs.:($$j) -
            ᶜSqₜᵖʲs.:($$j) * TD.internal_energy(thp, ᶜtsʲs.:($$j))

        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            ᶜSqₜᵖʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_tot)
    end
    return nothing
end
