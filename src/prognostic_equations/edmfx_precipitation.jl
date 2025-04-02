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
    (; ᶜΦ) = p.core

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)

        @. Yₜ.c.sgsʲs.:($$j).mse +=
            ᶜSqₜᵖʲs.:($$j) * (
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜtsʲs.:($$j),
                    ᶜΦ,
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

    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed

    # TODO what about the mass end energy outflow via bottom boundary?

    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).q_liq +=
            ᶜSqₗᵖʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_liq)

        @. Yₜ.c.sgsʲs.:($$j).q_ice +=
            ᶜSqᵢᵖʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_ice)

        @. Yₜ.c.sgsʲs.:($$j).q_rai +=
            ᶜSqᵣᵖʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_rai)

        @. Yₜ.c.sgsʲs.:($$j).q_sno +=
            ᶜSqₛᵖʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_sno)
    end
    return nothing
end
