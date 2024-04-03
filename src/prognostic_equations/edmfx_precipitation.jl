#####
##### EDMF precipitation
#####

edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model,
    precip_model,
) = nothing

function edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::PrognosticEDMFX,
    precip_model::Microphysics0Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜS_q_totʲs, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜS_q_totʲs.:($$j)[colidx]

        @. Yₜ.c.sgsʲs.:($$j).mse[colidx] +=
            ᶜS_q_totʲs.:($$j)[colidx] * (
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜtsʲs.:($$j)[colidx],
                    ᶜΦ[colidx],
                ) - TD.internal_energy(thermo_params, ᶜtsʲs.:($$j)[colidx])
            )

        @. Yₜ.c.sgsʲs.:($$j).q_tot[colidx] +=
            ᶜS_q_totʲs.:($$j)[colidx] * (1 - Y.c.sgsʲs.:($$j).q_tot[colidx])
    end
    return nothing
end

function edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::PrognosticEDMFX,
    precip_model::Microphysics1Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜtsʲs) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜSqₜᵖʲs.:($$j)[colidx]

        @. Yₜ.c.sgsʲs.:($$j).mse[colidx] +=
            ᶜSeₜᵖʲs.:($$j)[colidx] -
            ᶜSqₜᵖʲs.:($$j)[colidx] *
            TD.internal_energy(thp, ᶜtsʲs.:($$j)[colidx])

        @. Yₜ.c.sgsʲs.:($$j).q_tot[colidx] +=
            ᶜSqₜᵖʲs.:($$j)[colidx] * (1 - Y.c.sgsʲs.:($$j).q_tot[colidx])
    end
    return nothing
end
