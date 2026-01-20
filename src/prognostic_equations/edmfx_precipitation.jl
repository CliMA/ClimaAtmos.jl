#####
##### EDMF precipitation tendency
#####

"""
    edmfx_precipitation_tendency!(Yₜ, Y, p, t, turbconv_model, microphysics_model)

Applies precipitation tendencies to the EDMFX prognostic variables.

This function calculates and applies the changes in updraft prognostic variables
(`ρa`, `mse`, `q_tot`, `q_liq`, `q_ice`, `q_rai`, `q_sno`) due to precipitation
processes within the EDMFX framework. The specific precipitation sources
(`ᶜSqₜᵖʲs`, `ᶜSqₗᵖʲs`, etc., defined to be positive when representing a source) 
are precomputed and stored in `p.precomputed`.

The tendencies are applied to the updraft prognostic variables (`Yₜ.c.sgsʲs.:(j)`)
for each mass flux subdomain `j`.

Arguments:
- `Yₜ`: The tendency state vector.
- `Y`: The current state vector.
- `p`: The cache, containing precomputed quantities and parameters.
- `t`: The current simulation time.
- `turbconv_model`: The turbulence convection model (e.g., `PrognosticEDMFX`).
- `microphysics_model`: The precipitation model (e.g., `Microphysics0Moment`,
  `Microphysics1Moment`, `Microphysics2Moment`).

Returns: `nothing`, modifies `Yₜ` in place.
"""

edmfx_precipitation_tendency!(Yₜ, Y, p, t, turbconv_model, microphysics_model) =
    nothing

function edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::Microphysics0Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜSqₜᵖʲs, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)

        ᶜTʲ = @. lazy(TD.air_temperature(thermo_params, ᶜtsʲs.:($$j)))
        ᶜq_liq_raiʲ = @. lazy(TD.liquid_specific_humidity(thermo_params, ᶜtsʲs.:($$j)))
        ᶜq_ice_snoʲ = @. lazy(TD.ice_specific_humidity(thermo_params, ᶜtsʲs.:($$j)))
        @. Yₜ.c.sgsʲs.:($$j).mse +=
            ᶜSqₜᵖʲs.:($$j) * (
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜTʲ,
                    ᶜq_liq_raiʲ,
                    ᶜq_ice_snoʲ,
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
    microphysics_model::Microphysics1Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed

    # TODO what about the mass end energy outflow via bottom boundary?

    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜSqₗᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜSqᵢᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜSqᵣᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜSqₛᵖʲs.:($$j)
    end
    return nothing
end

function edmfx_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::Microphysics2Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs, ᶜSnₗᵖʲs, ᶜSnᵣᵖʲs) = p.precomputed

    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜSqₗᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜSqᵢᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜSqᵣᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜSqₛᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).n_liq += ᶜSnₗᵖʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).n_rai += ᶜSnᵣᵖʲs.:($$j)
    end
    return nothing
end
