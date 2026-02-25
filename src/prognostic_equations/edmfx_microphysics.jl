#####
##### EDMF precipitation tendency
#####

"""
    edmfx_microphysics_tendency!(Yₜ, Y, p, t, turbconv_model, microphysics_model)

Applies microphysics tendencies to the EDMFX prognostic variables.

This function calculates and applies the changes in updraft prognostic variables
(`ρa`, `mse`, `q_tot`, `q_liq`, `q_ice`, `q_rai`, `q_sno`) due to microphysics
processes within the EDMFX framework. The microphysics tendencies
(`ᶜSqₜᵐʲs`, `ᶜSqₗᵐʲs`, etc., defined to be positive when representing a source) 
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

edmfx_microphysics_tendency!(Yₜ, Y, p, t, turbconv_model, microphysics_model) =
    nothing

function edmfx_microphysics_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::Microphysics0Moment,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜSqₜᵐʲs, ᶜTʲs, ᶜq_tot_safeʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core

    for j in 1:n

        @. Yₜ.c.sgsʲs.:($$j).ρa += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵐʲs.:($$j)

        @. Yₜ.c.sgsʲs.:($$j).mse +=
            ᶜSqₜᵐʲs.:($$j) * (
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜTʲs.:($$j),
                    ᶜq_liq_raiʲs.:($$j),
                    ᶜq_ice_snoʲs.:($$j),
                    ᶜΦ,
                ) - TD.internal_energy(thermo_params, ᶜTʲs.:($$j))
            )

        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            ᶜSqₜᵐʲs.:($$j) * (1 - Y.c.sgsʲs.:($$j).q_tot)
    end
    return nothing
end

function edmfx_microphysics_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::Union{
        Microphysics1Moment,
        QuadratureMicrophysics{Microphysics1Moment},
    },
)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜSqₗᵐʲs, ᶜSqᵢᵐʲs, ᶜSqᵣᵐʲs, ᶜSqₛᵐʲs) = p.precomputed

    # TODO what about the mass end energy outflow via bottom boundary?

    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜSqₗᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜSqᵢᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜSqᵣᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜSqₛᵐʲs.:($$j)
    end
    return nothing
end

function edmfx_microphysics_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::Union{
        Microphysics2Moment,
        QuadratureMicrophysics{Microphysics2Moment},
    },
)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜSqₗᵐʲs, ᶜSqᵢᵐʲs, ᶜSqᵣᵐʲs, ᶜSqₛᵐʲs, ᶜSnₗᵐʲs, ᶜSnᵣᵐʲs) = p.precomputed

    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).q_liq += ᶜSqₗᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_ice += ᶜSqᵢᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_rai += ᶜSqᵣᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).q_sno += ᶜSqₛᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).n_liq += ᶜSnₗᵐʲs.:($$j)
        @. Yₜ.c.sgsʲs.:($$j).n_rai += ᶜSnᵣᵐʲs.:($$j)
    end
    return nothing
end
