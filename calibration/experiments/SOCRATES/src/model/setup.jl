"""
The SOCRATES setup.

ClimaAtmos's `Setups.ForcingFromFile` already composes exactly what a SOCRATES case needs — an
initial condition read from a column dataset, a composed external forcing, prescribed surface
temperature from the dataset's `ts`, and insolation from its `coszen`/`rsdt` — and it takes a
`ColumnDataset` rather than a path. Since [`socrates_forcing`](@ref) produces an in-memory
`ColumnDataset`, no bespoke setup type is needed: this is a constructor call, and every
`Setups` interface method comes from upstream.
"""

using ClimaAtmos: ClimaAtmos as CA
using Dates: Dates

"""Default aerodynamic roughness length [m] for the SOCRATES ocean surface."""
const DEFAULT_Z0 = 1.0e-4

"""
    socrates_setup(FT, case; z, dt_sec, start_date, forcing_terms, z0, refresh)

The `Setups.ForcingFromFile` for `case`, backed by in-memory SSCF forcing on levels `z`.

Surface fluxes use Monin–Obukhov with roughness `z0`; surface temperature and insolation come from
the forcing dataset (`ExternalTemperature` and `ExternalTVInsolation`, the `ForcingFromFile`
defaults).
"""
function socrates_setup(
    ::Type{FT},
    case::SocratesCase;
    z::AbstractVector,
    dt_sec::Real = DEFAULT_FORCING_DT,
    start_date::Dates.DateTime = simulation_start_date(case),
    forcing_terms = default_socrates_forcing_terms(case),
    z0::Real = DEFAULT_Z0,
    refresh::Bool = false,
) where {FT <: AbstractFloat}
    forcing = socrates_forcing(FT, case; z, dt_sec, start_date, forcing_terms, refresh)
    return SocratesSetup(
        CA.Setups.ForcingFromFile(
            forcing.dataset,
            Dates.format(start_date, "yyyymmdd");
            forcing,
            flux_scheme = CA.SurfaceConditions.MoninObukhov(; z0 = FT(z0)),
        ),
    )
end

"""
    SocratesSetup(inner)

`inner` with the initial condensate split out of `q_tot` instead of left in the vapour.

`ColumnProfiles` carries no condensate, so `column_profiles_ic` hands `physical_state` only
`(T, ρ, q_tot)` and its `q_liq`/`q_ice` default to zero. The observed `q_tot` includes cloud water,
so the column starts supersaturated by the observed condensate and the microphysics deposits it over
the first steps, releasing latent heat the observed `T` already contains — the phase it lands in, and
hence the column's character, is then set by the sampled timescales rather than by the observations.

Every other `Setups` method is `inner`'s.
"""
struct SocratesSetup{S}
    inner::S
end

"""
    CA.Setups.center_initial_condition(setup::SocratesSetup, local_geometry, params)

The observed state with `q_tot` partitioned at liquid saturation: the excess over
`q_vap_saturation(T, ρ, Liquid())` becomes `q_liq`, and `q_ice` starts at zero. Nothing is left to
condense, so no latent heat is released and the column starts at the observed temperature.
"""
function CA.Setups.center_initial_condition(
    setup::SocratesSetup,
    local_geometry,
    params,
)
    profiles = setup.inner.profiles
    (; z) = local_geometry.coordinates
    FT = typeof(z)
    T = FT(profiles.T(z))
    ρ = FT(profiles.ρ(z))
    q_tot = FT(profiles.q_tot(z))
    thermo = CA.Parameters.thermodynamics_params(params)
    q_sat_liq = CA.TD.q_vap_saturation(thermo, T, ρ, CA.TD.Liquid())
    return CA.Setups.physical_state(;
        T,
        ρ,
        q_tot,
        q_liq = max(zero(FT), q_tot - FT(q_sat_liq)),
        q_ice = zero(FT),
        u = FT(profiles.u(z)),
        v = FT(profiles.v(z)),
        tke = zero(FT),
    )
end

CA.Setups.surface_condition(setup::SocratesSetup, params) =
    CA.Setups.surface_condition(setup.inner, params)

CA.Setups.external_forcing(setup::SocratesSetup, ::Type{FT}) where {FT} =
    CA.Setups.external_forcing(setup.inner, FT)

CA.Setups.insolation_model(setup::SocratesSetup) =
    CA.Setups.insolation_model(setup.inner)

CA.Setups.surface_temperature_model(setup::SocratesSetup) =
    CA.Setups.surface_temperature_model(setup.inner)
