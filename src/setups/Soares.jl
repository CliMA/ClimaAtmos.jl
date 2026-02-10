"""
    Soares

The Soares setup described in [Soares2004](@cite), with a hydrostatically
balanced pressure profile. Profiles are sourced from AtmosphericProfilesLibrary.

## Example
```julia
setup = Soares(; prognostic_tke = true, thermo_params)
```
"""
struct Soares{P}
    prognostic_tke::Bool
    profiles::P
end

Soares(; prognostic_tke::Bool, thermo_params) =
    Soares(prognostic_tke, soares_profiles(thermo_params))

function soares_profiles(thermo_params)
    FT = eltype(thermo_params)
    θ = APL.Soares_θ_liq_ice(FT)
    q_tot = APL.Soares_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0 = FT(100000.0), θ, q_tot)
    u = APL.Soares_u(FT)
    tke = APL.Soares_tke_prescribed(FT)
    return (; θ, q_tot, p, u, tke)
end

function center_initial_condition(setup::Soares, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    θ = profiles.θ(z)
    q_tot = profiles.q_tot(z)
    p = profiles.p(z)
    T = TD.air_temperature(thermo_params, TD.pθ_li(), p, θ, q_tot)
    tke = prognostic_tke ? FT(0) : profiles.tke(z)

    return physical_state(; T, p, q_tot, u = profiles.u(z), tke)
end

function surface_condition(::Soares, params)
    FT = eltype(params)
    parameterization = MoninObukhov(;
        z0 = FT(0.16),
        θ_flux = FT(0.06),
        q_flux = FT(2.5e-5),
        ustar = FT(0.28),
    )
    return SurfaceState(;
        parameterization, T = FT(300), p = FT(1e5), q_vap = FT(5e-3),
    )
end
