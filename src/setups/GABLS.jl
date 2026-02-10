"""
    GABLS

The GABLS setup described in [Kosovic2000](@cite), with a hydrostatically
balanced pressure profile. Profiles are sourced from AtmosphericProfilesLibrary.

Surface temperature is time-varying: `T = 265 - 0.25t/3600`.

## Example
```julia
setup = GABLS(; prognostic_tke = true, thermo_params)
```
"""
struct GABLS{P}
    prognostic_tke::Bool
    profiles::P
end

GABLS(; prognostic_tke::Bool, thermo_params) =
    GABLS(prognostic_tke, gabls_profiles(thermo_params))

function gabls_profiles(thermo_params)
    FT = eltype(thermo_params)
    θ = APL.GABLS_θ_liq_ice(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0 = FT(100000.0), θ)
    u = APL.GABLS_u(FT)
    tke = APL.GABLS_tke_prescribed(FT)
    return (; θ, p, u, tke)
end

function center_initial_condition(setup::GABLS, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    θ = profiles.θ(z)
    p = profiles.p(z)
    T = TD.air_temperature(thermo_params, TD.pθ_li(), p, θ)
    tke = prognostic_tke ? FT(0) : profiles.tke(z)

    return physical_state(; T, p, u = profiles.u(z), tke)
end

function surface_condition(::GABLS, params)
    FT = eltype(params)
    p = FT(1e5)
    q_vap = FT(0)
    z0 = FT(0.1)
    parameterization = MoninObukhov(; z0)
    function surface_state(surface_coordinates, interior_z, t)
        _FT = eltype(surface_coordinates)
        SurfaceState(;
            parameterization,
            T = 265 - _FT(0.25) * _FT(t) / 3600,
            p,
            q_vap,
        )
    end
    return surface_state
end

coriolis_forcing(::GABLS, ::Type{FT}) where {FT} =
    (;
        prof_ug = APL.GABLS_geostrophic_ug(FT),
        prof_vg = APL.GABLS_geostrophic_vg(FT),
        coriolis_param = FT(1.39e-4),
    )
