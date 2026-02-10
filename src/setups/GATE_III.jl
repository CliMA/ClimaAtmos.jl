"""
    GATE_III

The GATE_III setup described in [Khairoutdinov2009](@cite), with a
hydrostatically balanced pressure profile. Uses T (not θ) for hydrostatic
integration. Profiles are sourced from AtmosphericProfilesLibrary.

## Example
```julia
setup = GATE_III(; prognostic_tke = true, thermo_params)
```
"""
struct GATE_III{P}
    prognostic_tke::Bool
    profiles::P
end

GATE_III(; prognostic_tke::Bool, thermo_params) =
    GATE_III(prognostic_tke, gate_iii_profiles(thermo_params))

function gate_iii_profiles(thermo_params)
    FT = eltype(thermo_params)
    T = APL.GATE_III_T(FT)
    q_tot = APL.GATE_III_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0 = FT(101500.0), T, q_tot)
    u = APL.GATE_III_u(FT)
    tke = APL.GATE_III_tke(FT)
    return (; T, q_tot, p, u, tke)
end

function center_initial_condition(setup::GATE_III, local_geometry, params)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    tke = prognostic_tke ? FT(0) : profiles.tke(z)

    return physical_state(;
        T = profiles.T(z),
        p = profiles.p(z),
        q_tot = profiles.q_tot(z),
        u = profiles.u(z),
        tke,
    )
end
