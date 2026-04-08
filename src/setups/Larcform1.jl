"""
    Larcform1

Single-column model setup for the Larcform1 arctic boundary layer case, based on
Pithan et al. (2016), JGR Atmospheres — SCM intercomparison for Arctic boundary layer.

Canonical conditions (Section 2.2):
- Location: 80°N
- Start date: 1 January (zero solar insolation)
- Initial surface temperature: 250 K
- Sea ice: 1 m thick, 100% concentration
- Geostrophic wind: 5 m/s throughout troposphere

Profiles are sourced from AtmosphericProfilesLibrary (branch jy/Larcform1).
"""
struct Larcform1{P}
    prognostic_tke::Bool
    profiles::P
end

Larcform1(; prognostic_tke::Bool = true, thermo_params) =
    Larcform1(prognostic_tke, larcform1_profiles(thermo_params))

function larcform1_profiles(thermo_params)
    FT = eltype(thermo_params)
    p_0 = FT(101300)
    q_tot = APL.Larcform1_q_tot(FT)
    T = APL.Larcform1_T(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, T, q_tot)
    u = APL.Larcform1_geostrophic_u(FT)
    v = APL.Larcform1_geostrophic_v(FT)
    return (; q_tot, T, p, u, v)
end

function center_initial_condition(setup::Larcform1, local_geometry, params)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup
    tke = prognostic_tke ? FT(0) : FT(0)
    return physical_state(;
        T = profiles.T(z),
        p = profiles.p(z),
        q_tot = profiles.q_tot(z),
        u = profiles.u(z),
        v = profiles.v(z),
        tke,
    )
end
