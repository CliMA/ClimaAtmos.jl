"""
    Larcform1

Single-column model setup for the Larcform1 arctic boundary layer case, based on
Pithan et al. (2016) — SCM intercomparison for the Arctic winter boundary layer.

Canonical conditions (Pithan 2016, Section 2):

  - Location: 80°N
  - Start date: 1 January (zero solar insolation)
  - Initial surface temperature: 250 K (sea ice)
  - Sea ice: 1 m thick, 100% concentration
  - Geostrophic wind: 5 m/s throughout troposphere

Profiles are sourced from AtmosphericProfilesLibrary. RH is specified with respect
to liquid water (Pithan 2016, Table 1). The humidity profile is split at the
tropopause: RH-derived q_tot below, fixed q_top above.

## Example

```julia
setup = Larcform1(; prognostic_tke = true, thermo_params)
```
"""
struct Larcform1{P}
    prognostic_tke::Bool
    profiles::P
end

Larcform1(; prognostic_tke::Bool = true, thermo_params) =
    Larcform1(prognostic_tke, larcform1_profiles(thermo_params))

function larcform1_profiles(thermo_params)
    FT = eltype(thermo_params)
    T_prof = APL.Larcform1_T(FT)
    RH_prof = APL.Larcform1_RH(FT)
    p_apl = APL.Larcform1_p(FT)  # analytic pressure used for RH → q_tot conversion
    z_trop = FT(APL.Larcform1_constants.z_tropopause)
    q_top = FT(APL.Larcform1_constants.q_top)
    Rv_over_Rd = TD.Parameters.Rv_over_Rd(thermo_params)

    # RH → q_tot using liquid-water saturation (Pithan 2016 Table 1 specifies wrt liquid)
    q_tot_prof = APL.ZProfile(
        z -> if z ≤ z_trop
            T = T_prof(z)
            p = p_apl(z)
            RH = RH_prof(z)
            p_v_sat = TD.saturation_vapor_pressure(thermo_params, T, TD.Liquid())
            ϵ = 1 / Rv_over_Rd
            e = p_v_sat * RH
            ϵ * e / (p - e + ϵ * e)
        else
            q_top
        end,
    )

    p_prof = hydrostatic_pressure_profile(;
        thermo_params,
        p_0 = FT(APL.Larcform1_constants.P_0),
        T = T_prof,
        q_tot = q_tot_prof,
    )
    u = APL.Larcform1_geostrophic_u(FT)
    v = APL.Larcform1_geostrophic_v(FT)
    return (; T = T_prof, q_tot = q_tot_prof, p = p_prof, u, v)
end

function center_initial_condition(setup::Larcform1, local_geometry, params; p_at_point = nothing)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    (; profiles) = setup
    tke = FT(0)
    return physical_state(;
        T = profiles.T(z),
        p = evaluate_pressure(profiles.p, z; p_at_point),
        q_tot = profiles.q_tot(z),
        u = profiles.u(z),
        v = profiles.v(z),
        tke,
    )
end

function surface_condition(::Larcform1, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    T_surface = FT(250)
    p_surface = FT(APL.Larcform1_constants.P_0)
    # Surface q_vap consistent with RH-wrt-liquid specification
    p_v_sat = TD.saturation_vapor_pressure(thermo_params, T_surface, TD.Liquid())
    ϵ_v = TD.Parameters.R_d(thermo_params) / TD.Parameters.R_v(thermo_params)
    q_vap = ϵ_v * p_v_sat / (p_surface - p_v_sat * (1 - ϵ_v))
    return (;
        flux_scheme = MoninObukhov(; z0 = FT(1e-3)),
        temperature = AnalyticTemperature(Returns(T_surface)),
        overrides = SurfaceBoundaryOverrides(p = p_surface, q_vap = FT(q_vap)),
    )
end

coriolis_forcing(::Larcform1, ::Type{FT}) where {FT} = (;
    prof_ug = APL.Larcform1_geostrophic_u(FT),
    prof_vg = APL.Larcform1_geostrophic_v(FT),
    coriolis_param = FT(1.432e-4),  # 2Ω sin(80°N)
)
