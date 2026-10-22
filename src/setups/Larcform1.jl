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

The case constants (lapse rate, surface temperature and pressure, tropopause
pressure, stratospheric humidity) come from AtmosphericProfilesLibrary. The
reference atmosphere built from them, a constant lapse rate up to the 300 hPa
tropopause and isothermal above it, uses `R_d` and `grav` from ClimaParams, as do
the tropopause height and the profiles that switch there. RH is specified with
respect to liquid water (Pithan 2016, Table 1). The humidity profile is split at
the tropopause: RH-derived q_tot below, fixed q_top above.

TKE starts from zero. The case has no prescribed TKE profile, so it takes no
`prognostic_tke` keyword and is meant to run with prognostic TKE.

# Examples

```julia
setup = Larcform1(; thermo_params)
```
"""
struct Larcform1{P}
    profiles::P
end

Larcform1(; thermo_params) = Larcform1(larcform1_profiles(thermo_params))

"""
    larcform1_reference_atmosphere(thermo_params)

Return the Larcform1 reference atmosphere of Pithan 2016, Table 1: temperature
falls at the constant lapse rate `γ` from `T_0` up to the tropopause at
`P_tropopause` and is isothermal above it. The pressure follows the dry
hypsometric equation with exponent `α = R_d γ / grav`, where `R_d` and `grav` come
from `thermo_params`.

# Returns

`(; z_trop, T, p)`: the tropopause height [m] and the profiles `z -> T` [K] and
`z -> p` [Pa].
"""
function larcform1_reference_atmosphere(thermo_params)
    FT = eltype(thermo_params)
    LC = APL.Larcform1_constants
    R_d = TD.Parameters.R_d(thermo_params)
    grav = TD.Parameters.grav(thermo_params)
    γ = FT(LC.γ)
    T_0 = FT(LC.T_0)
    P_0 = FT(LC.P_0)
    P_trop = FT(LC.P_tropopause)

    α = R_d * γ / grav
    z_trop = (T_0 / γ) * (1 - (P_trop / P_0)^α)
    T_trop = T_0 - γ * z_trop

    # The closures capture values only, not `FT`: model components must not hold types.
    T = APL.ZProfile(z -> z ≤ z_trop ? T_0 - γ * oftype(T_0, z) : T_trop)
    p = APL.ZProfile(
        z -> if z ≤ z_trop
            P_0 * (1 - γ * oftype(T_0, z) / T_0)^(1 / α)
        else
            P_trop * exp(-grav / (R_d * T_trop) * (oftype(T_0, z) - z_trop))
        end,
    )
    return (; z_trop, T, p)
end

"""
    larcform1_relative_humidity(p, P_0)

Relative humidity with respect to liquid water at pressure `p` [Pa] (Pithan 2016,
Table 1): 0.2 at pressures of 600 hPa and below, rising linearly in pressure to
0.8 at the surface pressure `P_0`.
"""
function larcform1_relative_humidity(p, P_0)
    FT = typeof(p)
    p_knee = FT(60000)
    return p ≤ p_knee ? FT(0.2) : FT(0.2) + FT(0.6) * (p - p_knee) / (P_0 - p_knee)
end

function larcform1_profiles(thermo_params)
    FT = eltype(thermo_params)
    # The analytic reference pressure, not the hydrostatic one below, sets RH and
    # thereby q_tot; the hydrostatic pressure in turn depends on q_tot.
    reference = larcform1_reference_atmosphere(thermo_params)
    (; z_trop) = reference
    T_prof = reference.T
    p_ref = reference.p
    P_0 = FT(APL.Larcform1_constants.P_0)
    q_top = FT(APL.Larcform1_constants.q_top)
    Rv_over_Rd = TD.Parameters.Rv_over_Rd(thermo_params)

    # RH → q_tot using liquid-water saturation (Pithan 2016 Table 1 specifies wrt liquid)
    q_tot_prof = APL.ZProfile(
        z -> if z ≤ z_trop
            T = T_prof(z)
            p = p_ref(z)
            RH = larcform1_relative_humidity(p, P_0)
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
        p_0 = P_0,
        T = T_prof,
        q_tot = q_tot_prof,
    )
    # Geostrophic wind of 5 m/s through the troposphere (Pithan 2016, Section 2)
    u_g = FT(5)
    u = APL.ZProfile(z -> z ≤ z_trop ? u_g : zero(u_g))
    v = APL.Larcform1_geostrophic_v(FT)
    return (; T = T_prof, q_tot = q_tot_prof, p = p_prof, u, v)
end

function center_initial_condition(setup::Larcform1, local_geometry, params)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    (; profiles) = setup
    tke = FT(0)
    return physical_state(;
        T = profiles.T(z),
        p = profiles.p(z),
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
        overrides = SurfaceBoundaryOverrides(q_vap = FT(q_vap)),
    )
end

const LARCFORM1_LATITUDE = 80  # Pithan 2016, Section 2 (degrees north)

function coriolis_forcing(setup::Larcform1, params)
    FT = eltype(params)
    # f = 2Ω sin(φ), with the planetary rotation rate Ω from ClimaParams
    coriolis_param = 2 * CAP.Omega(params) * sind(FT(LARCFORM1_LATITUDE))
    return (;
        prof_ug = setup.profiles.u,
        prof_vg = setup.profiles.v,
        coriolis_param,
    )
end
