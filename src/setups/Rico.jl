"""
    Rico

The RICO (Rain In Cumulus over the Ocean) setup described in [Rauber2007](@cite), with a
hydrostatically balanced pressure profile. Profiles are sourced from AtmosphericProfilesLibrary.

The `profiles` field stores precomputed atmospheric profile functions (computed
at construction time before broadcasting).

## Example
```julia
import Thermodynamics as TD
import ClimaParams as CP
FT = Float64
toml_dict = CP.create_toml_dict(FT)
thermo_params = TD.Parameters.ThermodynamicsParameters(toml_dict)
setup = Rico(; prognostic_tke = true, thermo_params)
```
"""
struct Rico{P}
    prognostic_tke::Bool
    profiles::P
end
Rico(; prognostic_tke, thermo_params) =
    Rico(prognostic_tke, rico_profiles(thermo_params))

"""
    rico_profiles(thermo_params)

Precompute the atmospheric profiles for the Rico case. Returns a NamedTuple of
interpolatable profile functions of height `z`.
"""
function rico_profiles(thermo_params)
    FT = eltype(thermo_params)
    p_0 = FT(101540.0)
    θ = APL.Rico_θ_liq_ice(FT)
    q_tot = APL.Rico_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    u = APL.Rico_u(FT)
    v = APL.Rico_v(FT)
    tke = APL.Rico_tke_prescribed(FT)
    return (; θ, q_tot, p, u, v, tke)
end

function center_initial_condition(setup::Rico, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    # Evaluate profiles at z
    θ_val = FT(profiles.θ(z))
    q_tot_val = FT(profiles.q_tot(z))
    p_val = FT(profiles.p(z))
    T = FT(TD.air_temperature(thermo_params, TD.pθ_li(), p_val, θ_val, q_tot_val))

    tke_val = prognostic_tke ? FT(0) : FT(profiles.tke(z))

    return physical_state(;
        T, p = p_val, q_tot = q_tot_val,
        u = FT(profiles.u(z)), v = FT(profiles.v(z)),
        tke = tke_val,
    )
end

function surface_condition(::Rico, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    T_surface = FT(299.8)
    p_surface = FT(101540)
    p_sat = TD.saturation_vapor_pressure(thermo_params, T_surface, TD.Liquid())
    ϵ_v = TD.Parameters.R_d(thermo_params) / TD.Parameters.R_v(thermo_params)
    q_vap = ϵ_v * p_sat / (p_surface - p_sat * (1 - ϵ_v))
    return (;
        T = T_surface,
        p = p_surface,
        q_vap = FT(q_vap),
        z0 = FT(1.5e-4),
    )
end

# ============================================================================
# SCM forcing profiles
# ============================================================================

function subsidence_forcing(::Rico, ::Type{FT}) where {FT}
    return APL.Rico_subsidence(FT)
end

function large_scale_advection_forcing(::Rico, ::Type{FT}) where {FT}
    return (; prof_dTdt = APL.Rico_dTdt(FT), prof_dqtdt = APL.Rico_dqtdt(FT))
end

function coriolis_forcing(::Rico, ::Type{FT}) where {FT}
    return (;
        prof_ug = APL.Rico_geostrophic_ug(FT),
        prof_vg = APL.Rico_geostrophic_vg(FT),
        coriolis_param = FT(4.5e-5),
    )
end
