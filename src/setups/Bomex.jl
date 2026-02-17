"""
    Bomex

The Bomex setup described in [Holland1973](@cite), with a hydrostatically
balanced pressure profile. Profiles are sourced from AtmosphericProfilesLibrary.

The `profiles` field stores precomputed atmospheric profile functions (computed
at construction time before broadcasting).

## Example
```julia
import Thermodynamics as TD
import ClimaParams as CP
FT = Float64
toml_dict = CP.create_toml_dict(FT)
thermo_params = TD.Parameters.ThermodynamicsParameters(toml_dict)
setup = Bomex(; prognostic_tke = true, thermo_params)
```
"""
struct Bomex{P}
    prognostic_tke::Bool
    profiles::P
end
Bomex(; prognostic_tke, thermo_params) =
    Bomex(prognostic_tke, bomex_profiles(thermo_params))

"""
    bomex_profiles(thermo_params)

Precompute the atmospheric profiles for the Bomex case, given thermodynamic
parameters `thermo_params`. 

Returns a NamedTuple of interpolatable profile functions of height `z`.
"""
function bomex_profiles(thermo_params)
    FT = eltype(thermo_params)
    p_0 = FT(101500.0)
    θ = APL.Bomex_θ_liq_ice(FT)
    q_tot = APL.Bomex_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    u = APL.Bomex_u(FT)
    tke = APL.Bomex_tke_prescribed(FT)
    return (; θ, q_tot, p, u, tke)
end

function center_initial_condition(setup::Bomex, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    # Evaluate profiles at z
    θ = FT(profiles.θ(z))
    q_tot = FT(profiles.q_tot(z))
    p = FT(profiles.p(z))
    T = FT(TD.air_temperature(thermo_params, TD.pθ_li(), p, θ, q_tot))

    tke = prognostic_tke ? FT(0) : FT(profiles.tke(z))

    return physical_state(; T, p, q_tot, u = FT(profiles.u(z)), tke)
end

function surface_condition(::Bomex, params)
    FT = eltype(params)
    return (;
        T = FT(300.4),
        p = FT(101500),
        q_vap = FT(0.02245),
        θ_flux = FT(8e-3),
        q_flux = FT(5.2e-5),
        z0 = FT(1e-4),
        ustar = FT(0.28),
    )
end

# ============================================================================
# SCM forcing profiles
# ============================================================================

function subsidence_forcing(::Bomex, ::Type{FT}) where {FT}
    return APL.Bomex_subsidence(FT)
end

function large_scale_advection_forcing(::Bomex, ::Type{FT}) where {FT}
    return (; prof_dTdt = APL.Bomex_dTdt(FT), prof_dqtdt = APL.Bomex_dqtdt(FT))
end

function coriolis_forcing(::Bomex, ::Type{FT}) where {FT}
    return (;
        prof_ug = APL.Bomex_geostrophic_u(FT),
        prof_vg = z -> FT(0),
        coriolis_param = FT(0.376e-4),
    )
end
