"""
    ISDAC

The ISDAC (Indirect and Semi-Direct Aerosol Campaign) setup, with a
hydrostatically balanced pressure profile. Profiles are sourced from
AtmosphericProfilesLibrary.

When `perturb` is true, applies pseudorandom temperature perturbations
of amplitude 0.1 K below 825 m.

## Example
```julia
setup = ISDAC(; prognostic_tke = true, perturb = false, thermo_params)
```
"""
struct ISDAC{P}
    prognostic_tke::Bool
    perturb::Bool
    profiles::P
end

ISDAC(; prognostic_tke::Bool, perturb::Bool = false, thermo_params) =
    ISDAC(prognostic_tke, perturb, isdac_profiles(thermo_params))

function isdac_profiles(thermo_params)
    FT = eltype(thermo_params)
    θ = APL.ISDAC_θ_liq_ice(FT)
    q_tot = APL.ISDAC_q_tot(FT)
    p = hydrostatic_pressure_profile(;
        thermo_params, θ, q_tot,
        p_0 = FT(102000), z_max = 5000,
    )
    u = APL.ISDAC_u(FT)
    v = APL.ISDAC_v(FT)
    tke = APL.ISDAC_tke(FT)
    return (; θ, q_tot, p, u, v, tke)
end

function center_initial_condition(setup::ISDAC, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, perturb, profiles) = setup

    θ = profiles.θ(z)
    if perturb && z < 825
        θ += FT(0.1) * randn(FT)
    end
    q_tot = profiles.q_tot(z)
    p = profiles.p(z)
    T = TD.air_temperature(thermo_params, TD.pθ_li(), p, θ, q_tot)
    tke = prognostic_tke ? profiles.tke(z) : FT(0)

    return physical_state(;
        T, p, q_tot,
        u = profiles.u(z), v = profiles.v(z),
        tke,
    )
end

function surface_condition(::ISDAC, params)
    FT = eltype(params)
    parameterization = MoninObukhov(; z0 = FT(4e-4))
    return SurfaceState(;
        parameterization, T = FT(267), p = FT(102000),
    )
end

subsidence_forcing(::ISDAC, ::Type{FT}) where {FT} = APL.ISDAC_subsidence(FT)

external_forcing(::ISDAC, ::Type{FT}) where {FT} = ISDACForcing()
radiation_model(::ISDAC, ::Type{FT}) where {FT} = RadiationISDAC{FT}()
