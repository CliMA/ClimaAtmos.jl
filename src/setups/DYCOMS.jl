"""
    DYCOMS{P, FT}

Unified struct for DYCOMS_RF01 ([Stevens2005](@cite)) and DYCOMS_RF02
([Ackerman2009](@cite)), with hydrostatically balanced pressure profiles
sourced from AtmosphericProfilesLibrary.

The two variants differ only in APL profiles, surface heat fluxes, and
geostrophic wind. Construct via `DYCOMS_RF01(; ...)` or `DYCOMS_RF02(; ...)`.

## Example
```julia
setup = DYCOMS_RF01(; prognostic_tke = true, thermo_params)
setup = DYCOMS_RF02(; prognostic_tke = true, thermo_params)
```
"""
struct DYCOMS{P, FT}
    prognostic_tke::Bool
    profiles::P
    shf::FT
    lhf::FT
    ug::FT
end

function DYCOMS_RF01(; prognostic_tke::Bool, thermo_params)
    FT = eltype(thermo_params)
    DYCOMS(prognostic_tke,
        dycoms_profiles(thermo_params;
            θ = APL.Dycoms_RF01_θ_liq_ice(FT),
            q_tot = APL.Dycoms_RF01_q_tot(FT),
            u = APL.Dycoms_RF01_u0(FT),
            v = APL.Dycoms_RF01_v0(FT),
        ), FT(15), FT(115), FT(7))
end

function DYCOMS_RF02(; prognostic_tke::Bool, thermo_params)
    FT = eltype(thermo_params)
    DYCOMS(prognostic_tke,
        dycoms_profiles(thermo_params;
            θ = APL.Dycoms_RF02_θ_liq_ice(FT),
            q_tot = APL.Dycoms_RF02_q_tot(FT),
            u = APL.Dycoms_RF02_u(FT),
            v = APL.Dycoms_RF02_v(FT),
        ), FT(16), FT(93), FT(5))
end

function dycoms_profiles(thermo_params; θ, q_tot, u, v)
    FT = eltype(thermo_params)
    p = hydrostatic_pressure_profile(; thermo_params, p_0 = FT(101780.0), θ, q_tot)
    tke = APL.Dycoms_RF01_tke_prescribed(FT)
    return (; θ, q_tot, u, v, p, tke)
end

function center_initial_condition(setup::DYCOMS, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup
    q_tot = profiles.q_tot(z)
    return physical_state(;
        T = TD.air_temperature(thermo_params, TD.pθ_li(),
            profiles.p(z), profiles.θ(z), q_tot),
        p = profiles.p(z),
        q_tot,
        u = profiles.u(z), v = profiles.v(z),
        tke = prognostic_tke ? FT(0) : profiles.tke(z),
    )
end

function surface_condition(setup::DYCOMS, params)
    FT = eltype(params)
    parameterization = MoninObukhov(;
        z0 = FT(1e-4), shf = setup.shf, lhf = setup.lhf, ustar = FT(0.25),
    )
    return SurfaceState(;
        parameterization, T = FT(292.5), p = FT(101780), q_vap = FT(0.01384),
    )
end

radiation_model(::DYCOMS, ::Type{FT}) where {FT} = RadiationDYCOMS{FT}()

subsidence_forcing(::DYCOMS, ::Type{FT}) where {FT} =
    let D = FT(3.75e-6)
        z -> -z * D
    end

coriolis_forcing(setup::DYCOMS, ::Type{FT}) where {FT} =
    (;
        prof_ug = Returns(setup.ug),
        prof_vg = Returns(FT(-5.5)),
        coriolis_param = FT(0), # TODO: check this
    )
