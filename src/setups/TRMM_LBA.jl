"""
    TRMM_LBA

The TRMM_LBA setup described in [Grabowski2006](@cite), with a hydrostatically
balanced pressure profile. Profiles are sourced from AtmosphericProfilesLibrary.

Surface fluxes are time-varying: `shf` and `lhf` follow a cosine ramp over
the first 5.25 hours.

## Example
```julia
setup = TRMM_LBA(; prognostic_tke = true, thermo_params)
```
"""
struct TRMM_LBA{P}
    prognostic_tke::Bool
    profiles::P
end

TRMM_LBA(; prognostic_tke::Bool, thermo_params) =
    TRMM_LBA(prognostic_tke, trmm_lba_profiles(thermo_params))

function trmm_lba_profiles(thermo_params)
    FT = eltype(thermo_params)
    p_0 = FT(99130.0)
    T_profile = APL.TRMM_LBA_T(FT)

    # Compute q_tot from measured pressure and relative humidity profiles
    Rv_over_Rd = TD.Parameters.Rv_over_Rd(thermo_params)
    measured_p = APL.TRMM_LBA_p(FT)
    measured_RH = APL.TRMM_LBA_RH(FT)
    measured_z_values = APL.TRMM_LBA_z(FT)
    measured_q_tot_values = map(measured_z_values) do z
        p_v_sat = TD.saturation_vapor_pressure(thermo_params, T_profile(z), TD.Liquid())
        denominator =
            measured_p(z) - p_v_sat +
            (1 / Rv_over_Rd) * p_v_sat * measured_RH(z) / 100
        q_v_sat = p_v_sat * (1 / Rv_over_Rd) / denominator
        return q_v_sat * measured_RH(z) / 100
    end
    q_tot = Intp.extrapolate(
        Intp.interpolate(
            (measured_z_values,),
            measured_q_tot_values,
            Intp.Gridded(Intp.Linear()),
        ),
        Intp.Flat(),
    )

    p = hydrostatic_pressure_profile(; thermo_params, p_0, T = T_profile, q_tot)
    u = APL.TRMM_LBA_u(FT)
    v = APL.TRMM_LBA_v(FT)
    tke = APL.TRMM_LBA_tke_prescribed(FT)
    return (; T = T_profile, q_tot, p, u, v, tke)
end

function center_initial_condition(setup::TRMM_LBA, local_geometry, params)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    (; prognostic_tke, profiles) = setup

    tke = prognostic_tke ? FT(0) : profiles.tke(z)

    return physical_state(;
        T = profiles.T(z),
        p = profiles.p(z),
        q_tot = profiles.q_tot(z),
        u = profiles.u(z), v = profiles.v(z),
        tke,
    )
end

function surface_condition(::TRMM_LBA, params)
    FT = eltype(params)
    T = FT(296.85)
    p = FT(99130)
    q_vap = FT(0.02245)
    z0 = FT(1e-4)
    ustar = FT(0.28)
    function surface_state(surface_coordinates, interior_z, t)
        _FT = eltype(surface_coordinates)
        value = cos(_FT(π) / 2 * (1 - _FT(t) / (_FT(5.25) * 3600)))
        shf = 270 * max(0, value)^_FT(1.5)
        lhf = 554 * max(0, value)^_FT(1.3)
        parameterization = MoninObukhov(; z0, shf, lhf, ustar)
        return SurfaceState(; parameterization, T, p, q_vap)
    end
    return surface_state
end

radiation_model(::TRMM_LBA, ::Type{FT}) where {FT} = RadiationTRMM_LBA(FT)
