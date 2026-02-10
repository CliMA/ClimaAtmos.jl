"""
    PrecipitatingColumn

A 1-dimensional precipitating column test using Rico-based profiles with
prescribed precipitation fields. Profiles are precomputed at construction time.
"""
struct PrecipitatingColumn{P}
    profiles::P
end

PrecipitatingColumn(; thermo_params) =
    PrecipitatingColumn(precipitating_column_profiles(thermo_params))

"""
    _prescribed_prof(FT, z_mid, z_max, val)

Return a Gaussian profile function centered at `z_mid`, zero above `z_max`.
"""
_prescribed_prof(::Type{FT}, z_mid, z_max, val) where {FT} =
    z -> z < z_max ? FT(val) * exp(-(z - FT(z_mid))^2 / 2 / FT(1e3)^2) : FT(0)

function precipitating_column_profiles(thermo_params)
    FT = eltype(thermo_params)
    p_0 = FT(101300.0)
    θ = APL.Rico_θ_liq_ice(FT)
    q_tot = APL.Rico_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    qR = _prescribed_prof(FT, 2000, 5000, 1e-6)
    qS = _prescribed_prof(FT, 5000, 8000, 2e-6)
    qL = _prescribed_prof(FT, 4000, 5500, 2e-5)
    qI = _prescribed_prof(FT, 6000, 9000, 1e-5)
    nL = _prescribed_prof(FT, 4000, 5500, 1e7)
    nR = _prescribed_prof(FT, 2000, 5000, 1e3)
    return (; θ, q_tot, p, qR, qS, qL, qI, nL, nR)
end

function center_initial_condition(setup::PrecipitatingColumn, local_geometry, params)
    thermo_params = CAP.thermodynamics_params(params)
    (; θ, q_tot, p, qR, qS, qL, qI, nL, nR) = setup.profiles
    (; z) = local_geometry.coordinates

    q_tot_z = q_tot(z)
    q_liq_z = qL(z) + qR(z)
    q_ice_z = qI(z) + qS(z)

    T = TD.air_temperature(
        thermo_params, TD.pθ_li(), p(z), θ(z), q_tot_z, q_liq_z, q_ice_z,
    )

    return physical_state(;
        T,
        p = p(z),
        q_tot = q_tot_z,
        q_liq = q_liq_z,
        q_ice = q_ice_z,
        q_rai = qR(z),
        q_sno = qS(z),
        n_liq = nL(z),
        n_rai = nR(z),
    )
end
