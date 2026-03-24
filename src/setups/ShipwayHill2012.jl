"""
    ShipwayHill2012

The initial condition described in [ShipwayHill2012](@cite), with a
hydrostatically balanced pressure profile.

B. J. Shipway and A. A. Hill.
Diagnosis of systematic differences between multiple parametrizations of warm
rain microphysics using a kinematic framework.
Quarterly Journal of the Royal Meteorological Society 138, 2196-2211 (2012).
"""
struct ShipwayHill2012{P}
    profiles::P
end

function ShipwayHill2012(; thermo_params)
    FT = eltype(thermo_params)
    z_values = FT[0, 740, 3260]
    rv_values = FT[0.015, 0.0138, 0.0024]
    θ_values = FT[297.9, 297.9, 312.66]

    linear_profile(zs, vals) = Intp.extrapolate(
        Intp.interpolate((zs,), vals, Intp.Gridded(Intp.Linear())),
        Intp.Linear(),
    )
    rv(z) = max(linear_profile(z_values, rv_values)(z), zero(z))
    q_tot(z) = rv(z) / (1 + rv(z))
    θ(z) = linear_profile(z_values, θ_values)(z)

    p_0 = FT(100_700)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    return (; θ, q_tot, p)
end

function center_initial_condition(setup::ShipwayHill2012, local_geometry, params)
    thermo_params = CAP.thermodynamics_params(params)
    (; θ, q_tot, p) = setup.profiles
    (; z) = local_geometry.coordinates

    q_tot_z = q_tot(z)
    T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z), q_tot_z)

    return physical_state(; T, p = p(z), q_tot = q_tot_z)
end

function surface_condition(::ShipwayHill2012, params)
    function surface_state(surface_coordinates, interior_z, t)
        FT = eltype(surface_coordinates)
        T = FT(297.9)
        p = FT(100700)
        rv_0 = FT(0.015)
        q_vap = rv_0 / (1 + rv_0)
        parameterization = SurfaceConditions.ExchangeCoefficients(; Cd = FT(0), Ch = FT(0))
        return SurfaceState(; parameterization, T, p, q_vap)
    end
    return surface_state
end

prescribed_flow_model(::ShipwayHill2012, ::Type{FT}) where {FT} =
    ShipwayHill2012VelocityProfile{FT}()
