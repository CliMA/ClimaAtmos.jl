#####
##### Initial conditions for a sphere
#####

function center_initial_condition_3d(
    local_geometry,
    params,
    atmos,
    perturb_initstate,
)

    thermo_params = CAP.thermodynamics_params(params)
    # Coordinates
    z = local_geometry.coordinates.z
    FT = eltype(z)

    # Constants from ClimaAtmos.Parameters
    grav = FT(CAP.grav(params))

    # Initial temperature and pressure
    temp_profile = TD.TemperatureProfiles.DecayingTemperatureProfile{FT}(
        thermo_params,
        FT(290),
        FT(220),
        FT(8e3),
    )
    T, p = temp_profile(thermo_params, z)
    if perturb_initstate
        T += rand(FT) * FT(0.1) * (z < 5000)
    end

    # Initial velocity
    u = FT(0)
    v = FT(0)
    uₕ_local = Geometry.UVVector(u, v)
    uₕ = Geometry.Covariant12Vector(uₕ_local, local_geometry)

    # Initial moisture
    q_tot = FT(0)

    # Initial values computed from the thermodynamic state
    ρ = TD.air_density(thermo_params, T, p)
    ts = TD.PhaseEquil_ρTq(thermo_params, ρ, T, q_tot)
    ᶜ𝔼_kwarg =
        energy_vars(thermo_params, ts, norm_sqr(uₕ_local) / 2, grav * z, atmos)
    # TODO: Include ability to handle nonzero initial cloud condensate

    return (;
        ρ,
        ᶜ𝔼_kwarg...,
        uₕ,
        moisture_vars(thermo_params, ts, atmos)...,
        precipitation_vars(FT, atmos)...,
        turbconv_vars(FT, atmos)...,
    )
end
