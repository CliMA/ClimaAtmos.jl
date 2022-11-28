#####
##### Initial conditions for a single column
#####

function center_initial_condition_column(
    local_geometry,
    params,
    atmos,
    perturb_initstate,
)
    thermo_params = CAP.thermodynamics_params(params)
    z = local_geometry.coordinates.z
    FT = eltype(z)

    R_d = FT(CAP.R_d(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))

    T = FT(300)
    p = MSLP * exp(-z * grav / (R_d * T))
    ρ = p / (R_d * T)
    ts = TD.PhaseDry_ρp(thermo_params, ρ, p)
    uₕ_local = Geometry.UVVector(FT(0), FT(0))
    ᶜ𝔼_kwarg =
        energy_vars(thermo_params, ts, norm_sqr(uₕ_local) / 2, grav * z, atmos)

    return (;
        ρ,
        ᶜ𝔼_kwarg...,
        uₕ = Geometry.Covariant12Vector(FT(0), FT(0)),
        precipitation_vars(FT, atmos)...,
        turbconv_vars(FT, atmos)...,
    )
end
