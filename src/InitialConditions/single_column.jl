#####
##### Initial conditions for a single column
#####

function center_initial_condition_column(
    local_geometry,
    params,
    energy_form,
    perturb_initstate,
    moisture_model,
    turbconv_model,
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

    if energy_form isa PotentialTemperature
        𝔼_kwarg = (; ρθ = ρ * TD.liquid_ice_pottemp(thermo_params, ts))
    elseif energy_form isa TotalEnergy
        𝔼_kwarg =
            (; ρe_tot = ρ * (TD.internal_energy(thermo_params, ts) + grav * z))
    elseif energy_form isa InternalEnergy
        𝔼_kwarg = (; ρe_int = ρ * TD.internal_energy(thermo_params, ts))
    end

    tc_kwargs = if turbconv_model isa Nothing
        NamedTuple()
    elseif turbconv_model isa TC.EDMFModel
        (;
            ρq_tot = FT(0), # TC needs this, for now.
            TC.cent_prognostic_vars_edmf(FT, turbconv_model)...,
        )
    end

    return (;
        ρ,
        𝔼_kwarg...,
        uₕ = Geometry.Covariant12Vector(FT(0), FT(0)),
        tc_kwargs...,
    )
end
