#####
##### Initial conditions for a box
#####

function center_initial_condition_box(
    local_geometry,
    params,
    energy_form,
    moisture_model,
    turbconv_model,
    precip_model,
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
    if energy_form isa PotentialTemperature
        ᶜ𝔼_kwarg = (; ρθ = ρ * TD.liquid_ice_pottemp(thermo_params, ts))
    elseif energy_form isa TotalEnergy
        K = norm_sqr(uₕ_local) / 2
        ᶜ𝔼_kwarg = (;
            ρe_tot = ρ * (TD.internal_energy(thermo_params, ts) + K + grav * z)
        )
    elseif energy_form isa InternalEnergy
        ᶜ𝔼_kwarg = (; ρe_int = ρ * TD.internal_energy(thermo_params, ts))
    end
    if moisture_model isa DryModel
        moisture_kwargs = NamedTuple()
    elseif moisture_model isa EquilMoistModel
        moisture_kwargs = (; ρq_tot = ρ * q_tot)
    elseif moisture_model isa NonEquilMoistModel
        moisture_kwargs = (;
            ρq_tot = ρ * q_tot,
            ρq_liq = ρ * TD.liquid_specific_humidity(thermo_params, ts),
            ρq_ice = ρ * TD.ice_specific_humidity(thermo_params, ts),
        )
    end
    # TODO: Include ability to handle nonzero initial cloud condensate

    tc_kwargs = if turbconv_model isa Nothing
        NamedTuple()
    elseif turbconv_model isa TC.EDMFModel
        TC.cent_prognostic_vars_edmf(FT, turbconv_model)
    end
    return (; ρ, ᶜ𝔼_kwarg..., uₕ, moisture_kwargs..., tc_kwargs...)
end
