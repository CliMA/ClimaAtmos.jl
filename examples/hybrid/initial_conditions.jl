##
## Initial conditions
##

function init_state(
    center_initial_condition,
    face_initial_condition,
    center_space,
    face_space,
    params,
    model_spec,
)
    (; energy_form, moisture_model, turbconv_model) = model_spec
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)
    c =
        center_initial_condition.(
            ᶜlocal_geometry,
            params,
            energy_form,
            moisture_model,
            turbconv_model,
        )
    f = face_initial_condition.(ᶠlocal_geometry, params, turbconv_model)
    Y = Fields.FieldVector(; c, f)
    return Y
end

function face_initial_condition(local_geometry, params, turbconv_model)
    z = local_geometry.coordinates.z
    FT = eltype(z)
    tc_kwargs = if turbconv_model isa Nothing
        NamedTuple()
    else
        TC.face_prognostic_vars_edmf(FT, local_geometry, turbconv_model)
    end
    (; w = Geometry.Covariant3Vector(FT(0)), tc_kwargs...)
end

function center_initial_condition_column(
    local_geometry,
    params,
    energy_form,
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

function center_initial_condition_baroclinic_wave(
    local_geometry,
    params,
    energy_form,
    moisture_model,
    turbconv_model;
    is_balanced_flow = false,
)

    thermo_params = CAP.thermodynamics_params(params)
    # Coordinates
    z = local_geometry.coordinates.z
    ϕ = local_geometry.coordinates.lat
    λ = local_geometry.coordinates.long
    FT = eltype(z)

    # Constants from ClimaAtmos.Parameters
    R_d = FT(CAP.R_d(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))
    Ω = FT(CAP.Omega(params))
    R = FT(CAP.planet_radius(params))

    # Constants required for dry initial conditions
    k = 3
    T_e = FT(310) # temperature at the equator
    T_p = FT(240) # temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005)
    A = 1 / Γ
    B = (T_0 - T_p) / T_0 / T_p
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p
    b = 2
    H = R_d * T_0 / grav
    z_t = FT(15e3)
    λ_c = FT(20)
    ϕ_c = FT(40)
    d_0 = R / 6
    V_p = FT(1)

    # Constants required for moist initial conditions
    p_w = FT(3.4e4)
    p_t = FT(1e4)
    q_t = FT(1e-12)
    q_0 = FT(0.018)
    ϕ_w = FT(40)
    ε = FT(0.608)

    # Initial virtual temperature and pressure
    τ_z_1 = exp(Γ * z / T_0)
    τ_z_2 = 1 - 2 * (z / b / H)^2
    τ_z_3 = exp(-(z / b / H)^2)
    τ_1 = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2 = C * τ_z_2 * τ_z_3
    τ_int_1 = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2 = C * z * τ_z_3
    I_T = cosd(ϕ)^k - k * (cosd(ϕ))^(k + 2) / (k + 2)
    T_v = (τ_1 - τ_2 * I_T)^(-1)
    p = MSLP * exp(-grav / R_d * (τ_int_1 - τ_int_2 * I_T))

    # Initial velocity
    U = grav * k / R * τ_int_2 * T_v * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
    u = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U)
    v = FT(0)
    if !is_balanced_flow
        F_z = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
        r = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
        c3 = cos(π * r / 2 / d_0)^3
        s1 = sin(π * r / 2 / d_0)
        cond = (0 < r < d_0) * (r != R * pi)
        u +=
            -16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
            sin(r / R) * cond
        v +=
            16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            cosd(ϕ_c) *
            sind(λ - λ_c) / sin(r / R) * cond
    end
    uₕ_local = Geometry.UVVector(u, v)
    uₕ = Geometry.Covariant12Vector(uₕ_local, local_geometry)

    # Initial moisture and temperature
    if moisture_model isa DryModel
        q_tot = FT(0)
    else
        q_tot =
            (p <= p_t) ? q_t :
            q_0 * exp(-(ϕ / ϕ_w)^4) * exp(-((p - MSLP) / p_w)^2)
    end
    T = T_v / (1 + ε * q_tot) # This is the formula used in the paper.
    # T = T_v * (1 + q_tot) / (1 + q_tot * CAP.molmass_ratio(params))
    # This is the actual formula, which would be consistent with TD.

    # Initial values computed from the thermodynamic state
    ts = TD.PhaseEquil_pTq(thermo_params, p, T, q_tot)
    ρ = TD.air_density(thermo_params, ts)
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

    # TODO: synchronize `ρθ_liq_ice`, `u`, `v`, `uₕ`, `ρ` with TC
    tc_kwargs = if turbconv_model isa Nothing
        NamedTuple()
    elseif turbconv_model isa TC.EDMFModel
        (;
            ρθ_liq_ice = FT(0),
            ρq_tot = FT(0),
            u = FT(0),
            v = FT(0),
            TC.cent_prognostic_vars_edmf(FT, turbconv_model)...,
        )
    end
    return (; ρ, ᶜ𝔼_kwarg..., uₕ, moisture_kwargs..., tc_kwargs...)
end

function center_initial_condition_sphere(
    local_geometry,
    params,
    energy_form,
    moisture_model,
    turbconv_model;
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
    T += rand(FT) * FT(0.1) * (z < 5000)

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

    # TODO: synchronize `ρθ_liq_ice`, `u`, `v`, `uₕ`, `ρ` with TC
    tc_kwargs = if turbconv_model isa Nothing
        NamedTuple()
    elseif turbconv_model isa TC.EDMFModel
        (;
            ρθ_liq_ice = FT(0),
            ρq_tot = FT(0),
            u = FT(0),
            v = FT(0),
            TC.cent_prognostic_vars_edmf(FT, turbconv_model)...,
        )
    end
    return (; ρ, ᶜ𝔼_kwarg..., uₕ, moisture_kwargs..., tc_kwargs...)
end
