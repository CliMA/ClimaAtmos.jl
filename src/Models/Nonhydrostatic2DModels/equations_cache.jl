const CM0M = CloudMicrophysics.Microphysics_0M
const CM1M = CloudMicrophysics.Microphysics_1M

@inline function precompute_microphysics_0M(ρq_tot, ρ, e_int, Φ, params)

    # saturation adjustment
    q_tot = ρq_tot / ρ
    ts = TD.PhaseEquil_ρeq(params, ρ, e_int, q_tot)
    q = TD.PhasePartition(params, ts)
    λ = TD.liquid_fraction(params, ts)
    I_l = TD.internal_energy_liquid(params, ts)
    I_i = TD.internal_energy_ice(params, ts)

    # precipitation removal source terms
    # (cached to avoid re-computing many times per time step)
    S_q_tot = CM0M.remove_precipitation(params, q)
    S_e_tot = (λ * I_l + (1 - λ) * I_i + Φ) * S_q_tot

    # temporarily dumping q.liq and q.ice into cache
    # for a quick way to visualise them in tests
    q_liq = q.liq
    q_ice = q.ice

    return (; S_q_tot, S_e_tot, q_liq, q_ice)
end

@inline function precompute_microphysics_1M(
    ρq_tot,
    ρq_rai,
    ρq_sno,
    ρ,
    e_int,
    Φ,
    params,
)

    FT = eltype(ρ)

    q_rai = ρq_rai / ρ
    q_sno = ρq_sno / ρ

    # saturation adjustment
    q_tot = ρq_tot / ρ
    ts = TD.PhaseEquil_ρeq(params, ρ, e_int, q_tot)

    q = TD.PhasePartition(params, ts)
    T = TD.air_temperature(params, ts)
    λ = TD.liquid_fraction(params, ts)
    I_d = TD.internal_energy_dry(params, ts)
    I_v = TD.internal_energy_vapor(params, ts)
    I_l = TD.internal_energy_liquid(params, ts)
    I_i = TD.internal_energy_ice(params, ts)
    L_f = TD.latent_heat_fusion(params, ts)

    _T_freeze = CLIMAParameters.Planet.T_freeze(params)
    _cv_l = CLIMAParameters.Planet.cv_l(params)

    # temporary vars for summimng different microphysics source terms
    S_q_rai::FT = FT(0)
    S_q_sno::FT = FT(0)
    S_q_tot::FT = FT(0)
    S_e_tot::FT = FT(0)

    # source of rain via autoconversion
    tmp = CM1M.conv_q_liq_to_q_rai(params, q.liq)
    S_q_rai += tmp
    #S_ql -= tmp
    S_e_tot -= tmp * (I_l + Φ)

    # source of snow via autoconversion
    tmp = CM1M.conv_q_ice_to_q_sno_no_supersat(params, q.ice)
    S_q_sno += tmp
    #S_qi -= tmp
    S_e_tot -= tmp * (I_i + Φ)

    # source of rain water via accretion cloud water - rain
    tmp = CM1M.accretion(
        params,
        CM1M.LiquidType(),
        CM1M.RainType(),
        q.liq,
        q_rai,
        ρ,
    )
    S_q_rai += tmp
    #S_ql -= tmp
    S_e_tot -= tmp * (I_l + Φ)

    # source of snow via accretion cloud ice - snow
    tmp =
        CM1M.accretion(params, CM1M.IceType(), CM1M.SnowType(), q.ice, q_sno, ρ)
    S_q_sno += tmp
    #S_qi -= tmp
    S_e_tot -= tmp * (I_i + Φ)

    # sink of cloud water via accretion cloud water - snow
    tmp = CM1M.accretion(
        params,
        CM1M.LiquidType(),
        CM1M.SnowType(),
        q.liq,
        q_sno,
        ρ,
    )
    if T < _T_freeze # cloud droplets freeze to become snow)
        S_q_sno += tmp
        #S_ql -= tmp
        S_e_tot -= tmp * (I_i + Φ)
    else # snow melts, both cloud water and snow become rain
        α = _cv_l / L_f * (T - _T_freeze)
        #S_ql -= tmp
        S_q_sno -= tmp * α
        S_q_rai += tmp * (1 + α)
        S_e_tot -= tmp * ((1 + α) * I_l - α * I_i + Φ)
    end

    # sink of cloud ice via accretion cloud ice - rain
    tmp1 =
        CM1M.accretion(params, CM1M.IceType(), CM1M.RainType(), q.ice, q_rai, ρ)
    # sink of rain via accretion cloud ice - rain
    tmp2 = CM1M.accretion_rain_sink(params, q.ice, q_rai, ρ)
    #S_qi -= tmp1
    S_e_tot -= tmp1 * (I_i + Φ)
    S_q_rai -= tmp2
    S_e_tot += tmp2 * L_f
    S_q_sno += tmp1 + tmp2

    # accretion rain - snow
    if T < _T_freeze
        tmp = CM1M.accretion_snow_rain(
            params,
            CM1M.SnowType(),
            CM1M.RainType(),
            q_sno,
            q_rai,
            ρ,
        )
        S_q_sno += tmp
        S_q_rai -= tmp
        S_e_tot += tmp * L_f
    else
        tmp = CM1M.accretion_snow_rain(
            params,
            CM1M.RainType(),
            CM1M.SnowType(),
            q_rai,
            q_sno,
            ρ,
        )
        S_q_sno -= tmp
        S_q_rai += tmp
        S_e_tot -= tmp * L_f
    end

    # rain evaporation sink (it already has negative sign for evaporation)
    tmp = CM1M.evaporation_sublimation(params, CM1M.RainType(), q, q_rai, ρ, T)
    S_q_rai += tmp
    S_e_tot -= tmp * (I_l + Φ)

    # snow sublimation/deposition source/sink
    tmp = CM1M.evaporation_sublimation(params, CM1M.SnowType(), q, q_sno, ρ, T)
    S_q_sno += tmp
    S_e_tot -= tmp * (I_i + Φ)

    # snow melt
    tmp = CM1M.snow_melt(params, q_sno, ρ, T)
    S_q_sno -= tmp
    S_q_rai += tmp
    S_e_tot -= tmp * L_f

    # total qt sink is the sum of precip sources
    S_q_tot = -S_q_rai - S_q_sno

    # temporarily dumping q.liq and q.ice into cache
    # for a quick way to visualise them in tests
    q_liq = q.liq
    q_ice = q.ice

    return (; S_q_tot, S_e_tot, S_q_rai, S_q_sno, q_liq, q_ice)
end

@inline function precompute_cache!(dY, Y, Ya, _...)
    error("not implemented for this model configuration.")
end

@inline function precompute_cache!(
    dY,
    Y,
    Ya,
    ::PotentialTemperature,
    ::Dry,
    ::NoPrecipitation,
    params,
    FT,
)
    ρ = Y.base.ρ
    ρθ = Y.thermodynamics.ρθ

    z = Fields.coordinate_field(axes(ρ)).z
    g::FT = CLIMAParameters.Planet.grav(params)

    # update cached gravitational potential (TODO - should be done only once)
    @. Ya.Φ = g * z
    # TODO: save ts into cache
    @. Ya.p = TD.air_pressure(params, TD.PhaseDry_ρθ(params, ρ, ρθ / ρ))
end

@inline function precompute_cache!(
    dY,
    Y,
    Ya,
    ::TotalEnergy,
    ::EquilibriumMoisture,
    ::PrecipitationRemoval,
    params,
    FT,
)
    # unpack state variables
    ρ = Y.base.ρ
    ρe_tot = Y.thermodynamics.ρe_tot
    ρq_tot = Y.moisture.ρq_tot

    z = Fields.coordinate_field(axes(ρ)).z
    g::FT = CLIMAParameters.Planet.grav(params)

    cρuₕ = Y.base.ρuh # Covariant12Vector on centers
    fρw = Y.base.ρw # Covariant3Vector on faces
    If2c = Operators.InterpolateF2C()
    cuvw =
        Geometry.Covariant123Vector.(cρuₕ ./ ρ) .+
        Geometry.Covariant123Vector.(If2c.(fρw) ./ ρ)

    # update cached gravitational potential (TODO - should be done only once)
    @. Ya.Φ = g * z
    # update cached kinetic energy
    @. Ya.K = norm_sqr(cuvw) / 2
    # update cached internal energy
    @. Ya.e_int = ρe_tot / ρ - Ya.Φ - Ya.K

    # update cached pressure
    @. Ya.p = TD.air_pressure(
        params,
        TD.PhaseEquil_ρeq(params, ρ, Ya.e_int, ρq_tot / ρ),
    )

    # update cached microphysics helper variables
    @. Ya.microphysics_cache =
        precompute_microphysics_0M(ρq_tot, ρ, Ya.e_int, Ya.Φ, $Ref(params))
end

@inline function precompute_cache!(
    dY,
    Y,
    Ya,
    ::TotalEnergy,
    ::EquilibriumMoisture,
    ::OneMoment,
    params,
    FT,
)
    # unpack state variables
    ρ = Y.base.ρ
    ρe_tot = Y.thermodynamics.ρe_tot
    ρq_tot = Y.moisture.ρq_tot
    ρq_rai = Y.precipitation.ρq_rai
    ρq_sno = Y.precipitation.ρq_sno

    z = Fields.coordinate_field(axes(ρ)).z
    g::FT = CLIMAParameters.Planet.grav(params)

    cρuₕ = Y.base.ρuh # Covariant12Vector on centers
    fρw = Y.base.ρw # Covariant3Vector on faces
    If2c = Operators.InterpolateF2C()
    cuvw =
        Geometry.Covariant123Vector.(cρuₕ ./ ρ) .+
        Geometry.Covariant123Vector.(If2c.(fρw) ./ ρ)

    # update cached gravitational potential (TODO - should be done only once)
    @. Ya.Φ = g * z
    # update cached kinetic energy
    @. Ya.K = norm_sqr(cuvw) / 2
    # update cached internal energy
    @. Ya.e_int = ρe_tot / ρ - Ya.Φ - Ya.K

    # update cached pressure
    @. Ya.p = TD.air_pressure(
        params,
        TD.PhaseEquil_ρeq(params, ρ, Ya.e_int, ρq_tot / ρ),
    )

    # update cached microphysics helper variables
    @. Ya.microphysics_cache = precompute_microphysics_1M(
        ρq_tot,
        ρq_rai,
        ρq_sno,
        ρ,
        Ya.e_int,
        Ya.Φ,
        $Ref(params),
    )
end
