@inline function precompute_microphysics(ρq_tot, ρ, e_int, Φ, params)

    # saturation adjustment
    q_tot = ρq_tot / ρ
    ts = Thermodynamics.PhaseEquil_ρeq(params, ρ, e_int, q_tot)
    q = Thermodynamics.PhasePartition(ts)
    λ = Thermodynamics.liquid_fraction(ts)
    I_l = Thermodynamics.internal_energy_liquid(ts)
    I_i = Thermodynamics.internal_energy_ice(ts)

    # precipitation removal source terms
    # (cached to avoid re-computing many times per time step)
    S_q_tot = CloudMicrophysics.Microphysics_0M.remove_precipitation(params, q)
    S_e_tot = (λ * I_l + (1 - λ) * I_i + Φ) * S_q_tot

    # temporarily dumping q.liq and q.ice into cache
    # for a quick way to visualise them in tests
    q_liq = q.liq
    q_ice = q.ice

    return (; S_q_tot, S_e_tot, q_liq, q_ice)
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
    # save pressure into cache
    @. Ya.p = Thermodynamics.air_pressure(Thermodynamics.PhaseDry_ρθ(
        params,
        ρ,
        ρθ / ρ,
    ))
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
    @. Ya.p = Thermodynamics.air_pressure(Thermodynamics.PhaseEquil_ρeq(
        params,
        ρ,
        Ya.e_int,
        ρq_tot / ρ,
    ))

    # update cached microphysics helper variables
    @. Ya.microphysics_cache =
        precompute_microphysics(ρq_tot, ρ, Ya.e_int, Ya.Φ, $Ref(params))
end
