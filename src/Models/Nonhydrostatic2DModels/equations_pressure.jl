@inline function calculate_pressure(Y, Ya, _...)
    error("not implemented for this model configuration.")
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::PotentialTemperature,
    ::Dry,
    params,
    Φ,
    K,
    FT,
)
    ρ = Y.base.ρ
    ρθ = Y.thermodynamics.ρθ

    p = @. Thermodynamics.air_pressure(Thermodynamics.PhaseDry_ρθ(
        params,
        ρ,
        ρθ / ρ,
    ))

    return p
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::TotalEnergy,
    ::EquilibriumMoisture,
    params,
    Φ,
    K,
    FT,
)
    ρ = Y.base.ρ
    ρe_tot = Y.thermodynamics.ρe_tot
    ρq_tot = Y.moisture.ρq_tot
    e_int = @. ρe_tot / ρ - K - Φ

    p = @. Thermodynamics.air_pressure(Thermodynamics.PhaseEquil_ρeq(
        params,
        ρ,
        e_int,
        ρq_tot / ρ,
    ))

    return p

end
