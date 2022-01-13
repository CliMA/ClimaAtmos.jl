@inline function calculate_pressure(Y, Ya, _...)
    error("not implemented for this model configuration.")
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::PotentialTemperature,
    ::Dry,
    params,
    FT,
)
    ρ = Y.base.ρ
    ρθ = Y.thermodynamics.ρθ

    ts = @. Thermodynamics.PhaseDry_ρθ(params, ρ, ρθ / ρ)
    p = @. Thermodynamics.air_pressure(ts)

    return p
end
