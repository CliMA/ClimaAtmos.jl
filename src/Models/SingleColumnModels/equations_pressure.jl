@inline function calculate_pressure(Y, Ya, _...)
    error("not implemented for this model configuration.")
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::AbstractBaseModelStyle,
    ::PotentialTemperature,
    ::Dry,
    params,
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
