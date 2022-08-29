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
    thermo_params = CAP.thermodynamics_params(params)
    ρ = Y.base.ρ
    ρθ = Y.thermodynamics.ρθ

    p = @. TD.air_pressure(
        thermo_params,
        TD.PhaseDry_ρθ(thermo_params, ρ, ρθ / ρ),
    )

    return p
end
