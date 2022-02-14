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
    # parameters
    cp_d::FT = CLIMAParameters.Planet.cp_d(params)
    cv_d::FT = CLIMAParameters.Planet.cv_d(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    p_0::FT = CLIMAParameters.Planet.MSLP(params)

    ρ = Y.base.ρ
    ρθ = Y.thermodynamics.ρθ

    p = @. cp_d * (R_d * ρθ / p_0)^(R_d / cv_d)

    return p
end
