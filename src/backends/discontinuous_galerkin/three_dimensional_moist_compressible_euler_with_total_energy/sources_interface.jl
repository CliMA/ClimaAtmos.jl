@inline calc_source!(source, ::ThreeDimensionalMoistCompressibleEulerWithTotalEnergy, _...) = nothing

@inline function calc_source!(
    source,
    balance_law::ThreeDimensionalMoistCompressibleEulerWithTotalEnergy, 
    ::DeepShellCoriolis, 
    state, 
    aux
)
    ρu = state.ρu

    Ω  = @SVector [-0, -0, balance_law.parameters.Ω]

    source.ρu += -2Ω × ρu

    nothing
end

@inline function calc_source!(
    source,
    balance_law::ThreeDimensionalMoistCompressibleEulerWithTotalEnergy,  
    ::ZeroMomentMicrophysics, 
    state, 
    aux
) 
    ρ    = state.ρ
    ρq   = state.ρq
    Φ    = aux.Φ
    eos  = balance_law.equation_of_state
    τ    = balance_law.parameters.τ_precip
    T_0  = balance_law.parameters.T_0 
    cv_l = balance_law.parameters.cv_l
  
    # we need the saturation excess in order to calculate the 
    # source terms for prognostic variables
    T    = calc_air_temperature(eos, state, aux, balance_law.parameters)
    qᵥₛ  = calc_saturation_specific_humidity(ρ, T, balance_law.parameters) 
    ρS   = max(0, ρq - ρ * qᵥₛ) # saturation excess
    Iₗ   = cv_l * (T - T_0) # liquid internal energy
  
    # source terms are proportional to the saturation excess
    source.ρ  -= ρS / τ
    source.ρe -= (Iₗ + Φ) * ρS / τ
    source.ρq -= ρS / τ
  end